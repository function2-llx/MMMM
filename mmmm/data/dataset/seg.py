from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import h5py
import numpy as np
from numpy import typing as npt
import pandas as pd
import torch
from torch.types import Device

from luolib.types import param3_t, tuple2_t, tuple3_t
from luolib.utils.misc import ensure_rgb
from monai import transforms as mt
from monai.data import MetaTensor
from monai.utils import ensure_tuple_rep

from mmmm.models import MMMMTokenizer
import mmmm.data.dataset._dataset as _dataset
from ..utils import prepare_vlm_inputs
from ..defs import PROCESSED_SEG_DATA_ROOT, Sparse, DataPoint, split_t

def get_seg_data_list(name: str, split: split_t):
    if split != 'train':
        raise NotImplementedError
    dataset_dir = PROCESSED_SEG_DATA_ROOT / name
    info = pd.read_csv(dataset_dir / 'info.csv', dtype={'key': 'string'})
    info.set_index('key', inplace=True)
    return [
        {
            'dataset_dir': dataset_dir,
            'key': key,
        }
        for key in info.index
    ]

@dataclass(kw_only=True)
class SegTransConf:
    scale_z: tuple2_t[float] = (0.5, 1.5)
    scale_z_p: float = 0.25
    scale_xy: tuple2_t[float] = (0.5, 1.5)
    base_vit_patch_size: param3_t[int] = 16
    num_pos: int = 40  # I've encountered cases setting this to larger than 48 causing NCCL timeout
    num_neg: int = 4

    force_fg_ratio: float = 0.9

def get_seg_transform(
    conf: _dataset.DatasetConf,
    tokenizer: MMMMTokenizer,
    inference: bool,
) -> Callable[[dict], DataPoint]:
    return mt.Compose([
        LoadSparse(),
        SamplePatch(
            max_vision_tokens=conf.max_vision_tokens,
            conf=conf.seg_trans,
            tokenizer=tokenizer,
            inference=inference,
        ),
        InputTransformD(),
    ])

class InputTransformD(mt.Transform):
    def __call__(self, data: dict) -> DataPoint:
        data = dict(data)
        img = data['image']
        if isinstance(img, MetaTensor):
            img = img.as_tensor()
        data['image'], _ = ensure_rgb(img)
        masks = data['masks']
        if isinstance(masks, MetaTensor):
            masks = masks.as_tensor()
        if masks.dtype != torch.bool:
            masks = masks.round().bool()
        data['masks'] = masks
        return data

class SamplePatch(mt.RandomizableTransform):
    def __init__(
        self,
        max_vision_tokens: int,
        conf: SegTransConf,
        tokenizer: MMMMTokenizer,
        inference: bool,
        device: Device = 'cpu',
    ):
        super().__init__()
        self.max_vision_tokens = max_vision_tokens
        self.conf = conf
        self.tokenizer = tokenizer
        self.device = device
        self.base_vit_patch_size: tuple3_t[int] = ensure_tuple_rep(base_vit_patch_size, 3)
        self.inference = inference

    def __call__(self, data: dict):
        """
        1. determine scale, sample patch size, vit patch size
        2. determine crop center
        3. (later) flip, rotate 90 (isotropic plane)
        Returns:
        """
        sparse: Sparse = data['sparse']

        # sample patch position
        position: npt.NDArray[np.int16]
        if self.R.uniform() < self.force_fg_ratio:
            # foreground oversampling
            c = self.R.randint(len(image_pos_classes))
            with h5py.File(patch_dir / 'class_positions.h5') as f:
                positions: h5py.Dataset = f[str(c)]
                position = positions[self.R.randint(positions.shape[0])]
        else:
            # sample a random patch position
            c = None
            position = np.array([self.R.randint(s) for s in patches_class_mask.shape[:-1]], dtype=np.int16)

        # sample negative classes
        neg_class_ids, = (~pos_class_mask).nonzero()
        # all negative classes for this patch:
        # - positive classes in the whole image but not in this patch
        # - negative classes for the whole image
        neg_classes: list[str] = [image_pos_classes[i] for i in neg_class_ids] + meta['negative_classes']
        neg_classes = self.R.choice(neg_classes, min(len(neg_classes), self.num_neg), replace=False).tolist()

        # sample positive classes
        if c is not None:
            pos_class_mask[c] = False
        pos_class_ids, = pos_class_mask.nonzero()
        num_pos = self.num_pos - (c is not None)
        pos_class_ids = self.R.choice(pos_class_ids, min(pos_class_ids.shape[0], num_pos), replace=False)
        if c is not None:
            pos_class_ids = np.insert(pos_class_ids, 0, c)
        pos_classes: list[str] = [image_pos_classes[i] for i in pos_class_ids]

        # construct image & masks patch
        data_dir = dataset_dir / 'data' / key
        modalities = meta['modalities']
        modality_id = self.R.randint(len(modalities))
        patch_slice = [slice(p, p + s) for p, s in zip(position, self.patch_size)]
        # TODO: support RGB
        image = np.load(data_dir / 'images.npy', 'r')[modality_id:modality_id + 1, *patch_slice]
        image = torch.as_tensor(np.array(image), device=self.device)
        pos_masks = np.load(data_dir / 'masks.npy', 'r')[:, *patch_slice]
        pos_masks = pos_masks[pos_class_ids]
        pos_masks = torch.as_tensor(np.array(pos_masks), device=self.device)

        # prepare sample output
        modality = modalities[modality_id]
        conversation, mask_classes = gen_seg_conversation(
            modality, pos_classes, neg_classes, self.tokenizer, self.R, inference=self.inference,
        )
        masks = pos_masks.new_zeros((len(mask_classes), *pos_masks.shape[1:]))
        pos_class_to_idx = {name: i for i, name in enumerate(pos_classes)}
        for i, name in enumerate(mask_classes):
            if (pos_idx := pos_class_to_idx.get(name, -1)) != -1:
                masks[i] = pos_masks[pos_idx]

        vlm_inputs, conversation_text = prepare_vlm_inputs(
            conversation, self.tokenizer, self.patch_size, self.vit_patch_size, self.inference,
        )
        if np.less(image.shape[1:], self.patch_size).any():
            mean, std = meta['mean'], meta['std']
            padder = mt.SpatialPad(self.patch_size)
            image = torch.cat([
                padder(image[i:i + 1], value=-mean[i] / std[i])
                for i in range(image.shape[0])
            ])
            masks = padder(masks)
        data = {
            'image': image,
            'spacing': sparse.spacing,
            'modality': modality,
            'masks': masks,
            'mask_classes': mask_classes,
            'vlm_inputs': vlm_inputs,
        }
        return data

class LoadSparse(mt.Transform):
    def __call__(self, data: Mapping):
        dataset_dir: Path = data['dataset_dir']
        key = data['key']
        data = dict(data)
        data['sparse'] = Sparse.from_json((dataset_dir / f'data/{key}/sparse.json').read_bytes())
        return data

def gen_seg_conversation(
    modality: str,
    pos_classes: list[str],
    neg_classes: list[str],
    tokenizer: MMMMTokenizer,
    R: np.random.RandomState | int,
    p_use_neg_mask: float = 1.,
    inference: bool = False,
):
    def _convert_list(names: Iterable[str], mask: bool):
        # FIXME: do not use special tokens explicitly in text
        if mask:
            names = map(tokenizer.wrap_name, names)
        return ', '.join(names)

    if isinstance(R, int):
        R = np.random.RandomState(R)

    # copy the input list because the shuffling is in-place
    pos_classes = list(pos_classes)
    R.shuffle(pos_classes)
    neg_classes = list(neg_classes)
    R.shuffle(neg_classes)

    # merge positive and negative classes with random order without shuffling
    pos_class_mask = torch.zeros(len(pos_classes) + len(neg_classes), dtype=torch.bool)
    pos_class_mask[R.choice(pos_class_mask.shape[0], len(pos_classes), replace=False)] = True
    pos_it, neg_it = map(iter, [pos_classes, neg_classes])
    classes = [
        next(pos_it) if m else next(neg_it)
        for m in pos_class_mask
    ]

    assert len(classes) > 0
    neg_mask = R.uniform() < p_use_neg_mask
    if neg_mask:
        prompt = f'For the given {modality} image, output the segmentation masks for the following objects: {_convert_list(classes, False)}.'
    else:
        prompt = f'For the given {modality} image, find the following objects, and output segmentation masks for the found objects: {_convert_list(classes, False)}. '
    if len(pos_classes) > 0:
        if len(neg_classes) > 0:
            response = f'The following objects are found: {_convert_list(pos_classes, True)}. ' + \
                       f'The following objects are not found: {_convert_list(neg_classes, neg_mask)}. '
            mask_classes = pos_classes + (neg_classes if neg_mask else [])
        else:
            response = f'All of the requested objects are found: {_convert_list(classes, True)}. '
            mask_classes = classes
    else:
        if neg_mask:
            response = f'None of the requested objects are found: {_convert_list(classes, True)}. '
            mask_classes = classes
        else:
            response = 'None of the requested objects are found. '
            mask_classes = []
    # mask_classes: the list of classes that with masks, following the order occurring in the conversation
    if inference:
        # TODO: refactor this function
        response = ''
    return [(prompt, response)], mask_classes
