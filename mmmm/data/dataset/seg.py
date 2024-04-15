from __future__ import annotations as _

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import einops
import numpy as np
import pandas as pd
import torch
from torch.types import Device

from luolib.types import tuple2_t
from luolib.utils import load_pt_zst
from monai import transforms as mt

from mmmm.models import MMMMTokenizer
import mmmm.data.dataset._dataset as _dataset
from ..defs import ConvTurn, DataPoint, PROCESSED_SEG_DATA_ROOT, Sparse, split_t
from ..utils import prepare_vlm_inputs
from .modality import gen_modality_conversation

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
    scale_z: tuple2_t[float]
    scale_z_p: float
    max_tokens_z: int
    scale_xy: tuple2_t[float]
    scale_xy_p: float
    aniso_ratio_range: tuple2_t[float] = (0.5, 2.)
    log_vit_patch_size_z_std = 0.25  # 2-sigma, 95.45%
    num_pos: int  # I've encountered cases setting this to larger than 48 causing NCCL timeout
    num_neg: int
    force_fg_ratio: float

def get_seg_transform(
    conf: _dataset.DatasetConf,
    tokenizer: MMMMTokenizer,
    inference: bool,
) -> Callable[[dict], DataPoint]:
    return mt.Compose([
        SamplePatch(conf, tokenizer, inference),
        # InputTransformD(),
    ])

def toss(R: np.random.RandomState, prob: float):
    return R.uniform() < prob

class SamplePatch(mt.Randomizable):
    def __init__(
        self,
        conf: _dataset.DatasetConf,
        tokenizer: MMMMTokenizer,
        inference: bool,
        device: Device = 'cpu',
    ):
        super().__init__()
        self.conf = conf
        self.tokenizer = tokenizer
        self.device = device
        self.inference = inference

    def gen_patch_info(self, sparse):
        conf = self.conf
        trans_conf = conf.seg_trans
        # 1. sample tokens_z, tokens_xy, thus obtain patch_size_xy
        if sparse.shape[0] == 1:
            tokens_z = 1
        else:
            # TODO: maybe there's a better approximation for tokens_z
            tokens_z = self.R.randint(1, trans_conf.max_tokens_z + 1)
        tokens_xy = int((conf.max_vision_tokens / tokens_z) ** 0.5)
        patch_size_xy = tokens_xy * conf.vit_patch_size_xy
        # 2. sample scale_xy
        if (max_scale_xy := max(sparse.shape[1:]) / patch_size_xy) <= trans_conf.scale_xy[0]:
            # the original image is too small, just resize to patch size
            scale_xy = max_scale_xy
        elif toss(self.R, trans_conf.scale_xy_p):
            scale_xy = self.R.uniform(
                trans_conf.scale_xy[0],
                min(trans_conf.scale_xy[1], max_scale_xy),
            )
        else:
            scale_xy = 1.
        # 3. sample scale_z
        if sparse.spacing[0] < 2 * min(sparse.spacing[1:]):
            # initialize with isotropic scale
            scale_z = scale_xy
            if toss(self.R, trans_conf.scale_z_p):
                scale_z *= self.R.uniform(
                    max(
                        trans_conf.scale_z[0],
                        trans_conf.aniso_ratio_range[0] * min(sparse.spacing[1:]) / sparse.spacing[0],
                    ),
                    min(
                        trans_conf.scale_z[1],
                        trans_conf.aniso_ratio_range[1] * min(sparse.spacing[1:]) / sparse.spacing[0],
                    ),
                )
        else:
            scale_z = 1.
        # 4. determine vit_patch_size_z
        if sparse.shape[0] == 1:
            vit_patch_size_z = 1
        else:
            spacing_xy = min(sparse.spacing[1:]) * scale_xy
            spacing_z = sparse.spacing[0] * scale_z
            log_vit_patch_size_z = self.R.normal(
                np.log2(conf.base_vit_patch_size_z * spacing_xy / spacing_z),
                trans_conf.log_vit_patch_size_z_std,
            )
            log_vit_patch_size_z = np.clip(
                np.rint(log_vit_patch_size_z), 0, conf.base_vit_patch_size_z.bit_length() - 1,
            )
            vit_patch_size_z = 1 << int(log_vit_patch_size_z)
        vit_patch_size = np.array((vit_patch_size_z, conf.vit_patch_size_xy, conf.vit_patch_size_xy))
        scale = np.array((scale_z, scale_xy, scale_xy))
        patch_size = np.array((tokens_z * vit_patch_size_z, patch_size_xy, patch_size_xy))
        return patch_size, scale, vit_patch_size

    def get_patch_start(self, maybe_patch_center: np.ndarray, effective_patch_size: np.ndarray, shape: np.ndarray):
        # TODO: add randomization
        patch_start = maybe_patch_center - (effective_patch_size >> 1)
        patch_start = np.clip(patch_start, 0, shape - effective_patch_size)
        return patch_start

    def _is_positive(self, patch_mask_size: int, mask_size: int):
        """check should a mask be considered positive in a patch"""
        return 2 * patch_mask_size >= mask_size

    def __call__(self, data: dict) -> DataPoint:
        data = dict(data)
        conf = self.conf
        trans_conf = conf.seg_trans
        data_dir = data['dataset_dir'] / 'data' / data['key']
        sparse = Sparse.from_json((data_dir / 'sparse.json').read_bytes())
        annotation = sparse.annotation
        # 1. generate patch information
        patch_size, scale, vit_patch_size = self.gen_patch_info(sparse)
        patch_tokens, _rem = np.divmod(patch_size, vit_patch_size)
        assert np.array_equiv(_rem, 0)
        effective_patch_size = np.minimum(np.ceil(patch_size * scale).astype(np.int64), sparse.shape)
        # 2. sample patch position
        if self.R.uniform() < trans_conf.force_fg_ratio:
            # foreground oversampling
            # TODO: handle data with bbox only
            c = self.R.randint(len(annotation.mask))
            # use str for mmap, will be fixed in PyTorch 2.3: https://github.com/pytorch/pytorch/pull/116104
            class_positions: torch.Tensor = torch.load(str(data_dir / 'class_positions.pt'), mmap=True)
            class_offsets: torch.Tensor = torch.load(data_dir / 'class_offsets.pt')
            position_idx = self.R.randint(class_offsets[c], class_offsets[c + 1])
            maybe_patch_center = class_positions[position_idx].numpy()
            patch_start = self.get_patch_start(maybe_patch_center, effective_patch_size, sparse.shape)
        else:
            patch_start = self.R.randint(sparse.shape - effective_patch_size)
        patch_slice = [
            slice(start, start + size)
            for start, size in zip(patch_start, effective_patch_size)
        ]
        # 3. crop patch & mask (without further affine transform)
        if len(sparse.modalities) == 1:
            modality = sparse.modalities[0]
            modality_slice = slice(None)
        else:
            # NOTE: currently, it is assumed that there will not be multiple RGB images
            modality_idx = self.R.randint(len(sparse.modalities))
            modality = sparse.modalities[modality_idx]
            modality_slice = slice(modality_idx, modality_idx + 1)
        images: torch.HalfTensor = torch.load(str(data_dir / 'images.pt'), mmap=True)
        patch = images[modality_slice, *patch_slice]
        masks: torch.BoolTensor = load_pt_zst(data_dir / 'masks.pt.zst')
        patch_masks = masks[:, *patch_slice]
        # TODO: crop bbox
        # 4. determine positive & negative classes within the cropped patch
        patch_mask_sizes: list[int] = einops.reduce(patch_masks, 'c ... -> c', 'sum').tolist()
        if len(dict(annotation.mask)) != len(annotation.mask):
            raise NotImplementedError
        anatomy_pos, anatomy_neg = [], list(sparse.anatomy.neg)
        anomaly_pos, anomaly_neg = [], list(sparse.anomaly.neg)
        for c, (name, mask_size) in enumerate(annotation.mask):
            patch_mask_size = patch_mask_sizes[c]
            if self._is_positive(patch_mask_size, mask_size):
                if name in sparse.anatomy.pos:
                    anatomy_pos.append(name)
                else:
                    anomaly_pos.append(name)
            elif patch_mask_size == 0:
                if name in sparse.anatomy.pos:
                    anatomy_neg.append(name)
                else:
                    anomaly_neg.append(name)
        # restricted number of classes for anatomy
        if len(anatomy_pos) > trans_conf.num_pos:
            anatomy_pos = self.R.choice(anatomy_pos, trans_conf.num_pos, replace=False).tolist()
        if len(anatomy_neg) > trans_conf.num_neg:
            anatomy_neg = self.R.choice(anatomy_neg, trans_conf.num_neg, replace=False).tolist()
        # 5. apply affine transform!
        affine_trans = mt.Compose(
            [
                *[
                    mt.RandFlipD(['image', 'masks'], 0.5, i)
                    for i in range(3)
                ],
                mt.RandRotate90D(['image', 'masks'], 0.5, spatial_axes=(1, 2)),
                mt.AffineD(
                    ['image', 'masks'],
                    scale_params=scale.tolist(),
                    spatial_size=patch_size.tolist(),
                ),
                mt.ToTensor(track_meta=False),
            ],
            lazy=True,
        )
        affine_trans.set_random_state(state=self.R)
        _dict_data = affine_trans({'image': patch, 'masks': patch_masks})
        patch, patch_masks = _dict_data['image'], _dict_data['masks'].round().bool()
        # 6. generate conversation
        conversation = gen_modality_conversation(modality, self.R)
        mask_classes = []
        conversation_anatomy, mask_classes_anatomy = gen_anatomy_conversation(
            anatomy_pos, anatomy_neg, self.tokenizer, self.R,
        )
        conversation.extend(conversation_anatomy)
        mask_classes.extend(mask_classes_anatomy)
        conversation_anomaly, mask_classes_anomaly = gen_anomaly_conversation(
            anomaly_pos, anomaly_neg, sparse.anomaly.complete, self.tokenizer, self.R,
        )
        conversation.extend(conversation_anomaly)
        mask_classes.extend(mask_classes_anomaly)
        vlm_inputs, conversation_text = prepare_vlm_inputs(
            conversation, self.tokenizer, patch_tokens.prod(), self.inference,
        )
        # TODO: handle normalization
        if patch.shape[0] == 1:
            # ensure RGB
            patch = einops.repeat(patch, '1 ... -> c ...', c=3).contiguous()
        data_point: DataPoint = {
            'image': patch,
            # TODO: apply transform on grounding image
            'grounding_image': patch,
            'patch_size': tuple(vit_patch_size.tolist()),
            'vlm_inputs': vlm_inputs,
            'mask': self._create_mask_label(annotation.mask, mask_classes, patch, patch_masks),
            'mask_index': torch.ones(len(mask_classes), dtype=torch.bool),
            'bbox': torch.empty(0, 2, 3),
            'bbox_index': torch.zeros(0, dtype=torch.bool),
        }
        return data_point

    def _create_mask_label(
        self,
        mask: list[tuple[str, int]],
        mask_classes: list[str],
        patch: torch.Tensor,
        patch_masks: torch.BoolTensor,
    ) -> torch.BoolTensor:
        class_to_idx = {name: i for i, (name, _) in enumerate(mask)}
        mask_label = torch.zeros(len(mask_classes), *patch.shape[1:], dtype=torch.bool)
        for i, mask_class in enumerate(mask_classes):
            if (c := class_to_idx.get(mask_class)) is not None:
                mask_label[i] = patch_masks[c]
        return mask_label

# class InputTransformD(mt.Transform):
#     def __call__(self, data: dict) -> DataPoint:
#         data = dict(data)
#         img = data['image']
#         if isinstance(img, MetaTensor):
#             img = img.as_tensor()
#         data['image'], _ = ensure_rgb(img)
#         masks = data['masks']
#         if isinstance(masks, MetaTensor):
#             masks = masks.as_tensor()
#         if masks.dtype != torch.bool:
#             masks = masks.round().bool()
#         data['masks'] = masks
#         return data

def gen_anatomy_conversation(
    pos_classes: list[str],
    neg_classes: list[str],
    tokenizer: MMMMTokenizer,
    R: np.random.RandomState,
    p_use_neg_mask: float = 1.,
) -> tuple[list[ConvTurn], list[str]]:
    """
    Returns:
      - conversation
      - class names, following the order occurring in the conversation
    """
    def _convert_list(names: Iterable[str], mask: bool):
        # FIXME: do not use special tokens explicitly in text
        if mask:
            names = map(tokenizer.wrap_name, names)
        return ', '.join(names)

    if len(pos_classes) == 0 and len(neg_classes) == 0:
        return [], []
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
        prompt = f'Output the segmentation masks for the following objects: {_convert_list(classes, False)}.'
    else:
        prompt = f'Find the following objects, and output segmentation masks for the found objects: {_convert_list(classes, False)}. '
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
    return [ConvTurn(prompt, response)], mask_classes

def gen_anomaly_conversation(
    pos_classes: list[str],
    neg_classes: list[str],
    _complete: bool,
    tokenizer: MMMMTokenizer,
    R: np.random.RandomState | int,
    p_use_neg_mask: float = 1.,
):
    return gen_anatomy_conversation(pos_classes, neg_classes, tokenizer, R, p_use_neg_mask)
