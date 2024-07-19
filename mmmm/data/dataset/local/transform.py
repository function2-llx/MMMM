from __future__ import annotations as _

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import cytoolz
import einops
import math
import numpy as np
import numpy.typing as npt
import orjson
import pandas as pd
import torch
from torch.types import Device
import torchvision.transforms.v2.functional as tvtf

from luolib.transforms.box_ops import apply_affine_to_boxes_int
from luolib.types import tuple2_t, tuple3_t
from luolib.utils import load_pt_zst
from luolib.utils.misc import ceil_divide
from monai import transforms as mt
from monai.data import MetaTensor
from monai.data.box_utils import CenterSizeMode, box_area, convert_box_mode, spatial_crop_boxes
from monai.transforms import generate_spatial_bounding_box
from monai.utils import GridSamplePadMode, InterpolateMode

import mmmm.data.dataset._dataset as _dataset
from mmmm.data.defs import DataPoint, PROCESSED_LOCAL_DATA_ROOT, mmmm_debug, Split
from mmmm.data.sparse import Sparse
from mmmm.data.target_tax import TargetCategory, get_target_tax
from mmmm.data.utils import prepare_vlm_inputs
from mmmm.tokenizer import MMMMTokenizer
from ..misc import gen_modality_conv, get_max_scale_for_size, intensity_norm, toss, get_max_resize
from .template import gen_anomaly_conv, gen_general_conv

__all__ = [
    'get_local_data_list',
    'get_local_transform',
    'LocalTransConf',
]

def get_local_data_list(name: str, split: Split):
    dataset_dir = PROCESSED_LOCAL_DATA_ROOT / name
    if (split_path := dataset_dir / 'split.json').exists():
        split_dict = orjson.loads(split_path.read_bytes())
        keys = set(split_dict[split])
    else:
        keys = None
        if split != Split.TRAIN:
            raise ValueError
    info = pd.read_csv(dataset_dir / 'info.csv', dtype={'key': 'string'})
    info.set_index('key', inplace=True)
    return [
        {
            'dataset': name,
            'dataset_dir': dataset_dir,
            'key': key,
        }
        for key in info.index if keys is None or key in keys
    ]

@dataclass(kw_only=True)
class LocalTransConf:
    """
    Attributes:
        neg_grounding_prob: the probability of negative targets are forced grounded
        vlm: whether the transform is performed for VLM
    """
    max_vision_tokens: int
    max_tokens_z: int
    log2_patch_size_z_std = 0.25  # 2-sigma, 95.45%
    num_pos: int
    num_neg: int
    modality_prob: float = 0.8
    grounding_prob: float = 0.99
    neg_grounding_prob: float = 0.2

def get_local_transform(
    conf: _dataset.DatasetConf,
    tokenizer: MMMMTokenizer | None,
    inference: bool,
) -> Callable[[dict], DataPoint]:
    return LocalTransform(conf, tokenizer, inference)

def _get_patch_size_xy(size: npt.NDArray[np.int64], scale: float, stride: int, max_tokens: int) -> tuple2_t[int]:
    size_scaled = size / scale
    # smaller_size, larger_size = size_scaled.sort()
    smaller_idx = size_scaled.argmin()
    max_smaller_tokens = math.floor(max_tokens ** 0.5)
    smaller_tokens = ceil_divide(size_scaled[smaller_idx], stride)
    if smaller_tokens > max_smaller_tokens:
        patch_size = max_smaller_tokens * stride
        return patch_size, patch_size
    larger_tokens = min(max_tokens // smaller_tokens, ceil_divide(size_scaled[smaller_idx ^ 1], stride).astype(np.int64))
    ret = np.empty(2, dtype=np.int64)
    ret[smaller_idx] = smaller_tokens * stride
    ret[smaller_idx ^ 1] = larger_tokens * stride
    return tuple(ret.tolist())

def norm_boxes(boxes: torch.LongTensor, norm_size: Sequence[int]) -> torch.DoubleTensor:
    norm_size_t = einops.repeat(torch.tensor(norm_size), 'd -> (l2 d)', l2=2)
    boxes_normed = boxes.double() / norm_size_t
    boxes_normed = convert_box_mode(boxes_normed, dst_mode=CenterSizeMode)
    return boxes_normed

class LocalTransform(mt.Randomizable):
    def __init__(
        self,
        conf: _dataset.DatasetConf,
        tokenizer: MMMMTokenizer | None,
        inference: bool,
        device: Device = 'cpu',
    ):
        super().__init__()
        self.conf = conf
        self.trans_conf = conf.local_trans
        self.tokenizer = tokenizer
        self.device = device
        self.inference = inference
        self.target_tax = get_target_tax()

    def _get_category(self, name: str):
        return self.target_tax[name].category

    def _sample_targets(self, targets: Iterable[str], limit: int, category: str | None = None) -> list[str]:
        if category is None:
            targets = list(targets)
        else:
            targets = [*filter(lambda target: self._get_category(target) == category, targets)]
        if len(targets) > limit:
            targets = self.R.choice(targets, limit, replace=False).tolist()
        return targets

    def _spatial_transform(
        self,
        image: torch.Tensor,
        masks: torch.BoolTensor | None,
        boxes: torch.LongTensor | None,
        resize: tuple3_t[int],
        stride: tuple3_t[int],
    ) -> tuple[torch.Tensor, torch.BoolTensor | None, torch.LongTensor | None]:
        keys = ['image']
        if masks is not None:
            keys.append('masks')
        affine_trans = mt.Compose(
            [
                mt.ResizeD(keys, resize, mode=InterpolateMode.TRILINEAR),
                mt.DivisiblePadD(keys, stride),
                *[
                    mt.RandFlipD(keys, 0.5, i)
                    for i in range(3)
                ],
                mt.RandRotate90D(keys, 0.75, spatial_axes=(1, 2)),
            ],
            lazy=True,
            overrides={
                'image': {'padding_mode': GridSamplePadMode.ZEROS},
                'masks': {'padding_mode': GridSamplePadMode.ZEROS},
            }
        )
        affine_trans.set_random_state(state=self.R)
        _dict_data = {'image': image}
        if masks is not None:
            _dict_data['masks'] = masks
        _dict_data = affine_trans(_dict_data)
        image_t: MetaTensor = _dict_data['image']
        if masks is None:
            masks_t = None
        else:
            masks_t = _dict_data['masks'].round().bool().as_tensor()
        if boxes is None:
            boxes_t = None
        else:
            boxes_t = apply_affine_to_boxes_int(boxes, image_t.affine.inverse())
        return image_t.as_tensor(), masks_t, boxes_t

    def __call__(self, data: dict):
        data = dict(data)
        dataset_name = data['dataset']
        conf = self.conf
        trans_conf = conf.local_trans
        data_dir: Path = data['dataset_dir'] / 'data' / data['key']
        sparse = Sparse.from_json((data_dir / 'sparse.json').read_bytes())
        images: torch.Tensor = load_pt_zst(data_dir / 'images.pt.zst')
        if len(sparse.modalities) == 1:
            modality = sparse.modalities[0]
            modality_slice = slice(None)
        else:
            # NOTE: currently, it is assumed that there will not be multiple RGB images
            modality_idx = self.R.randint(len(sparse.modalities))
            modality = sparse.modalities[modality_idx]
            modality_slice = slice(modality_idx, modality_idx + 1)
        image = tvtf.to_dtype(images[modality_slice], scale=True)
        if (mask_path := data_dir / 'masks.pt.zst').exists():
            masks: torch.BoolTensor | None = load_pt_zst(mask_path)
        else:
            masks = None

        targets = {
            target.name: target
            for target in cytoolz.concat(sparse.targets.values())
        }
        neg_targets = list(cytoolz.concat(sparse.neg_targets.values()))
        # 5. sample grounding classes, generate conversations
        conv = []
        grounding_classes = []
        grounding = toss(self.R, trans_conf.grounding_prob)
        neg_grounding = toss(self.R, trans_conf.neg_grounding_prob) if grounding else False
        conv_anatomy, grounding_classes_anatomy = gen_general_conv(
            self._sample_targets(targets, trans_conf.num_pos, TargetCategory.ANATOMY),
            self._sample_targets(neg_targets, trans_conf.num_neg, TargetCategory.ANATOMY),
            grounding,
            neg_grounding,
            self.tokenizer,
            self.target_tax,
            self.R,
        )
        conv.extend(conv_anatomy)
        grounding_classes.extend(grounding_classes_anatomy)
        conv_anomaly, grounding_classes_anomaly = gen_anomaly_conv(
            self._sample_targets(targets, trans_conf.num_pos, TargetCategory.ANOMALY),
            self._sample_targets(neg_targets, trans_conf.num_neg, TargetCategory.ANOMALY),
            grounding,
            neg_grounding,
            self.tokenizer,
            self.target_tax,
            dataset_name,
            self.R,
        )
        conv.extend(conv_anomaly)
        grounding_classes.extend(grounding_classes_anomaly)
        if len(conv) == 0 or toss(self.R, trans_conf.modality_prob):
            # this also avoid an empty conversation
            conv = gen_modality_conv(modality, self.R) + conv
        # resize image
        if (size_z := image.shape[1]) <= trans_conf.max_tokens_z:
            patch_size_z = pool_size_z = stride_z = 1
            tokens_z = size_z
        else:
            pool_size_z = conf.base_pool_size_z
            log2_patch_size_z = self.R.normal(
                np.log2(size_z / (pool_size_z * trans_conf.max_tokens_z)),
                trans_conf.log2_patch_size_z_std,
            )
            log2_patch_size_z = np.clip(
                np.rint(log2_patch_size_z), 0, conf.base_vit_patch_size_z.bit_length() - 1,
            )
            patch_size_z = 1 << int(log2_patch_size_z)
            stride_z = patch_size_z * pool_size_z
            tokens_z = min(math.ceil(size_z / stride_z), trans_conf.max_tokens_z)
        patch_size = (patch_size_z, conf.vit_patch_size_xy, conf.vit_patch_size_xy)
        stride = (stride_z, conf.stride_xy, conf.stride_xy)
        pool_size = (pool_size_z, conf.pool_size_xy, conf.pool_size_xy)
        resize_shape = (
            min(size_z, tokens_z * stride_z),  # do not resize z if unnecessary
            *get_max_resize(
                image.shape[2:],
                conf.stride_xy,
                trans_conf.max_vision_tokens // tokens_z,
            ),
        )
        if dataset_name == 'VinDr-CXR':
            # Do you like hard coding? Yes, I don't.
            sem_masks = None
            boxes_list = []
            index_offsets = torch.empty(len(grounding_classes), 2, dtype=torch.long)
            index_offset = 0
            for i, class_name in enumerate(grounding_classes):
                if target := targets.get(class_name):
                    boxes_ = torch.from_numpy(target.boxes)
                    boxes_list.append(boxes_)
                    _num = boxes_.shape[0]
                else:
                    _num = 0
                index_offsets[i] = torch.tensor((index_offset, index_offset := index_offset + _num))
            if len(boxes_list) == 0:
                boxes = torch.empty(0, 6, dtype=torch.long)
            else:
                boxes = torch.cat(boxes_list)
        else:
            sem_masks = torch.zeros(len(grounding_classes), *image.shape[1:])
            for i, class_name in enumerate(grounding_classes):
                if target := targets.get(class_name):
                    target: Sparse.Target
                    sem_masks[i] = einops.reduce(masks[slice(*target.index_offset)], 'c ... -> ...', 'any')
            boxes = None
            index_offsets = None
        image, sem_masks, boxes = self._spatial_transform(image, sem_masks, boxes, resize_shape, stride)
        if boxes is not None:
            boxes = norm_boxes(boxes, image.shape[1:])
        if image.shape[0] == 1:
            # ensure RGB
            image = einops.repeat(image, '1 ... -> c ...', c=3).contiguous()
        # no normalization for grounding image, following SegVol. don't panic! they always append a `MinMaxNormalization` after intensity normalization
        grounding_image = image
        image = intensity_norm(image)
        vlm_inputs, conversation_text = prepare_vlm_inputs(
            conv,
            self.tokenizer,
            (np.array(image.shape[1:]) // stride).prod().item(),
            inference=self.inference,
            grounding=grounding,
            max_seq_len=conf.max_seq_len,
            bop_weight=conf.bop_weight,
        )
        data_point = {
            'src': (dataset_name, data['key']),
            'image': image,
            # TODO: apply transform on grounding image
            'grounding_image': grounding_image,
            'patch_size': patch_size,
            'pool_size': pool_size,
            'vlm_inputs': vlm_inputs,
            'masks': sem_masks,
            'boxes': boxes,
            'index_offsets': index_offsets,
            'instance_mask': boxes is not None,
        }
        return data_point
