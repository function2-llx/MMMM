from __future__ import annotations as _

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import cytoolz
import einops
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from torch.types import Device
import torchvision.transforms.v2.functional as tvtf

from luolib.types import tuple2_t
from luolib.utils import load_pt_zst
from mmmm.data.dataset.local.template import gen_anomaly_conversation, gen_general_conv
from monai import transforms as mt
from monai.apps.detection.transforms.box_ops import apply_affine_to_boxes
from monai.data.box_utils import box_area, clip_boxes_to_image

import mmmm.data.dataset._dataset as _dataset
from mmmm.tokenizer import MMMMTokenizer
from mmmm.data.dataset.misc import gen_modality_conv, intensity_norm, toss
from mmmm.data.defs import DataPoint, PROCESSED_LOCAL_DATA_ROOT, split_t
from mmmm.data.sparse import Sparse
from mmmm.data.target_tax import TargetCategory, get_target_tax
from mmmm.data.utils import prepare_vlm_inputs

__all__ = [
    'get_local_data_list',
    'get_local_transform',
    'LocalTransConf',
]

def get_local_data_list(name: str, split: split_t):
    if split != 'train':
        raise NotImplementedError
    dataset_dir = PROCESSED_LOCAL_DATA_ROOT / name
    info = pd.read_csv(dataset_dir / 'info.csv', dtype={'key': 'string'})
    info.set_index('key', inplace=True)
    return [
        {
            'dataset': name,
            'dataset_dir': dataset_dir,
            'key': key,
        }
        for key in info.index
    ]

@dataclass(kw_only=True)
class LocalTransConf:
    max_vision_tokens: int
    scale_z: tuple2_t[float]
    scale_z_p: float
    max_tokens_z: int
    scale_xy: tuple2_t[float]
    scale_xy_p: float
    aniso_ratio_range: tuple2_t[float] = (0.5, 2.)
    log2_vit_patch_size_z_std = 0.5  # 2-sigma, 95.45%
    num_pos: int  # I've encountered cases setting this to larger than 48 causing NCCL timeout
    num_neg: int
    mask_th_abs: int = 1000
    mask_th_rel: float = 0.5
    box_th: float = 0.5
    grounding_prob: float = 0.99

def get_local_transform(
    conf: _dataset.DatasetConf,
    tokenizer: MMMMTokenizer,
    inference: bool,
) -> Callable[[dict], DataPoint]:
    return SamplePatch(conf, tokenizer, inference)

_ignore_anomalies = {'nodule/mass', 'other lesion'}

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
        self.trans_conf = conf.seg_trans
        self.tokenizer = tokenizer
        self.device = device
        self.inference = inference
        self.target_tax = get_target_tax()

    def gen_patch_info(self, sparse: Sparse):
        conf = self.conf
        trans_conf = conf.seg_trans
        # 1. sample tokens_z, tokens_xy, thus obtain patch_size_xy
        if sparse.shape[0] == 1:
            tokens_z = 1
        else:
            # TODO: maybe there's a better approximation for tokens_z
            tokens_z = self.R.randint(1, trans_conf.max_tokens_z + 1)
        tokens_xy = int((trans_conf.max_vision_tokens / tokens_z) ** 0.5)
        patch_size_xy = tokens_xy * conf.stride_xy
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
            pool_size_z = 1
            vit_patch_size_z = 1
        else:
            pool_size_z = conf.base_pool_size_z
            spacing_xy = min(sparse.spacing[1:]) * scale_xy
            spacing_z = sparse.spacing[0] * scale_z
            log2_vit_patch_size_z = self.R.normal(
                np.log2(conf.base_vit_patch_size_z * spacing_xy / spacing_z),
                trans_conf.log2_vit_patch_size_z_std,
            )
            log2_vit_patch_size_z = np.clip(
                np.rint(log2_vit_patch_size_z), 0, conf.base_vit_patch_size_z.bit_length() - 1,
            )
            vit_patch_size_z = 1 << int(log2_vit_patch_size_z)
        vit_patch_size = np.array((vit_patch_size_z, conf.vit_patch_size_xy, conf.vit_patch_size_xy))
        scale = np.array((scale_z, scale_xy, scale_xy))
        patch_size = np.array((tokens_z * vit_patch_size_z * pool_size_z, patch_size_xy, patch_size_xy))
        pool_size = np.array((pool_size_z, conf.pool_size_xy, conf.pool_size_xy))
        return patch_size, scale, vit_patch_size, pool_size

    def get_patch_start(self, ref_patch_center: npt.NDArray[np.int64], effective_patch_size: np.ndarray, shape: np.ndarray):
        # TODO: add randomization
        patch_start = ref_patch_center - (effective_patch_size >> 1)
        patch_start = np.clip(patch_start, 0, shape - effective_patch_size)
        return patch_start

    def filter_target(
        self,
        target: Sparse.Target,
        patch_masks: torch.BoolTensor | None,
        patch_start: npt.NDArray[np.int64],
        patch_size: npt.NDArray[np.int64],
    ) -> tuple[list[int], torch.LongTensor, int]:
        """
        Returns:
            - indexes among all instances for certain instances of this target
            - boxes for certain instances
            - number of instances that is uncertain
        """
        conf = self.trans_conf
        # NOTE: boxes cropping is applied here
        boxes = torch.from_numpy(target.boxes - einops.repeat(patch_start, 'd -> (l2 d)', l2=2))
        clipped_boxes, keep = clip_boxes_to_image(boxes, patch_size)
        if clipped_boxes.shape[0] == 0:
            return [], clipped_boxes, 0
        else:
            if patch_masks is None:
                keep = box_area(clipped_boxes) >= box_area(boxes[keep]) * self.trans_conf.box_th
                mask_indexes = []
            else:
                mask_sizes = target.mask_sizes[keep]
                mask_indexes = torch.arange(*target.index_offset)[keep]
                patch_mask_sizes = einops.reduce(
                    patch_masks[slice(*target.index_offset)], 'n ... -> n', 'sum',
                ).numpy()
                keep = (patch_mask_sizes >= mask_sizes * conf.mask_th_rel) & (patch_mask_sizes >= conf.mask_th_abs)
                mask_indexes = mask_indexes[keep].tolist()
            boxes = clipped_boxes[keep]
            num_uncertain = clipped_boxes.shape[0] - keep.sum().item()
            return mask_indexes, boxes, num_uncertain

    def _get_category(self, name: str):
        if name in _ignore_anomalies:
            return TargetCategory.ANOMALY
        return self.target_tax[name].category
    def _sample_targets(self, category: str, targets: Iterable[str], limit: int) -> list[str]:
        targets = [*filter(lambda target: self._get_category(target) == category, targets)]
        if len(targets) > limit:
            targets = self.R.choice(targets, limit, replace=False).tolist()
        return targets

    def __call__(self, data: dict) -> DataPoint:
        data = dict(data)
        dataset_name = data['dataset']
        conf = self.conf
        trans_conf = conf.seg_trans
        data_dir: Path = data['dataset_dir'] / 'data' / data['key']
        sparse = Sparse.from_json((data_dir / 'sparse.json').read_bytes())
        # 1. generate patch information
        patch_size, scale, vit_patch_size, pool_size = self.gen_patch_info(sparse)
        stride = vit_patch_size * pool_size
        patch_tokens, _rem = np.divmod(patch_size, stride)
        assert np.array_equiv(_rem, 0)
        effective_patch_size = np.minimum(np.ceil(patch_size * scale).astype(np.int64), sparse.shape)
        # 2. sample patch position
        if (class_positions_path := data_dir / 'class_positions.pt').exists():
            class_positions: torch.Tensor = torch.load(class_positions_path, mmap=True)
            # foreground oversampling. not using a force fg ratio sinc it may result in zero classes
            ref: Sparse.Target = self.R.choice(list(cytoolz.concat(sparse.targets.values())))
            position_idx = self.R.randint(*ref.position_offset)
            ref_patch_center = class_positions[position_idx].numpy()
            patch_start = self.get_patch_start(ref_patch_center, effective_patch_size, sparse.shape)
        else:
            patch_start = self.R.randint(sparse.shape - effective_patch_size + 1)
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
        images: torch.Tensor = load_pt_zst(data_dir / 'images.pt.zst')
        patch = tvtf.to_dtype(images[modality_slice, *patch_slice], scale=True)
        if (mask_path := data_dir / 'masks.pt.zst').exists():
            whole_masks: torch.BoolTensor = load_pt_zst(mask_path)
            patch_masks = whole_masks[:, *patch_slice]
        else:
            patch_masks = None
        # NOTE: will apply cropping to boxes later
        # 4. determine positive & negative classes within the cropped patch
        targets = {}
        neg_targets = list(cytoolz.concat(sparse.neg_targets.values()))
        for target in cytoolz.concat(sparse.targets.values()):
            target: Sparse.Target
            mask_indexes, boxes, num_uncertain = self.filter_target(
                target, patch_masks, patch_start, patch_size,
            )
            if boxes.shape[0] == 0 and num_uncertain == 0:
                neg_targets.append(target.name)
            elif boxes.shape[0] > 0:
                targets[target.name] = {
                    'mask_indexes': mask_indexes,
                    'boxes': boxes,
                    'num_uncertain': num_uncertain,
                    'semantic': target.semantic,
                }
        neg_targets = set(neg_targets)
        # 5. apply affine transform!
        affine_trans = mt.Compose(
            [
                *[
                    mt.RandFlipD(['image', 'masks'], 0.5, i, allow_missing_keys=True)
                    for i in range(3)
                ],
                mt.RandRotate90D(['image', 'masks'], 0.5, spatial_axes=(1, 2), allow_missing_keys=True),
                mt.AffineD(
                    ['image', 'masks'],
                    scale_params=scale.tolist(),
                    spatial_size=patch_size.tolist(),
                    allow_missing_keys=True,
                ),
            ],
            lazy=True,
        )
        affine_trans.set_random_state(state=self.R)
        _dict_data = {'image': patch}
        if patch_masks is not None:
            _dict_data['masks'] = patch_masks
        _dict_data = affine_trans(_dict_data)
        patch = _dict_data['image']
        if patch_masks is not None:
            patch_masks = _dict_data['masks'].round().bool().as_tensor()
        # NOTE: will apply affine transform to boxes later
        trans_affine = patch.affine
        # 6. generate conversation
        conv = gen_modality_conv(modality, self.R)
        req_classes = []
        grounding = toss(self.R, trans_conf.grounding_prob)
        conv_anatomy, req_classes_anatomy = gen_general_conv(
            self._sample_targets(TargetCategory.ANATOMY, targets, trans_conf.num_pos),
            self._sample_targets(TargetCategory.ANATOMY, neg_targets, trans_conf.num_neg),
            grounding,
            self.tokenizer,
            self.target_tax,
            self.R,
        )
        conv.extend(conv_anatomy)
        req_classes.extend(req_classes_anatomy)
        conv_anomaly, req_classes_anomaly = gen_anomaly_conversation(
            self._sample_targets(TargetCategory.ANOMALY, targets, trans_conf.num_pos),
            self._sample_targets(TargetCategory.ANOMALY, neg_targets, trans_conf.num_neg),
            sparse.complete_anomaly,
            grounding,
            self.tokenizer,
            self.target_tax,
            dataset_name,
            self.R,
        )
        conv.extend(conv_anomaly)
        req_classes.extend(req_classes_anomaly)
        vlm_inputs, conversation_text = prepare_vlm_inputs(
            conv,
            self.tokenizer,
            patch_tokens.prod().item(),
            inference=self.inference,
            grounding=grounding,
            max_seq_len=conf.max_seq_len,
        )
        # 7. final preparation
        if patch.shape[0] == 1:
            # ensure RGB
            patch = einops.repeat(patch, '1 ... -> c ...', c=3).contiguous()
        patch = intensity_norm(patch)
        if grounding:
            # masks = torch.zeros(len(req_classes), *patch.shape[1:], dtype=torch.bool)
            if patch_masks is None:
                pass
            _data = {
                'mask_indexes': [],
                'boxes': [],
                'num_uncertain': torch.empty(len(req_classes), dtype=torch.bool),
                'semantic': torch.empty(len(req_classes), dtype=torch.bool),
                'index_offsets': torch.empty(len(req_classes), 2, dtype=torch.int64),
            }
            index_offset = 0
            for i, req_class in enumerate(req_classes):
                if req_class in neg_targets:
                    _num = 0
                    _data['num_uncertain'][i] = 0
                    _data['semantic'][i] = False
                elif (req_target := targets.get(req_class)) is not None:
                    _data['mask_indexes'].extend(req_target['mask_indexes'])
                    boxes = req_target['boxes']
                    _num = boxes.shape[0]
                    _data['boxes'].append(boxes)
                    _data['num_uncertain'][i] = req_target['num_uncertain']
                    _data['semantic'][i] = req_target['semantic']
                _data['index_offsets'][i] = torch.tensor([index_offset, (index_offset := index_offset + _num)])
            if index_offset == 0:
                _data['boxes'] = torch.empty(0, 6, dtype=torch.int64)
            else:
                _data['boxes'] = torch.cat(_data['boxes'], dim=0)
        else:
            _data = {
                'mask_indexes': [],
                'boxes': torch.empty(0, 6, dtype=torch.int64),
                'num_uncertain': torch.zeros(len(req_classes), dtype=torch.int64),
                'semantic': torch.zeros(len(req_classes), dtype=torch.bool),
                'index_offsets': torch.zeros(len(req_classes), 2, dtype=torch.int64),
            }
        boxes = _data['boxes']
        apply_affine_to_boxes(boxes, trans_affine.inverse())
        data_point: DataPoint = {
            'image': patch,
            # TODO: apply transform on grounding image
            'grounding_image': patch,
            'patch_size': tuple(vit_patch_size.tolist()),
            'pool_size': tuple(pool_size.tolist()),
            'vlm_inputs': vlm_inputs,
            'masks': None if patch_masks is None else patch_masks[_data['mask_indexes']],
            'boxes': _data['boxes'],
            'num_uncertain': _data['num_uncertain'],
            'semantic': _data['semantic'],
            'index_offsets': _data['index_offsets'],
        }
        return data_point

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
