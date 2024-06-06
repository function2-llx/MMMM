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
from luolib.types import tuple2_t
from luolib.utils import load_pt_zst
from luolib.utils.misc import ceil_divide
from monai import transforms as mt
from monai.data import MetaTensor
from monai.data.box_utils import CenterSizeMode, box_area, convert_box_mode, spatial_crop_boxes
from monai.transforms import generate_spatial_bounding_box
from monai.utils import GridSamplePadMode

import mmmm.data.dataset._dataset as _dataset
from mmmm.data.defs import DataPoint, PROCESSED_LOCAL_DATA_ROOT, mmmm_debug, Split
from mmmm.data.sparse import Sparse
from mmmm.data.target_tax import TargetCategory, get_target_tax
from mmmm.data.utils import prepare_vlm_inputs
from mmmm.tokenizer import MMMMTokenizer
from ..misc import gen_modality_conv, get_max_scale_for_size, intensity_norm, toss
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
    """
    max_vision_tokens: int
    scale_z: tuple2_t[float]
    scale_z_p: float
    max_tokens_z: int
    scale_xy: tuple2_t[float]
    scale_xy_p: float
    aniso_ratio_range: tuple2_t[float] = (0.5, 4.)
    log2_vit_patch_size_z_std = 0.5  # 2-sigma, 95.45%
    num_pos: int
    num_neg: int
    mask_th_abs: int = 1000
    mask_th_rel: float = 0.5
    box_th: float = 0.5
    grounding_prob: float = 0.99
    neg_grounding_prob: float = 0.2
    vlm: bool = True

def get_local_transform(
    conf: _dataset.DatasetConf,
    tokenizer: MMMMTokenizer | None,
    inference: bool,
) -> Callable[[dict], DataPoint]:
    return SamplePatch(conf, tokenizer, inference)

_ignore_anomalies = {'nodule/mass', 'other lesion'}

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

class SamplePatch(mt.Randomizable):
    def __init__(
        self,
        conf: _dataset.DatasetConf,
        tokenizer: MMMMTokenizer | None,
        inference: bool,
        device: Device = 'cpu',
    ):
        """
        Args:
            vlm: whether the transform is performed for VLM
        """
        super().__init__()
        self.conf = conf
        self.trans_conf = conf.local_trans
        if self.trans_conf.vlm:
            assert tokenizer is not None
        self.tokenizer = tokenizer
        self.device = device
        self.inference = inference
        self.target_tax = get_target_tax()

    def gen_patch_info(self, sparse: Sparse):
        conf = self.conf
        trans_conf = conf.local_trans
        # 1. sample tokens_z, tokens_xy, thus obtain patch_size_xy
        if sparse.shape[0] == 1:
            tokens_z = 1
        else:
            # TODO: maybe there's a better approximation for tokens_z
            tokens_z = self.R.randint(1, trans_conf.max_tokens_z + 1)
        # 2. sample scale_xy
        tokens_xy_total = trans_conf.max_vision_tokens // tokens_z
        min_scale_xy = 1. if tokens_z == 1 else trans_conf.scale_xy[0]
        max_scale_xy = min(
            1 / get_max_scale_for_size(
                sparse.shape[1:], conf.stride_xy, tokens_xy_total,
            ),
            trans_conf.scale_xy[1],
        )
        if max_scale_xy <= min_scale_xy:
            # the original image is too small, just scale as much as possible
            scale_xy = max_scale_xy
        elif toss(self.R, trans_conf.scale_xy_p):
            scale_xy = self.R.uniform(min_scale_xy, max_scale_xy)
        else:
            scale_xy = 1.
        spacing_xy = min(sparse.spacing[1:]) * scale_xy
        # 3. sample scale_z
        if sparse.shape[0] == 1:
            scale_z = 1.
        else:
            spacing_z = np.maximum(sparse.spacing[0], trans_conf.aniso_ratio_range[0] * spacing_xy)
            if spacing_z < trans_conf.aniso_ratio_range[1] * spacing_xy and toss(self.R, trans_conf.scale_z_p):
                spacing_z *= self.R.uniform(
                    max(
                        trans_conf.scale_z[0],
                        trans_conf.aniso_ratio_range[0] * spacing_xy / spacing_z,
                    ),
                    min(
                        trans_conf.scale_z[1],
                        trans_conf.aniso_ratio_range[1] * spacing_xy / spacing_z,
                    ),
                )
            scale_z = spacing_z / sparse.spacing[0]
        # 4. determine vit_patch_size_z
        if sparse.shape[0] == 1:
            pool_size_z = 1
            vit_patch_size_z = 1
        else:
            pool_size_z = conf.base_pool_size_z
            log2_vit_patch_size_z = self.R.normal(
                np.log2(conf.base_vit_patch_size_z * spacing_xy / spacing_z),
                trans_conf.log2_vit_patch_size_z_std,
            )
            log2_vit_patch_size_z = np.clip(
                np.rint(log2_vit_patch_size_z),
                0, conf.base_vit_patch_size_z.bit_length() - 1,
            )
            vit_patch_size_z = 1 << int(log2_vit_patch_size_z)
        vit_patch_size = np.array((vit_patch_size_z, conf.vit_patch_size_xy, conf.vit_patch_size_xy))
        scale = np.array((scale_z, scale_xy, scale_xy))
        patch_size_z = tokens_z * vit_patch_size_z * pool_size_z
        patch_size_xy = _get_patch_size_xy(
            sparse.shape[1:], scale_xy, conf.stride_xy, tokens_xy_total,
        )
        patch_size = np.array((patch_size_z, *patch_size_xy))
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
        origin_boxes = torch.from_numpy(target.boxes)
        boxes, keep = spatial_crop_boxes(origin_boxes, patch_start, patch_start + patch_size)
        if boxes.shape[0] == 0:
            return [], boxes, 0
        else:
            if patch_masks is None:
                keep = box_area(boxes) >= box_area(origin_boxes[keep]) * self.trans_conf.box_th
                mask_indexes = []
                num_uncertain = boxes.shape[0] - keep.sum().item()
            else:
                _center = patch_size >> 1
                mask_sizes = torch.from_numpy(target.mask_sizes)[keep]
                mask_indexes = torch.arange(*target.index_offset)[keep]
                patch_masks = patch_masks[slice(*target.index_offset)][keep]
                patch_mask_sizes = einops.reduce(patch_masks, 'n ... -> n', 'sum')
                keep = (
                    (patch_mask_sizes >= conf.mask_th_abs) |
                    (patch_mask_sizes >= mask_sizes * conf.mask_th_rel) |
                    patch_masks[:, *_center]
                )
                mask_indexes = mask_indexes[keep].tolist()
                num_uncertain = boxes.shape[0] - keep.sum().item() - (patch_mask_sizes == 0).sum().item()
            boxes = boxes[keep]
            return mask_indexes, boxes, num_uncertain

    def _get_category(self, name: str):
        if name in _ignore_anomalies:
            return TargetCategory.ANOMALY
        return self.target_tax[name].category

    def _sample_targets(self, targets: Iterable[str], limit: int, category: str | None = None) -> list[str]:
        if category is None:
            targets = list(targets)
        else:
            targets = [*filter(lambda target: self._get_category(target) == category, targets)]
        if len(targets) > limit:
            targets = self.R.choice(targets, limit, replace=False).tolist()
        return targets

    def _affine_transform(
        self,
        patch: torch.Tensor,
        patch_masks: torch.BoolTensor | None,
        boxes: torch.LongTensor,
        patch_size: Sequence[int],
        scale: Sequence[float],
    ) -> tuple[torch.Tensor, torch.BoolTensor | None, torch.LongTensor]:
        affine_trans = mt.Compose(
            [
                # NOTE: scaling should be performed firstly since spatial axis order might change
                mt.AffineD(
                    ['image', 'masks'],
                    scale_params=scale.tolist(),
                    spatial_size=patch_size.tolist(),
                    allow_missing_keys=True,
                ),
                *[
                    mt.RandFlipD(['image', 'masks'], 0.5, i, allow_missing_keys=True)
                    for i in range(3)
                ],
                mt.RandRotate90D(['image', 'masks'], 0.75, spatial_axes=(1, 2), allow_missing_keys=True),
            ],
            lazy=True,
            overrides={
                'image': {'padding_mode': GridSamplePadMode.ZEROS},
                'masks': {'padding_mode': GridSamplePadMode.ZEROS},
            }
        )
        affine_trans.set_random_state(state=self.R)
        _dict_data = {'image': patch}
        if patch_masks is not None:
            _dict_data['masks'] = patch_masks
        _dict_data = affine_trans(_dict_data)
        patch_t: MetaTensor = _dict_data['image']
        if patch_masks is None:
            patch_masks_t = None
            boxes_t = apply_affine_to_boxes_int(boxes, patch_t.affine.inverse())
        else:
            patch_masks_t = _dict_data['masks'].round().bool().as_tensor()
            boxes_t = torch.empty(patch_masks_t.shape[0], 6, dtype=torch.int64)
            for i, mask in enumerate(patch_masks_t):
                # setting allow_smaller=False to make MONAI happy
                _start, _end = generate_spatial_bounding_box(mask[None], allow_smaller=False)
                boxes_t[i] = torch.tensor([*_start, *_end])
        return patch_t.as_tensor(), patch_masks_t, boxes_t

    def __call__(self, data: dict) -> DataPoint:
        data = dict(data)
        dataset_name = data['dataset']
        conf = self.conf
        trans_conf = conf.local_trans
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
                target, patch_masks, patch_start, effective_patch_size,
            )
            # NOTE: a target will be included for training if any:
            #   - totally negative in the patch
            #   - at least one instance certainly presents in the patch
            #   - only the presence of the target is known, but no localized information
            #     - indicated with num_uncertain = -1
            #     - not applicable for local type data
            if boxes.shape[0] == 0 and num_uncertain == 0:
                neg_targets.append(target.name)
            elif boxes.shape[0] > 0:
                targets[target.name] = {
                    'mask_indexes': mask_indexes,
                    'boxes': boxes,
                    'num_uncertain': num_uncertain,
                    'semantic': target.semantic,
                }

        # 5. generate conversation
        if trans_conf.vlm:
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
                sparse.complete_anomaly,
                grounding,
                neg_grounding,
                self.tokenizer,
                self.target_tax,
                dataset_name,
                self.R,
            )
            conv.extend(conv_anomaly)
            grounding_classes.extend(grounding_classes_anomaly)
            if len(conv) == 0 or toss(self.R, 0.9):
                # this also avoid an empty conversation
                conv = gen_modality_conv(modality, self.R) + conv
            vlm_inputs, conversation_text = prepare_vlm_inputs(
                conv,
                self.tokenizer,
                patch_tokens.prod().item(),
                inference=self.inference,
                grounding=grounding,
                max_seq_len=conf.max_seq_len,
                bop_weight=conf.bop_weight,
            )
        else:
            grounding = True
            pos_classes = self._sample_targets(targets, trans_conf.num_pos)
            neg_classes = self._sample_targets(neg_targets, trans_conf.num_neg)
            grounding_classes = pos_classes + neg_classes
            vlm_inputs = None

        # 6. prepare the data point!
        if grounding:
            targets_data = {
                'mask_indexes': [],
                'boxes': [],
                'num_uncertain': torch.empty(len(grounding_classes), dtype=torch.int64),
                'semantic': torch.empty(len(grounding_classes), dtype=torch.bool),
                'index_offsets': torch.empty(len(grounding_classes), 2, dtype=torch.int64),
            }
            index_offset = 0
            for i, grounding_class in enumerate(grounding_classes):
                if (req_target := targets.get(grounding_class)) is None:
                    _num = 0
                    targets_data['num_uncertain'][i] = 0
                    # no ground truth, we can give instance segmentation a try
                    targets_data['semantic'][i] = False
                else:
                    targets_data['mask_indexes'].extend(req_target['mask_indexes'])
                    boxes = req_target['boxes']
                    _num = boxes.shape[0]
                    targets_data['boxes'].append(boxes)
                    targets_data['num_uncertain'][i] = req_target['num_uncertain']
                    targets_data['semantic'][i] = req_target['semantic']
                targets_data['index_offsets'][i] = torch.tensor([index_offset, (index_offset := index_offset + _num)])
            if index_offset == 0:
                targets_data['boxes'] = torch.empty(0, 6, dtype=torch.int64)
            else:
                targets_data['boxes'] = torch.cat(targets_data['boxes'], dim=0)
        else:
            targets_data = {
                'mask_indexes': [],
                'boxes': torch.empty(0, 6, dtype=torch.int64),
                'num_uncertain': torch.full((len(grounding_classes), ), -1, dtype=torch.int64),
                'semantic': torch.empty(len(grounding_classes), dtype=torch.bool),
                'index_offsets': torch.zeros(len(grounding_classes), 2, dtype=torch.int64),
            }
        if patch_masks is not None:
            # select masks before affine transform to save computation
            if len(mask_indexes := targets_data['mask_indexes']) > 0:
                patch_masks = patch_masks[mask_indexes]
            else:
                patch_masks = None
        patch, patch_masks, boxes = self._affine_transform(
            patch, patch_masks, targets_data['boxes'], patch_size, scale,
        )
        boxes = norm_boxes(boxes, patch.shape[1:])
        if patch.shape[0] == 1:
            # ensure RGB
            patch = einops.repeat(patch, '1 ... -> c ...', c=3).contiguous()
        if not mmmm_debug():
            patch = intensity_norm(patch)
        index_offsets = targets_data['index_offsets']
        # prepare semantic label
        if patch_masks is None:
            semantic_masks = semantic_boxes = None
        else:
            num_targets = index_offsets.shape[0]
            semantic_masks = torch.empty(num_targets, 1, *patch_masks.shape[1:], dtype=torch.bool)
            semantic_boxes = torch.empty(num_targets, 6, dtype=torch.int64)
            for i, index_offset in enumerate(index_offsets):
                semantic_masks[i] = einops.reduce(patch_masks[slice(*index_offset)], 'c ... -> 1 ...', 'any')
                _start, _end = generate_spatial_bounding_box(semantic_masks[i])
                semantic_boxes[i] = torch.tensor([*_start, *_end])
            semantic_boxes = norm_boxes(semantic_boxes, patch.shape[1:])

        data_point: DataPoint = {
            'src': (dataset_name, data['key']),
            'image': patch,
            # TODO: apply transform on grounding image
            'grounding_image': patch,
            'patch_size': tuple(vit_patch_size.tolist()),
            'pool_size': tuple(pool_size.tolist()),
            'vlm_inputs': vlm_inputs,
            'grounding_classes': grounding_classes,  # oh
            'masks': patch_masks,
            'boxes': boxes,
            'semantic_masks': semantic_masks,
            'semantic_boxes': semantic_boxes,
            'num_uncertain': targets_data['num_uncertain'],
            'semantic': targets_data['semantic'],
            'index_offsets': index_offsets,
        }
        return data_point
