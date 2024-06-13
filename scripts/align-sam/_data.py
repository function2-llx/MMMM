from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import cytoolz
import einops
import math
import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as nnf
from lightning.fabric.utilities.distributed import DistributedSamplerWrapper
from torch.utils.data import Dataset as _TorchDataset, Sampler

from monai.data import DataLoader
from monai.data.box_utils import spatial_crop_boxes
import monai.transforms as mt
from monai.utils import GridSamplePadMode
from luolib.datamodule import ExpDataModuleBase
from luolib.types import tuple2_t
from luolib.utils import load_pt_zst
from luolib.utils.misc import ceil_divide

from mmmm.data import get_target_tax
from mmmm.data.dataset import DatasetSpec
from mmmm.data.dataset.local import get_local_data_list
from mmmm.data.dataset.misc import get_max_scale_for_size, toss
from mmmm.data.defs import Split
from mmmm.data.sparse import Sparse

def _is_power_of_2(x: int):
    return x & (x - 1) == 0

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

@dataclass(kw_only=True)
class TransConf:
    low_res_p: float
    max_vision_tokens: int
    max_vision_tokens_lr: int
    max_vision_tokens_2d: int
    max_vision_tokens_2d_lr: int
    scale_z: tuple2_t[float]
    scale_z_p: float
    max_tokens_z: int
    max_tokens_z_lr: int
    scale_xy: tuple2_t[float]
    scale_xy_p: float
    aniso_ratio_range: tuple2_t[float] = (0.5, 3.)
    log2_vit_patch_size_z_std: float
    num_pos: int
    num_neg: int
    full_size_ratio: float
    force_fg_ratio: float = 0.66
    scale_intensity_prob: float = 0.15
    scale_intensity_factor: float = 0.1
    shift_intensity_prob: float = 0.15
    shift_intensity_offset: float = 0.1

@dataclass(kw_only=True)
class DatasetConf:
    datasets: list[DatasetSpec]
    base_vit_patch_size_z: int
    vit_patch_size_xy: int
    trans: TransConf

    def __post_init__(self):
        assert _is_power_of_2(self.vit_patch_size_xy)
        assert _is_power_of_2(self.base_vit_patch_size_z)

class SamplePatch(mt.Randomizable):
    def __init__(self, conf: DatasetConf):
        super().__init__()
        self.conf = conf
        self.trans_conf = conf.trans
        self.target_tax = get_target_tax()

    def gen_patch_size_info(self, sparse: Sparse):
        """
        Returns:
            - patch size (actual input size for the vision model)
            - scale:
        """
        conf = self.conf
        trans_conf = conf.trans
        size_z: int = sparse.shape[0].item()
        if toss(self.R, trans_conf.low_res_p):
            max_vision_tokens = trans_conf.max_vision_tokens_lr
            max_vision_tokens_2d = trans_conf.max_vision_tokens_2d_lr
            max_tokens_z = trans_conf.max_tokens_z_lr
        else:
            max_vision_tokens = trans_conf.max_vision_tokens
            max_vision_tokens_2d = trans_conf.max_vision_tokens_2d
            max_tokens_z = trans_conf.max_tokens_z
        if size_z == 1 or toss(self.R, trans_conf.full_size_ratio):
            if size_z <= max_tokens_z:
                vit_patch_size_z = 1
                tokens_z = size_z
            else:
                log2_vit_patch_size_z = self.R.normal(
                    np.log2(size_z / max_tokens_z),
                    trans_conf.log2_vit_patch_size_z_std,
                )
                log2_vit_patch_size_z = np.clip(
                    np.rint(log2_vit_patch_size_z), 0, conf.base_vit_patch_size_z.bit_length() - 1,
                )
                vit_patch_size_z = 1 << int(log2_vit_patch_size_z)
                tokens_z = min(math.ceil(size_z / vit_patch_size_z), max_tokens_z)
            patch_size_z = tokens_z * vit_patch_size_z
            tokens_xy_total = max_vision_tokens_2d if size_z == 1 else max_vision_tokens // tokens_z
            scale_xy = 1 / get_max_scale_for_size(
                sparse.shape[1:],
                conf.vit_patch_size_xy,
                tokens_xy_total,
            )
            scale_z = size_z / patch_size_z
        else:
            tokens_z = min(max_tokens_z, size_z)
            # I know, it is always 3D here
            tokens_xy_total = max_vision_tokens_2d if size_z == 1 else max_vision_tokens // tokens_z
            # 2. sample scale_xy
            min_scale_xy = trans_conf.scale_xy[0]
            max_scale_xy = min(
                1 / get_max_scale_for_size(
                    sparse.shape[1:], conf.vit_patch_size_xy, tokens_xy_total,
                ),
                trans_conf.scale_xy[1],
            )
            if max_scale_xy <= min_scale_xy:
                # the original image within the plane is too small, just scale to use full size
                scale_xy = max_scale_xy
            elif toss(self.R, trans_conf.scale_xy_p):
                scale_xy = self.R.uniform(min_scale_xy, max_scale_xy)
            else:
                scale_xy = 1.
            spacing_xy = min(sparse.spacing[1:]) * scale_xy
            # 3. sample scale_z & vit_patch_size_z
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
            # 4. sample vit_patch_size_z
            log2_vit_patch_size_z = self.R.normal(
                np.log2(conf.base_vit_patch_size_z * spacing_xy / spacing_z),
                trans_conf.log2_vit_patch_size_z_std,
            )
            log2_vit_patch_size_z = np.clip(
                np.rint(log2_vit_patch_size_z),
                0, conf.base_vit_patch_size_z.bit_length() - 1,
            )
            vit_patch_size_z = 1 << int(log2_vit_patch_size_z)
            patch_size_z = tokens_z * vit_patch_size_z

        vit_patch_size = np.array((vit_patch_size_z, conf.vit_patch_size_xy, conf.vit_patch_size_xy))
        scale = np.array((scale_z, scale_xy, scale_xy))
        patch_size_xy = _get_patch_size_xy(
            sparse.shape[1:], scale_xy, conf.vit_patch_size_xy, tokens_xy_total,
        )
        patch_size = np.array((patch_size_z, *patch_size_xy))
        return patch_size, scale, vit_patch_size

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
    ) -> tuple[list[int], torch.LongTensor]:
        """
        Returns:
            - indexes among all instances for certain instances of this target
            - boxes for all instances
        """
        # NOTE: boxes cropping is applied here
        origin_boxes = torch.from_numpy(target.boxes)
        boxes, keep = spatial_crop_boxes(origin_boxes, patch_start, patch_start + patch_size)
        if boxes.shape[0] == 0 or patch_masks is None:
            # NOTE: when only bounding boxes are available, it must be 2D image and always uses full image
            mask_indexes = []
        else:
            _center = patch_size >> 1
            mask_indexes = torch.arange(*target.index_offset)[keep]
            patch_masks = patch_masks[slice(*target.index_offset)][keep]
            keep = einops.reduce(patch_masks, 'n ... -> n', 'any')
            mask_indexes = mask_indexes[keep].tolist()
            boxes = boxes[keep]
        return mask_indexes, boxes

    def _sample_targets(self, targets: Iterable[str], limit: int) -> list[str]:
        targets = list(targets)
        if len(targets) > limit:
            targets = self.R.choice(targets, limit, replace=False).tolist()
        return targets

    def _affine_transform(
        self,
        patch: torch.ByteTensor,
        masks: torch.BoolTensor | None,
        patch_size: np.ndarray,
        scale: np.ndarray,
    ) -> tuple[torch.FloatTensor, torch.BoolTensor | None]:
        if masks is None:
            keys = ['image']
        else:
            keys = ['image', 'masks']
        # interpolation may not be as accurate as grid sampling, but probably faster
        resize = np.minimum(np.maximum((patch.shape[1:] / scale).round().astype(np.int64), 1), patch_size)
        if not np.array_equal(resize, patch.shape[1:]):
            resize = resize.tolist()
            patch = nnf.interpolate(patch[None], resize, mode='nearest-exact')[0]
            if masks is not None:
                masks = nnf.interpolate(masks[None].byte(), resize, mode='nearest-exact')[0].bool()
        flip_rotate_trans = mt.Compose(
            [
                # NOTE: scaling should be performed firstly since spatial axis order might change after random rotation
                mt.SpatialPadD(keys, patch_size),
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
            },
        )
        flip_rotate_trans.set_random_state(state=self.R)
        _dict_data = {'image': patch, 'masks': masks}
        _dict_data = flip_rotate_trans(_dict_data)
        patch_t = _dict_data['image'].as_tensor()
        masks_t = None if masks is None else (_dict_data['masks'] > 0.5).as_tensor()
        return patch_t, masks_t

    def __call__(self, data: dict):
        data = dict(data)
        dataset_name = data['dataset']
        conf = self.conf
        trans_conf = conf.trans
        data_dir: Path = data['dataset_dir'] / 'data' / data['key']
        sparse = Sparse.from_json((data_dir / 'sparse.json').read_bytes())

        # 1. generate patch information
        patch_size, scale, vit_patch_size = self.gen_patch_size_info(sparse)
        patch_tokens, _rem = np.divmod(patch_size, vit_patch_size)
        assert np.array_equiv(_rem, 0)
        effective_patch_size = np.minimum(np.ceil(patch_size * scale).astype(np.int64), sparse.shape)

        # 2. sample patch position
        if (class_positions_path := data_dir / 'class_positions.pt').exists() and toss(self.R, trans_conf.force_fg_ratio):
            class_positions: torch.Tensor = torch.load(class_positions_path, mmap=True)
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
        # 4. determine positive & negative classes within the cropped patch
        targets = {}
        neg_targets = list(cytoolz.concat(sparse.neg_targets.values()))
        if (mask_path := data_dir / 'masks.pt.zst').exists():
            whole_masks: torch.BoolTensor = load_pt_zst(mask_path)
            patch_masks = whole_masks[:, *patch_slice]
            is_pos = einops.reduce(patch_masks, 'c ... -> c', 'any')
            for target in cytoolz.concat(sparse.targets.values()):
                target: Sparse.Target
                index_slice = slice(*target.index_offset)
                if is_pos[index_slice].any():
                    targets[target.name] = index_slice
                else:
                    neg_targets.append(target.name)
            pos_classes = self._sample_targets(targets, trans_conf.num_pos)
            pos_masks = torch.cat([
                einops.reduce(patch_masks[targets[name]], 'c ... -> 1 ...', 'any')
                for name in pos_classes
            ])
        else:
            assert sum(1 for _ in cytoolz.concat(sparse.targets.values())) == 0
            pos_classes = []
            pos_masks = None

        neg_classes = self._sample_targets(neg_targets, trans_conf.num_neg)
        if len(sparse.modalities) == 1:
            modality_slice = slice(None)
        else:
            # NOTE: currently, it is assumed that there will not be multiple RGB images
            modality_idx = self.R.randint(len(sparse.modalities))
            modality_slice = slice(modality_idx, modality_idx + 1)
        images: torch.ByteTensor = load_pt_zst(data_dir / 'images.pt.zst')
        patch = images[modality_slice, *patch_slice]
        patch, pos_masks = self._affine_transform(patch, pos_masks, patch_size, scale)
        patch = patch.float() / 255

        # TODO: maybe shuffle the classes, though it should have no effect
        classes = pos_classes + neg_classes
        neg_masks = torch.zeros(len(neg_classes), *patch.shape[1:], dtype=torch.bool)
        if pos_masks is None:
            masks = neg_masks
        else:
            masks = torch.cat([pos_masks, neg_masks])

        if patch.shape[0] == 1:
            # ensure RGB
            patch = einops.repeat(patch, '1 ... -> c ...', c=3).contiguous()
        data_point = {
            'src': (dataset_name, data['key']),
            'image': patch,
            'patch_size': tuple(vit_patch_size.tolist()),
            'classes': classes,
            'masks': masks,
        }
        return data_point

class Dataset(_TorchDataset):
    def __init__(self, conf: DatasetConf):
        super().__init__()
        self.conf = conf
        self.data_lists = [
            get_local_data_list(dataset.name, Split.TRAIN)
            for dataset in conf.datasets
        ]
        trans_conf = conf.trans
        self.transform = mt.Compose([
            SamplePatch(conf),
            mt.RandScaleIntensityD(
                'image', prob=trans_conf.scale_intensity_prob, factors=trans_conf.scale_intensity_factor,
            ),
            mt.RandShiftIntensityD(
                'image', prob=trans_conf.shift_intensity_prob, offsets=trans_conf.shift_intensity_offset,
            ),
        ])

    @property
    def dataset_weights(self):
        weights = torch.tensor(
            [
                dataset.weight * len(data_list)
                for dataset, data_list in zip(self.conf.datasets, self.data_lists)
            ],
            dtype=torch.float64,
        )
        return weights

    def __getitem__(self, index: tuple[int, int]):
        dataset_idx, sub_idx = index
        data_list = self.data_lists[dataset_idx]
        data = data_list[sub_idx]
        return self.transform(data)

class NestedRandomSampler(Sampler):
    def __init__(self, dataset: Dataset, num_samples: int, seed: int = 42):
        super().__init__()
        self.dataset = dataset
        self.dataset_weights = dataset.dataset_weights
        self.num_samples = num_samples
        self.G = torch.Generator()
        self.G.manual_seed(seed)

    @property
    def num_datasets(self):
        return self.dataset_weights.shape[0]

    def __iter__(self) -> Iterator[tuple[int, int]]:
        cnt = torch.zeros(self.num_datasets, dtype=torch.int64)
        buffer = [torch.empty(0, dtype=torch.int64) for _ in range(self.num_datasets)]
        for dataset_idx in torch.multinomial(
            self.dataset_weights, self.num_samples, True, generator=self.G,
        ):
            if cnt[dataset_idx] == buffer[dataset_idx].shape[0]:
                buffer[dataset_idx] = torch.randperm(len(self.dataset.data_lists[dataset_idx]))
                cnt[dataset_idx] = 0
            sub_idx = buffer[dataset_idx][cnt[dataset_idx]]
            cnt[dataset_idx] += 1
            yield dataset_idx.item(), sub_idx.item()

    def __len__(self):
        return self.num_samples

def _collate_fn(batch: list):
    ret = {}
    for x in batch:
        for key, value in x.items():
            ret.setdefault(key, []).append(value)
    return ret

class DataModule(ExpDataModuleBase):
    def __init__(self, *, dataset: DatasetConf, **kwargs):
        super().__init__(**kwargs)
        self.dataset_conf = dataset

    def train_dataloader(self):
        dataset = Dataset(self.dataset_conf)
        conf = self.dataloader_conf
        assert conf.train_batch_size is not None and conf.num_batches is not None
        sampler = NestedRandomSampler(dataset, conf.num_batches * conf.train_batch_size * self.world_size)
        if self.world_size > 1:
            # TODO: make this lazy (_DatasetSamplerWrapper). currently, it will consume the whole sampler at once
            sampler = DistributedSamplerWrapper(
                sampler,
                num_replicas=self.world_size,
                rank=self.trainer.global_rank,
                shuffle=False,
            )
        return DataLoader(
            dataset,
            batch_size=conf.train_batch_size,
            sampler=sampler,
            num_workers=conf.num_workers,
            pin_memory=conf.pin_memory,
            prefetch_factor=conf.prefetch_factor,
            persistent_workers=conf.persistent_workers,
            collate_fn=_collate_fn,
        )
