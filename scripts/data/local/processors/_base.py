from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial
import gc
import itertools as it
from logging import Logger
from pathlib import Path
from typing import TypeVar

import cytoolz
import einops
import numpy as np
from numpy import typing as npt
import orjson
import pandas as pd
from scipy.stats import norm
import torch
from torch.nn import functional as nnf
from torchvision.transforms.v2 import functional as tvtf
from torchvision.transforms.v2.functional import to_dtype

from luolib import transforms as lt
from luolib.transforms import affine_resize
from luolib.transforms.box_ops import apply_affine_to_boxes_int
from luolib.types import tuple3_t
from luolib.utils import as_tensor, fall_back_none, get_cuda_device, process_map, save_pt_zst
from monai import transforms as mt
from monai.data import MetaTensor, convert_box_to_standard_mode
from monai.data.box_utils import CornerCornerModeTypeB, box_centers, clip_boxes_to_image
from monai.transforms import generate_spatial_bounding_box

from mmmm.data import load_target_tax
from mmmm.data.defs import ORIGIN_LOCAL_DATA_ROOT, PROCESSED_LOCAL_DATA_ROOT, Split
from mmmm.data.sparse import Sparse
from mmmm.data.target_tax import TargetCategory
from monai.utils import GridSampleMode

@dataclass(kw_only=True)
class DataPoint:
    """
    Attributes:
        key: unique identifier within the dataset
        images: co-registered images, map: modality ↦ file path
        complete_anomaly: all anomalies observable in the images are included in the label
    """
    key: str
    images: dict[str, Path]
    complete_anomaly: bool = False
    extra: ... = None

@dataclass(kw_only=True)
class SegDataPoint(DataPoint):
    pass

"""
1. multi-label, multi-file (single-channel, binary values)
2. multi-class, single-file (single-channel, multiple values)
"""

@dataclass(kw_only=True)
class MultiLabelMultiFileDataPoint(SegDataPoint):
    """
    Attributes:
        masks: list of (target name, path to the segmentation mask)
    """
    masks: list[tuple[str, Path]]

@dataclass(kw_only=True)
class MultiClassDataPoint(SegDataPoint):
    label: Path
    class_mapping: dict[int, str]

_CLIP_LOWER = norm.cdf(-3)
_CLIP_UPPER = norm.cdf(3)

def clip_intensity(image: torch.Tensor, exclude_min: bool = False) -> mt.SpatialCrop:
    """clip the intensity in-place
    Returns:
        the cropper
    """
    x = image.view(image.shape[0], -1)
    if image.dtype == torch.uint8:
        minv = image.new_tensor(0.)
    else:
        ref = x
        if exclude_min:
            assert x.shape[0] == 1
            ref = ref[ref > ref.min()][None]
        minv = lt.quantile(ref, _CLIP_LOWER, 1, True)
        maxv = lt.quantile(ref, _CLIP_UPPER, 1, True)
        x.clamp_(minv, maxv)
    select_mask = image.new_empty((1, *image.shape[1:]), dtype=torch.bool)
    torch.any(x > minv, dim=0, keepdim=True, out=select_mask.view(1, -1))
    roi_start, roi_end = generate_spatial_bounding_box(select_mask)
    return mt.SpatialCrop(roi_start=roi_start, roi_end=roi_end)

class SkipException(Exception):
    pass

tensor_t = TypeVar('tensor_t', bound=torch.Tensor)

class Processor(ABC):
    """
    TODO: check aspect ratio
    Attributes:
        name: name of the dataset to be processed by the processor
        orientation: if orientation is None, will determine it from the spacing
        min_aniso_ratio: minimum value for spacing_z / spacing_xy
        cuda_cache_th: cuda cache usage threshold to empty cache (in GiB)
        semantic_targets: classes that merge multiple instances into a single mask
    """
    name: str
    max_workers: int | None = None
    chunksize: int | None = None
    orientation: str | None = None
    max_smaller_edge: int = 512
    min_aniso_ratio: float = 0.5
    mask_batch_size: int = 8
    max_class_positions: int = 10000
    cuda_cache_th: int = 15
    semantic_targets: set[str] = set()
    affine_atol: float = 1e-2
    assert_local: bool = True
    clip_min: bool = False

    def __init__(self, logger: Logger, *, max_workers: int, chunksize: int, override: bool):
        self.tax = load_target_tax()
        self.logger = logger
        if self.max_workers is None or override:
            self.max_workers = max_workers
        if self.chunksize is None or override:
            self.chunksize = chunksize
        self._check_targets(self.semantic_targets)

    @property
    def device(self):
        return get_cuda_device()

    def _check_cuda_cache(self):
        device = get_cuda_device()
        cuda_cache = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
        if cuda_cache > self.cuda_cache_th << 30:
            gc.collect()
            torch.cuda.empty_cache()

    def _get_category(self, name: str):
        return self.tax[name].category

    @property
    def dataset_root(self):
        return ORIGIN_LOCAL_DATA_ROOT / self.name

    @property
    def output_name(self) -> str:
        return self.name

    @property
    def output_root(self):
        return PROCESSED_LOCAL_DATA_ROOT / self.output_name

    @property
    def case_data_root(self):
        return self.output_root / 'data'

    @abstractmethod
    def get_data_points(self) -> tuple[list[DataPoint], dict[Split, list[str]] | None]:
        pass

    def image_loader(self, path: Path) -> MetaTensor:
        raise NotImplementedError

    def load_images(self, data_point: DataPoint) -> tuple[list[str], MetaTensor]:
        modalities = []
        images = []
        affine = None
        has_rgb = False
        for modality, path in data_point.images.items():
            if has_rgb:
                raise ValueError('multiple images including RGB is not supported')
            modalities.append(modality)
            image = self.image_loader(path)
            if affine is None:
                affine = image.affine
            else:
                self._check_affine(affine, image.affine, atol=0)
            # is_natural_list.append(is_natural)
            if image.shape[0] == 3:
                has_rgb = True
            images.append(image)
        return modalities, torch.cat(images).to(device=get_cuda_device())

    def mask_loader(self, path: Path) -> MetaTensor:
        raise NotImplementedError

    def _ensure_binary_mask(self, mask: torch.Tensor):
        assert not mask.is_floating_point()
        assert (0 <= mask).all() and (mask <= 1).all()
        return mask.bool()

    def _check_affine(self, affine1: torch.Tensor, affine2: torch.Tensor, atol: float | None = None):
        atol = fall_back_none(atol, self.affine_atol)
        assert torch.allclose(affine1, affine2, atol=atol)

    def create_seg_annotations(self, targets: list[str], masks: MetaTensor) -> tuple[list[str], set[str], MetaTensor | None, None]:
        pos_mask = einops.reduce(masks, 'c ... -> c', 'any').cpu().numpy()
        targets = np.array(targets)
        pos_targets = targets[pos_mask].tolist()
        neg_targets = set(targets[~pos_mask].tolist())
        assert len(set(pos_targets) & neg_targets) == 0
        if pos_mask.any():
            masks = masks[pos_mask]
        else:
            masks = None
        return pos_targets, neg_targets, masks, None

    def _load_multi_class_masks(self, label_path: Path, class_mapping: dict[int, str]):
        targets = list(class_mapping.values())
        label: MetaTensor = self.mask_loader(label_path).to(dtype=torch.int16, device=self.device)
        assert label.shape[0] == 1
        class_ids = torch.tensor([c for c in class_mapping], dtype=torch.int16, device=self.device)
        for _ in range(label.ndim - 1):
            class_ids = class_ids[..., None]  # make broadcastable
        masks = label == class_ids
        return targets, masks

    def load_masks(self, data_point: SegDataPoint, images: MetaTensor) -> tuple[list[str], MetaTensor]:
        if isinstance(data_point, MultiLabelMultiFileDataPoint):
            targets, mask_paths = zip(*data_point.masks)
            targets = list(targets)
            # NOTE: make sure that mask loader returns bool tensor
            mask_list: list[MetaTensor] = list(map(self.mask_loader, mask_paths))
            affine = mask_list[0].affine
            for mask in mask_list[1:]:
                self._check_affine(affine, mask.affine)
            masks: MetaTensor = torch.cat(mask_list).to(device=self.device)
            masks.affine = affine
            masks = self._ensure_binary_mask(masks)
        elif isinstance(data_point, MultiClassDataPoint):
            targets, masks = self._load_multi_class_masks(data_point.label, data_point.class_mapping)
        else:
            raise NotImplementedError
        return targets, masks

    def load_annotations(
        self, data_point: DataPoint, images: MetaTensor,
    ) -> tuple[list[str], set[str], MetaTensor | None, torch.Tensor | None]:
        """
        NOTE: metadata for mask should be preserved to match with the image
        Args:
            images: the loaded images, can be useful when some metadata (e.g., affine, shape) is needed
        Returns:
            - list of positive targets
            - set of negative targets
            - segmentation masks (dtype = torch.bool).
              should follow the order of positive targets.
              return meta tensor for affine checking.
              None when unavailable.
            - bounding box in StandardMode (CornerCornerModeTypeA).
              should follow the order of positive targets.
              None when unavailable.
        """
        if isinstance(data_point, SegDataPoint):
            targets, masks = self.load_masks(data_point, images)
            return self.create_seg_annotations(targets, masks)
        else:
            raise NotImplementedError

    def _collect_info(self, data_point: DataPoint):
        if (path := self.case_data_root / data_point.key / 'info.pkl').exists():
            return pd.read_pickle(path)
        return None

    def process(self, limit: int | tuple[int, int] | None = None, empty_cache: bool = False, raise_error: bool = False):
        data_points, split = self.get_data_points()
        assert len(data_points) > 0
        assert len(data_points) == len(set(data_point.key for data_point in data_points)), "key must be unique within the dataset"
        pending_data_points = [*filter(lambda p: not (self.case_data_root / p.key).exists(), data_points)]
        if isinstance(limit, int):
            pending_data_points = pending_data_points[:limit]
        elif isinstance(limit, tuple) and len(limit) == 2:
            pending_data_points = pending_data_points[slice(*limit)]
        if self.orientation is None:
            self.logger.warning('orientation is not specified, will infer from the metadata')
        if len(pending_data_points) > 0:
            self.logger.info(f'{len(pending_data_points)} data points to be processed')
            self.case_data_root.mkdir(parents=True, exist_ok=True)
            process_map(
                partial(self.process_data_point, empty_cache=empty_cache, raise_error=raise_error),
                pending_data_points,
                max_workers=self.max_workers, chunksize=self.chunksize, ncols=80,
            )
        info_list: list[dict | None] = process_map(
            self._collect_info, data_points,
            max_workers=self.max_workers, chunksize=10, disable=True,
        )
        if split:
            split = {
                str(key): value
                for key, value in split.items()
            }
            (self.output_root / 'split.json').write_bytes(orjson.dumps(split, option=orjson.OPT_INDENT_2))
        info_list = [*filter(lambda x: x is not None, info_list)]
        if len(info_list) > 0:
            info = pd.DataFrame.from_records(info_list, index='key')
            info.to_csv(self.output_root / 'info.csv')
            info.to_excel(self.output_root / 'info.xlsx', freeze_panes=(1, 1))

    def get_orientation(self, images: MetaTensor):
        if self.orientation is not None:
            return self.orientation
        codes = ['RAS', 'ASR', 'SRA']
        diff = np.empty(len(codes))
        shape_diff = np.empty(len(codes), dtype=np.int64)
        dummy = MetaTensor(torch.empty(0, *images.shape[1:], device=images.device), images.affine)
        for i, code in enumerate(codes):
            orientation = mt.Orientation(code)
            dummy_t: MetaTensor = orientation(dummy)
            diff[i] = abs(dummy_t.pixdim[1] - dummy_t.pixdim[2])
            shape_diff[i] = abs(dummy_t.shape[2] - dummy_t.shape[3])
        if diff.max() - diff.min() > 1e-3 * diff.min():
            orientation = codes[diff.argmin()]
        elif shape_diff.min() == 0 and shape_diff.max() != 0:
            # select plane with edges of equal lengths
            orientation = codes[shape_diff.argmin()]
        else:
            # fall back to SRA
            orientation = 'SRA'
        return orientation

    def compute_resize(
        self, spacing: torch.DoubleTensor, shape: tuple3_t[int],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
        if self.max_smaller_edge < (smaller_edge := min(shape[1:])):
            scale_xy = smaller_edge / self.max_smaller_edge
        else:
            scale_xy = 1.
        new_spacing_xy = spacing[1:].min().item() * scale_xy
        new_spacing_z = max(spacing[0].item(), new_spacing_xy * self.min_aniso_ratio)
        new_spacing = np.array([new_spacing_z, new_spacing_xy, new_spacing_xy])
        scale_z = new_spacing_z / spacing[0].item()
        scale = np.array([scale_z, scale_xy, scale_xy])
        new_shape = (np.array(shape) / scale).round().astype(np.int64)
        return new_spacing, new_shape

    def _is_unknown_target(self, target: str):
        return target not in self.tax

    def _check_targets(self, targets: Iterable[str]):
        unknown_targets = [*filter(self._is_unknown_target, targets)]
        if len(unknown_targets) > 0:
            print('unknown targets:', unknown_targets)
            raise ValueError

    @staticmethod
    def _generate_bbox_from_mask(masks: torch.BoolTensor) -> torch.LongTensor:
        bbox_t = mt.BoundingRect()
        boxes = bbox_t(masks).astype(np.int64)
        boxes = convert_box_to_standard_mode(boxes, CornerCornerModeTypeB)
        return torch.from_numpy(boxes)

    def _group_targets(
        self, targets: list[str], masks: torch.BoolTensor | None, boxes: torch.LongTensor | None,
    ) -> tuple[list[Sparse.Target], torch.BoolTensor | None, torch.LongTensor | None]:
        if len(targets) == 0:
            return [], None, None
        groups = []
        permute = []
        if masks is not None:
            assert boxes is None
            boxes = Processor._generate_bbox_from_mask(masks)
        class_positions = []
        index_offset = position_offset = 0
        for target, group in cytoolz.groupby(lambda x: x[1], enumerate(targets)).items():
            indexes: list[int] = [i for i, _ in group]
            permute.extend(indexes)
            semantic = target in self.semantic_targets
            num_instances = len(group)
            if semantic:
                assert num_instances == 1
            target_boxes = boxes[indexes]
            if masks is None:
                positions = box_centers(target_boxes).floor().long()
                mask_sizes = None
            else:
                target_masks = masks[indexes]
                mask_sizes = target_masks.new_empty((target_masks.shape[0], ), dtype=torch.int64)
                for i in range(0, target_masks.shape[0], self.mask_batch_size):
                    mask_sizes[i:i + self.mask_batch_size] = (
                        einops.reduce(target_masks[i:i + self.mask_batch_size], 'n ... -> n', 'sum')
                    )
                mask_sizes = mask_sizes.cpu().numpy()
                merged_mask = einops.reduce(target_masks, 'n ... -> ...', 'any')
                positions = merged_mask.nonzero()
            if positions.shape[0] > self.max_class_positions:
                positions = positions[
                    # deterministic? random!
                    torch.randint(positions.shape[0], (self.max_class_positions,), device=positions.device),
                ]
            class_positions.append(positions)
            groups.append(
                Sparse.Target(
                    name=target,
                    semantic=semantic,
                    position_offset=(position_offset, (position_offset := position_offset + positions.shape[0])),
                    index_offset=(index_offset, (index_offset := index_offset + num_instances)),
                    mask_sizes=mask_sizes,
                    boxes=target_boxes.numpy(),
                ),
            )
        class_positions = torch.cat(class_positions)
        if masks is not None:
            masks = masks[permute]
        return groups, masks, class_positions

    def process_data_point(self, data_point: DataPoint, empty_cache: bool, raise_error: bool):
        """
        Returns:
            (metadata, human-readable information saved in csv) if process success, else None
        """
        self.key = key = data_point.key
        try:
            if empty_cache:
                self._check_cuda_cache()
            modalities, images = self.load_images(data_point)
            targets, neg_targets, masks, boxes = self.load_annotations(data_point, images)
            if targets:
                if masks is None:
                    if self.assert_local:
                        assert boxes is not None
                        assert len(targets) == boxes.shape[0]
                else:
                    assert boxes is None
                    assert len(targets) == masks.shape[0]
            if len(self.semantic_targets) > 0:
                assert boxes is None, 'only masks can be merged'
            self._check_targets(targets)
            self._check_targets(neg_targets)
            # 1. orientation
            if boxes is not None:
                assert (boxes[:, 3:] <= boxes.new_tensor(images.shape[1:])).all()
            # start to accumulate for boxes transform
            origin_affine = images.affine
            if images.shape[1] > 1:
                orient = mt.Orientation(self.get_orientation(images))
                images: MetaTensor = orient(images).contiguous()  # type: ignore
                if masks is not None:
                    masks: MetaTensor = orient(masks).contiguous()  # type: ignore
            if masks is not None:
                assert images.shape[1:] == masks.shape[1:]
                self._check_affine(images.affine, masks.affine)
            spacing: torch.DoubleTensor = images.pixdim
            info = {
                'key': key,
                'label': len(targets) > 0,
                **{f'shape-o-{i}': s for i, s in enumerate(images.shape[1:])},
                **{f'space-o-{i}': s.item() for i, s in enumerate(spacing)}
            }
            # 2. clip intensity, and crop the images & masks
            cropper = clip_intensity(images, exclude_min=self.clip_min)
            images: MetaTensor = cropper(images)  # type: ignore
            # 3. compute resize (default: adapt to self.max_smaller_edge and self.min_aniso_ratio)
            new_spacing, new_shape = self.compute_resize(spacing, images.shape[1:])
            info.update({
                **{f'shape-{i}': s.item() for i, s in enumerate(new_shape)},
                **{f'space-{i}': s.item() for i, s in enumerate(new_spacing)},
            })
            # 4. apply resize & intensity normalization, save processed results
            # 4.1. normalize and save images
            images, mean, std = self.normalize_image(images, new_shape, GridSampleMode.BICUBIC)
            save_dir = self.case_data_root / f'.{key}'
            save_dir.mkdir(exist_ok=True, parents=True)
            # 4.2. save image
            save_pt_zst(
                tvtf.to_dtype(as_tensor(images).cpu(), torch.uint8, scale=True),
                save_dir / f'images.pt.zst',
            )
            # 4.3. resize, filter, compress, and save masks & class positions
            if masks is not None:
                masks = cropper(masks)  # type: ignore
                masks = self.resize_masks(masks, new_shape)
                assert einops.reduce(masks, 'c ... -> c', 'any').all()
            elif boxes is not None:
                # apply the accumulated transform to boxes
                boxes = apply_affine_to_boxes_int(boxes, torch.linalg.solve(images.affine, origin_affine))
                boxes, keep = clip_boxes_to_image(boxes, new_shape)
                assert keep.all()
            targets, masks, class_positions = self._group_targets(targets, masks, boxes)
            if masks is not None:
                save_pt_zst(as_tensor(masks).cpu(), save_dir / 'masks.pt.zst')
            if class_positions is not None:
                # the size of clas_positions is small, and we can use mmap, thus not compressed
                torch.save(class_positions.cpu(), save_dir / 'class_positions.pt')
            assert len(targets) > 0 or len(neg_targets) > 0
            # 4.4 handle and save sparse information
            sparse = Sparse(
                spacing=new_spacing,
                shape=new_shape,
                mean=mean.cpu().numpy(),
                std=std.cpu().numpy(),
                modalities=modalities,
                targets=cytoolz.groupby(lambda target: self._get_category(target.name), targets),  # type: ignore
                neg_targets=cytoolz.groupby(lambda name: self._get_category(name), neg_targets),  # type: ignore
                complete_anomaly=data_point.complete_anomaly,
                extra=data_point.extra,
            )
            for category in TargetCategory:
                sparse.targets.setdefault(category, [])
                sparse.neg_targets.setdefault(category, [])
            (save_dir / 'sparse.json').write_text(sparse.to_json())
            # 4.5. save info, wait for collection
            pd.to_pickle(info, save_dir / 'info.pkl')
            # 5. complete
            save_dir.rename(save_dir.with_name(key))
        except SkipException:
            self.logger.info(f'skip {key}')
            (self.case_data_root / data_point.key).mkdir(parents=True)
        except Exception as e:
            self.logger.error(key)
            self.logger.error(e)
            if raise_error:
                raise e
            else:
                import traceback
                self.logger.error(traceback.format_exc())

    def normalize_image(
        self, images: MetaTensor, new_shape: npt.NDArray[np.int64], mode: GridSampleMode,
    ) -> tuple[MetaTensor, torch.Tensor, torch.Tensor]:
        # 1. rescale to [0, 1]
        if images.dtype == torch.uint8:
            images = to_dtype(images, scale=True)
        else:
            images = images.float()
            minv, maxv = images.amin((1, 2, 3), keepdim=True), images.amax((1, 2, 3), keepdim=True)
            images = (images - minv) / (maxv - minv)
        # 2. resize
        images = affine_resize(images, new_shape)
        images.clamp_(0, 1)
        # 3. calculate mean & std on non-zero values
        mean = images.new_empty((images.shape[0], ))
        std = images.new_empty((images.shape[0], ))
        for i in range(images.shape[0]):
            fg = images[i][images[i] > 0]
            mean[i] = fg.mean()
            std[i] = fg.std()
        return images, mean, std

    def resize_masks(self, masks: torch.BoolTensor, new_shape: npt.NDArray[np.int64]) -> torch.BoolTensor:
        new_shape = tuple(new_shape.tolist())
        if masks.shape[1:] == new_shape:
            resized_masks = masks
        else:
            resized_masks = masks.new_empty((masks.shape[0], *new_shape))
            for i in range(0, masks.shape[0], self.mask_batch_size):
                batch = masks[None, i:i + self.mask_batch_size]
                batch = nnf.interpolate(batch.float(), new_shape, mode='trilinear')
                resized_masks[i:i + self.mask_batch_size] = batch[0] > 0.5
        return resized_masks

INF_SPACING = 1e8

class _LoaderBase:
    def _adapt_to_3d(self, image: MetaTensor):
        image = image.unsqueeze(1)
        # the original affine information of a 2D image is discarded
        image.affine = np.diag([INF_SPACING, 1, 1, 1])
        return image

class DefaultImageLoaderMixin(_LoaderBase):
    image_reader = None
    image_dtype = None
    assert_gray_scale: bool = False

    def _check_natural_image(self, image: MetaTensor):
        if image.shape[0] == 4:
            # check RGBA
            assert image.dtype == torch.uint8
            assert (image[3] == 255).all()
            image = image[:3]
        assert image.shape[0] in (1, 3)
        if self.assert_gray_scale and image.shape[0] != 1:
            # check gray scale
            assert (image[0] == image[1]).all() and (image[0] == image[2]).all()
            image = image[0:1]
        return image

    def image_loader(self, path: Path) -> MetaTensor:
        loader = mt.LoadImage(self.image_reader, image_only=True, dtype=self.image_dtype, ensure_channel_first=True)
        image = loader(path)
        if image.ndim == 3:
            image = self._check_natural_image(image)
            image = self._adapt_to_3d(image)
        else:
            assert image.ndim == 4
        return image

class NaturalImageLoaderMixin(DefaultImageLoaderMixin):
    image_reader = 'pilreader'
    image_dtype = torch.uint8

class DefaultMaskLoaderMixin(_LoaderBase):
    mask_reader = None
    # int16 should be enough, right?
    mask_dtype = torch.int16

    def mask_loader(self, path: Path) -> MetaTensor:
        loader = mt.LoadImage(self.mask_reader, image_only=True, dtype=self.mask_dtype, ensure_channel_first=True)
        mask = loader(path)
        if mask.ndim == 3:
            mask = self._adapt_to_3d(mask)
        return mask
