from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
import gc
import itertools as it
from logging import Logger
from pathlib import Path

import cytoolz
import einops
import numpy as np
from numpy import typing as npt
import pandas as pd
from scipy.stats import norm
import torch
from torch.nn import functional as nnf
from torchvision.transforms.v2 import functional as tvtf
from torchvision.transforms.v2.functional import to_dtype

from luolib import transforms as lt
from luolib.transforms import affine_resize
from luolib.types import tuple3_t
from luolib.utils import as_tensor, fall_back_none, get_cuda_device, process_map, save_pt_zst
from monai import transforms as mt
from monai.apps.detection.transforms.box_ops import apply_affine_to_boxes
from monai.data import MetaTensor, convert_box_to_standard_mode
from monai.data.box_utils import CornerCornerModeTypeB, box_centers, clip_boxes_to_image
from monai.transforms import generate_spatial_bounding_box

from mmmm.data import load_target_tax
from mmmm.data.defs import ORIGIN_SEG_DATA_ROOT, PROCESSED_SEG_DATA_ROOT
from mmmm.data.sparse import Sparse

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

def clip_intensity(image: torch.Tensor) -> mt.SpatialCrop:
    """clip the intensity in-place
    Returns:
        the cropper
    """
    x = image.view(image.shape[0], -1)
    if image.dtype == torch.uint8:
        minv = image.new_tensor(0.)
    else:
        minv = lt.quantile(x, _CLIP_LOWER, 1, True)
        maxv = lt.quantile(x, _CLIP_UPPER, 1, True)
        x.clamp_(minv, maxv)
    select_mask = image.new_empty((1, *image.shape[1:]), dtype=torch.bool)
    torch.any(x > minv, dim=0, keepdim=True, out=select_mask.view(1, -1))
    roi_start, roi_end = generate_spatial_bounding_box(select_mask)
    return mt.SpatialCrop(roi_start=roi_start, roi_end=roi_end)

class Processor(ABC):
    """
    TODO: check aspect ratio
    Attributes:
        name: name of the dataset to be processed by the processor
        orientation: if orientation is None, will determine it from the spacing
        min_aniso_ratio: minimum value for spacing_z / spacing_xy
        cuda_cache_th: cuda cache usage threshold to empty cache (in GiB)
        merged_targets: classes that merge multiple instances into a single mask, and will be excluded from bbox calculation from the mask
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
    merged_targets: set[str] = set()
    affine_atol: float = 1e-2

    def __init__(self, logger: Logger, *, max_workers: int, chunksize: int, override: bool):
        self.tax = load_target_tax()
        self.logger = logger
        if self.max_workers is None or override:
            self.max_workers = max_workers
        if self.chunksize is None or override:
            self.chunksize = chunksize

    def _get_category(self, name: str):
        return self.tax[name].category

    @property
    def dataset_root(self):
        return ORIGIN_SEG_DATA_ROOT / self.name

    @property
    def output_name(self) -> str:
        return self.name

    @property
    def output_root(self):
        return PROCESSED_SEG_DATA_ROOT / self.output_name

    @property
    def case_data_root(self):
        return self.output_root / 'data'

    @abstractmethod
    def get_data_points(self) -> list[DataPoint]:
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
        assert ((mask == 0) | (mask == 1)).all()
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

    def load_masks(self, data_point: SegDataPoint, images: MetaTensor) -> tuple[list[str], MetaTensor]:
        device = get_cuda_device()
        if isinstance(data_point, MultiLabelMultiFileDataPoint):
            targets, mask_paths = zip(*data_point.masks)
            targets = list(targets)
            # NOTE: make sure that mask loader returns bool tensor
            mask_list: list[MetaTensor] = process_map(
                self.mask_loader, mask_paths,
                new_mapper=False, disable=True, max_workers=min(4, len(mask_paths)),
            )
            affine = mask_list[0].affine
            for mask in mask_list[1:]:
                self._check_affine(affine, mask.affine)
            masks: MetaTensor = torch.cat(mask_list).to(device=device)
            masks.affine = affine
            masks = self._ensure_binary_mask(masks)
        elif isinstance(data_point, MultiClassDataPoint):
            class_mapping = data_point.class_mapping
            targets = list(class_mapping.values())
            label: MetaTensor = self.mask_loader(data_point.label).to(dtype=torch.int16, device=device)
            assert label.shape[0] == 1
            class_ids = torch.tensor([c for c in class_mapping], dtype=torch.int16, device=device)
            for _ in range(label.ndim - 1):
                class_ids = class_ids[..., None]  # make broadcastable
            masks = label == class_ids
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

    def process(self, limit: int | None = None, empty_cache: bool = False, raise_error: bool = False):
        data_points = self.get_data_points()
        assert len(data_points) > 0
        assert len(data_points) == len(set(data_point.key for data_point in data_points)), "key must be unique within the dataset"
        pending_data_points = [*filter(lambda p: not (self.case_data_root / p.key).exists(), data_points)]
        if limit is not None:
            pending_data_points = pending_data_points[:limit]
        if self.orientation is None:
            self.logger.warning('orientation is not specified, will infer from the metadata')
        if len(pending_data_points) > 0:
            self.logger.info(f'{len(pending_data_points)} data points to be processed')
            self.case_data_root.mkdir(parents=True, exist_ok=True)
            process_map(
                self.process_data_point,
                pending_data_points, it.repeat(empty_cache), it.repeat(raise_error),
                max_workers=self.max_workers, chunksize=self.chunksize, ncols=80,
            )
        info_list: list[dict | None] = process_map(
            self._collect_info, data_points,
            max_workers=self.max_workers, chunksize=10, disable=True,
        )
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

    def _check_targets(self, targets: Iterable[str]):
        unknown_targets = [*filter(lambda name: name not in self.tax, targets)]
        if len(unknown_targets) > 0:
            print('unknown targets:', unknown_targets)
            raise ValueError

    @staticmethod
    def _generate_bbox_from_mask(masks: torch.BoolTensor) -> torch.LongTensor:
        bbox_t = mt.BoundingRect()
        boxes = bbox_t(masks).astype(np.int64)
        boxes = convert_box_to_standard_mode(boxes, CornerCornerModeTypeB)
        return torch.from_numpy(boxes)

    def _convert_annotations(
        self, targets: list[str], masks: torch.BoolTensor | None, boxes: torch.LongTensor | None,
    ) -> tuple[list[Sparse.Annotation], torch.LongTensor | None]:
        ret = []
        class_positions = None if masks is None and boxes is None else []
        if masks is not None:
            mask_sizes: list[int] = einops.reduce(masks, 'c ... -> c', 'sum').tolist()
            assert boxes is None
            boxes = Processor._generate_bbox_from_mask(masks)
        offset = 0
        for target, group in cytoolz.groupby(lambda x: x[1], enumerate(targets)).items():
            indexes: list[int] = [i for i, _ in group]
            merged = target in self.merged_targets
            num = len(group)
            if merged:
                assert num == 1
            class_boxes = None if boxes is None else torch.stack([boxes[i] for i in indexes])
            if masks is not None:
                merged_mask = einops.reduce(masks[indexes], 'n ... -> ...', 'any')
                positions = merged_mask.nonzero().short()
            elif class_boxes is not None:
                positions = box_centers(class_boxes).floor().short()
            if class_positions is not None:
                if positions.shape[0] > self.max_class_positions:
                    positions = positions[
                        # deterministic? random!
                        torch.randint(positions.shape[0], (self.max_class_positions,), device=positions.device),
                    ]
                class_positions.append(positions)
                position_offset = (offset, (offset := offset + positions.shape[0]))
            else:
                position_offset = None
            ret.append(
                Sparse.Annotation(
                    name=target,
                    num=num,
                    merged=merged,
                    position_offset=position_offset,
                    boxes=None if boxes is None else torch.stack([boxes[i] for i in indexes]).numpy(),
                    masks=None if masks is None else [
                        Sparse.Annotation.MaskInfo(i, mask_sizes[i])
                        for i in indexes
                    ],
                ),
            )
        if class_positions is not None:
            class_positions = torch.cat(class_positions)
        return ret, class_positions

    def process_data_point(self, data_point: DataPoint, empty_cache: bool, raise_error: bool):
        """
        Returns:
            (metadata, human-readable information saved in csv) if process success, else None
        """
        self.key = key = data_point.key
        try:
            if empty_cache:
                device = get_cuda_device()
                cuda_cache = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
                if cuda_cache > self.cuda_cache_th << 30:
                    gc.collect()
                    torch.cuda.empty_cache()
            modalities, images = self.load_images(data_point)
            targets, neg_targets, masks, boxes = self.load_annotations(data_point, images)
            if len(self.merged_targets) > 0:
                assert boxes is None, 'only masks can be merged'
            self._check_targets(targets)
            self._check_targets(neg_targets)
            # 1. orientation
            # start to accumulate for boxes transform
            if boxes is not None:
                assert (boxes[:, 3:] <= boxes.new_tensor(images.shape[1:])).all()
            origin_affine = images.affine
            orient = mt.Orientation(self.get_orientation(images))
            images: MetaTensor = orient(images).contiguous()  # type: ignore
            if masks is not None:
                masks: MetaTensor = orient(masks).contiguous()  # type: ignore
                assert images.shape[1:] == masks.shape[1:]
                self._check_affine(images.affine, masks.affine)
            spacing: torch.DoubleTensor = images.pixdim
            info = {
                'key': key,
                **{f'shape-o-{i}': s for i, s in enumerate(images.shape[1:])},
                **{f'space-o-{i}': s.item() for i, s in enumerate(spacing)}
            }
            # 2. clip intensity, and crop the images & masks
            cropper = clip_intensity(images)
            images: MetaTensor = cropper(images)  # type: ignore
            # 3. compute resize (default: adapt to self.max_smaller_edge and self.min_aniso_ratio)
            new_spacing, new_shape = self.compute_resize(spacing, images.shape[1:])
            info.update({
                **{f'shape-{i}': s.item() for i, s in enumerate(new_shape)},
                **{f'space-{i}': s.item() for i, s in enumerate(new_spacing)},
            })
            # 4. apply resize & intensity normalization, save processed results
            # 4.1. normalize and save images
            images, mean, std = self.normalize_image(images, new_shape)
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
                save_pt_zst(as_tensor(masks).cpu(), save_dir / 'masks.pt.zst')
            else:
                # apply the accumulated transform to boxes
                boxes_f = apply_affine_to_boxes(boxes, torch.linalg.solve(images.affine, origin_affine))
                boxes_f, keep = clip_boxes_to_image(boxes_f, new_shape)
                assert keep.all()
                boxes = torch.empty_like(boxes_f, dtype=torch.int64)
                boxes[:, :3] = boxes_f[:, :3].floor()
                boxes[:, 3:] = boxes_f[:, 3:].ceil()
            annotations, class_positions = self._convert_annotations(targets, masks, boxes)
            if class_positions is not None:
                # the size of clas_positions is small, and we can use mmap, thus not compressed
                torch.save(class_positions.cpu(), save_dir / 'class_positions.pt')
            annotations = cytoolz.groupby(lambda annotation: self._get_category(annotation.name), annotations)
            neg_targets = cytoolz.groupby(lambda name: self._get_category(name), neg_targets)
            # 4.4 handle and save sparse information
            sparse = Sparse(
                spacing=new_spacing,
                shape=new_shape,
                mean=mean.cpu().numpy(),
                std=std.cpu().numpy(),
                modalities=modalities,
                annotations=annotations,
                neg_targets={category: set(targets) for category, targets in neg_targets.items()},
                complete_anomaly=data_point.complete_anomaly,
                extra=data_point.extra,
            )
            (save_dir / 'sparse.json').write_text(sparse.to_json())
            # 4.5. save info, wait for collection
            pd.to_pickle(info, save_dir / 'info.pkl')
            # 5. complete
            save_dir.rename(save_dir.with_name(key))
        except Exception as e:
            self.logger.error(key)
            self.logger.error(e)
            if raise_error:
                raise e
            else:
                import traceback
                self.logger.error(traceback.format_exc())

    def normalize_image(self, images: MetaTensor, new_shape: npt.NDArray[np.int64]) -> tuple[MetaTensor, torch.Tensor, torch.Tensor]:
        # 1. rescale to [0, 1]
        if images.dtype == torch.uint8:
            images = to_dtype(images, scale=True)
        else:
            images = images.float()
            minv, maxv = images.amin((1, 2, 3), keepdim=True), images.amax((1, 2, 3), keepdim=True)
            images = (images - minv) / (maxv - minv)
        # 2. resize
        images = affine_resize(images, new_shape, dtype=torch.float16).float()
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
