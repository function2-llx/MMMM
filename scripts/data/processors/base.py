from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
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
from torchvision.transforms import v2 as tvt
from torchvision.transforms.v2 import functional as tvtf

from luolib import transforms as lt
from luolib.utils import get_cuda_device, process_map
from monai.data import MetaTensor
from monai import transforms as mt
from monai.utils import GridSampleMode

from mmmm.data.defs import Meta, ORIGIN_SEG_DATA_ROOT, PROCESSED_SEG_DATA_ROOT
from mmmm.data.seg_tax import SegClass

ImageLoader: type = Callable[[Path], MetaTensor]

@dataclass
class DataPoint:
    key: str
    """unique identifier in the dataset"""
    images: dict[str, Path]
    """co-registered images, map: modality ↦ file path"""

"""
1. multi-label, multi-file (single-channel, binary values)
2. multi-class, single-file (single-channel, multiple values)
3. multi-label, single-file (multi-channel, binary values)
"""

@dataclass
class MultiLabelMultiFileDataPoint(DataPoint):
    masks: dict[str, Path]
    """map: seg-tax name ↦ path to the segmentation mask in the image"""

@dataclass
class MultiClassDataPoint(DataPoint):
    label: Path
    class_mapping: dict[int, str]

_CLIP_LOWER = norm.cdf(-3)
_CLIP_UPPER = norm.cdf(3)

def clip_intensity(img: torch.Tensor, is_natural: bool) -> torch.BoolTensor:
    """clip the intensity in-place
    Args:
        is_natural: whether the modality of the image is natural (RGB or gray scale)
    Returns:
        the crop mask after clipping
    """
    x = img.view(img.shape[0], -1)
    if is_natural:
        minv = img.new_tensor(0.)
    else:
        minv = lt.quantile(x, _CLIP_LOWER, 1, True)
        maxv = lt.quantile(x, _CLIP_UPPER, 1, True)
        x.clamp_(minv, maxv)
    crop_mask = img.new_zeros((1, *img.shape[1:]), dtype=torch.bool)
    torch.any(x > minv, dim=0, keepdim=True, out=crop_mask.view(1, -1))
    return crop_mask

def crop(images: MetaTensor, masks: torch.BoolTensor, crop_mask: torch.BoolTensor) -> tuple[MetaTensor, torch.BoolTensor]:
    data = {
        'images': images,
        'masks': masks,
        'crop_mask': crop_mask,
    }
    data = mt.CropForegroundD(['images', 'masks'], 'crop_mask', start_coord_key=None, end_coord_key=None)(data)
    return data['images'], data['masks']

def is_natural_modality(modality: str):
    return modality.startswith('RGB') or modality.startswith('gray')

def is_rgb_modality(modality: str):
    return modality.startswith('RGB')

class Processor(ABC):
    name: str
    """name of the dataset to be processed by the processor"""
    max_workers: int | None = None
    chunksize: int | None = None
    orientation: str | None = None
    """if orientation is None, will determine it from the spacing"""
    max_smaller_edge: int = 512
    min_aniso_ratio: float = 0.7
    """minimum value for spacing_z / spacing_xy"""
    mask_batch_size: int = 32

    def __init__(self, seg_tax: dict[str, SegClass], logger: Logger, *, max_workers: int, chunksize: int, override: bool):
        self.seg_tax = seg_tax
        self.logger = logger
        if self.max_workers is None or override:
            self.max_workers = max_workers
        if self.chunksize is None or override:
            self.chunksize = chunksize

    @property
    def dataset_root(self):
        return ORIGIN_SEG_DATA_ROOT / self.name

    @property
    def output_name(self) -> str:
        return self.name

    @property
    def output_root(self):
        return PROCESSED_SEG_DATA_ROOT / self.output_name

    @abstractmethod
    def get_data_points(self) -> list[DataPoint]:
        pass

    @abstractmethod
    def get_image_loader(self) -> ImageLoader:
        pass

    def load_images(self, data_point: DataPoint) -> tuple[list[str], MetaTensor]:
        loader = self.get_image_loader()
        modalities = []
        images = []
        for modality, path in data_point.images.items():
            modalities.append(modality)
            images.append(loader(path))
        return modalities, torch.cat(images).to(get_cuda_device())

    @abstractmethod
    def get_mask_loader(self) -> ImageLoader:
        pass

    def load_masks(self, data_point: DataPoint) -> tuple[list[str], torch.BoolTensor]:
        """
        Returns:
            seg-tax name ↦ segmentation mask
        """
        loader = self.get_mask_loader()
        device = get_cuda_device()
        if isinstance(data_point, MultiLabelMultiFileDataPoint):
            masks = data_point.masks
            classes, mask_paths = zip(*masks.items())
            classes = list(classes)
            # NOTE: make sure that mask loader returns bool tensor
            masks = process_map(
                loader, mask_paths,
                new_mapper=False, disable=True, max_workers=min(4, len(mask_paths)),
            )
            masks = torch.cat(masks).to(dtype=torch.bool, device=device)
        elif isinstance(data_point, MultiClassDataPoint):
            class_mapping = data_point.class_mapping
            label = loader(data_point.label).to(dtype=torch.int16, device=device)
            assert label.shape[0] == 1
            class_ids = torch.tensor([c for c in class_mapping], dtype=torch.int16, device=device)
            for _ in range(label.ndim - 1):
                class_ids = class_ids[..., None]  # make broadcastable
            classes = list(class_mapping.values())
            masks = label == class_ids
        else:
            raise NotImplementedError
        return classes, masks

    def process(self):
        data_points = self.get_data_points()
        assert len(data_points) > 0
        assert len(data_points) == len(set(data_point.key for data_point in data_points))
        if (meta_path := self.output_root / 'meta.pkl').exists():
            processed_meta: pd.DataFrame = pd.read_pickle(meta_path)
        else:
            processed_meta = pd.DataFrame()
        data_points = list(filter(lambda p: p.key not in processed_meta.index, data_points))
        if len(data_points) == 0:
            return
        self.logger.info(f'{len(data_points)} data points to be processed')
        (self.output_root / 'data').mkdir(parents=True, exist_ok=True)
        meta, info = zip(
            *process_map(
                self.process_data_point,
                data_points,
                max_workers=self.max_workers, chunksize=self.chunksize, ncols=80,
            )
        )
        meta = pd.concat([processed_meta, pd.DataFrame.from_records(meta, index='key')])
        info = pd.DataFrame.from_records(info, index='key')
        if (info_path := self.output_root / 'info.csv').exists():
            processed_info = pd.read_csv(info_path, dtype={'key': 'string'}).set_index('key')
            info = pd.concat([processed_info, info])
        info.to_csv(info_path)
        info.to_excel(info_path.with_suffix('.xlsx'), freeze_panes=(1, 1))
        meta.to_pickle(meta_path)

    def get_orientation(self, images: MetaTensor):
        if self.orientation is not None:
            return self.orientation
        codes = ['RAS', 'ASR', 'SRA']
        diff = np.empty(len(codes))
        dummy = MetaTensor(torch.empty(1, 1, 1, 1), images.affine)
        for i, code in enumerate(codes):
            orientation = mt.Orientation(code)
            spacing = orientation(dummy).pixdim
            diff[i] = abs(spacing[1] - spacing[2])
        orientation = codes[diff.argmin()]
        return orientation

    def orient(self, images: MetaTensor, masks: torch.BoolTensor) -> tuple[MetaTensor, torch.BoolTensor]:
        orientation = self.get_orientation(images)
        trans = mt.Orientation(orientation)
        images, masks = map(lambda x: trans(x).contiguous(), [images, masks])
        return images, masks

    def process_data_point(self, data_point: DataPoint) -> tuple[dict, dict] | None:
        """
        Returns:
            (metadata, human-readable information saved in csv) if process success, else None
        """
        self.key = key = data_point.key
        try:
            modalities, images = self.load_images(data_point)
            is_natural = is_natural_modality(modalities[0])
            assert all(is_natural == is_natural_modality(modality) for modality in modalities[1:])
            if any(is_rgb_modality(modality) for modality in modalities):
                assert len(modalities) == 1, 'multiple RGB images is not supported'
            classes, masks = self.load_masks(data_point)
            for name in classes:
                assert name in self.seg_tax
            images, masks = self.orient(images, masks)
            assert images.shape[1:] == masks.shape[1:]
            info = {
                'key': key,
                **{f'shape-o-{i}': s for i, s in enumerate(images.shape[1:])},
                **{f'space-o-{i}': s.item() for i, s in enumerate(images.pixdim)}
            }
            # 1. clip intensity, compute crop mask
            crop_mask = clip_intensity(images, is_natural)
            # 2. crop images and masks
            images, masks = crop(images, masks, crop_mask)
            # 3. compute resize (adapt to self.max_smaller_edge and self.min_aniso_ratio)
            if self.max_smaller_edge < (smaller_edge := min(images.shape[2:])):
                scale_xy = smaller_edge / self.max_smaller_edge
            else:
                scale_xy = 1.
            new_spacing_xy = images.pixdim[2:].min().item() * scale_xy
            new_spacing_z = max(images.pixdim[0].item(), new_spacing_xy * self.min_aniso_ratio)
            new_spacing = np.array([new_spacing_z, new_spacing_xy, new_spacing_xy])
            scale_z = new_spacing_z / images.pixdim[0].item()
            scale = np.array([scale_z, scale_xy, scale_xy])
            new_shape = np.round(np.array(images.shape[1:]) / scale).astype(np.int32)
            info.update({
                **{f'shape-{i}': s.item() for i, s in enumerate(new_shape)},
                **{f'space-{i}': s.item() for i, s in enumerate(new_spacing)},
            })
            # 4. apply resize & intensity normalization, save processed results
            save_dir = self.output_root / 'data' / key
            save_dir.mkdir(exist_ok=True, parents=True)
            # for modality, image in images.items():
            images, mean, std = self.normalize_image(images, is_natural, new_shape)
            np.save(save_dir / f'images.npy', images.cpu().numpy().astype(np.float16))
            masks = self.resize_masks(masks, new_shape)
            positive_mask: torch.BoolTensor = einops.reduce(masks > 0, 'c ... -> c', 'any')
            masks = masks[positive_mask]
            np.save(save_dir / 'masks.npy', masks.cpu().numpy())
            positive_classes = [name for i, name in enumerate(classes) if positive_mask[i]]
            negative_classes = [name for i, name in enumerate(classes) if not positive_mask[i]]
            # TODO: save mask sizes
            meta: Meta = {
                'key': data_point.key,
                'spacing': new_spacing,
                'shape': new_shape,
                'mean': mean.cpu().numpy(),
                'std': std.cpu().numpy(),
                'modalities': modalities,
                'positive_classes': positive_classes,
                'negative_classes': negative_classes,
            }
            return meta, info
        except Exception as e:
            self.logger.error(key)
            self.logger.error(e)
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def normalize_image(self, images: torch.Tensor, is_natural: bool, new_shape: npt.NDArray[np.int32]):
        # 1. translate intensity to [0, ...] for padding during resizing
        if not is_natural:
            images = images - einops.reduce(images, 'c ... -> c', 'min')
        # 2. resize
        if new_shape[0] == images.shape[1]:
            if not np.array_equal(new_shape[1:], images.shape[2:]):
                images = tvtf.resize(
                    images, new_shape[1:].tolist(), tvt.InterpolationMode.BICUBIC, antialias=True,
                )
        else:
            scale = images.shape[1:] / new_shape
            anti_aliasing_filter = mt.GaussianSmooth((scale - 1) / 2)
            filtered = anti_aliasing_filter(images)
            resizer = mt.Affine(
                scale_params=scale.tolist(),
                spatial_size=new_shape.tolist(),
                mode=GridSampleMode.BICUBIC,
                image_only=True,
            )
            images = resizer(filtered)
        # 3. rescale intensity fo [0, 1]
        images.clamp_(0)
        maxv = 255 if is_natural else einops.reduce(images, 'c ... -> c', 'max')
        images = images / maxv
        # 4. Z-score normalization or by pre-defined statistics
        if is_natural:
            mean = images.new_tensor([[[[0.48145466, 0.4578275, 0.40821073]]]])
            std = images.new_tensor([[[[0.26862954, 0.26130258, 0.27577711]]]])
        else:
            mean = images.as_tensor().new_empty(images.shape[0], 1, 1, 1)
            std = images.as_tensor().new_empty(images.shape[0], 1, 1, 1)
            for i in range(images.shape[0]):
                fg = images[i][images[i] > 0]
                mean[i] = fg.mean()
                std[i] = fg.std()
        images = (images - mean) / std
        return images, mean[:, 0, 0, 0], std[:, 0, 0, 0]

    def resize_masks(self, masks: torch.BoolTensor, new_shape: npt.NDArray[np.int32]) -> torch.BoolTensor:
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

class Default3DImageLoaderMixin:
    image_reader = None

    def get_image_loader(self) -> ImageLoader:
        return mt.LoadImage(self.image_reader, image_only=True, dtype=None, ensure_channel_first=True)

class Binary3DMaskLoaderMixin:
    mask_reader = None

    def get_mask_loader(self) -> ImageLoader:
        return mt.LoadImage(self.mask_reader, image_only=True, dtype=torch.bool, ensure_channel_first=True)

class MultiClass3DMaskLoaderMixin:
    mask_reader = None

    def get_mask_loader(self) -> ImageLoader:
        return mt.LoadImage(self.mask_reader, image_only=True, dtype=torch.int16, ensure_channel_first=True)
