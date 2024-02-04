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

from mmmm.data.defs import ORIGIN_SEG_DATA_ROOT, PROCESSED_SEG_DATA_ROOT
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

_CLIP_LOWER = norm.cdf(-3)
_CLIP_UPPER = norm.cdf(3)

def clip_intensity(img: torch.Tensor, is_natural: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """clip the intensity in-place
    Args:
        is_natural: whether the modality of the image is natural (RGB or gray scale)
    Returns:
        intensity lower & upper threshold for clipping
    """
    if is_natural:
        minv = img.new_tensor(0.)
        maxv = img.new_tensor(255.)
    else:
        minv = lt.quantile(img, _CLIP_LOWER)
        maxv = lt.quantile(img, _CLIP_UPPER)
        img.clamp_(minv, maxv)
    return minv, maxv

def crop(images: dict[str, MetaTensor], masks: dict[str, MetaTensor], crop_mask: torch.Tensor) -> tuple[dict[str, MetaTensor], dict[str, MetaTensor]]:
    data = {
        **{
            ('image', modality): image
            for modality, image in images.items()
        },
        **{
            ('mask', name): mask
            for name, mask in masks.items()
        },
        'crop_mask': crop_mask,
    }
    data = mt.CropForegroundD(data.keys(), 'crop_mask', start_coord_key=None, end_coord_key=None)(data)
    data.pop('crop_mask')
    images = {modality: image for (image_type, modality), image in data.items() if image_type == 'image'}
    masks = {name: mask for (image_type, name), mask in data.items() if image_type == 'mask'}
    return images, masks

def is_natural_modality(modality: str):
    return modality.startswith('RGB') or modality.startswith('gray')

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
    mask_batch_size: int = 8

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

    def load_images(self, data_point: DataPoint):
        loader = self.get_image_loader()
        device = get_cuda_device()
        return {
            modality: loader(path).to(device)
            for modality, path in data_point.images.items()
        }

    @abstractmethod
    def get_mask_loader(self) -> ImageLoader:
        pass

    def load_masks(self, data_point: DataPoint) -> dict[str, MetaTensor]:
        """
        Returns:
            seg-tax name ↦ segmentation mask
        """
        loader = self.get_mask_loader()
        if isinstance(data_point, MultiLabelMultiFileDataPoint):
            mask_paths = list(data_point.masks.values())
            masks = process_map(
                loader, mask_paths,
                new_mapper=False, disable=True, max_workers=min(4, len(mask_paths)),
            )
            masks = torch.stack(masks).to(get_cuda_device())
            return {
                name: masks[i]
                for i, name in enumerate(data_point.masks)
            }
        else:
            raise NotImplementedError

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

    def orient(self, images: dict[str, MetaTensor], masks: dict[str, MetaTensor]):
        if self.orientation is None:
            image = cytoolz.first(images.values())
            codes = ['RAS', 'ASR', 'SRA']
            diff = np.empty(len(codes))
            dummy = MetaTensor(torch.empty(1, 1, 1, 1), image.affine)
            for i, code in enumerate(codes):
                orientation = mt.Orientation(code, lazy=True)
                spacing = orientation(dummy).pixdim
                diff[i] = abs(spacing[1] - spacing[2])
            orientation = codes[diff.argmin()]
        else:
            orientation = self.orientation

        def _apply(data: dict[str, MetaTensor]) -> dict[str, MetaTensor]:
            keys = list(data.keys())
            data = mt.OrientationD(keys, orientation)(data)
            data = mt.ToTensorD(keys)(data)
            return data

        return _apply(images), _apply(masks)

    def process_data_point(self, data_point: DataPoint) -> tuple[dict, dict] | None:
        """
        Returns:
            (metadata, human-readable information saved in csv) if process success, else None
        """
        key = data_point.key
        try:
            images = self.load_images(data_point)
            masks = self.load_masks(data_point)
            images, masks = self.orient(images, masks)
            # check shape & spacing consistency
            sample_image = cytoolz.first(images.values())
            shape = sample_image.shape
            assert all(image.shape == shape for image in images.values())
            assert all(mask.shape == shape for mask in masks.values())

            info = {
                'key': key,
                **{f'shape-o-{i}': s for i, s in enumerate(shape[1:])},
                **{f'space-o-{i}': s.item() for i, s in enumerate(sample_image.pixdim)}
            }
            # 1. clip intensity, compute crop mask
            crop_mask = torch.zeros_like(sample_image, dtype=torch.bool)  # 0 for cropping
            for modality, image in images.items():
                minv, maxv = clip_intensity(image, is_natural_modality(modality))
                crop_mask |= image > minv
            # 2. crop images and masks
            images, masks = crop(images, masks, crop_mask)
            # 3. compute resize (adapt to self.max_smaller_edge and self.min_aniso_ratio)
            sample_image = cytoolz.first(images.values())
            if self.max_smaller_edge < (smaller_edge := min(sample_image.shape[2:])):
                scale_xy = smaller_edge / self.max_smaller_edge
            else:
                scale_xy = 1.
            new_spacing_xy = sample_image.pixdim[2:].min().item() * scale_xy
            new_spacing_z = max(sample_image.pixdim[0].item(), new_spacing_xy * self.min_aniso_ratio)
            new_spacing = np.array([new_spacing_z, new_spacing_xy, new_spacing_xy])
            scale_z = new_spacing_z / sample_image.pixdim[0].item()
            scale = np.array([scale_z, scale_xy, scale_xy])
            new_shape = np.round(np.array(sample_image.shape[1:]) / scale).astype(np.int32)
            # 4. apply resize & intensity normalization, save processed results
            save_dir = self.output_root / 'data' / key
            save_dir.mkdir(exist_ok=True, parents=True)
            for modality, image in images.items():
                image = self.normalize_image(image, modality, new_shape)
                np.save(save_dir / f'{modality}.npy', image.cpu().numpy().astype(np.float16))
            masks_save_dir = save_dir / 'masks'
            masks_save_dir.mkdir(exist_ok=True, parents=True)
            masks, positive_classes = self.resize_masks(masks, new_shape)
            for name, mask in masks.items():
                np.save(masks_save_dir / f'{name}.npy', mask.cpu().numpy())
            info.update({
                **{f'shape-{i}': s.item() for i, s in enumerate(new_shape)},
                **{f'space-{i}': s.item() for i, s in enumerate(new_spacing)},
            })
            meta = {
                'key': data_point.key,
                'spacing': new_spacing,
                'shape': new_shape,
                'positive_classes': positive_classes,
            }
            return meta, info
        except Exception as e:
            self.logger.error(key)
            self.logger.error(e)
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def normalize_image(self, img: torch.Tensor, modality: str, new_shape: npt.NDArray[np.int32]):
        is_natural = is_natural_modality(modality)
        # 1. translate intensity to [0, ...] for padding during resizing
        if not is_natural:
            img = img - img.min()
        # 2. resize
        if new_shape[0] == img.shape[1]:
            if not np.array_equal(new_shape[1:], img.shape[2:]):
                img = tvtf.resize(
                    img, new_shape[1:].tolist(), tvt.InterpolationMode.BICUBIC, antialias=True,
                )
        else:
            scale = img.shape[1:] / new_shape
            anti_aliasing_filter = mt.GaussianSmooth((scale - 1) / 2)
            filtered = anti_aliasing_filter(img)
            resizer = mt.Affine(
                scale_params=scale.tolist(),
                spatial_size=new_shape.tolist(),
                mode=GridSampleMode.BICUBIC,
                image_only=True,
            )
            img = resizer(filtered)
        # 3. rescale intensity fo [0, 1]
        img.clamp_(0)
        maxv = 255 if is_natural else img.max()
        return img / maxv

    def resize_masks(self, masks: dict[str, torch.Tensor], new_shape: npt.NDArray[np.int32]) -> tuple[dict[str, torch.Tensor], list[str]]:
        new_shape = tuple(new_shape.tolist())
        do_resize = cytoolz.first(masks.values()).shape[1:] == new_shape
        if do_resize:
            masks = dict(masks)
        positive_classes = []
        for batch in cytoolz.partition_all(self.mask_batch_size, masks.items()):
            names, batch = zip(*batch)
            batch = torch.stack(batch)
            if do_resize:
                batch = nnf.interpolate(batch.float(), new_shape, mode='trilinear')
                batch = batch > 0.5
            positive_mask = einops.reduce(batch, 'c ... -> c', 'any')
            for i, name in enumerate(names):
                if do_resize:
                    masks[name] = batch[i]
                if positive_mask[i]:
                    positive_classes.append(name)
        return masks, positive_classes

class Default3DLoaderMixin:
    reader = None

    def get_image_loader(self) -> ImageLoader:
        return mt.LoadImage(self.reader, image_only=True, dtype=None, ensure_channel_first=True)

    def get_mask_loader(self) -> ImageLoader:
        return mt.LoadImage(self.reader, image_only=True, dtype=torch.bool, ensure_channel_first=True)
