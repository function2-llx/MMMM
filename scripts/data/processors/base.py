from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from io import BytesIO
from logging import Logger
from pathlib import Path

import einops
import numpy as np
from numpy import typing as npt
import pandas as pd
from scipy.stats import norm
import torch
from torch.nn import functional as nnf
from torchvision.transforms import v2 as tvt
from torchvision.transforms.v2 import functional as tvtf
import zstandard as zstd

from luolib import transforms as lt
from luolib.utils import get_cuda_device, process_map
from monai import transforms as mt
from monai.data import MetaTensor
from monai.utils import GridSampleMode

from mmmm.data import load_target_tax
from mmmm.data.defs import Annotation, ORIGIN_SEG_DATA_ROOT, PROCESSED_SEG_DATA_ROOT, Sparse

ImageLoader: type = Callable[[Path], MetaTensor]

@dataclass
class DataPoint:
    key: str
    """unique identifier within the dataset"""
    images: dict[str, Path]
    """co-registered images, map: modality ↦ file path"""

"""
1. multi-label, multi-file (single-channel, binary values)
2. multi-class, single-file (single-channel, multiple values)
3. multi-label, single-file (multi-channel, binary values)
"""

@dataclass
class MultiLabelMultiFileDataPoint(DataPoint):
    masks: list[tuple[str, Path]]
    """list of (target name, path to the segmentation mask)"""

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
    min_aniso_ratio: float = 0.5
    """minimum value for spacing_z / spacing_xy"""
    mask_batch_size: int = 32
    complete_anomaly: bool = False

    def __init__(self, logger: Logger, *, max_workers: int, chunksize: int, override: bool):
        self.tax = load_target_tax()
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

    @property
    def case_data_root(self):
        return self.output_root / 'data'

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

    def load_masks(self, data_point: DataPoint) -> tuple[torch.BoolTensor, list[str]]:
        """
        Returns:
            - segmentation mask
            - target names corresponding to the channel dimension
        """
        loader = self.get_mask_loader()
        device = get_cuda_device()
        if isinstance(data_point, MultiLabelMultiFileDataPoint):
            targets, mask_paths = zip(*data_point.masks)
            targets = list(targets)
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
            targets = list(class_mapping.values())
            masks = label == class_ids
        else:
            raise NotImplementedError
        return masks, targets

    def process(self):
        data_points = self.get_data_points()
        assert len(data_points) > 0
        assert len(data_points) == len(set(data_point.key for data_point in data_points)), "key must be unique within the dataset"
        data_points = [*filter(lambda p: not (self.case_data_root / p.name).exists(), data_points)]
        if len(data_points) == 0:
            return
        self.logger.info(f'{len(data_points)} data points to be processed')
        self.case_data_root.mkdir(parents=True, exist_ok=True)
        process_map(
            self.process_data_point,
            data_points,
            max_workers=self.max_workers, chunksize=self.chunksize, ncols=80,
        )
        # meta = pd.concat([processed_meta, pd.DataFrame.from_records(meta, index='key')])
        # info = pd.DataFrame.from_records(info, index='key')
        # if (info_path := self.output_root / 'info.csv').exists():
        #     processed_info = pd.read_csv(info_path, dtype={'key': 'string'}).set_index('key')
        #     info = pd.concat([processed_info, info])
        # info.to_csv(info_path)
        # info.to_excel(info_path.with_suffix('.xlsx'), freeze_panes=(1, 1))
        # meta.to_pickle(meta_path)

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

    def compute_resize(self, images: MetaTensor) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int32]]:
        if self.max_smaller_edge < (smaller_edge := min(images.shape[2:])):
            scale_xy = smaller_edge / self.max_smaller_edge
        else:
            scale_xy = 1.
        new_spacing_xy = images.pixdim[1:].min().item() * scale_xy
        new_spacing_z = max(images.pixdim[0].item(), new_spacing_xy * self.min_aniso_ratio)
        new_spacing = np.array([new_spacing_z, new_spacing_xy, new_spacing_xy])
        scale_z = new_spacing_z / images.pixdim[0].item()
        scale = np.array([scale_z, scale_xy, scale_xy])
        new_shape = (np.array(images.shape[1:]) / scale).round().astype(np.int32)
        return new_spacing, new_shape

    def _check_targets(self, targets: list[str]):
        for name in targets:
            assert name in self.tax

    def _generate_bbox_from_mask(self, mask: torch.BoolTensor) -> npt.NDArray[np.float64]:
        bbox_t = mt.BoundingRect()
        bbox = bbox_t(mask[None])
        return bbox[0].astype(np.float64)

    def process_data_point(self, data_point: DataPoint):
        """
        Returns:
            (metadata, human-readable information saved in csv) if process success, else None
        """
        self.key = key = data_point.key
        try:
            # TODO: control images.dtype
            modalities, images = self.load_images(data_point)
            is_natural = is_natural_modality(modalities[0])
            assert all(is_natural == is_natural_modality(modality) for modality in modalities[1:])
            if any(is_rgb_modality(modality) for modality in modalities):
                assert len(modalities) == 1, 'multiple RGB images is not supported'
            masks, targets = self.load_masks(data_point)
            self._check_targets(targets)
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
            # 3. compute resize (default: adapt to self.max_smaller_edge and self.min_aniso_ratio)
            new_spacing, new_shape = self.compute_resize(images)
            info.update({
                **{f'shape-{i}': s.item() for i, s in enumerate(new_shape)},
                **{f'space-{i}': s.item() for i, s in enumerate(new_spacing)},
            })
            # 4. apply resize & intensity normalization, save processed results
            # 4.1. normalize and save images
            images, mean, std = self.normalize_image(images, is_natural, new_shape)
            save_dir = self.case_data_root / f'.{key}'
            save_dir.mkdir(exist_ok=True, parents=True)
            # 4.2. save image
            torch.save(images.half().cpu(), save_dir / f'images.pt')
            # 4.3. resize, filter, compress, and save masks
            masks = self.resize_masks(masks, new_shape)
            pos_target_mask: torch.BoolTensor = einops.reduce(masks > 0, 'c ... -> c', 'any')
            # filter positive masks
            masks = masks[pos_target_mask]
            pos_targets = [name for i, name in enumerate(targets) if pos_target_mask[i]]
            neg_targets = [name for i, name in enumerate(targets) if not pos_target_mask[i]]
            with BytesIO() as buffer, open(save_dir / 'masks.pt.zst', 'wb') as f:
                torch.save(masks.cpu(), buffer)
                f.write(zstd.compress(buffer.getvalue()))
            # 4.4 handle and save sparse information
            sparse = Sparse(
                new_spacing,
                new_shape,
                mean.cpu().numpy(),
                std.cpu().numpy(),
                modalities,
                Sparse.Anatomy(
                    [*filter(lambda name: self.tax[name].category == 'anatomy', pos_targets)],
                    [*filter(lambda name: self.tax[name].category == 'anatomy', neg_targets)],
                ),
                Sparse.Anomaly(
                    [*filter(lambda name: self.tax[name].category == 'anomaly', pos_targets)],
                    [*filter(lambda name: self.tax[name].category == 'anomaly', neg_targets)],
                    self.complete_anomaly,
                ),
            )
            pd.to_pickle(sparse, 'sparse.pkl')
            # 4.5. handle and save annotation information
            annotation = Annotation(
                [(name, masks[i].sum().item()) for i, name in enumerate(pos_targets)],
                [
                    (name, self._generate_bbox_from_mask(masks[i]))
                    for i, name in enumerate(pos_targets)
                ],
            )
            pd.to_pickle(annotation, 'annotation.pkl')
            # 5. complete
            save_dir.rename(save_dir.with_name(key))
        except Exception as e:
            self.logger.error(key)
            self.logger.error(e)
            import traceback
            self.logger.error(traceback.format_exc())

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
