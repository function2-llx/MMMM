from abc import ABC, abstractmethod
from dataclasses import dataclass
import gc
import itertools as it
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

from luolib import transforms as lt
from luolib.transforms import affine_resize
from luolib.types import tuple3_t
from luolib.utils import as_tensor, get_cuda_device, process_map, save_pt_zst
from monai import transforms as mt
from monai.data import MetaTensor
from monai.utils import GridSampleMode

from mmmm.data import load_target_tax
from mmmm.data.defs import ORIGIN_SEG_DATA_ROOT, PROCESSED_SEG_DATA_ROOT, Sparse

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

"""
1. multi-label, multi-file (single-channel, binary values)
2. multi-class, single-file (single-channel, multiple values)
"""

@dataclass(kw_only=True)
class MultiLabelMultiFileDataPoint(DataPoint):
    """
    Attributes:
        masks: list of (target name, path to the segmentation mask)
    """
    masks: list[tuple[str, Path]]

@dataclass(kw_only=True)
class MultiClassDataPoint(DataPoint):
    label: Path
    class_mapping: dict[int, str]

_CLIP_LOWER = norm.cdf(-3)
_CLIP_UPPER = norm.cdf(3)

def clip_intensity(img: torch.Tensor) -> torch.BoolTensor:
    """clip the intensity in-place
    Returns:
        the crop mask after clipping
    """
    x = img.view(img.shape[0], -1)
    if img.dtype == torch.uint8:
        minv = img.new_tensor(0.)
    else:
        minv = lt.quantile(x, _CLIP_LOWER, 1, True)
        maxv = lt.quantile(x, _CLIP_UPPER, 1, True)
        x.clamp_(minv, maxv)
    crop_mask = img.new_zeros((1, *img.shape[1:]), dtype=torch.bool)
    torch.any(x > minv, dim=0, keepdim=True, out=crop_mask.view(1, -1))
    return crop_mask

def crop(images: torch.Tensor, masks: torch.BoolTensor, crop_mask: torch.BoolTensor) -> tuple[MetaTensor, torch.BoolTensor]:
    data = {
        'images': images,
        'masks': masks,
        'crop_mask': crop_mask,
    }
    data = mt.CropForegroundD(['images', 'masks'], 'crop_mask', start_coord_key=None, end_coord_key=None)(data)
    return data['images'], data['masks']

class Processor(ABC):
    """
    TODO: check aspect ratio
    Attributes:
        name: name of the dataset to be processed by the processor
        orientation: if orientation is None, will determine it from the spacing
        min_aniso_ratio: minimum value for spacing_z / spacing_xy
        do_normalize: whether to normalize the image during pre-processing; otherwise, during training
        cuda_cache_th: cuda cache usage threshold to empty cache (in GiB)
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
    bbox_ignore_targets: set[str] = set()

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

    def _check_affine(self, affine1: torch.Tensor, affine2: torch.Tensor, atol: float = 1e-2):
        assert torch.allclose(affine1, affine2, atol=atol)

    def load_masks(self, data_point: DataPoint) -> tuple[MetaTensor, list[str]]:
        """
        NOTE: metadata for mask should be preserved to match with the image
        Returns:
            - a tensor of segmentation masks (dtype = torch.bool)
            - target names corresponding to the channel dimension
        """
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
            label: MetaTensor = self.mask_loader(data_point.label).to(dtype=torch.int16, device=device)
            assert label.shape[0] == 1
            class_ids = torch.tensor([c for c in class_mapping], dtype=torch.int16, device=device)
            for _ in range(label.ndim - 1):
                class_ids = class_ids[..., None]  # make broadcastable
            targets = list(class_mapping.values())
            masks = label == class_ids
        else:
            raise NotImplementedError
        return masks, targets

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
        shape_diff = np.empty(len(codes), dtype=np.int32)
        dummy = MetaTensor(torch.empty(1, *images.shape[1:], device=images.device), images.affine)
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

    def orient(self, images: MetaTensor, masks: MetaTensor) -> tuple[MetaTensor, MetaTensor]:
        orientation = self.get_orientation(images)
        trans = mt.Orientation(orientation)
        images, masks = map(lambda x: trans(x).contiguous(), [images, masks])
        return images, masks

    def compute_resize(self, spacing: torch.DoubleTensor, shape: tuple3_t[int]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int16]]:
        if self.max_smaller_edge < (smaller_edge := min(shape[1:])):
            scale_xy = smaller_edge / self.max_smaller_edge
        else:
            scale_xy = 1.
        new_spacing_xy = spacing[1:].min().item() * scale_xy
        new_spacing_z = max(spacing[0].item(), new_spacing_xy * self.min_aniso_ratio)
        new_spacing = np.array([new_spacing_z, new_spacing_xy, new_spacing_xy])
        scale_z = new_spacing_z / spacing[0].item()
        scale = np.array([scale_z, scale_xy, scale_xy])
        new_shape = (np.array(shape) / scale).round().astype(np.int16)
        return new_spacing, new_shape

    def _check_targets(self, targets: list[str]):
        unknown_targets = [name for name in targets if name not in self.tax]
        if len(unknown_targets) > 0:
            print('unknown targets:', unknown_targets)
            raise ValueError

    def _generate_bbox_from_mask(self, masks: torch.BoolTensor) -> list[Sparse.BBox]:
        if masks.shape[0] == 0:
            return []
        bbox_t = mt.BoundingRect()
        bbox = bbox_t(masks)
        bbox = bbox.reshape((-1, 3, 2)) / np.array(masks.shape[1:])[:, None]
        center = bbox.mean(axis=2)
        shape = bbox[..., 1] - bbox[..., 0]
        return [Sparse.BBox(c, s) for c, s in zip(center, shape)]

    def _compute_class_positions(self, masks: torch.BoolTensor) -> tuple[torch.ShortTensor, torch.LongTensor]:
        ret = []
        offsets = torch.empty(masks.shape[0] + 1, dtype=torch.long, device=masks.device)
        offsets[0] = 0
        for i, mask in enumerate(masks):
            positions = mask.nonzero().short()
            if positions.shape[0] > self.max_class_positions:
                # deterministic? random!
                positions = positions[torch.randint(positions.shape[0], (self.max_class_positions,), device=mask.device)]
            offsets[i + 1] = offsets[i] + positions.shape[0]
            ret.append(positions)
        return torch.cat(ret), offsets

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
                if cuda_cache > self.cuda_cache_th * 1024 ** 3:
                    gc.collect()
                    torch.cuda.empty_cache()
            modalities, images = self.load_images(data_point)
            masks, targets = self.load_masks(data_point)
            self._check_targets(targets)
            images, masks = self.orient(images, masks)
            assert images.shape[1:] == masks.shape[1:]
            self._check_affine(images.affine, masks.affine)
            spacing: torch.DoubleTensor = images.pixdim
            info = {
                'key': key,
                **{f'shape-o-{i}': s for i, s in enumerate(images.shape[1:])},
                **{f'space-o-{i}': s.item() for i, s in enumerate(spacing)}
            }
            # 1. clip intensity, compute crop mask
            crop_mask = clip_intensity(images)
            # 2. crop images and masks
            images, masks = crop(images, masks, crop_mask)
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
            masks = self.resize_masks(masks, new_shape)
            pos_target_mask: torch.BoolTensor = einops.reduce(masks > 0, 'c ... -> c', 'any')
            # filter positive masks
            masks = masks[pos_target_mask]
            pos_targets = [name for i, name in enumerate(targets) if pos_target_mask[i]]
            neg_targets = [name for i, name in enumerate(targets) if not pos_target_mask[i]]
            if masks.shape[0] > 0:
                save_pt_zst(as_tensor(masks).cpu(), save_dir / 'masks.pt.zst')
                class_positions, class_offsets = self._compute_class_positions(masks)
                torch.save(class_positions.cpu(), save_dir / 'class_positions.pt')
                torch.save(class_offsets.cpu(), save_dir / 'class_offsets.pt')
            # 4.4 handle and save sparse information
            sparse = Sparse(
                spacing=new_spacing,
                shape=new_shape,
                mean=mean.cpu().numpy(),
                std=std.cpu().numpy(),
                modalities=modalities,
                anatomy=Sparse.Anatomy(
                    [*filter(lambda name: self.tax[name].category == 'anatomy', pos_targets)],
                    [*filter(lambda name: self.tax[name].category == 'anatomy', neg_targets)],
                ),
                anomaly=Sparse.Anomaly(
                    [*filter(lambda name: self.tax[name].category == 'anomaly', pos_targets)],
                    [*filter(lambda name: self.tax[name].category == 'anomaly', neg_targets)],
                    data_point.complete_anomaly,
                ),
                annotation=Sparse.Annotation(
                    [(name, masks[i].sum().item()) for i, name in enumerate(pos_targets)],
                    [
                        (pos_targets[i], bbox)
                        for i, bbox in enumerate(self._generate_bbox_from_mask(masks))
                        if pos_targets[i] not in self.bbox_ignore_targets
                    ],
                ),
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

    def normalize_image(self, images: torch.Tensor, new_shape: npt.NDArray[np.int32]) -> tuple3_t[torch.Tensor]:
        # 1. rescale to [0, 1]
        if images.dtype == torch.uint8:
            images = images.float() / 255
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

INF_SPACING = 1e8

class _LoaderBase:
    def _adapt_to_3d(self, image: MetaTensor):
        image = image.unsqueeze(1)
        image.affine[0, 0] = INF_SPACING
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