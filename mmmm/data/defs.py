from __future__ import annotations as _

from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Literal, NamedTuple, TypedDict

from mashumaro import pass_through
from mashumaro.config import BaseConfig
from mashumaro.mixins.orjson import DataClassORJSONMixin
import nibabel as nib
import numpy as np
import numpy.typing as npt
import orjson
import torch

from luolib.types import PathLike, tuple3_t
from luolib.utils import load_pt_zst

DATA_ROOT = Path('data')
ORIGIN_DATA_ROOT = DATA_ROOT / 'origin'
PROCESSED_DATA_ROOT = DATA_ROOT / 'processed'
ORIGIN_SEG_DATA_ROOT = ORIGIN_DATA_ROOT / 'image'
ORIGIN_VL_DATA_ROOT = ORIGIN_DATA_ROOT / 'vision-language'
PROCESSED_SEG_DATA_ROOT = PROCESSED_DATA_ROOT / 'image'
PROCESSED_VL_DATA_ROOT = PROCESSED_DATA_ROOT / 'vision-language'
PROCESSED_VG_DATA_ROOT = PROCESSED_DATA_ROOT / 'visual-grounding'

def _numpy_field(dtype: np.dtype):
    return field(metadata={'serialize': pass_through, 'deserialize': partial(np.array, dtype=dtype)})

@dataclass
class Sparse(DataClassORJSONMixin):
    """
    Attributes:
        modalities: all images of different modalities must be co-registered
        mean: mean intensity for each modality
        complete_anomaly: indicating that `pos` covers all anomalies in the image
    """
    spacing: npt.NDArray[np.float64] = _numpy_field(np.float64)
    shape: npt.NDArray[np.int16] = _numpy_field(np.int16)
    modalities: list[str]
    mean: npt.NDArray[np.float32] = _numpy_field(np.float32)
    std: npt.NDArray[np.float32] = _numpy_field(np.float32)

    @dataclass(kw_only=True)
    class Annotation:
        """
        indistinguishable instances of the same class
        Attributes:
            name: class name
            num: number of instances (== len(boxes) == len(masks), if available)
            merged: whether different instances are merged, bbox makes little sense in this case and is set to None
            boxes: use MONAI's StandardMode (CornerCornerModeTypeA)
            masks: mask index of each instance corresponding to the mask file; if None, no mask for the instance available
        """
        name: str
        num: int
        boxes: npt.NDArray[np.float64] | None = field(
            metadata={
                'serialize': pass_through,
                'deserialize': lambda boxes: None if boxes is None else np.array(boxes, dtype=np.float64),
            },
        )
        merged: bool

        @dataclass
        class MaskInfo:
            index: int
            size: int = 0
        masks: list[MaskInfo] | None

    pos_anatomy: list[Annotation]
    neg_anatomy: set[str]
    pos_anomaly: list[Annotation]
    neg_anomaly: set[str]
    complete_anomaly: bool
    extra: Any = None

    class Config(BaseConfig):
        orjson_options = orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_INDENT_2

def convert_to_slicer(data_dir: PathLike, output_dir: PathLike | None = None, multiclass: bool = True):
    """convert the processed data by MMMM to the format readable by Slicer"""
    data_dir = Path(data_dir)
    output_dir = data_dir / 'slicer' if output_dir is None else Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    img = torch.load(data_dir / 'images.pt')
    sparse = Sparse.from_json((data_dir / 'sparse.json').read_bytes())
    for i, modality in enumerate(sparse.modalities):
        nib.save(
            nib.Nifti1Image(img[i].float().numpy(), np.diag([*sparse.spacing, 1])),
            output_dir / f'{modality}.nii.gz',
        )
    masks: torch.BoolTensor = load_pt_zst(data_dir / 'masks.pt.zst')
    if multiclass:
        seg = torch.zeros(masks.shape[1:], dtype=torch.int16)
        for c in range(masks.shape[0]):
            seg[masks[c]] = c + 1
    else:
        seg = masks
    nib.save(
        nib.Nifti1Image(seg.numpy(), np.diag([*sparse.spacing, 1])),
        output_dir / 'seg.nii.gz',
    )

class DataPoint(TypedDict):
    """
    Attributes:
        mask: (c, *spatial)
        mask_index: select targets from text that corresponds to the mask
        bbox: (c, [center, size]), or (c, 2, 3)
    """
    image: torch.Tensor
    grounding_image: torch.Tensor | None
    patch_size: tuple3_t[int]
    pool_size: tuple3_t[int]
    vlm_inputs: dict[str, torch.Tensor]
    mask: torch.BoolTensor
    mask_index: torch.BoolTensor
    bbox: torch.Tensor
    bbox_index: torch.BoolTensor

class Batch(TypedDict):
    image: list[torch.Tensor]
    grounding_image: list[torch.Tensor | None]
    patch_size: list[tuple3_t[int]]
    pool_size: list[tuple3_t[int]]
    vlm_inputs: dict[str, torch.Tensor]
    mask: list[torch.BoolTensor]
    mask_index: list[torch.BoolTensor]
    bbox: list[torch.Tensor]
    bbox_index: list[torch.BoolTensor]

split_t = Literal['train', 'val']
CE_IGNORE_INDEX = -100

class ConvTurn(NamedTuple):
    prompt: str
    response: str
