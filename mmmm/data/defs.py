from __future__ import annotations

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
import pandas as pd
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
        normalized: whether the images are normalized during pre-processing
        anatomy: information for generating general conversation related to anatomy targets
        anomaly: information for generating general conversation related to anomaly targets
    """
    spacing: npt.NDArray[np.float64] = _numpy_field(np.float64)
    shape: npt.NDArray[np.int16] = _numpy_field(np.int16)
    modalities: list[str]
    mean: npt.NDArray[np.float32] = _numpy_field(np.float32)
    std: npt.NDArray[np.float32] = _numpy_field(np.float32)
    normalized: bool

    @dataclass
    class Anatomy:
        """
        Attributes:
            pos: anatomical structures that are assured to be observable in the image
            neg: anatomical structures that are assured to be unobservable in the image
        """
        pos: set[str]
        neg: set[str]
    anatomy: Anatomy

    @dataclass
    class Anomaly:
        """
        Attributes:
            pos: anomalies that are assured to be observable in the image; multiple instances is not supported yet
            neg: anomalies that are assured to be unobservable in the image
            complete: indicating that `pos` covers all anomalies in the image
        """
        pos: set[str]
        neg: set[str]
        complete: bool
    anomaly: Anomaly

    @dataclass
    class BBox:
        center: npt.NDArray[np.float64] = _numpy_field(np.float64)
        size: npt.NDArray[np.float64] = _numpy_field(np.float64)

    @dataclass
    class Annotation:
        """
        Attributes:
            mask: list of (name, mask size), where the order corresponds to the channel dimension of the mask
                file, and names may repeat for multiple anomalies with the same name
            bbox: list of (target name, 3D bounding box coordinates), coordinates range: [0, shape - 1]
        """
        mask: list[tuple[str, int]]
        bbox: list[tuple[str, Sparse.BBox]]
    annotation: Annotation

    extra: Any = None

    class Config(BaseConfig):
        orjson_options = orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_INDENT_2

def convert_to_slicer(data_dir: PathLike, output_dir: PathLike | None = None, multiclass: bool = True):
    """convert the processed data by MMMM to the format readable by Slicer"""
    data_dir = Path(data_dir)
    output_dir = data_dir / 'slicer' if output_dir is None else Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    img = torch.load(data_dir / 'images.pt')
    sparse: Sparse = pd.read_pickle(data_dir / 'sparse.pkl')
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
    vlm_inputs: dict[str, torch.Tensor]
    mask: torch.BoolTensor
    mask_index: torch.BoolTensor
    bbox: torch.Tensor
    bbox_index: torch.BoolTensor

class Batch(TypedDict):
    image: list[torch.Tensor]
    grounding_image: list[torch.Tensor | None]
    patch_size: list[tuple3_t[int]]
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
