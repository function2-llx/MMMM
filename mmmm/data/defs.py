from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mashumaro.mixins.orjson import DataClassORJSONMixin
import nibabel as nib
import numpy as np
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

@dataclass(kw_only=True)
class Sparse(DataClassORJSONMixin):
    """
    Attributes:
        modalities: all images of different modalities must be co-registered
        mean: mean intensity for each modality
        normalized: whether the images are normalized during pre-processing
        anatomy: information for generating general conversation related to anatomy targets
        anomaly: information for generating general conversation related to anomaly targets
    """
    spacing: tuple3_t[float]
    shape: tuple3_t[int]
    modalities: list[str]
    mean: list[float]
    std: list[float]
    normalized: bool

    @dataclass
    class Anatomy:
        """
        Attributes:
            pos: anatomical structures that are assured to be observable in the image
            neg: anatomical structures that are assured to be unobservable in the image
        """
        pos: list[str]
        neg: list[str]
    anatomy: Anatomy

    @dataclass
    class Anomaly:
        """
        Attributes:
            pos: anomalies that are assured to be observable in the image
            neg: anomalies that are assured to be unobservable in the image
            complete: indicating that `pos` covers all anomalies in the image
        """
        pos: list[str]
        neg: list[str]
        complete: bool
    anomaly: Anomaly

    @dataclass
    class BBox:
        center: tuple3_t[float]
        shape: tuple3_t[float]

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

def encode_patch_size(patch_size: tuple3_t[int]):
    return ','.join(map(str, patch_size))

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
