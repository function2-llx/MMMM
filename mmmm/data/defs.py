from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy import typing as npt

from luolib.types import tuple3_t

DATA_ROOT = Path('data')
ORIGIN_DATA_ROOT = DATA_ROOT / 'origin'
PROCESSED_DATA_ROOT = DATA_ROOT / 'processed'
ORIGIN_SEG_DATA_ROOT = ORIGIN_DATA_ROOT / 'image'
PROCESSED_SEG_DATA_ROOT = PROCESSED_DATA_ROOT / 'image'

@dataclass
class Sparse:
    spacing: npt.NDArray[np.float64]
    shape: npt.NDArray[np.int32]
    mean: npt.NDArray[np.floating]
    """mean intensity for each modality"""
    std: npt.NDArray[np.floating]
    modalities: list[str]
    """all images of different modalities must be co-registered"""

    @dataclass
    class Anatomy:
        pos: list[str]
        """anatomical structures that are assured to be observable in the image"""
        neg: list[str]
        """anatomical structures that are assured to be unobservable in the image"""
    anatomy: Anatomy

    @dataclass
    class Anomaly:
        pos: list[tuple[str, int]]
        """anomalies that are assured to be observable in the image, with number of instances"""
        neg: list[str]
        """anomalies that are assured to be unobservable in the image"""
        complete: bool
        """indicating that `pos` covers all anomalies in the image"""
    anomaly: Anomaly

@dataclass
class Annotation:
    mask: list[str]
    """names of targets each of which have a segmentation mask available,
    corresponding to the channel dimension of the mask file, and may repeat for multiple anomalies with the same name
    """
    bbox: list[tuple[str, npt.NDArray[np.float64]]]
    """list of (target name, 3D bounding box coordinates)"""

def encode_patch_size(patch_size: tuple3_t[int]):
    return ','.join(map(str, patch_size))
