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
    """
    Attributes:
        mean: mean intensity for each modality
        modalities: all images of different modalities must be co-registered
    """
    spacing: npt.NDArray[np.float64]
    shape: npt.NDArray[np.int32]
    mean: npt.NDArray[np.floating]
    std: npt.NDArray[np.floating]
    modalities: list[str]

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
            pos: anomalies that are assured to be observable in the image, with number of instances
            neg: anomalies that are assured to be unobservable in the image
            complete: indicating that `pos` covers all anomalies in the image
        """
        pos: list[tuple[str, int]]
        neg: list[str]
        complete: bool
    anomaly: Anomaly

@dataclass
class Annotation:
    """
    Attributes:
        mask: list of (name, mask size), where the order corresponds to the channel dimension of the mask
            file, and names may repeat for multiple anomalies with the same name
        bbox: list of (target name, 3D bounding box coordinates)
    """
    mask: list[tuple[str, int]]
    bbox: list[tuple[str, npt.NDArray[np.float64]]]

def encode_patch_size(patch_size: tuple3_t[int]):
    return ','.join(map(str, patch_size))
