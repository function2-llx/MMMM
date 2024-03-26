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
class Meta:
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
        neg: list[str]
    anatomy: Anatomy

    @dataclass
    class Anomaly:
        pos: list[tuple[str, int]]
        neg: list[str]
        nil: bool
    anomaly: Anomaly

def encode_patch_size(patch_size: tuple3_t[int]):
    return ','.join(map(str, patch_size))
