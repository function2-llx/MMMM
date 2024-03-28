from pathlib import Path
from typing import TypedDict

import numpy as np
from numpy import typing as npt

from luolib.types import tuple3_t

DATA_ROOT = Path('data')
ORIGIN_DATA_ROOT = DATA_ROOT / 'origin'
PROCESSED_DATA_ROOT = DATA_ROOT / 'processed'
ORIGIN_SEG_DATA_ROOT = ORIGIN_DATA_ROOT / 'image'
ORIGIN_VL_DATA_ROOT = ORIGIN_DATA_ROOT / 'vision-language'
PROCESSED_SEG_DATA_ROOT = PROCESSED_DATA_ROOT / 'image'
PROCESSED_VL_DATA_ROOT = PROCESSED_DATA_ROOT / 'vision-language'
PROCESSED_VG_DATA_ROOT = PROCESSED_DATA_ROOT / 'visual-grounding'

class Meta(TypedDict):
    spacing: npt.NDArray[np.float64]
    shape: npt.NDArray[np.int32]
    mean: npt.NDArray[np.floating]
    std: npt.NDArray[np.floating]
    modalities: list[str]
    positive_classes: list[str]
    negative_classes: list[str]

def encode_patch_size(patch_size: tuple3_t[int]):
    return ','.join(map(str, patch_size))
