from __future__ import annotations as _

from dataclasses import field
from functools import partial
from pathlib import Path
from typing import Literal, NamedTuple, TypedDict

from mashumaro import pass_through
import numpy as np
import torch

from luolib.types import tuple3_t

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
