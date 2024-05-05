from __future__ import annotations as _

import os
from pathlib import Path
from typing import Literal, NamedTuple, TypedDict

import torch

from luolib.types import tuple3_t
from monai.utils import str2bool

def debug() -> bool:
    val = os.environ.get('MMMM_DEBUG', False)
    return val if isinstance(val, bool) else str2bool(val)

DATA_ROOT = Path('data')
ORIGIN_DATA_ROOT = DATA_ROOT / 'origin'
PROCESSED_DATA_ROOT = DATA_ROOT / ('processed-debug' if debug() else 'processed')
ORIGIN_LOCAL_DATA_ROOT = ORIGIN_DATA_ROOT / 'local'
ORIGIN_VL_DATA_ROOT = ORIGIN_DATA_ROOT / 'vision-language'
PROCESSED_LOCAL_DATA_ROOT = PROCESSED_DATA_ROOT / 'local'
PROCESSED_VL_DATA_ROOT = PROCESSED_DATA_ROOT / 'vision-language'
PROCESSED_VG_DATA_ROOT = PROCESSED_DATA_ROOT / 'visual-grounding'

class DataPoint(TypedDict):
    """
    Attributes:
        index_offsets: (n, 2) for each target, indicating the label index offsets for
        num_uncertain: number of uncertain instances for the target.
            -1 to ignore this target when calculating grounding loss
        semantic: the mask label for this class is semantic
    """
    image: torch.Tensor
    grounding_image: torch.Tensor
    patch_size: tuple3_t[int]
    pool_size: tuple3_t[int]
    vlm_inputs: dict[str, torch.Tensor]
    masks: torch.BoolTensor | None
    boxes: torch.Tensor
    index_offsets: torch.LongTensor
    num_uncertain: torch.LongTensor
    semantic: torch.BoolTensor

class Batch(TypedDict):
    image: list[torch.Tensor]
    grounding_image: list[torch.Tensor]
    patch_size: list[tuple3_t[int]]
    pool_size: list[tuple3_t[int]]
    vlm_inputs: dict[str, torch.Tensor]
    masks: list[torch.BoolTensor | None]
    boxes: list[torch.Tensor]
    index_offsets: list[torch.LongTensor]
    num_uncertain: list[torch.LongTensor]
    semantic: list[torch.BoolTensor]

split_t = Literal['train', 'val']
CE_IGNORE_INDEX = -100

class ConvTurn(NamedTuple):
    prompt: str
    response: str
