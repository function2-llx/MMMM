from __future__ import annotations

from collections.abc import Sequence

import math

import numpy as np
import torch

from luolib.types import tuple2_t, tuple3_t

from mmmm.data.defs import ConvTurn

PROMPTS = [
    'What is the modality of this image?',
    'What type of imaging modality is used to acquire the given image?',
]

RESPONSES = [
    'The modality of this image is {}.',
]

def gen_modality_conv(modality: str, R: np.random.RandomState) -> list[ConvTurn]:
    return [
        ConvTurn(
            R.choice(PROMPTS),
            R.choice(RESPONSES).format(modality)
        ),
    ]

def toss(R: np.random.RandomState, prob: float):
    return R.uniform() < prob

def intensity_norm(
    image: torch.Tensor,
    mean: tuple3_t[float] = (0.48145466, 0.4578275, 0.40821073),
    std: tuple3_t[float] = (0.26862954, 0.26130258, 0.27577711),
):
    """default mean and std is adopted from CogVLM (, which is from CLIP)"""
    mean = image.new_tensor(mean)
    std = image.new_tensor(std)
    return (image - mean.view(-1, 1, 1, 1)) / std.view(-1, 1, 1, 1)

@np.vectorize
def _solve(a: float, M: int):
    """find max integer t s.t. t⌈at⌉ ≤ M"""
    aM = a * M
    n = math.ceil(aM ** 0.5)
    if aM > (n - 1) * n:
        t = M // n
    else:
        t = math.floor((n - 1) / a)
    return t

def get_max_scale_for_size(size: Sequence[int], stride: int, max_tokens: int) -> float:
    """find maximum scale parameter (for size, instead of spacing) s, s.t. s * size has at most max_tokens"""
    size = np.array(size)
    assert size.shape[0] == 2
    gcd = np.gcd(size, stride)
    size_p = size // gcd
    stride //= gcd
    ps = stride * np.flip(size_p)
    t = _solve(ps / np.flip(ps), max_tokens)
    scale = (t * stride / size_p).max()
    return scale.item()

def get_max_resize(size: Sequence[int], stride: int, max_tokens: int) -> tuple2_t[int]:
    scale = get_max_scale_for_size(size, stride, max_tokens)
    resize = np.multiply(size, scale).round().astype(np.int64)
    return tuple(resize.tolist())
