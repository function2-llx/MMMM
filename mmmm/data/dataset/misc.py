from __future__ import annotations

from collections.abc import Sequence

import math
from pathlib import Path
from typing import Sequence

import einops
import monai.transforms as mt
import numpy as np
import torch
from monai.data import MetaTensor, convert_box_mode
from monai.data.box_utils import CenterSizeMode
from monai.utils import InterpolateMode, GridSamplePadMode
from torchvision.io import read_image
from torchvision.transforms import v2 as tvt

from luolib.transforms.box_ops import apply_affine_to_boxes_int
from luolib.types import tuple2_t, tuple3_t, PathLike
from luolib.utils import load_pt_zst

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

def load_image_data(image_path: PathLike) -> torch.Tensor:
    image_path = Path(image_path)
    if image_path.name.endswith('.pt'):
        image = torch.load(image_path)
    elif image_path.name.endswith('.pt.zst'):
        image = load_pt_zst(image_path)
    else:
        image = read_image(str(image_path))
        image = einops.rearrange(image, 'c h w -> c 1 h w')
    return image

def get_patch_size_z(
    base_patch_size_z: int,
    base_pool_size_z: int,
    size_z: int,
    max_tokens_z: int,
    log2_patch_size_z_std: float | None = None,
    R: np.random.RandomState | None = None,
):
    if size_z <= max_tokens_z:
        patch_size_z = pool_size_z = stride_z = 1
        tokens_z = size_z
    else:
        pool_size_z = base_pool_size_z
        if log2_patch_size_z_std is None:
            log2_patch_size_z = np.log2(size_z / (pool_size_z * max_tokens_z))
        else:
            log2_patch_size_z = R.normal(
                np.log2(size_z / (pool_size_z * max_tokens_z)),
                log2_patch_size_z_std,
            )
        log2_patch_size_z = np.clip(
            np.rint(log2_patch_size_z), 0, base_patch_size_z.bit_length() - 1,
        )
        patch_size_z = 1 << int(log2_patch_size_z)
        stride_z = patch_size_z * pool_size_z
        tokens_z = min(math.ceil(size_z / stride_z), max_tokens_z)
    return patch_size_z, pool_size_z, stride_z, tokens_z


def spatial_transform_image_labels(
    image: torch.Tensor,
    masks: torch.BoolTensor | None,
    boxes: torch.LongTensor | None,
    resize: tuple3_t[int],
    stride: tuple3_t[int],
    rand_flip: bool = False,
    rand_rotate: bool = False,
    R: np.random.RandomState | None = None,
) -> tuple[torch.Tensor, torch.BoolTensor | None, torch.LongTensor | None]:
    keys = ['image']
    if masks is not None:
        keys.append('masks')
    transforms = [
        mt.ResizeD(keys, resize, mode=InterpolateMode.TRILINEAR),
        mt.DivisiblePadD(keys, stride),
    ]
    if rand_flip:
        transforms.extend([
            mt.RandFlipD(keys, 0.5, i)
            for i in range(3)
        ])
    if rand_rotate:
        transforms.append(
            mt.RandRotate90D(keys, 0.75, spatial_axes=(1, 2)),
        )
    affine_trans = mt.Compose(
        transforms,
        lazy=True,
        overrides={
            'image': {'padding_mode': GridSamplePadMode.ZEROS},
            'masks': {'padding_mode': GridSamplePadMode.ZEROS},
        }
    )
    affine_trans.set_random_state(state=R)
    _dict_data = {'image': image}
    if masks is not None:
        _dict_data['masks'] = masks
    _dict_data = affine_trans(_dict_data)
    image_t: MetaTensor = _dict_data['image']
    if masks is None:
        masks_t = None
    else:
        masks_t = _dict_data['masks'].round().bool().as_tensor()
    if boxes is None:
        boxes_t = None
    else:
        boxes_t = apply_affine_to_boxes_int(boxes, image_t.affine.inverse())
    return image_t.as_tensor(), masks_t, boxes_t

def norm_boxes(boxes: torch.LongTensor, norm_size: Sequence[int]) -> torch.DoubleTensor:
    norm_size_t = einops.repeat(torch.tensor(norm_size), 'd -> (l2 d)', l2=2)
    boxes_normed = boxes.double() / norm_size_t
    boxes_normed = convert_box_mode(boxes_normed, dst_mode=CenterSizeMode)
    return boxes_normed
