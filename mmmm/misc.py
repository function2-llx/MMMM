from pathlib import Path

import einops
import math
import numpy as np
import torch
from torchvision.io import read_image
import torchvision.transforms.v2.functional as tvtf

from luolib.types import PathLike
from luolib.utils import load_pt_zst
from luolib.utils.misc import ensure_rgb
import monai.transforms as mt
from monai.utils import InterpolateMode, convert_to_tensor

from mmmm.data.dataset.misc import get_max_resize, intensity_norm

def image_transform(
    image_path: PathLike,
    max_vision_tokens,
    max_tokens_z: int = 4,
    base_patch_size_z: int = 16,
    base_pool_size_z: int = 2,
    patch_size_xy: int = 16,
    pool_size_xy: int = 2,
):
    image_path_str = str(image_path)
    if image_path_str.endswith('.pt.zst'):
        image = load_pt_zst(Path(image_path))
    else:
        image = read_image(image_path_str)
        image = einops.rearrange(image, 'c h w -> c 1 h w')
    image = tvtf.to_dtype(image, torch.float, scale=True)
    if (size_z := image.shape[1]) <= max_tokens_z:
        patch_size_z = pool_size_z = stride_z = 1
        tokens_z = size_z
    else:
        pool_size_z = base_pool_size_z
        log2_patch_size_z = np.log2(size_z / (pool_size_z * max_tokens_z)),
        log2_patch_size_z = np.clip(
            np.rint(log2_patch_size_z), 0, base_patch_size_z.bit_length() - 1,
        )
        patch_size_z = 1 << int(log2_patch_size_z)
        stride_z = patch_size_z * pool_size_z
        tokens_z = min(math.ceil(size_z / stride_z), max_tokens_z)
    patch_size = (patch_size_z, patch_size_xy, patch_size_xy)
    stride_xy = patch_size_xy * pool_size_xy
    stride = (stride_z, stride_xy, stride_xy)
    resize_shape = (
        min(size_z, tokens_z * stride_z),  # do not resize z if unnecessary
        *get_max_resize(
            image.shape[2:],
            stride_xy,
            max_vision_tokens // tokens_z,
        ),
    )
    if resize_shape != image.shape[1:]:
        resize = mt.Resize(resize_shape, mode=InterpolateMode.TRILINEAR, anti_aliasing=True)
        image = resize(image)
    image = mt.DivisiblePad(stride)(image)
    image = convert_to_tensor(image)
    image, _ = ensure_rgb(image, contiguous=True)
    image = intensity_norm(image)
    pool_size = (pool_size_z, pool_size_xy, pool_size_xy)
    num_vision_tokens = (np.array(image.shape[1:]) // stride).prod().item()
    return image, patch_size, pool_size, num_vision_tokens
