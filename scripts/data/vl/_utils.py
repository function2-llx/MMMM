import gc
from itertools import starmap

import math
import torch
from torchvision.transforms.v2.functional import to_dtype

from luolib.utils import get_cuda_device
from monai.transforms import generate_spatial_bounding_box
from monai.utils import InterpolateMode
import monai.transforms as mt

def crop_resize(image: torch.Tensor) -> torch.Tensor | None:
    crop_mask = (image > 0).any(0, keepdim=True)
    if not crop_mask.any():
        return None
    start, end = generate_spatial_bounding_box(crop_mask)
    image = image[:, *starmap(slice, zip(start, end))]
    max_tokens_z = min(4, image.shape[1])
    max_smaller_edge = int((256 / max_tokens_z) ** 0.5) * 32
    resize_shape = [min(max_tokens_z * 32, image.shape[1]), *image.shape[2:]]
    if (_base := min(resize_shape[1:])) > max_smaller_edge:
        for j in (1, 2):
            resize_shape[j] = math.ceil(resize_shape[j] * max_smaller_edge / _base)
    if (resize_shape := tuple(resize_shape)) != image.shape[1:]:
        image = to_dtype(image, scale=True)
        resize = mt.Resize(resize_shape, mode=InterpolateMode.TRILINEAR, anti_aliasing=True)
        image = resize(image)
        image = to_dtype(image.as_tensor(), dtype=torch.uint8, scale=True)
    return image

def check_cuda_cache(cuda_cache_th: int = 10):
    device = get_cuda_device()
    cuda_cache = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    if cuda_cache > cuda_cache_th * 1024 ** 3:
        gc.collect()
        torch.cuda.empty_cache()
