from functools import partial
from pathlib import Path

import cytoolz
import monai.transforms as mt

import nibabel as nib
import numpy as np
from jsonargparse import CLI
from monai.data import MetaTensor
import torchvision.transforms.v2.functional as tvtf
import torch
from monai.utils import InterpolateMode

from luolib.transforms import quantile
from luolib.utils import load_pt_zst, save_pt_zst, process_map, get_cuda_device
from mmmm.data.dataset.misc import get_max_resize
from mmmm.data.defs import PROCESSED_VG_DATA_ROOT

def process(image_path: Path):
    torch.cuda.empty_cache()
    device = get_cuda_device()
    assert image_path.name.endswith('.nii.gz')
    key = image_path.name[:-len('.nii.gz')]
    parent = image_path.parent
    save_dir = parent / f'{key}-t'
    if save_dir.exists():
        return
    image: MetaTensor = mt.LoadImage(ensure_channel_first=True)(image_path)
    masks: MetaTensor = load_pt_zst(parent / f'{key}_seg.pt.zst')
    assert torch.allclose(image.affine, masks.affine)
    image, masks = map(
        cytoolz.compose(
            torch.Tensor.contiguous,
            mt.Orientation('SRA'),
            partial(torch.Tensor.cuda, device=device),
        ),
        (image, masks),
    )
    min_v = quantile(image, 0.5 / 100)
    max_v = quantile(image, 99.5 / 100)
    image.clip_(min_v, max_v)
    image = (image - min_v) / (max_v - min_v)
    cropper = mt.CropForegroundD(['image', 'masks'], source_key='image', allow_smaller=False)
    image, masks = cytoolz.get(['image', 'masks'], cropper({'image': image, 'masks': masks}))
    resize_shape = (
        min(128, image.shape[1]),
        *get_max_resize(image.shape[2:], 32, 64)
    )
    resize = mt.ResizeD(
        ['image', 'masks'],
        resize_shape,
        mode=[InterpolateMode.TRILINEAR, InterpolateMode.NEAREST_EXACT],
        dtype=[image.dtype, torch.uint8],
        anti_aliasing=False,
    )
    image, masks = cytoolz.get(['image', 'masks'], resize({'image': image, 'masks': masks}))
    masks = masks.bool()
    image, masks = map(MetaTensor.as_tensor, (image, masks))
    image = tvtf.to_dtype(image, dtype=torch.uint8, scale=True)
    tmp_save_dir = parent / f'.{key}-t'
    tmp_save_dir.mkdir(exist_ok=True)
    save_pt_zst(image.cpu(), tmp_save_dir / f'{key}.pt.zst')
    save_pt_zst(masks.cpu(), tmp_save_dir / f'{key}_seg.pt.zst')
    tmp_save_dir.rename(save_dir)

def main(max_workers: int = 8, d_range: tuple[int | None, int | None] = (None, None)):
    all_files = list((PROCESSED_VG_DATA_ROOT / 'CT-RATE/image').glob('*/*.nii.gz'))
    d_range = list(d_range)
    if d_range[0] is None:
        d_range[0] = 0
    if d_range[1] is None:
        d_range[1] = len(all_files)
    all_files = all_files[slice(*d_range)]
    process_map(
        process, all_files, max_workers=max_workers, chunksize=1, dynamic_ncols=True,
    )

if __name__ == '__main__':
    CLI(main)
