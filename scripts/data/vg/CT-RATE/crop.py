from functools import partial
from pathlib import Path

import cytoolz
from jsonargparse import CLI
from monai.data import MetaTensor
import monai.transforms as mt
from monai.utils import InterpolateMode
import torch
import torchvision.transforms.v2.functional as tvtf

from luolib.transforms import quantile
from luolib.utils import get_cuda_device, load_pt_zst, process_map, save_pt_zst
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
    seg_path = parent / f'{key}_seg.pt.zst'
    image: MetaTensor = mt.LoadImage(ensure_channel_first=True)(image_path)
    if seg_path.exists():
        masks: MetaTensor = load_pt_zst(parent / f'{key}_seg.pt.zst')
        assert torch.allclose(image.affine, masks.affine)
    else:
        masks = None
    convert_fn = cytoolz.compose(
        torch.Tensor.contiguous,
        mt.Orientation('SRA'),
        partial(torch.Tensor.cuda, device=device),
    )
    image = convert_fn(image)
    if masks is not None:
        masks = convert_fn(masks)
    min_v = quantile(image, 0.5 / 100)
    max_v = quantile(image, 99.5 / 100)
    image.clip_(min_v, max_v)
    image = (image - min_v) / (max_v - min_v)
    cropper = mt.CropForegroundD(['image', 'masks'], source_key='image', allow_smaller=False, allow_missing_keys=True)
    data = {'image': image}
    if masks is not None:
        data['masks'] = masks
    data = cropper(data)
    image = data['image']
    if masks is not None:
        masks = data['masks']
    resize_shape = (
        min(128, image.shape[1]),
        *get_max_resize(image.shape[2:], 32, 64)
    )
    resize = mt.Resize(
        resize_shape,
        mode=InterpolateMode.TRILINEAR,
        anti_aliasing=False,
    )
    image = resize(image).as_tensor()
    if masks is not None:
        mask_batch_size = 16
        results = []
        for i in range(0, masks.shape[0], mask_batch_size):
            result = resize(masks[i:i + mask_batch_size])
            results.append(result.as_tensor() > 0.5)
        masks = torch.cat(results, dim=0)
    image = tvtf.to_dtype(image, dtype=torch.uint8, scale=True)
    tmp_save_dir = parent / f'.{key}-t'
    tmp_save_dir.mkdir(exist_ok=True)
    save_pt_zst(image.cpu(), tmp_save_dir / f'{key}.pt.zst')
    if masks is not None:
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
