import itertools as it
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from luolib.transforms import sliding_window_sum
from luolib.types import tuple3_t
from luolib.utils import get_cuda_device, process_map

from mmmm.data.defs import PROCESSED_SEG_DATA_ROOT

mask_batch_size: int = 4
patch_size: tuple3_t[int] = (96, 224, 224)

def _load_mask(mask_dir: Path, name: str):
    return name, np.load(mask_dir / f'{name}.npy')

def process_case(dataset_dir: Path, key: str):
    case_dir = dataset_dir / 'data' / key
    masks = torch.as_tensor(np.load(case_dir / 'masks.npy'), device=get_cuda_device())
    patch_sum = []
    dtype = torch.int32 if masks[0].numel() < torch.iinfo(torch.int32).max else torch.int64
    mask_sizes = masks.new_empty((masks.shape[0], 1, 1, 1), dtype=dtype)
    for i in range(0, masks.shape[0], mask_batch_size):
        batch = masks[i:i + mask_batch_size]
        mask_sizes[i:i + mask_batch_size] = batch.sum(dim=(1, 2, 3), keepdim=True, dtype=dtype)
        patch_sum.append(sliding_window_sum(batch, patch_size, dtype))
    patch_sum = torch.cat(patch_sum)
    # patch contains 50% of the mask or 1000 mask voxels is positive
    positive_th = (mask_sizes * 0.5).clamp(max=1000)
    positive_mask = patch_sum > positive_th
    save_dir = dataset_dir / 'patch' / ','.join(map(str, patch_size)) / f'.{key}'
    save_dir.mkdir(exist_ok=True, parents=True)
    np.save(save_dir / 'positive_mask.npy', positive_mask.cpu().numpy())
    with h5py.File(save_dir / 'class_to_patch.h5', 'w') as f:
        for i in range(masks.shape[0]):
            positions = positive_mask[i].nonzero().short()
            f.create_dataset(str(i), data=positions.cpu().numpy())
    save_dir.rename(save_dir.with_name(key))

def process_dataset(dataset_dir: Path):
    save_root = dataset_dir / 'patch' / ','.join(map(str, patch_size))
    meta: pd.DataFrame = pd.read_pickle(dataset_dir / 'meta.pkl')
    keys = list(filter(lambda name: not (save_root / name).exists(), meta.index))
    process_map(
        process_case,
        it.repeat(dataset_dir), keys,
        ncols=80, max_workers=3, chunksize=1, total=len(keys),
    )

def main():
    process_dataset(PROCESSED_SEG_DATA_ROOT / 'TotalSegmentator')
    # for dataset_dir in PROCESSED_SEG_DATA_ROOT.iterdir():
    #     print(dataset_dir.name)

if __name__ == '__main__':
    main()
