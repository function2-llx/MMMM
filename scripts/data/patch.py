import itertools as it
from pathlib import Path

import einops
import h5py
import numpy as np
import pandas as pd
import torch

from luolib import transforms as lt
from luolib.types import tuple3_t
from luolib.utils import get_cuda_device, process_map

from mmmm.data.defs import PROCESSED_SEG_DATA_ROOT, encode_patch_size

mask_batch_size: int = 4
patch_size: tuple3_t[int] = (96, 224, 224)

def _load_mask(mask_dir: Path, name: str):
    return name, np.load(mask_dir / f'{name}.npy')

def _get_center_slices(shape: tuple3_t[int]):
    ret = [
        slice(s >> 1, (s >> 1) + 1) if s <= ps
        else slice(ps >> 1, s - ps + 1 + (ps >> 1))
        for s, ps in zip(shape, patch_size)
    ]
    return ret

def process_case(dataset_dir: Path, key: str):
    case_dir = dataset_dir / 'data' / key
    masks = torch.as_tensor(np.load(case_dir / 'masks.npy'), device=get_cuda_device())
    patches_sum = []
    dtype = torch.int32 if masks[0].numel() < torch.iinfo(torch.int32).max else torch.int64
    mask_sizes = masks.new_empty((masks.shape[0], 1, 1, 1), dtype=dtype)
    for i in range(0, masks.shape[0], mask_batch_size):
        batch = masks[i:i + mask_batch_size]
        mask_sizes[i:i + mask_batch_size] = batch.sum(dim=(1, 2, 3), keepdim=True, dtype=dtype)
        patches_sum.append(lt.sliding_window_sum(batch, patch_size, dtype))
    patches_sum = torch.cat(patches_sum)
    # a patch is "significant" for a class iff any:
    # - contains at least 80% voxels of this class
    # - center voxel is this class
    significant_mask = (patches_sum > (mask_sizes * 0.8)) | masks[:, *_get_center_slices(masks.shape[1:])]
    patches_class_mask = patches_sum > 0

    save_dir = dataset_dir / 'patch' / encode_patch_size(patch_size) / f'.{key}'
    save_dir.mkdir(exist_ok=True, parents=True)
    with h5py.File(save_dir / 'class_positions.h5', 'w') as f:
        for i in range(masks.shape[0]):
            if significant_mask[i].any():
                positions = significant_mask[i].nonzero().short()
            else:
                # no position is significant for this class according to previous rules, use the 80% percentile instead
                class_significant_mask = patches_sum[i] >= lt.quantile(patches_sum[i][patches_sum[i] > 0], 0.8)
                positions = class_significant_mask.nonzero().short()
            assert positions.shape[0] > 0
            f.create_dataset(str(i), data=positions.cpu().numpy())
    # convert to channel last for efficient (position -> class) query
    patches_class_mask = einops.rearrange(patches_class_mask, 'c ... -> ... c')
    np.save(save_dir / 'patches_class_mask.npy', patches_class_mask.cpu().numpy())
    save_dir.rename(save_dir.with_name(key))

def process_dataset(dataset_dir: Path):
    save_root = dataset_dir / 'patch' / encode_patch_size(patch_size)
    meta: pd.DataFrame = pd.read_pickle(dataset_dir / 'meta.pkl')
    keys = list(filter(lambda name: not (save_root / name).exists(), meta.index))
    process_map(
        process_case,
        it.repeat(dataset_dir), keys,
        ncols=80, max_workers=4, chunksize=1, total=len(keys),
    )

def main():
    process_dataset(PROCESSED_SEG_DATA_ROOT / 'AMOS22')
    # for dataset_dir in PROCESSED_SEG_DATA_ROOT.iterdir():
    #     print(dataset_dir.name)

if __name__ == '__main__':
    main()
