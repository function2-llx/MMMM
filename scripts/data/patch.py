from pathlib import Path

import cytoolz
import itertools as it
import numpy as np
import pandas as pd
import torch

from luolib.transforms import sliding_window_sum
from luolib.types import tuple3_t
from luolib.utils import get_cuda_device, process_map

from mmmm.data.defs import PROCESSED_SEG_DATA_ROOT

mask_batch_size: int = 16
patch_size: tuple3_t[int] = (96, 224, 224)

def _load_mask(mask_dir: Path, name: str):
    return name, np.load(mask_dir / f'{name}.npy')

def process_case(case_dir: Path, positive_classes: list[str]):
    device = get_cuda_device()
    masks = process_map(
        _load_mask,
        it.repeat(case_dir / 'masks'), positive_classes,
        new_mapper=False, disable=True, max_workers=16,
    )
    masks = dict(masks)
    for batch in cytoolz.partition_all(mask_batch_size, masks.items()):
        # process_map(lambda name: np.load(case_dir / 'masks' / f'{name}.npy'), batch_classes, )
        names, batch = zip(*batch)
        batch = torch.as_tensor(np.concatenate(batch), device=device)
        s = sliding_window_sum(batch, patch_size)
        print(s.shape)

def process_dataset(dataset_dir: Path):
    meta: pd.DataFrame = pd.read_pickle(dataset_dir / 'meta.pkl')
    meta = meta.head(5)
    process_map(
        process_case,
        meta.index.map(lambda key: dataset_dir / 'data' / key), meta['positive_classes'],
        ncols=80, max_workers=0, chunksize=1,
    )

def main():
    process_dataset(PROCESSED_SEG_DATA_ROOT / 'TotalSegmentator')
    # for dataset_dir in PROCESSED_SEG_DATA_ROOT.iterdir():
    #     print(dataset_dir.name)

if __name__ == '__main__':
    main()
