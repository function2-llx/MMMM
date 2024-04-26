from pathlib import Path
import shutil

import torch

from luolib.utils import process_map, save_pt_zst
from mmmm.data.defs import PROCESSED_DATA_ROOT

src_root = PROCESSED_DATA_ROOT / 'image'
target_root = PROCESSED_DATA_ROOT / 'image-compressed'

def process(src_dir: Path):
    target_dir = target_root / src_dir.relative_to(src_root)
    if target_dir.exists():
        return
    tmp_target_dir = target_dir.with_name(f'.{target_dir.name}')
    if tmp_target_dir.exists():
        shutil.rmtree(tmp_target_dir)
    else:
        tmp_target_dir.mkdir(parents=True)
    for filepath in src_dir.iterdir():
        if filepath.name == 'images.pt':
            images: torch.ByteTensor = torch.load(filepath)
            save_pt_zst(images, tmp_target_dir / 'images.pt.zst')
        else:
            (tmp_target_dir / filepath.name).hardlink_to(filepath)
    tmp_target_dir.rename(target_dir)

def main():
    for dataset_dir in src_root.glob('*/'):
        target_dataset_dir = target_root / dataset_dir.relative_to(src_root)
        target_dataset_dir.mkdir(exist_ok=True, parents=True)
        for name in ['info.csv', 'info.xlsx']:
            if not (target_dataset_dir / name).exists():
                (target_dataset_dir / name).hardlink_to(dataset_dir / name)
        src_dirs = list((dataset_dir / 'data').glob('*/'))
        process_map(process, src_dirs, max_workers=16, desc=dataset_dir.name)

if __name__ == '__main__':
    main()
