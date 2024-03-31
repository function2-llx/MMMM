# convert .nii.gz to .nii.zst
import gzip
from pathlib import Path
import shutil

import zstandard as zstd

from luolib.utils import process_map

def convert(seg_dir: Path):
    output_dir = seg_dir.parent / '.segmentations-zstd'
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    for seg_path in seg_dir.glob('*.nii.gz'):
        with open(seg_path, 'rb') as f:
            data_gz = f.read()
        data = gzip.decompress(data_gz)
        data_zst = zstd.compress(data)
        with open(output_dir / seg_path.with_suffix('.zst').name, 'wb') as f:
            f.write(data_zst)
    output_dir.rename(output_dir.with_name('segmentations-zstd'))

def main():
    data_dir = Path('data/origin/image/TotalSegmentator/Totalsegmentator_dataset_v201')
    items = []
    for seg_dir in data_dir.glob('*/segmentations'):
        if not (seg_dir.parent / 'segmentations-zstd').exists():
            items.append(seg_dir)
    process_map(convert, items, ncols=80, max_workers=12)

if __name__ == '__main__':
    main()
