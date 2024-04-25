from copy import copy
from pathlib import Path

import orjson
import torch

from luolib.utils import get_cuda_device, process_map, save_pt_zst

from _utils import crop_resize, check_cuda_cache

RP_dir = Path('data/processed/vision-language/Radiopaedia')
RP_image_dir = RP_dir / 'images'
save_dir = Path('data/processed/vl-compressed/Radiopaedia')
image_save_dir = save_dir / 'images'
cuda_cache_th = 10

def process_study(study: dict):
    try:
        study = copy(study)
        for i, image_path in enumerate(study['image']):
            check_cuda_cache(cuda_cache_th)
            image_path = Path(image_path)
            save_path = image_save_dir / image_path.relative_to(RP_image_dir).with_suffix('.pt.zst')
            study['image'][i] = str(save_path)
            if save_path.exists():
                continue
            save_path.parent.mkdir(exist_ok=True, parents=True)
            image = torch.load(image_path, map_location=get_cuda_device())
            image = crop_resize(image)
            save_pt_zst(image.cpu(), save_path, atomic=True)
        return study
    except Exception as e:
        print(e)
        return None

def process_split(split: str):
    data_list = orjson.loads((RP_dir / f'{split}.json').read_bytes())
    results = process_map(process_study, data_list, max_workers=8, chunksize=8, ncols=80)
    Path(save_dir / f'{split}.json').write_bytes(orjson.dumps(results, option=orjson.OPT_INDENT_2))

def main():
    for split in ['train', 'validate', 'test']:
        process_split(split)

if __name__ == '__main__':
    main()
