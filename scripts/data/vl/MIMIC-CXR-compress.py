from copy import copy
from pathlib import Path

import einops
import orjson
from torchvision.io import read_image

from luolib.utils import get_cuda_device, process_map, save_pt_zst
from _utils import crop_resize, check_cuda_cache

save_dir = Path('data/processed/vl-compressed/MIMIC-CXR')
cuda_cache_th = 10

def process_study(study: dict):
    try:
        study = copy(study)
        for i, image_path in enumerate(study['image']):
            check_cuda_cache()
            image_path = Path(image_path)
            save_path = save_dir / image_path.relative_to('data/origin/vision-language/MIMIC-CXR-JPG').with_suffix('.pt.zst')
            study['image'][i] = str(save_path)
            if study['modality'][i] == 'X-Ray':
                study['modality'][i] = 'X-ray'
            if save_path.exists():
                continue
            save_path.parent.mkdir(exist_ok=True, parents=True)
            image = read_image(str(image_path))
            image = image.to(device=get_cuda_device())
            image = einops.rearrange(image, 'c h w -> c 1 h w')
            image = crop_resize(image)
            save_pt_zst(image.cpu(), save_path, atomic=True)
        return study
    except Exception as e:
        print(e)
        return None

def process_split(split: str):
    data_list = orjson.loads((Path('data/processed/vision-language/MIMIC-CXR') / f'{split}.json').read_bytes())
    results = process_map(process_study, data_list, max_workers=8, chunksize=8, ncols=80)
    Path(save_dir / f'{split}.json').write_bytes(orjson.dumps(results, option=orjson.OPT_INDENT_2))

def main():
    for split in ['train', 'validate', 'test']:
        process_split(split)

if __name__ == '__main__':
    main()
