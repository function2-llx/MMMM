from copy import copy

import math
from pathlib import Path

import orjson
import torch
from torchvision.transforms.v2.functional import to_dtype

from luolib.utils import get_cuda_device, process_map, save_pt_zst
import monai.transforms as mt
from monai.utils import InterpolateMode

RP_dir = Path('data/processed/vision-language/Radiopaedia')
RP_image_dir = RP_dir / 'images'
save_dir = Path('data/processed/vl-compressed/Radiopaedia')
image_save_dir = save_dir / 'images'

def process_study(study: dict):
    study = copy(study)
    for i, image_path in enumerate(study['image']):
        image_path = Path(image_path)
        save_path = image_save_dir / image_path.relative_to(RP_image_dir).with_suffix('.pt.zst')
        study['image'][i] = str(save_path)
        if save_path.exists():
            continue
        save_path.parent.mkdir(exist_ok=True, parents=True)
        image = torch.load(str(image_path), mmap=True)
        max_tokens_z = min(4, image.shape[1])
        max_smaller_edge = int((256 / max_tokens_z) ** 0.5) * 32
        resize_shape = [min(max_tokens_z * 32, image.shape[1]), *image.shape[2:]]
        if (_base := min(resize_shape[1:])) > max_smaller_edge:
            for j in (1, 2):
                resize_shape[j] = math.ceil(resize_shape[j] * max_smaller_edge / _base)
        if (resize_shape := tuple(resize_shape)) != image.shape[1:]:
            image = image.to(device=get_cuda_device())
            image = to_dtype(image, scale=True)
            resize = mt.Resize(resize_shape, mode=InterpolateMode.TRILINEAR, anti_aliasing=True)
            image = resize(image)
            image = to_dtype(image.as_tensor(), dtype=torch.uint8, scale=True)
            image = image.cpu()
        save_pt_zst(image, save_path)
    return study

def process_split(split: str):
    data_list = orjson.loads((RP_dir / f'{split}.json').read_bytes())
    results = process_map(process_study, data_list, max_workers=8, chunksize=8, ncols=80)
    Path(save_dir / f'{split}.json').write_bytes(orjson.dumps(results, option=orjson.OPT_INDENT_2))

def main():
    for split in ['train', 'validate', 'test']:
        process_split(split)

if __name__ == '__main__':
    main()
