from datetime import datetime
import json
from pathlib import Path

import cytoolz
import einops
import numpy as np
import orjson
import torch
from torchvision.io import read_image
from tqdm import tqdm

from luolib.utils import file_append, process_map

from mmmm.data.defs import ORIGIN_VL_DATA_ROOT, PROCESSED_VL_DATA_ROOT, PROCESSED_DATA_ROOT

ORIGIN_RP_DATA_ROOT = ORIGIN_VL_DATA_ROOT / 'RP3D-Image' / 'raw_images'
now = datetime.now()
log_path = PROCESSED_DATA_ROOT / '.logs' / 'vl' / now.strftime("%Y-%m-%d") / f'{now.strftime("%H:%M:%S")}.log'
log_path.parent.mkdir(exist_ok=True, parents=True)

def convert_path(radfm_path: str) -> Path:
    path = radfm_path.replace(
        '/mnt/petrelfs/share_data/zhangxiaoman/DATA/Radio_VQA/processed_file/npys',
        str(PROCESSED_VL_DATA_ROOT / 'Radiopaedia' / 'images'),
    )
    for suffix in ['.nii.gz', '.npy']:
        if path.endswith(suffix):
            path = path[:-len(suffix)] + '.pt'
    return Path(path)

def process_text(json_file: str, train_val: bool = False):
    with open(ORIGIN_VL_DATA_ROOT / 'RadFM_data_csv' / 'data_csv' / json_file) as f:
        data = json.load(f)
    processed_data = []
    for item in tqdm(data):
        valid_idx = [
            i for i, path in enumerate(item['image_path'])
            if convert_path(path).exists()
        ]
        if len(valid_idx) == 0:
            continue
        if isinstance(item['finding'], str) and item['finding'].strip():
            processed_data.append(
                {
                    'image': [str(convert_path(item['image_path'][i])) for i in valid_idx],
                    'modality': [item['image_modality'][i] for i in valid_idx],
                    'findings': item['image_caption'],
                    # 'impression': item['impression'],
                    'vqa': item['qa_list'],
                }
            )

    if train_val:
        np.random.RandomState(233).shuffle(processed_data)
        num_val = 250
        train_data = processed_data[:-num_val]
        val_data = processed_data[-num_val:]
        (PROCESSED_VL_DATA_ROOT / 'Radiopaedia' / 'train.json').write_bytes(
            orjson.dumps(train_data, option=orjson.OPT_INDENT_2),
        )
        (PROCESSED_VL_DATA_ROOT / 'Radiopaedia' / 'validate.json').write_bytes(
            orjson.dumps(val_data, option=orjson.OPT_INDENT_2)
        )
    else:
        (PROCESSED_VL_DATA_ROOT / 'Radiopaedia' / 'test.json').write_bytes(
            orjson.dumps(processed_data, option=orjson.OPT_INDENT_2)
        )

overwrite: bool = False

def info(message: str):
    file_append(log_path, message)

def process_image(image_dir: Path):
    key = str(image_dir.relative_to(ORIGIN_RP_DATA_ROOT))
    save_path = PROCESSED_VL_DATA_ROOT / 'Radiopaedia' / 'images' / f'{key}.pt'
    if save_path.exists() and not overwrite:
        return
    image_list = []
    slice_paths = sorted(image_dir.iterdir(), key=lambda x: int(x.stem))
    if int(slice_paths[0].stem) != 1 or int(slice_paths[-1].stem) != len(slice_paths):
        info(f'inconsistent slice numbers: {key}')
        return
    has_rgb = False
    for slice_path in slice_paths:
        try:
            image_array = read_image(str(slice_path))
        except:
            info(f'image contains bad slice: {key}/{slice_path.name}')
            break
        assert 1 <= image_array.shape[0] <= 4, slice_path
        if image_array.shape[0] == 2 or image_array.shape[0] == 4:
            # remove the alpha channel
            image_array = image_array[:-1]
        if image_array.shape[0] == 3:
            has_rgb = True
        image_list.append(image_array)
        if image_array.shape[1:] != image_list[0].shape[1:]:
            info(f'inconsistent slice shapes: {key}')
            break
    else:
        if has_rgb:
            for i, image in enumerate(image_list):
                if image.shape[0] == 1:
                    image_list[i] = einops.repeat(image, '1 h w -> c h w', c=3)
        image_array = einops.rearrange(image_list, 'd c h w -> c d h w')
        save_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_save_path = save_path.with_name(f'.{save_path.name}')
        torch.save(image_array, tmp_save_path)
        tmp_save_path.rename(save_path)

def list_image_dirs(case_dir: Path):
    return list(
        cytoolz.concat(study_dir.iterdir() for study_dir in case_dir.iterdir()),
    )

def process_images():
    output_root = PROCESSED_VL_DATA_ROOT / 'Radiopaedia'
    (output_root / 'images').mkdir(parents=True, exist_ok=True)
    if (image_dirs_path := output_root / 'image_dirs.txt').exists():
        image_dirs = image_dirs_path.read_text().splitlines()
        image_dirs = [*map(Path, image_dirs)]
    else:
        image_dirs_list = process_map(
            list_image_dirs,
            list(ORIGIN_RP_DATA_ROOT.iterdir()),
            max_workers=12, chunksize=12, desc='get image dirs',
        )
        image_dirs = sorted(cytoolz.concat(image_dirs_list))
        image_dirs_path.write_text('\n'.join(map(str, image_dirs)))

    process_map(process_image, image_dirs, max_workers=8, chunksize=1)

def process():
    (PROCESSED_VL_DATA_ROOT / 'Radiopaedia').mkdir(parents=True, exist_ok=True)
    # process_images()
    process_text('radiology_train.json', train_val=True)
    process_text('radiology_test.json', train_val=False)

if __name__ == '__main__':
    process()
