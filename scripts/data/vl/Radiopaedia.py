from builtins import isinstance
import cv2
from einops import repeat
import json
from multiprocessing import Pool
import os
from PIL import Image
import random
import torch
from tqdm import tqdm
from typing import List

from mmmm.data.defs import ORIGIN_VL_DATA_ROOT, PROCESSED_VL_DATA_ROOT

ORIGIN_RP_DATA_ROOT = ORIGIN_VL_DATA_ROOT / 'RP3D-Image' / 'raw_images'

def process_text(json_file: str, train_val: bool = False):
    with open(ORIGIN_VL_DATA_ROOT / 'RadFM_data_csv' / 'data_csv' / json_file) as f:
        data = json.load(f)
        
    processed_data = []
    for item in data:
        valid = True
        for path in item['image_path']:
            if not os.path.exists(path.replace('/mnt/petrelfs/share_data/zhangxiaoman/DATA/Radio_VQA/processed_file/npys', str(PROCESSED_VL_DATA_ROOT / 'Radiopaedia' / 'images')) \
                        .replace('.nii.gz', '.pt') \
                        .replace('.npy', '.pt') \
                    ):
                valid = False
                break
        if not valid:
            continue

        text = ''
        if isinstance(item['finding'], str):
            text += 'Findings: ' + item['finding']
            if isinstance(item['impression'], str):
                if text:
                    text += ' '
                text += 'Impression: ' + item['impression']
            if text:
                processed_data.append(
                    {
                        'image': [
                            path.replace('/mnt/petrelfs/share_data/zhangxiaoman/DATA/Radio_VQA/processed_file/npys/', str(PROCESSED_VL_DATA_ROOT / 'Radiopaedia' / 'images/'))
                                .replace('.nii.gz', '.pt')
                                .replace('.npy', '.pt')
                            for path in item['image_path']
                        ],
                        'caption': text,
                        'qa_list': item['qa_list'],
                    }
                )

    if train_val:
        random.shuffle(processed_data)
        split = int(len(data) * 0.8)
        train_data = processed_data[:split]
        val_data = processed_data[split:]
        
        with open(PROCESSED_VL_DATA_ROOT / 'Radiopaedia' / 'train.json', 'w') as f:
            json.dump(train_data, f, indent=4)
        with open(PROCESSED_VL_DATA_ROOT / 'Radiopaedia' / 'validate.json', 'w') as f:
            json.dump(val_data, f, indent=4)

    else:
        with open(PROCESSED_VL_DATA_ROOT / 'Radiopaedia' / 'test.json', 'w') as f:
            json.dump(processed_data, f, indent=4)

def check_image(image_path: str):
    try:
        Image.open(image_path).load()
        return True
    except:
        return False

def process_cases(cases: List[str], i):
    print(f"Processing {i} - {i + 1000}")
    for case in tqdm(cases[i: i + 1000]):
        for subset in os.listdir(ORIGIN_RP_DATA_ROOT / case):
            for image in os.listdir(ORIGIN_RP_DATA_ROOT / case /subset):
                slices = os.listdir(ORIGIN_RP_DATA_ROOT / case / subset / image)
                image_list = []
                bad = False
                for slice in slices:
                    if not check_image(str(ORIGIN_RP_DATA_ROOT / case / subset / image / slice)):
                        break
                    image_array = cv2.imread(str(ORIGIN_RP_DATA_ROOT / case / subset / image / slice))
                    image_array = torch.tensor(image_array)
                    if len(image_array.shape) == 2:
                        image_array = repeat(image_array, 'h w -> h w c', c=3)
                    image_list.append(image_array)
                if bad:
                    print(f'Bad image: {case}/{subset}/{image}')
                    continue
                if len(image_list) == 1:
                    image_array = image_list[0].permute(2, 0, 1)
                else:
                    image_array = torch.stack(image_list).permute(3, 1, 2, 0)
                torch.save(image_array, PROCESSED_VL_DATA_ROOT / 'Radiopaedia' / 'images' / case / subset / f'{image}.pt')


def process_images():
    (PROCESSED_VL_DATA_ROOT / 'Radiopaedia' / 'images').mkdir(parents=True, exist_ok=True)
    cases = os.listdir(ORIGIN_RP_DATA_ROOT)
    with Pool() as p:
        p.starmap(process_cases, [(cases, i) for i in range(0, len(cases), 2000)])

def process():
    (PROCESSED_VL_DATA_ROOT / 'Radiopaedia').mkdir(parents=True, exist_ok=True)
    # process_images()
    process_text('radiology_train.json', train_val=True)
    process_text('radiology_test.json', train_val=False)

if __name__ == '__main__':
    process()