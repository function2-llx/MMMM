import json
import os
import shutil
from tqdm import tqdm

from mmmm.data.defs import ORIGIN_VL_DATA_ROOT, PROCESSED_VL_DATA_ROOT

def process_text(jsonl_file: str, out_file: str):
    with open(ORIGIN_VL_DATA_ROOT / 'pmc_oa' / jsonl_file, 'r') as f:
        lines = f.readlines()
        data = [json.loads(line) for line in lines]
    
    processed_data = [
        {
            'image': item['image'],
            'caption': item['caption'],
        }
        for item in data
    ]

    with open(PROCESSED_VL_DATA_ROOT / 'PMC-OA' / out_file, 'w') as f:
        json.dump(processed_data, f, indent=4)

def process_images():
    (PROCESSED_VL_DATA_ROOT / 'PMC-OA' / 'images').mkdir(parents=True, exist_ok=True)
    for img in tqdm(os.listdir(ORIGIN_VL_DATA_ROOT / 'pmc_oa' / 'caption_T060_filtered_top4_sep_v0_subfigures')):
        shutil.copyfile(
            ORIGIN_VL_DATA_ROOT / 'pmc_oa' / 'caption_T060_filtered_top4_sep_v0_subfigures' / img,
            PROCESSED_VL_DATA_ROOT / 'PMC-OA' / 'images' / img
        )

def process():
    (PROCESSED_VL_DATA_ROOT / 'PMC-OA').mkdir(parents=True, exist_ok=True)
    process_text('train.jsonl', 'train.json')
    process_text('valid.jsonl', 'validate.json')
    process_text('test.jsonl', 'test.json')
    process_images()

if __name__ == '__main__':
    process()