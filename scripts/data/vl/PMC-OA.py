import json
import os
import shutil
from tqdm import tqdm

from mmmm.data.defs import ORIGIN_VL_DATA_ROOT, PROCESSED_VL_DATA_ROOT

def process_text(jsonl_file: str, out_file: str):
    with open(ORIGIN_VL_DATA_ROOT / 'pmc_oa' / jsonl_file, 'r') as f:
        lines = f.readlines()
        data = [json.loads(line) for line in lines]
    
    data = [
        {
            'image': str(ORIGIN_VL_DATA_ROOT / 'caption_T060_filtered_top4_sep_v0_subfigures' / item['image']),
            'caption': item['caption'],
        }
        for item in data
    ]

    with open(PROCESSED_VL_DATA_ROOT / 'PMC-OA' / out_file, 'w') as f:
        json.dump(data, f, indent=4)

def process():
    (PROCESSED_VL_DATA_ROOT / 'PMC-OA').mkdir(parents=True, exist_ok=True)
    process_text('train.jsonl', 'train.json')
    process_text('valid.jsonl', 'validate.json')
    process_text('test.jsonl', 'test.json')

if __name__ == '__main__':
    process()