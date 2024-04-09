import json
import os
import shutil
from tqdm import tqdm

from mmmm.data.defs import ORIGIN_VL_DATA_ROOT, PROCESSED_VL_DATA_ROOT

def process_text(json_file: str):
    with open(ORIGIN_VL_DATA_ROOT / 'Slake1.0' / json_file) as f:
        data = json.load(f)
    
    data = [
        {
            'image': str(ORIGIN_VL_DATA_ROOT / 'Slake1.0' / 'imgs' / item['img_name']),
            'question': item['question'],
            'answer': item['answer'],
        }
        for item in data
    ]

    with open(PROCESSED_VL_DATA_ROOT / 'Slake' / json_file, 'w') as f:
        json.dump(data, f, indent=4)

def process():
    (PROCESSED_VL_DATA_ROOT / 'Slake').mkdir(parents=True, exist_ok=True)
    process_text('train.json')
    process_text('validate.json')
    process_text('test.json')

if __name__ == '__main__':
    process()