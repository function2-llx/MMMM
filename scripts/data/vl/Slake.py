import json
import os
import shutil
from tqdm import tqdm

from mmmm.data.defs import ORIGIN_VL_DATA_ROOT, PROCESSED_VL_DATA_ROOT

def process_text(json_file: str):
    with open(ORIGIN_VL_DATA_ROOT / 'Slake1.0' / json_file) as f:
        data = json.load(f)
    
    processed_data = [
        {
            'image': item['img_name'],
            'question': item['question'],
            'answer': item['answer'],
            'type': item['answer_type'],
        }
        for item in data
    ]

    with open(PROCESSED_VL_DATA_ROOT / 'Slake' / json_file, 'w') as f:
        json.dump(processed_data, f, indent=4)

def process_images():
    (PROCESSED_VL_DATA_ROOT / 'Slake' / 'images').mkdir(parents=True, exist_ok=True)
    for img in tqdm(os.listdir(ORIGIN_VL_DATA_ROOT / 'Slake1.0' / 'imgs')):
        (PROCESSED_VL_DATA_ROOT / 'Slake' / 'images' / img).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ORIGIN_VL_DATA_ROOT / 'Slake1.0' / 'imgs' / img / 'source.jpg', PROCESSED_VL_DATA_ROOT / 'Slake' / 'images' / img / 'source.jpg')

def process():
    (PROCESSED_VL_DATA_ROOT / 'Slake').mkdir(parents=True, exist_ok=True)
    process_text('train.json')
    process_text('validate.json')
    process_text('test.json')
    process_images()

if __name__ == '__main__':
    process()