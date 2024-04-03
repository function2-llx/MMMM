import json
import os
import shutil
from tqdm import tqdm

from mmmm.data.defs import ORIGIN_VL_DATA_ROOT, PROCESSED_VL_DATA_ROOT

def process_text(json_file: str):
    with open(ORIGIN_VL_DATA_ROOT / 'VQA-RAD' / json_file) as f:
        data = json.load(f)
    
    test_data = [
        {
            'image': item['image_name'],
            'question': item['question'],
            'answer': item['answer'],
        }
        for item in data
        if item['phrase_type'].startswith('test')
    ]

    train_val_data = [item for item in data if item not in test_data]
    train_data = train_val_data[:int(len(data) * 0.8)]
    val_data = train_val_data[int(len(data) * 0.8):]

    with open(PROCESSED_VL_DATA_ROOT / 'VQA-RAD' / 'train.json', 'w') as f:
        json.dump(train_data, f, indent=4)
    with open(PROCESSED_VL_DATA_ROOT / 'VQA-RAD' / 'validate.json', 'w') as f:
        json.dump(val_data, f, indent=4)
    with open(PROCESSED_VL_DATA_ROOT / 'VQA-RAD' / 'test.json', 'w') as f:
        json.dump(test_data, f, indent=4)

def process_images():
    (PROCESSED_VL_DATA_ROOT / 'VQA-RAD' / 'images').mkdir(parents=True, exist_ok=True)
    for img in tqdm(os.listdir(ORIGIN_VL_DATA_ROOT / 'VQA-RAD' / 'VQA_RAD Image Folder')):
        shutil.copyfile(
            ORIGIN_VL_DATA_ROOT / 'VQA-RAD' / 'VQA_RAD Image Folder' / img,
            PROCESSED_VL_DATA_ROOT / 'VQA-RAD' / 'images' / img
        )

def process():
    os.makedirs(PROCESSED_VL_DATA_ROOT / 'VQA-RAD', exist_ok=True)
    process_text('VQA_RAD Dataset Public.json')
    process_images()

if __name__ == '__main__':
    process()