import json
import os
import shutil
from tqdm import tqdm

from mmmm.data.defs import ORIGIN_VL_DATA_ROOT, PROCESSED_VL_DATA_ROOT

def process():
    os.makedirs(PROCESSED_VL_DATA_ROOT / 'VQA-RAD', exist_ok=True)
    with open(ORIGIN_VL_DATA_ROOT / 'VQA-RAD' / 'VQA_RAD Dataset Public.json') as f:
        data = json.load(f)
    
    test_data = [
        {
            'image': str(ORIGIN_VL_DATA_ROOT / 'VQA-RAD' / 'VQA_RAD Image Folder' / item['image_name']),
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

if __name__ == '__main__':
    process()