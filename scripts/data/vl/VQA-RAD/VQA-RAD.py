import json
import os
import random

from mmmm.data.defs import ORIGIN_VL_DATA_ROOT, PROCESSED_VL_DATA_ROOT

def process():
    os.makedirs(PROCESSED_VL_DATA_ROOT / 'VQA-RAD', exist_ok=True)
    with open(ORIGIN_VL_DATA_ROOT / 'VQA-RAD' / 'VQA_RAD Dataset Public.json') as f:
        data = json.load(f)

    data = sorted(data, key=lambda x: x['image_name'])
    test_data = []
    train_val_data = []

    test_vqa = []
    train_val_vqa = []
    img = ''
    for item in data:
        if item['image_name'] != img:
            if test_vqa:
                    test_data.append(
                        {
                            'image': [str(ORIGIN_VL_DATA_ROOT / 'VQA-RAD' / 'VQA_RAD Image Folder' / img)],
                            'vqa': test_vqa
                        }
                    )
            if train_val_vqa:
                    train_val_data.append(
                        {
                            'image': [str(ORIGIN_VL_DATA_ROOT / 'VQA-RAD' / 'VQA_RAD Image Folder' / img)],
                            'vqa': train_val_vqa
                        }
                    )
            img = item['image_name']
            test_vqa = []
            train_val_vqa = []
        if item['phrase_type'].startswith('test'):
            test_vqa.append(
                {
                    'question': item['question'],
                    'answer': item['answer'],
                }
            )
        else:
            train_val_vqa.append(
                {
                    'question': item['question'],
                    'answer': item['answer'],
                }
            )

    random.shuffle(train_val_data)
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