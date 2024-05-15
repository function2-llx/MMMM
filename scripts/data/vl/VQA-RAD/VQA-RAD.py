import json
import os
import random
import shutil

from tqdm import tqdm

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
    for item in tqdm(data):
        if item['image_name'] != img:
            if test_vqa:
                origin_image_path = ORIGIN_VL_DATA_ROOT / 'VQA-RAD' / 'VQA_RAD Image Folder' / img
                save_image_path = PROCESSED_VL_DATA_ROOT / f'VQA-RAD/images/{img}'
                save_image_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(origin_image_path, save_image_path)
                test_data.append(
                    {
                        'image': [str(save_image_path)],
                        'vqa': test_vqa
                    }
                )
            if train_val_vqa:
                origin_image_path = ORIGIN_VL_DATA_ROOT / 'VQA-RAD' / 'VQA_RAD Image Folder' / img
                save_image_path = PROCESSED_VL_DATA_ROOT / f'VQA-RAD/images/{img}'
                save_image_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(origin_image_path, save_image_path)
                train_val_data.append(
                    {
                        'image': [str(save_image_path)],
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
                    'answer': str(item['answer']),
                }
            )
        else:
            train_val_vqa.append(
                {
                    'question': item['question'],
                    # converting to str since it might be an int, humor
                    'answer': str(item['answer']),
                }
            )

    random.shuffle(train_val_data)
    train_data = train_val_data
    # train_data = train_val_data[:int(len(train_val_data) * 0.8)]
    # val_data = train_val_data[int(len(train_val_data) * 0.8):]

    with open(PROCESSED_VL_DATA_ROOT / 'VQA-RAD' / 'train.json', 'w') as f:
        json.dump(train_data, f, indent=4)
    # with open(PROCESSED_VL_DATA_ROOT / 'VQA-RAD' / 'validate.json', 'w') as f:
    #     json.dump(val_data, f, indent=4)
    with open(PROCESSED_VL_DATA_ROOT / 'VQA-RAD' / 'test.json', 'w') as f:
        json.dump(test_data, f, indent=4)

if __name__ == '__main__':
    process()
