import json
import os
import shutil
from tqdm import tqdm

from mmmm.data.defs import ORIGIN_VL_DATA_ROOT, PROCESSED_VL_DATA_ROOT

def process_text(txt_file: str, out_file: str, test: bool =False):
    with open(ORIGIN_VL_DATA_ROOT / 'VQA-Med' / txt_file, 'r') as f:
        data = f.readlines()
    
    processed_data = []
    for item in data:
        item = item.split('|')
        processed_data.append(
            {
                'image': item[0] + '.jpg',
                'question': item[2 if test else 1],
                'answer': item[3 if test else 2].strip(),
            }
        )
        
    with open(PROCESSED_VL_DATA_ROOT / 'VQA-Med' / out_file, 'w') as f:
        json.dump(processed_data, f, indent=4)

def process_images(img_dir: str, out_dir: str):
    os.makedirs(PROCESSED_VL_DATA_ROOT / 'VQA-Med' / out_dir, exist_ok=True)
    for img in tqdm(os.listdir(ORIGIN_VL_DATA_ROOT / 'VQA-Med' / img_dir)):
        shutil.copy(
            ORIGIN_VL_DATA_ROOT / 'VQA-Med' / img_dir / img,
            PROCESSED_VL_DATA_ROOT / 'VQA-Med' / out_dir / img
        )

def process():
    os.makedirs(PROCESSED_VL_DATA_ROOT / 'VQA-Med', exist_ok=True)
    process_text('ImageClef-2019-VQA-Med-Training/All_QA_Pairs_train.txt', 'train.json')
    process_text('ImageClef-2019-VQA-Med-Validation/All_QA_Pairs_val.txt', 'validate.json')
    process_text('VQAMed2019Test/VQAMed2019_Test_Questions_w_Ref_Answers.txt', 'test.json', test=True)
    process_images('ImageClef-2019-VQA-Med-Training/Train_images', 'images')
    process_images('ImageClef-2019-VQA-Med-Validation/Val_images', 'images')
    process_images('VQAMed2019Test/VQAMed2019_Test_Images', 'images')

if __name__ == '__main__':
    process()