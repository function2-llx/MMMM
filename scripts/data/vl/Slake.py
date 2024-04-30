import json
import os
import shutil
from tqdm import tqdm

from mmmm.data.defs import ORIGIN_VL_DATA_ROOT, PROCESSED_VL_DATA_ROOT

def process_text(json_file: str):
    with open(ORIGIN_VL_DATA_ROOT / 'Slake1.0' / json_file) as f:
        data = json.load(f)

    data = sorted(data, key=lambda x: x['img_name'])
    processed_data = []

    vqa = []
    img = ''
    modality = ''
    for item in data:
        if item['img_name'] != img:
            if vqa:
                processed_data.append(
                    {
                        'image': [str(ORIGIN_VL_DATA_ROOT / 'Slake1.0' / 'imgs' / img)],
                        'modality': modality,
                        'vqa': vqa
                    }
                )
            img = item['img_name']
            vqa = []
            modality = item['modality']
        if item['q_lang'] == 'en':
            vqa.append(
                {
                    'question': item['question'],
                    'answer': item['answer'],
                }
            )
        
    with open(PROCESSED_VL_DATA_ROOT / 'Slake' / json_file, 'w') as f:
        json.dump(processed_data, f, indent=4, ensure_ascii=False)

def process():
    (PROCESSED_VL_DATA_ROOT / 'Slake').mkdir(parents=True, exist_ok=True)
    process_text('train.json')
    process_text('validate.json')
    process_text('test.json')

if __name__ == '__main__':
    process()