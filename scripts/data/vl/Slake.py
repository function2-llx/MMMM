import json
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
    for item in tqdm(data):
        if item['img_name'] != img:
            if vqa:
                origin_image_path = ORIGIN_VL_DATA_ROOT / 'Slake1.0' / 'imgs' / img
                save_image_path = PROCESSED_VL_DATA_ROOT / f'Slake/images/{img}'
                save_image_path.parent.mkdir(exist_ok=True, parents=True)
                shutil.copy(origin_image_path, save_image_path)
                processed_data.append(
                    {
                        'image': [str(save_image_path)],
                        'modality': modality,
                        'vqa': vqa,
                    }
                )
            img = item['img_name']
            vqa = []
            modality = item['modality']
        if item['q_lang'] == 'en' and (question := item['question'].strip()) and (answer := item['answer'].strip()):
            vqa.append(
                {
                    'question': question,
                    'answer': answer,
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
