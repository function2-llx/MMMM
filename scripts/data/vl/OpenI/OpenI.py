import csv
import json
import shutil

import numpy as np
from tqdm import tqdm

from mmmm.data.defs import ORIGIN_VL_DATA_ROOT, PROCESSED_VL_DATA_ROOT

def process(csv_file: str):
    (save_dir := PROCESSED_VL_DATA_ROOT / 'OpenI' / 'images').mkdir(parents=True, exist_ok=True)
    with open(ORIGIN_VL_DATA_ROOT / 'OpenI' / csv_file) as f:
        reader = csv.DictReader(f)
        data = []
        for i, item in tqdm(list(enumerate(reader)), ncols=80):
            origin_image_paths = list((ORIGIN_VL_DATA_ROOT / 'OpenI' / 'images' / 'images_normalized').glob(f'{i}_IM*.dcm.png'))
            if len(origin_image_paths) > 0 and (findings := item['findings'].strip()) and (impression := item['impression'].strip()):
                save_paths = []
                for origin_path in origin_image_paths:
                    save_path = save_dir / origin_path.name
                    shutil.copy(origin_path, save_path)
                    save_paths.append(str(save_path))
                data.append(
                    {
                        'image': save_paths,
                        'modality': ['X-ray'] * len(origin_image_paths),
                        'findings': findings,
                        'impression': impression,
                    }
                )
    np.random.RandomState(30924).shuffle(data)

    train_data = data[:int(len(data) * 0.9)]
    # val_data = data[int(len(data) * 0.8):int(len(data) * 0.9)]
    test_data = data[int(len(data) * 0.9):]
    # TODO: follow the split of https://github.com/omar-mohamed/GPT2-Chest-X-Ray-Report-Generation, as suggested by RadFM
    with open(PROCESSED_VL_DATA_ROOT / 'OpenI' / 'train.json', 'w') as f:
        json.dump(train_data, f, indent=4)
    # with open(PROCESSED_VL_DATA_ROOT / 'OpenI' / 'validate.json', 'w') as f:
    #     json.dump(val_data, f, indent=4)
    with open(PROCESSED_VL_DATA_ROOT / 'OpenI' / 'test.json', 'w') as f:
        json.dump(test_data, f, indent=4)

if __name__ == '__main__':
    process('indiana_reports.csv')
