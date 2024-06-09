import json
import pandas as pd

from mmmm.data.defs import ORIGIN_VL_DATA_ROOT, PROCESSED_VL_DATA_ROOT


def process():
    data = pd.read_csv(ORIGIN_VL_DATA_ROOT / 'ROCOv2' / 'train_captions.csv')
    data = [
        {
            'image': [str(ORIGIN_VL_DATA_ROOT / 'ROCOv2' / 'train' / (row['ID'] + '.jpg'))],
            'caption': row['Caption'],
        }
        for _, row in data.iterrows()
    ]
    (PROCESSED_VL_DATA_ROOT / 'ROCOv2').mkdir(parents=True, exist_ok=True)
    with open(PROCESSED_VL_DATA_ROOT / 'ROCOv2' / 'train.json', 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    process()
