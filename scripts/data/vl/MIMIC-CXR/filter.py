from pathlib import Path

import orjson
import pandas as pd
from tqdm import tqdm

from mmmm.data.defs import PROCESSED_VL_DATA_ROOT, ORIGIN_VL_DATA_ROOT

labels = [
    ('Atelectasis', 'atelectasis'),
    ('Cardiomegaly', 'cardiomegaly'),
    ('Consolidation', 'pulmonary consolidation'),
    ('Edema', 'pulmonary edema'),
    ('Enlarged Cardiomediastinum', 'widened mediastinum'),
    ('Fracture', 'rib fracture'),
    ('Lung Lesion', 'lung nodule'),  # looks like they refer to the same thing in MIMIC-CXR
    ('Lung Opacity', 'pulmonary opacification'),
    ('Pleural Effusion', 'pleural effusion'),
    # ('Pleural Other'),
    # ('Pneumonia', ),
    ('Pneumothorax', 'pneumothorax'),
]

def main():
    data_dir = PROCESSED_VL_DATA_ROOT / f'MIMIC-CXR'
    prefix = 'data/processed/vl-compressed'
    prefix_len = len(prefix)
    new_prefix = 'data/processed/vision-language'
    label_df = pd.read_csv(
        ORIGIN_VL_DATA_ROOT / 'MIMIC-CXR-JPG/mimic-cxr-2.0.0-chexpert.csv',
        dtype={'subject_id': 'string', 'study_id': 'string'},
    )
    label_df.set_index(['subject_id', 'study_id'], inplace=True)
    for split in ['train', 'validate', 'test']:
        data = orjson.loads((data_dir / f'{split}.json').read_bytes())
        for item in tqdm(data):
            for key in ['findings', 'impression']:
                item[key] = item[key].replace('\n', '')
            for i, image_path in enumerate(item['image']):
                if i == 0:
                    patient, study = Path(image_path).parent.parts[-2:]
                assert image_path.startswith(prefix)
                item['image'][i] = new_prefix + image_path[prefix_len:]
            label = label_df.loc[patient[1:], study[1:]]
            if label['No Finding'] == 1:
                anomaly_pos = []
                anomaly_neg = [name for _, name in labels]
            else:
                anomaly_pos = []
                anomaly_neg = []
                for anomaly_key, name in labels:
                    if label[anomaly_key] == 1:
                        anomaly_pos.append(name)
                    elif label[anomaly_key] != -1:
                        anomaly_neg.append(name)
            item['anomaly_pos'] = anomaly_pos
            item['anomaly_neg'] = anomaly_neg
        (data_dir / f'{split}-filtered.json').write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))

if __name__ == '__main__':
    main()
