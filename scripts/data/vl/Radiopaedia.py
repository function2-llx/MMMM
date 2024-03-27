import json
import os
import random
import shutil
from tqdm import tqdm

from mmmm.data.defs import ORIGIN_VL_DATA_ROOT, PROCESSED_VL_DATA_ROOT

def process_text(json_file: str, train_val: bool = False):
    with open(ORIGIN_VL_DATA_ROOT / 'RadFM_data_csv' / 'data_csv' / json_file) as f:
        data = json.load(f)
        
    new_data = [
        {
            "image": [
                path.replace("/mnt/petrelfs/share_data/zhangxiaoman/DATA/Radio_VQA/processed_file/npys/", "/data/MMMM/data/processed/vision-language/radiopaedia/images/")
                    .replace(".nii.gz", ".pt")
                    .replace(".npy", ".pt")
                for path in item["image_path"]
            ],
            "image_modality": item["image_modality"],
            "plane_projection": item["plane_projection"],
            "aux_modality": item["aux_modality"],
            "caption": item["image_caption"],
            "qa_list": item["qa_list"],
            "link": item["link"],
            "title": item["title"],
            "case_discussion": item["case_discussion"].replace("Case Discussion", ""),
            "finding": item["finding"],
            "impression": item["impression"],
        }
        for item in data
    ]

    if train_val:
        random.shuffle(new_data)
        split = int(len(new_data) * 0.8)
        train_data = new_data[:split]
        val_data = new_data[split:]
        
        with open(PROCESSED_VL_DATA_ROOT / 'Radiopaedia' / 'train.json', 'w') as f:
            json.dump(train_data, f, indent=4)
        with open(PROCESSED_VL_DATA_ROOT / 'Radiopaedia' / 'validate.json', 'w') as f:
            json.dump(val_data, f, indent=4)

    else:
        with open(PROCESSED_VL_DATA_ROOT / 'Radiopaedia' / 'test.json', 'w') as f:
            json.dump(new_data, f, indent=4)

def process():
    (PROCESSED_VL_DATA_ROOT / 'Radiopaedia').mkdir(parents=True, exist_ok=True)
    process_text('radiology_train.json', train_val=True)
    process_text('radiology_test.json', train_val=False)

if __name__ == '__main__':
    process()