import csv
from pathlib import Path
import re

import orjson
import pandas as pd
from tqdm import tqdm

from mmmm.data.defs import ORIGIN_VL_DATA_ROOT, PROCESSED_VL_DATA_ROOT

def process(csv_file: str):
    pattern = re.compile(r'CXR(\d+)_(1_)?IM-\d{4}-\d{4}\.png')
    split_map = {}
    for split in ['train', 'test']:
        split_df = pd.read_csv(Path(__file__).parent / f'{split}ing_set.csv', index_col='Image Index')
        print(split_df.shape)
        for image_idx, row in split_df.iterrows():
            uid = int(pattern.match(image_idx).group(1))
            if (_split := split_map.get(uid)) is None:
                split_map[uid] = split
            else:
                assert _split == split_map
    split_data = {'train': [], 'test': []}
    (image_dir := PROCESSED_VL_DATA_ROOT / 'OpenI' / 'images').mkdir(parents=True, exist_ok=True)
    with open(ORIGIN_VL_DATA_ROOT / 'OpenI' / csv_file) as f:
        reader = csv.DictReader(f)
        for i, item in tqdm(list(enumerate(reader))):
            image_paths = list((ORIGIN_VL_DATA_ROOT / f'OpenI/images/images_normalized/{i}_IM*.dcm.png').iterdir())
            processed_image_paths = []
            if len(image_paths) > 0 and (findings := item['findings'].strip()) and (impression := item['impression'].strip()):
                for j, image_path in enumerate(image_paths):
                    (link_path := image_dir / image_path.name).link_to(image_path)
                    processed_image_paths.append(str(link_path))
                split_data[split_map[item['uid']]].append(
                    {
                        'image': processed_image_paths,
                        'modality': ['X-ray'] * len(processed_image_paths),
                        'findings': findings,
                        'impression': impression,
                    }
                )
    for split, data in split_data.items():
        (PROCESSED_VL_DATA_ROOT / f'OpenI/{split}.json').write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))

if __name__ == '__main__':
    process('indiana_reports.csv')
