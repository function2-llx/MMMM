import orjson

import numpy as np
from torchvision.io import read_image

from luolib.utils import process_map
from mmmm.data.defs import ORIGIN_VL_DATA_ROOT, PROCESSED_VL_DATA_ROOT

check_image: bool = True

def process_item(json_str: str):
    item: dict = orjson.loads(json_str)
    path = ORIGIN_VL_DATA_ROOT / 'pmc_oa/caption_T060_filtered_top4_sep_v0_subfigures' / item['image']
    if check_image:
        try:
            read_image(str(path))
        except:
            print(item['image'])
            return None
    caption: str = item['caption']
    caption = caption.strip()
    if len(caption) < 10:
        return None
    if caption[0].islower():
        caption = caption[0].upper() + caption[1:]
    if caption[-1] != '.':
        caption += '.'
    return {
        'image': str(path),
        'caption': caption,
    }

def process():
    (PROCESSED_VL_DATA_ROOT / 'PMC-OA').mkdir(parents=True, exist_ok=True)
    item_strs = (ORIGIN_VL_DATA_ROOT / 'pmc_oa' / 'pmc_oa.jsonl').read_text().strip().splitlines()
    data = process_map(process_item, item_strs, max_workers=16, chunksize=16, dynamic_ncols=True)
    data = [*filter(lambda x: x is not None, data)]
    np.random.RandomState(16358).shuffle(data)
    num_val = 500
    train_data, val_data = data[:-num_val], data[-num_val:]
    (PROCESSED_VL_DATA_ROOT / 'PMC-OA' / 'train.json').write_bytes(
        orjson.dumps(train_data, option=orjson.OPT_INDENT_2),
    )
    (PROCESSED_VL_DATA_ROOT / 'PMC-OA' / 'validate.json').write_bytes(
        orjson.dumps(val_data, option=orjson.OPT_INDENT_2),
    )

if __name__ == '__main__':
    process()
