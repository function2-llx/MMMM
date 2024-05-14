import orjson
from tqdm import tqdm

from mmmm.data.defs import PROCESSED_VL_DATA_ROOT

def main():
    data_dir = PROCESSED_VL_DATA_ROOT / f'MIMIC-CXR'
    prefix = 'data/processed/vl-compressed'
    prefix_len = len(prefix)
    new_prefix = 'data/processed/vision-language'
    for split in ['train', 'validate', 'test']:
        data = orjson.loads((data_dir / f'{split}.json').read_bytes())
        for item in tqdm(data):
            for key in ['findings', 'impression']:
                item[key] = item[key].replace('\n', '')
            for i, image_path in enumerate(item['image']):
                assert image_path.startswith(prefix)
                item['image'][i] = new_prefix + image_path[prefix_len:]

        (data_dir / f'{split}-filtered.json').write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))

if __name__ == '__main__':
    main()
