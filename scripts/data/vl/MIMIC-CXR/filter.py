import orjson
from tqdm import tqdm

from mmmm.data.defs import PROCESSED_VL_DATA_ROOT

def main():
    data_dir = PROCESSED_VL_DATA_ROOT / f'MIMIC-CXR'
    for split in ['train', 'validate', 'test']:
        data = orjson.loads((data_dir / f'{split}.json').read_bytes())
        for item in tqdm(data):
            for key in ['findings', 'impression']:
                item[key] = item[key].replace('\n', '')
        (data_dir / f'{split}-filtered.json').write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))

if __name__ == '__main__':
    main()
