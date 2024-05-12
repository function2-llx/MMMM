import orjson

from mmmm.data.defs import PROCESSED_VL_DATA_ROOT

def main():
    data_dir = PROCESSED_VL_DATA_ROOT / f'MIMIC-CXR'
    for split in ['train', 'val', 'test']:
        data = orjson.loads((data_dir / f'{split}.json').read_bytes())
        for item in data:
            for key in ['findings', 'impression']:
                item[key] = item['key'].replace('\n', '')
        

if __name__ == '__main__':
    main()
