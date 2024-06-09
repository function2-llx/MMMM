import json
import shutil

import orjson
import pandas as pd

from mmmm.data.defs import PROCESSED_VL_DATA_ROOT

def _get_data(dataset: str, split: str, suffix: str) -> list[dict]:
    if not (dst := PROCESSED_VL_DATA_ROOT / dataset / f'archive/{split}-processed-{suffix}.json').exists():
        shutil.copy(PROCESSED_VL_DATA_ROOT / dataset / f'{split}-processed.json', dst)
    return orjson.loads(dst.read_bytes())

def fix(dataset: str):
    for split in ['test', 'train']:
        df = pd.read_csv(PROCESSED_VL_DATA_ROOT / dataset / f'{split}-processed.csv')
        data = _get_data(dataset, split, 'to-fix')
        for item, processed_report in zip(data, df['processed']):
            item['processed_report'] = processed_report
        (PROCESSED_VL_DATA_ROOT / dataset / f'{split}-processed.json').write_bytes(
            orjson.dumps(data, option=orjson.OPT_INDENT_2),
        )

def clean(dataset: str):
    for split in ['test', 'train']:
        data = _get_data(dataset, split, 'to-clean')
        output = []
        for item in data:
            valid = True
            try:
                if not item['processed_report'].split('Impression:')[1].strip():
                    valid = False
                if not item['processed_report'].split('Findings:')[1].split('Impression:')[0].strip():
                    valid = False
            except:
                valid = False
            if valid:
                output.append(item)
            else:
                print(item)
        print(f'{dataset} {split}: {len(data)} -> {len(output)}')
        with open(PROCESSED_VL_DATA_ROOT / dataset / f'{split}-processed.json', 'w') as f:
            json.dump(output, f, indent=2)

def main():
    # fix('OpenI')
    # clean('OpenI')
    # fix('CT-RATE')
    clean('CT-RATE')

if __name__ == '__main__':
    main()
