import json
from collections import Counter

import cytoolz

from mmmm.data.defs import ORIGIN_VL_DATA_ROOT

def main():
    data = json.loads((ORIGIN_VL_DATA_ROOT / 'RadFM_data_csv' / 'data_csv' / 'radiology_train.json').read_bytes())
    counter = Counter(
        cytoolz.concat(
            map(lambda x: None if x is None else x.lower(), item['plane_projection'])
            for item in data
        ),
    )
    counter = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    print(json.dumps(counter, indent=4, ensure_ascii=False))

if __name__ == '__main__':
    main()
