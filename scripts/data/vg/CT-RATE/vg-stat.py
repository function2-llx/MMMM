from collections import Counter
from pathlib import Path

import orjson
from tqdm import tqdm

from mmmm.data.defs import PROCESSED_VG_DATA_ROOT

def main():
    data: list[dict] = orjson.loads((PROCESSED_VG_DATA_ROOT / 'CT-RATE' / 'train.json').read_bytes())
    cnt = Counter()
    for item in tqdm(data):
        item_targets = [tag['target'] for tag in item['tags']]
        for target in set(item_targets):
            cnt[target] += 1
    cnt = sorted(cnt.items(), key=lambda x: x[1], reverse=True)
    Path('vg-stat.json').write_bytes(orjson.dumps(dict(cnt), option=orjson.OPT_INDENT_2))

if __name__ == '__main__':
    main()
