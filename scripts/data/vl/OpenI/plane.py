from pathlib import Path

import orjson
import pandas as pd

from mmmm.data.defs import ORIGIN_VL_DATA_ROOT, PROCESSED_VL_DATA_ROOT, Split

dataset = 'OpenI'
df = pd.read_csv(ORIGIN_VL_DATA_ROOT / dataset / 'indiana_projections.csv', index_col='filename')
projection: pd.Series = df['projection']
output_dir = PROCESSED_VL_DATA_ROOT / dataset

def process(split: Split):
    ref = pd.read_csv(PROCESSED_VL_DATA_ROOT / dataset / f'{split}-processed.csv')
    data: list[dict] = orjson.loads((PROCESSED_VL_DATA_ROOT / dataset / f'{split}-processed copy.json').read_bytes())
    for item in data:
        planes = []
        for image_path in item['image']:
            plane = projection.get(Path(image_path).name)
            if plane is not None:
                plane = plane.lower()
            planes.append(plane)
        item['plane'] = planes
    (PROCESSED_VL_DATA_ROOT / dataset / f'{split}-processed.json').write_bytes(
        orjson.dumps(data, option=orjson.OPT_INDENT_2),
    )

def main():
    process(Split.TRAIN)
    process(Split.TEST)

if __name__ == '__main__':
    main()
