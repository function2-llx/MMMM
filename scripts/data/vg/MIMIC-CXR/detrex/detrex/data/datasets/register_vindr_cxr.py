import os
from functools import partial
from pathlib import Path

import orjson

from detectron2.data import DatasetCatalog, MetadataCatalog

thing_classes = [
    'Aortic enlargement',
    'Atelectasis',
    'Calcification',
    'Cardiomegaly',
    'Clavicle fracture',
    'Consolidation',
    'Edema',
    'Emphysema',
    'Enlarged PA',
    'ILD',
    'Infiltration',
    'Lung cavity',
    'Lung cyst',
    'Lung Opacity',
    'Mediastinal shift',
    'Nodule/Mass',
    'Pleural effusion',
    'Pleural thickening',
    'Pneumothorax',
    'Pulmonary fibrosis',
    'Rib fracture',
]

def _dataset_func(split: str, sub: bool):
    data_dir = Path(os.environ['DETECTRON2_DATASETS']) / 'VinDr-CXR'
    if sub:
        data_dir = data_dir.with_name(f'{data_dir.name}-sub')
    data = orjson.loads((data_dir / f'{split}.json').read_bytes())
    for item in data:
        item['file_name'] = str(data_dir / item['file_name'])

    return data[:10]

def _register():
    for split in ['train', 'test']:
        for sub in [True, False]:
            name = f'vindr-cxr_{split}'
            if sub:
                name += '-sub'
            DatasetCatalog.register(name, partial(_dataset_func, split, sub))
            meta = MetadataCatalog.get(name)
            meta.thing_classes = thing_classes

_register()
