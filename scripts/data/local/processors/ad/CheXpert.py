from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from tqdm import tqdm

from monai.data import MetaTensor

from mmmm.data.defs import Split
from mmmm.data.sparse import Sparse

from .._base import DataPoint, NaturalImageLoaderMixin, Processor

@dataclass(kw_only=True)
class CheXpertDataPoint(DataPoint):
    split: Split

class CheXpertProcessor(NaturalImageLoaderMixin, Processor):
    name = 'CheXpert'
    assert_local = False
    anomalies = [
        ('Enlarged Cardiomediastinum', 'widened mediastinum'),
        ('Cardiomegaly', 'cardiomegaly'),
        ('Lung Opacity', 'pulmonary opacification'),
        # ('Lung Lesion', 'lung lesion'),
        ('Edema', 'pulmonary edema'),
        ('Consolidation', 'pulmonary consolidation'),
        # ('Pneumonia', 'pneumonia'),
        ('Atelectasis', 'atelectasis'),
        ('Pneumothorax', 'pneumothorax'),
        ('Pleural Effusion', 'pleural effusion'),
        # 'Pleural Other',  too abstract
        ('Fracture', 'rib fracture'),
    ]

    @property
    def dataset_root(self):
        return super().dataset_root / 'chexpertchestxrays-u20210408'

    def get_data_points(self):
        ret = []
        split_dict = {}
        self.label = {}
        for split in Split:
            if split == Split.TRAIN:
                label_file = self.dataset_root / 'train_visualCheXbert.csv'
            else:
                stem = 'valid' if split == Split.VAL else 'test'
                label_file = self.dataset_root / f'CheXpert-v1.0/{stem}.csv'
            self.label[split] = label = pd.read_csv(label_file)
            keys = []
            for _, info in tqdm(label.iterrows(), total=label.shape[0]):
                path = Path(info['Path'])
                patient, study, filename = *path.parts[-3:-1], path.stem
                key = f'{patient}-{study}-{filename}'
                keys.append(key)
                ret.append(
                    CheXpertDataPoint(
                        key=key,
                        images={'X-ray': self.dataset_root / path},
                        split=split,
                        complete_anomaly=True,
                    )
                )
            split_dict[split] = keys
            label['_keys'] = keys
            label.set_index('_keys', inplace=True)
        return ret, split_dict

    def _load_anomalies(self, data_point: CheXpertDataPoint) -> tuple[set[str], set[str]]:
        info: pd.Series = self.label[data_point.split].loc[data_point.key]
        if info['No Finding']:
            return set(), {target for _, target in self.anomalies}
        pos, neg = [], []
        for (anomaly, target) in self.anomalies:
            (pos if info[anomaly] else neg).append(target)
        return set(pos), set(neg)

    def load_annotations(
        self, data_point: CheXpertDataPoint, images: MetaTensor,
    ) -> tuple[list[str], set[str], MetaTensor | None, torch.Tensor | None]:
        pos, neg = self._load_anomalies(data_point)
        return list(pos), neg, None, None

    def _group_targets(
        self, targets: list[str], _masks: ..., _boxes: ...,
    ) -> tuple[list[Sparse.Target], torch.BoolTensor | None, torch.LongTensor | None]:
        targets = [
            Sparse.Target(
                name=target,
                semantic=False,
                position_offset=None,
                index_offset=None,
                mask_sizes=None,
                boxes=None,
            )
            for target in targets
        ]
        return targets, None, None
