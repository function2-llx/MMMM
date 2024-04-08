from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

from monai.data import MetaTensor

from ._base import DataPoint, Default3DImageLoaderMixin, Default3DMaskLoaderMixin, Processor

@dataclass(kw_only=True)
class Prostate158DataPoint(DataPoint):
    anatomy: Path
    tumor: Path | None = None

class Prostate158Processor(Default3DImageLoaderMixin, Default3DMaskLoaderMixin, Processor):
    name = 'Prostate158'
    orientation = 'SRA'

    @property
    def dataset_root(self):
        return super().dataset_root / 'prostate158_train'

    def load_masks(self, data_point: Prostate158DataPoint) -> tuple[MetaTensor, list[str]]:
        anatomy: MetaTensor = self.mask_loader(data_point.anatomy)
        anatomy_masks = torch.cat([anatomy == 1, anatomy == 2])
        targets = ['transition zone of prostate', 'peripheral zone of prostate']
        if data_point.tumor is None:
            masks = anatomy_masks
        else:
            tumor_mask = self.mask_loader(data_point.tumor)
            self._check_affine(anatomy.affine, tumor_mask.affine)
            masks = torch.cat([anatomy_masks, tumor_mask == 3])
            targets.append('prostate cancer')
        return masks, targets

    def get_data_points(self):
        df = pd.concat([
            pd.read_csv(self.dataset_root / f'{split}.csv', dtype={'ID': 'string'}).set_index('ID')
            for split in ['train', 'valid']
        ])
        return [
            Prostate158DataPoint(
                key=key,
                images={'T2 MRI': self.dataset_root / info['t2']},
                anatomy=self.dataset_root / info['t2_anatomy_reader1'],
                tumor=None if pd.isna(tumor := info['t2_tumor_reader1']) else self.dataset_root / tumor,
            )
            for key, info in df.iterrows()
        ]
