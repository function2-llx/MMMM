from dataclasses import dataclass
from pathlib import Path

import torch

from monai.data import MetaTensor
from ._base import DefaultImageLoaderMixin, DefaultMaskLoaderMixin, Processor, SegDataPoint

@dataclass(kw_only=True)
class RibFracDataPoint(SegDataPoint):
    label: Path

class RibFracProcessor(DefaultImageLoaderMixin, DefaultMaskLoaderMixin, Processor):
    name = 'RibFrac'
    orientation = 'SRA'

    def load_masks(self, data_point: RibFracDataPoint, images: MetaTensor) -> tuple[list[str], MetaTensor]:
        label = self.mask_loader(data_point.label)
        label = label.to(device=self.device)
        label_ids = label.unique()
        max_label_id = label_ids.max().item()
        assert label_ids.min() == 0 and max_label_id + 1 == label_ids.shape[0]
        label_ids = label_ids[label_ids != 0]
        masks: MetaTensor = label == label_ids[:, None, None, None]  # type: ignore
        return ['rib fracture'] * max_label_id, masks

    def get_data_points(self):
        ret = []
        for label_path in self.dataset_root.glob('Part*/RibFrac*-label.nii.gz'):
            key = label_path.name[:-len('-label.nii.gz')]
            image_path = label_path.with_name(f'{key}-image.nii.gz')
            ret.append(
                RibFracDataPoint(
                    key=key,
                    images={'CT': image_path},
                    label=label_path,
                )
            )
        return ret
