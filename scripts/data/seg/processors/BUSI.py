from pathlib import Path

import einops
import torch

from monai import transforms as mt
from monai.data import MetaTensor

from ._base import DataPoint, INF_SPACING, MultiLabelMultiFileDataPoint, NaturalImageLoaderMixin, Processor

class BUSIProcessor(NaturalImageLoaderMixin, Processor):
    name = 'BUSI'

    def mask_loader(self, path: Path):
        loader = mt.LoadImage(dtype=torch.bool, ensure_channel_first=True)
        m: MetaTensor = loader(path)
        m = m[:, None]
        m.affine[0, 0] = INF_SPACING
        return m

    def load_masks(self, data_point: DataPoint) -> tuple[torch.BoolTensor, list[str]]:
        masks, targets = super().load_masks(data_point)
        return einops.reduce(masks, 'c 1 h w -> 1 1 h w', 'any'), ['breast cancer']

    def get_data_points(self) -> list[DataPoint]:
        ret = []
        for grade in ['normal', 'benign', 'malignant']:
            grade_dir = self.dataset_root / 'Dataset_BUSI_with_GT' / grade
            for image_path in grade_dir.glob(f'{grade} (*).png'):
                key = image_path.stem
                ret.append(
                    MultiLabelMultiFileDataPoint(
                        key=key,
                        images={'ultrasound': image_path},
                        complete_anomaly=True,
                        masks=[
                            ('breast cancer', mask_path)
                            for mask_path in grade_dir.glob(f'{key}_mask*.png')
                        ],
                        extra={'grade': grade},
                    ),
                )
        return ret
