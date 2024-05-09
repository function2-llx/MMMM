from pathlib import Path

import einops
import torch

from monai.data import MetaTensor
from ._base import DataPoint, DefaultMaskLoaderMixin, MultiLabelMultiFileDataPoint, NaturalImageLoaderMixin, Processor

class BUSIProcessor(NaturalImageLoaderMixin, DefaultMaskLoaderMixin, Processor):
    name = 'BUSI'
    mask_dtype = torch.bool
    assert_gray_scale = False

    def mask_loader(self, path: Path) -> MetaTensor:
        mask = super().mask_loader(path)
        if mask.shape[0] in (2, 4):
            # handle the alpha channel
            assert mask[-1].all() or not mask[-1].any()
            mask = mask[:-1]
        assert mask.shape[0] in (1, 3)
        for i in range(1, mask.shape[0]):
            assert (mask[0] == mask[i]).all()
        return mask[0:1]

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
