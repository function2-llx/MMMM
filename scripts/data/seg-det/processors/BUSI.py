import einops
import torch

from ._base import DataPoint, DefaultMaskLoaderMixin, MultiLabelMultiFileDataPoint, NaturalImageLoaderMixin, Processor

class BUSIProcessor(NaturalImageLoaderMixin, DefaultMaskLoaderMixin, Processor):
    name = 'BUSI'
    mask_dtype = torch.bool

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
