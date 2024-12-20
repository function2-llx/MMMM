import numpy as np
import torch

from mmmm.data.defs import Split
from ._base import DataPoint, DefaultImageLoaderMixin, DefaultMaskLoaderMixin, MultiClassDataPoint, Processor

class BraTS2023SegmentationProcessor(DefaultImageLoaderMixin, DefaultMaskLoaderMixin, Processor):
    orientation = 'SRA'

    @property
    def output_name(self):
        return f"BraTS2023-{self.name.split('-')[-1]}"

    def load_masks(self, data_point: MultiClassDataPoint, *args, **kwargs):
        targets, masks = super().load_masks(data_point, *args, **kwargs)
        mask_map = {target: mask for mask, target in zip(masks, targets)}
        masks = torch.stack([
            mask_map['necrotic tumor core'],
            mask_map['peritumoral edema'],
            mask_map['necrotic tumor core'] | mask_map['enhancing tumor'],
        ])
        targets = ['necrotic tumor core', 'peritumoral edema', 'glioma']
        return targets, masks

    def get_data_points(self):
        modality_map = {
            't1c': 'T1CE MRI',
            't1n': 'T1 MRI',
            't2f': 'T2-FLAIR MRI',
            't2w': 'T2 MRI',
        }
        class_mapping = {
            1: 'necrotic tumor core',
            2: 'peritumoral edema',
            3: 'enhancing tumor',
        }
        ret = []
        for subject_dir in self.dataset_root.glob('*/*'):
            if not (subject_dir.is_dir() and subject_dir.name.startswith(self.dataset_root.name)):
                continue
            key = subject_dir.name
            label_path = subject_dir / f'{key}-seg.nii.gz'
            if not label_path.exists():
                continue
            ret.append(
                MultiClassDataPoint(
                    key=key,
                    images={
                        modality: subject_dir / f'{key}-{modality_suffix}.nii.gz'
                        for modality_suffix, modality in modality_map.items()
                    },
                    label=label_path,
                    class_mapping=class_mapping,
                ),
            )
        return ret, None

class BraTS2023GLIProcessor(BraTS2023SegmentationProcessor):
    name = 'BraTS2023/BraTS-GLI'
    num_val = 50

    def get_data_points(self):
        data_points, _ = super().get_data_points()
        R = np.random.RandomState(233)
        keys = [data_point.key for data_point in data_points]
        R.shuffle(keys)
        return data_points, {
            Split.TRAIN: keys[:-self.num_val],
            Split.VAL: keys[-self.num_val:],
        }

class BraTS2023MENProcessor(BraTS2023SegmentationProcessor):
    name = 'BraTS2023/BraTS-MEN'

class BraTS2023METProcessor(BraTS2023SegmentationProcessor):
    name = 'BraTS2023/BraTS-MET'

class BraTS2023PEDProcessor(BraTS2023SegmentationProcessor):
    name = 'BraTS2023/BraTS-PED'

class BraTS2023SSAProcessor(BraTS2023SegmentationProcessor):
    name = 'BraTS2023/BraTS-SSA'
