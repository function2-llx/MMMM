from functools import cached_property
from pathlib import Path

import numpy as np
import torch

from luolib import transforms as lt
from luolib.datamodule import ExpDataModuleBase
from luolib.types import tuple3_t
from luolib.utils.misc import ensure_rgb
from monai import transforms as mt
from monai.networks import one_hot

class InputTransformD(mt.Transform):
    def __init__(self, num_fg_classes: int):
        self.num_fg_classes = num_fg_classes

    def __call__(self, data: dict):
        img, _ = ensure_rgb(data['img'].as_tensor())
        seg = one_hot(data['seg'].as_tensor(), self.num_fg_classes + 1, dim=0, dtype=torch.bool)
        seg = seg[1:]  # remove bg
        return {
            'case': data['case'],
            'img': img,
            'seg': seg,
        }

class DataModuleDebug(ExpDataModuleBase):
    def __init__(
        self,
        *,
        patch_size: tuple3_t[int],
        num_fg_classes: int = 15,
        val_num: int = 10,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.data_dir = Path('nnUNet-data/preprocessed/Dataset219_AMOS2022_postChallenge_task2/nnUNetPlans_3d_fullres')
        self.patch_size = patch_size
        self.val_num = val_num
        self.num_fg_classes = num_fg_classes
        self.class_names = ['spleen', 'right kidney', 'left kidney', 'gall bladder', 'esophagus', 'liver', 'stomach', 'arota', 'postcava', 'pancreas', 'right adrenal gland', 'left adrenal gland', 'duodenum', 'bladder', 'prostate | uterus']

    def train_transform(self):
        return mt.Compose(
            [
                lt.nnUNetLoaderD('case', self.data_dir, unravel_class_locations=True),
                mt.SpatialPadD(['img', 'seg'], self.patch_size),
                lt.OneOf(
                    [
                        mt.RandSpatialCropD(['img', 'seg'], self.patch_size, random_center=True, random_size=False),
                        mt.RandCropByLabelClassesD(
                            ['img', 'seg'],
                            'seg',
                            spatial_size=self.patch_size,
                            # who knows why they pop? https://github.com/Project-MONAI/MONAI/blob/1.3.0/monai/transforms/croppad/dictionary.py#L1153
                            indices_key='class_locations',
                            warn=False,
                        ),
                    ],
                    weights=(2, 1),
                ),
                InputTransformD(self.num_fg_classes),
            ],
        )

    def val_transform(self):
        return mt.Compose(
            [
                lt.nnUNetLoaderD('case', self.data_dir),
                InputTransformD(self.num_fg_classes),
            ],
        )

    @cached_property
    def all_data(self):
        ret = [
            {'case': case_file.stem}
            for case_file in self.data_dir.glob('*.npz')
        ]
        ret = sorted(ret, key=lambda x: x['case'])
        R = np.random.RandomState(42)
        R.shuffle(ret)
        assert len(ret) == 360
        ret = ret[:120]  # guess why
        return ret

    def train_data(self):
        return self.all_data[:-self.val_num]

    def val_data(self):
        return self.all_data[-self.val_num:]