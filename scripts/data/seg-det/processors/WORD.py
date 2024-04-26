from copy import copy

import numpy as np
import pandas as pd

from monai.data import MetaTensor

from mmmm.data.defs import ORIGIN_SEG_DATA_ROOT

from ._base import DefaultImageLoaderMixin, DefaultMaskLoaderMixin, Processor, MultiClassDataPoint

class WORDProcessor(DefaultImageLoaderMixin, DefaultMaskLoaderMixin, Processor):
    name = 'WORD'
    orientation = 'SRA'

    @property
    def dataset_root(self):
        return super().dataset_root / 'WORD-V0.1.0'

    def get_data_points(self):
        ret = []
        class_mapping = {
            1: 'liver',
            2: 'spleen',
            3: 'left kidney',
            4: 'right kidney',
            5: 'stomach',
            6: 'gallbladder',
            7: 'esophagus',
            8: 'pancreas',
            9: 'duodenum',
            10: 'colon',
            11: 'intestines',
            12: 'adrenal gland',
            13: 'rectum',
            14: 'bladder',
            15: 'head of left femur',
            16: 'head of right femur',
        }
        for label_dir in ['labelsTr', 'labelsVal', 'addition_validation_from_LiTS/lablesTs']:
            label_dir = self.dataset_root / label_dir
            for label_path in label_dir.glob('*.nii.gz'):
                key = label_path.name[:-len('.nii.gz')]
                if key.endswith('_word_label'):
                    key = key[:-len('_word_label')]
                    image_path = ORIGIN_SEG_DATA_ROOT / f'MSD/Task03_Liver/imagesTr/{key}.nii.gz'
                else:
                    image_path = label_dir.parent / label_dir.name.replace('labels', 'images') / f'{key}.nii.gz'
                ret.append(
                    MultiClassDataPoint(
                        key=key,
                        images={'CT': image_path},
                        label=label_path,
                        class_mapping=class_mapping,
                    ),
                )
        return ret
