from copy import copy

import pandas as pd

from mmmm.data.defs import Split
from ._base import DefaultImageLoaderMixin, DefaultMaskLoaderMixin, MultiClassDataPoint, Processor

class AMOS22Processor(DefaultImageLoaderMixin, DefaultMaskLoaderMixin, Processor):
    name = 'AMOS22'
    max_workers = 8

    def get_data_points(self):
        ret = []
        class_mapping_base = {
            1: 'spleen',
            2: 'right kidney',
            3: 'left kidney',
            4: 'gallbladder',
            5: 'esophagus',
            6: 'liver',
            7: 'stomach',
            8: 'aorta',
            9: 'inferior vena cava',
            10: 'pancreas',
            11: 'right adrenal gland',
            12: 'left adrenal gland',
            13: 'duodenum',
            14: 'bladder',
            # 15: 'prostate/uterus'
        }
        # uterus / prostate may also be a negative class for male / female
        class_mapping_male = copy(class_mapping_base)
        class_mapping_male[15] = 'prostate'
        class_mapping_male[16] = 'uterus'
        class_mapping_female = copy(class_mapping_base)
        class_mapping_female[15] = 'uterus'
        class_mapping_female[16] = 'prostate'
        class_mappings = {'M': class_mapping_male, 'F': class_mapping_female}
        meta = pd.read_csv(self.dataset_root / 'labeled_data_meta-fix.csv', index_col='amos_id')
        split_dict = {
            Split.TRAIN: [],
            Split.VAL: [],
        }
        for split_key, split in [
            ('Tr', Split.TRAIN),
            ('Va', Split.VAL),
        ]:
            for label_path in (self.dataset_root / 'amos22' / f'labels{split_key}').glob(f'*.nii.gz'):
                case = label_path.name[:-len('.nii.gz')]
                case_id = int(case.split('_')[1])
                modality = 'CT' if case_id <= 500 else 'MRI'
                key = case
                data_point = MultiClassDataPoint(
                    key=key,
                    images={modality: self.dataset_root / 'amos22' / f'images{split_key}' / f'{case}.nii.gz'},
                    label=label_path,
                    class_mapping=class_mappings[meta.loc[case_id, "Patient's Sex"]],
                )
                ret.append(data_point)
                split = Split.TRAIN if 'Tr' else Split.VAL
                split_dict[split].append(key)
        return ret
