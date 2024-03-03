from copy import copy

import pandas as pd

from .base import Default3DImageLoaderMixin, MultiClass3DMaskLoaderMixin, Processor, MultiClassDataPoint

class AMOS22Processor(Default3DImageLoaderMixin, MultiClass3DMaskLoaderMixin, Processor):
    name = 'AMOS22'
    max_workers = 8

    def get_data_points(self):
        ret = []
        suffix = '.nii.gz'
        mapping = {
            "1": "spleen",
            "2": "right kidney",
            "3": "left kidney",
            "4": "gallbladder",
            "5": "esophagus",
            "6": "liver",
            "7": "stomach",
            "8": "aorta",
            "9": "inferior vena cava",
            "10": "pancreas",
            "11": "right adrenal gland",
            "12": "left adrenal gland",
            "13": "duodenum",
            "14": "bladder",
            # "15": "prostate/uterus"
        }
        mapping = {int(k): v for k, v in mapping.items()}
        meta = pd.read_csv(self.dataset_root / 'labeled_data_meta-fix.csv', index_col='amos_id')
        for split in ['Tr', 'Va']:
            for label_path in (self.dataset_root / 'amos22' / f'labels{split}').glob(f'*{suffix}'):
                case = label_path.name[:-len(suffix)]
                case_id = int(case.split('_')[1])
                modality = 'CT' if case_id <= 500 else 'MRI'
                class_mapping = copy(mapping)
                class_mapping[15] = 'prostate' if meta.loc[case_id, "Patient's Sex"] == 'M' else 'uterus'
                # uterus / prostate may also be a negative class for male / female
                data_point = MultiClassDataPoint(
                    case,
                    {modality: self.dataset_root / 'amos22' / f'images{split}' / f'{case}{suffix}'},
                    label_path,
                    class_mapping,
                )
                ret.append(data_point)
        return ret
