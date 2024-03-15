from copy import copy

import numpy as np
import pandas as pd

from monai.data import MetaTensor

from mmmm.data.defs import ORIGIN_SEG_DATA_ROOT

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
        class15_mapping = {'M': 'prostate', 'F': 'uterus'}
        for split in ['Tr', 'Va']:
            for label_path in (self.dataset_root / 'amos22' / f'labels{split}').glob(f'*{suffix}'):
                case = label_path.name[:-len(suffix)]
                case_id = int(case.split('_')[1])
                modality = 'CT' if case_id <= 500 else 'MRI'
                class_mapping = copy(mapping)
                class_mapping[15] = class15_mapping[meta.loc[case_id, "Patient's Sex"]]
                # uterus / prostate may also be a negative class for male / female
                data_point = MultiClassDataPoint(
                    case,
                    {modality: self.dataset_root / 'amos22' / f'images{split}' / f'{case}{suffix}'},
                    label_path,
                    class_mapping,
                )
                ret.append(data_point)
        return ret

class AMOS22DebugProcessor(AMOS22Processor):
    name = 'AMOS22-debug'
    mask_batch_size = 8

    @property
    def dataset_root(self):
        return ORIGIN_SEG_DATA_ROOT / 'AMOS22'

    def compute_resize(self, images: MetaTensor):
        shape = np.array(images.shape[1:])
        spacing = images.pixdim.numpy()
        # from nnU-Net
        new_spacing = np.array([2.0, 0.712890625, 0.712890625])
        new_shape = (shape * spacing / new_spacing).round().astype(np.int32)
        return new_spacing, new_shape
