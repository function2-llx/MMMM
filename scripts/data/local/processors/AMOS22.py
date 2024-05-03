from copy import copy

import pandas as pd

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
        for split in ['Tr', 'Va']:
            for label_path in (self.dataset_root / 'amos22' / f'labels{split}').glob(f'*.nii.gz'):
                case = label_path.name[:-len('.nii.gz')]
                case_id = int(case.split('_')[1])
                modality = 'CT' if case_id <= 500 else 'MRI'
                data_point = MultiClassDataPoint(
                    key=case,
                    images={modality: self.dataset_root / 'amos22' / f'images{split}' / f'{case}.nii.gz'},
                    label=label_path,
                    class_mapping=class_mappings[meta.loc[case_id, "Patient's Sex"]],
                )
                ret.append(data_point)
        return ret

# class AMOS22DebugProcessor(AMOS22Processor):
#     name = 'AMOS22-debug'
#     mask_batch_size = 8
#
#     @property
#     def dataset_root(self):
#         return ORIGIN_SEG_DATA_ROOT / 'AMOS22'
#
#     def compute_resize(self, images: MetaTensor):
#         shape = np.array(images.shape[1:])
#         spacing = images.pixdim.numpy()
#         # from nnU-Net
#         new_spacing = np.array([2.0, 0.712890625, 0.712890625])
#         new_shape = (shape * spacing / new_spacing).round().astype(np.int32)
#         return new_spacing, new_shape
