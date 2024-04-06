from ._base import Default3DImageLoaderMixin, Default3DMaskLoaderMixin, MultiClassDataPoint, Processor

class MRSpineSegProcessor(Default3DImageLoaderMixin, Default3DMaskLoaderMixin, Processor):
    name = 'MRSpineSeg'

    def get_data_points(self):
        class_mapping = {
            1: 'sacrum',
            2: 'L5 vertebra',
            3: 'L4 vertebra',
            4: 'L3 vertebra',
            5: 'L2 vertebra',
            6: 'L1 vertebra',
            7: 'T12 vertebra',
            8: 'T11 vertebra',
            9: 'T10 vertebra',
            10: 'T9 vertebra',
            11: 'L5-S1 intervertebral disc',
            12: 'L4-L5 intervertebral disc',
            13: 'L3-L4 intervertebral disc',
            14: 'L2-L3 intervertebral disc',
            15: 'L1-L2 intervertebral disc',
            16: 'T12-L1 intervertebral disc',
            17: 'T11-T12 intervertebral disc',
            18: 'T10-T11 intervertebral disc',
            19: 'T9-T10 intervertebral disc',
        }
        ret = []
        for label_path in self.dataset_root.glob(f'train/Mask/mask_case*.nii.gz'):
            key = label_path.name[len('mask_'):-len('.nii.gz')]
            ret.append(
                MultiClassDataPoint(
                    key=key,
                    images={'T2 MRI': label_path.parents[1] / 'MR' / f'{key.capitalize()}.nii.gz'},
                    label=label_path,
                    class_mapping=class_mapping,
                )
            )
        return ret
