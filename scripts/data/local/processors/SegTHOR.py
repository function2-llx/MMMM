from ._base import DefaultImageLoaderMixin, DefaultMaskLoaderMixin, MultiClassDataPoint, Processor

class SegTHORProcessor(DefaultImageLoaderMixin, DefaultMaskLoaderMixin, Processor):
    name = 'SegTHOR'
    orientation = 'SRA'

    def get_data_points(self):
        # https://github.com/chestnut111/SegTHOR2019
        class_mapping = {
            1: 'esophagus',
            2: 'heart',
            3: 'trachea',
            4: 'aorta',
        }
        ret = []
        for case_dir in (self.dataset_root / 'train').glob('Patient_*/'):
            key = case_dir.name
            ret.append(
                MultiClassDataPoint(
                    key=key,
                    images={'CT': case_dir / f'{key}.nii.gz'},
                    label=case_dir / 'GT.nii.gz',
                    class_mapping=class_mapping,
                ),
            )
        return ret
