from ._base import DefaultImageLoaderMixin, DefaultMaskLoaderMixin, MultiLabelMultiFileDataPoint, Processor

class PARSE2022Processor(DefaultImageLoaderMixin, DefaultMaskLoaderMixin, Processor):
    name = 'PARSE2022'
    orientation = 'SRA'

    def get_data_points(self):
        ret = []
        for case_dir in (self.dataset_root / 'train').iterdir():
            key = case_dir.name
            ret.append(
                MultiLabelMultiFileDataPoint(
                    key=key,
                    images={'CT': case_dir / 'image' / f'{key}.nii.gz'},
                    masks=[('pulmonary artery', case_dir / 'label' / f'{key}.nii.gz')],
                ),
            )
        return ret
