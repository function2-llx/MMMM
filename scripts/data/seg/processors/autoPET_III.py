from ._base import DefaultImageLoaderMixin, DefaultMaskLoaderMixin, MultiLabelMultiFileDataPoint, Processor

class AutoPETIIIProcessor(DefaultImageLoaderMixin, DefaultMaskLoaderMixin, Processor):
    name = 'autoPET-III'
    orientation = 'SRA'
    bbox_ignore_targets = ['tumor']

    @property
    def dataset_root(self):
        return super().dataset_root / 'Autopet'

    def get_data_points(self):
        ret = []
        for mask_path in (self.dataset_root / 'labelsTr').glob('*.nii.gz'):
            key = mask_path.name[:-len('.nii.gz')]
            ret.append(
                MultiLabelMultiFileDataPoint(
                    key=key,
                    images={
                        'PET': self.dataset_root / 'imagesTr' / f'{key}_0000.nii.gz',
                        'CT': self.dataset_root / 'imagesTr' / f'{key}_0001.nii.gz',
                    },
                    masks=[('tumor', mask_path)],
                    complete_anomaly=True,
                ),
            )
        return ret
