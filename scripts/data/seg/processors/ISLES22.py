from ._base import Default3DMaskLoaderMixin, Default3DImageLoaderMixin, MultiLabelMultiFileDataPoint, Processor

class ISLES22Processor(Default3DImageLoaderMixin, Default3DMaskLoaderMixin, Processor):
    name = 'ISLES22'

    @property
    def dataset_root(self):
        return super().dataset_root / 'ISLES-2022'

    def get_data_points(self):
        ret = []
        mask_root = self.dataset_root / 'derivatives'
        for mask_path in mask_root.glob('*/*/*_msk.nii.gz'):
            key = mask_path.name[:-len('_msk.nii.gz')]
            img_dir = self.dataset_root / mask_path.parent.relative_to(mask_root)
            ret.append(
                MultiLabelMultiFileDataPoint(
                    key=key,
                    images={
                        modality: img_dir / dirname / f'{key}_{suffix}.nii.gz'
                        for modality, dirname, suffix in [
                            ('DW MRI', 'dwi', 'dwi'),
                            ('ADC MRI', 'dwi', 'adc'),
                            # FLAIR is not co-registered
                            # https://grand-challenge.org/forums/forum/ischemic-stroke-lesion-segmentation-challenge-672/topic/flair-versus-other-modalities-969/
                            # ('T2-FLAIR MRI', 'anat', 'FLAIR'),
                        ]
                    },
                    masks=[('stroke lesion', mask_path)]
                ),
            )
        return ret
