from ._base import Default3DImageLoaderMixin, Default3DMaskLoaderMixin, MultiLabelMultiFileDataPoint, Processor

class ATLASProcessor(Default3DImageLoaderMixin, Default3DMaskLoaderMixin, Processor):
    name = 'ATLAS'
    orientation = 'SRA'

    @property
    def dataset_root(self):
        return super().dataset_root / 'ATLAS_2'

    def get_data_points(self):
        ret = []
        for mask_path in (self.dataset_root / 'Training').glob('*/*/*/anat/*_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz'):
            key = mask_path.name[:-len('_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz')]
            ret.append(
                MultiLabelMultiFileDataPoint(
                    key=key,
                    images={'T1 MRI': mask_path.parent / f'{key}_space-MNI152NLin2009aSym_T1w.nii.gz'},
                    masks=[('stroke lesion', mask_path)]
                ),
            )
        return ret
