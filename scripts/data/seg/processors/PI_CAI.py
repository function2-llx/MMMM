import torch

from ._base import Default3DImageLoaderMixin, Default3DMaskLoaderMixin, MultiLabelMultiFileDataPoint, Processor

class PI_CAI2022Processor(Default3DImageLoaderMixin, Default3DMaskLoaderMixin, Processor):
    name = 'PI-CAI'
    orientation = 'SRA'

    def _check_binary_mask(self, masks: torch.Tensor):
        assert masks.shape[0] == 1
        masks[masks >= 1] = 1
        super()._check_binary_mask(masks)

    def get_data_points(self):
        ret = []
        for label_path in (self.dataset_root / 'picai_labels/csPCa_lesion_delineations/human_expert/resampled').glob('*.nii.gz'):
            key = label_path.name[:-len('.nii.gz')]
            patient_id, study_id = key.split('_')
            ret.append(
                MultiLabelMultiFileDataPoint(
                    key=key,
                    images={'T2 MRI': self.dataset_root / f'public_images/{patient_id}/{key}_t2w.mha'},
                    masks=[('prostate cancer', label_path)],
                ),
            )
        return ret
