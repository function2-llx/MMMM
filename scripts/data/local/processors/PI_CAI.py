import cytoolz
import torch

from ._base import DefaultImageLoaderMixin, DefaultMaskLoaderMixin, MultiLabelMultiFileDataPoint, Processor

class PI_CAIProcessor(DefaultImageLoaderMixin, DefaultMaskLoaderMixin, Processor):
    name = 'PI-CAI'
    orientation = 'SRA'
    semantic_targets = {'prostate cancer'}

    def _ensure_binary_mask(self, mask: torch.Tensor):
        assert mask.shape[0] == 1
        mask[mask >= 1] = 1
        return super()._ensure_binary_mask(mask)

    def get_data_points(self):
        ret = []
        for label_path in (self.dataset_root / 'picai_labels/csPCa_lesion_delineations/human_expert/resampled').glob('*.nii.gz'):
            key = label_path.name[:-len('.nii.gz')]
            patient_id, study_id = key.split('_')
            patient_dir = cytoolz.first(self.dataset_root.glob(f'picai_public_images_fold*/{patient_id}'))
            ret.append(
                MultiLabelMultiFileDataPoint(
                    key=key,
                    images={'T2 MRI': patient_dir / f'{key}_t2w.mha'},
                    masks=[('prostate cancer', label_path)],
                ),
            )
        return ret
