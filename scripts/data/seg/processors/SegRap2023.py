import torch

from monai.data import MetaTensor
from ._base import Default3DImageLoaderMixin, Default3DMaskLoaderMixin, MultiLabelMultiFileDataPoint, Processor

class SegRap2023Processor(Default3DImageLoaderMixin, Default3DMaskLoaderMixin, Processor):
    name = 'SegRap2023'
    orientation = 'SRA'

    def load_masks(self, data_point: MultiLabelMultiFileDataPoint) -> tuple[MetaTensor, list[str]]:
        masks, targets = super().load_masks(data_point)
        li = targets.index('left mandible')
        ri = targets.index('right mandible')
        mandible_mask = masks[li] | masks[ri]
        masks = torch.cat([masks[:li], masks[li + 1:ri], masks[ri + 1:], mandible_mask[None]])
        targets.remove('left mandible')
        targets.remove('right mandible')
        targets.append('mandible')
        return masks, targets

    def get_data_points(self):
        ret = []
        name_mapping = {
            'Brain': 'brain',
            'BrainStem': 'brainstem',
            'Chiasm': 'optic chiasm',
            'TemporalLobe_L': 'left temporal lobe',
            'TemporalLobe_R': 'right temporal lobe',
            'Hippocampus_L': 'left hippocampus',
            'Hippocampus_R': 'right hippocampus',
            'Eye_L': 'left eye',
            'Eye_R': 'right eye',
            'Lens_L': 'lens of left eye',
            'Lens_R': 'lens of right eye',
            'OpticNerve_L': 'left optic nerve',
            'OpticNerve_R': 'right optic nerve',
            'MiddleEar_L': 'left middle ear',
            'MiddleEar_R': 'right middle ear',
            'IAC_L': 'left internal auditory canal',
            'IAC_R': 'right internal auditory canal',
            'TympanicCavity_L': 'left tympanic cavity',
            'TympanicCavity_R': 'right tympanic cavity',
            'VestibulSemi_L': 'left semicircular canal',
            'VestibulSemi_R': 'right semicircular canal',
            'Cochlea_L': 'left cochlea',
            'Cochlea_R': 'right cochlea',
            'ETbone_L': 'left eustachian tube',
            'ETbone_R': 'right eustachian tube',
            'Pituitary': 'pituitary',
            'OralCavity': 'oral cavity',
            'Mandible_L': 'left mandible',
            'Mandible_R': 'right mandible',
            'Submandibular_L': 'left submandibular gland',
            'Submandibular_R': 'right submandibular gland',
            'Parotid_L': 'left parotid gland',
            'Parotid_R': 'right parotid gland',
            'Mastoid_L': 'left mastoid bone',
            'Mastoid_R': 'right mastoid bone',
            'TMjoint_L': 'left temporomandibular joint',
            'TMjoint_R': 'right temporomandibular joint',
            'SpinalCord': 'spinal cord',
            'Esophagus': 'esophagus',
            'Larynx': 'larynx',
            'Larynx_Glottic': 'glottis',
            'Larynx_Supraglot': 'supraglottis',
            'PharynxConst': 'pharynx',
            'Thyroid': 'thyroid',
            'Trachea': 'trachea',
        }
        for case_idx in range(120):
            key = f'segrap_{case_idx:04d}'
            ret.append(
                MultiLabelMultiFileDataPoint(
                    key=key,
                    images={
                        'CT': self.dataset_root / f'SegRap2023_Training_Set_120cases/{key}/image.nii.gz',
                        'contrast-enhanced CT': self.dataset_root / f'SegRap2023_Training_Set_120cases/{key}/image_contrast.nii.gz',
                    },
                    masks=[
                        (name, self.dataset_root / f'SegRap2023_Training_Set_120cases_Update_Labels/{key}/{mask_key}.nii.gz')
                        for mask_key, name in name_mapping.items()
                    ],
                ),
            )
        return ret
