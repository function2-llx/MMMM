from pathlib import Path

import einops
import numpy as np

from monai import transforms as mt
from monai.data import MetaTensor

from ._base import Default3DImageLoaderMixin, MultiClassDataPoint, Processor

class CHAOSProcessor(Default3DImageLoaderMixin, Processor):
    name = 'CHAOS'
    class_mapping = {
        63: 'liver',
        126: 'right kidney',
        189: 'left kidney',
        252: 'spleen',
    }

    def mask_loader(self, mask_dir: Path):
        loader = mt.LoadImage(dtype=np.uint8)
        mask = einops.rearrange(
            [loader(path) for path in sorted(mask_dir.iterdir())],
            'd h w -> 1 h w d',
        )
        return mask.flip(dims=(1, 2))

    def orient(self, images: MetaTensor, masks: MetaTensor) -> tuple[MetaTensor, MetaTensor]:
        masks.affine = images.affine
        return super().orient(images, masks)

    def _prepare(self, key: str, modality: str, image_dir: Path, mask_dir: Path) -> MultiClassDataPoint:
        return MultiClassDataPoint(
            key=key,
            images={modality: image_dir},
            complete_anomaly=True,
            label=mask_dir,
            class_mapping=self.class_mapping,
        )

    def get_data_points(self):
        # https://www.dropbox.com/s/b8ka7tcxm45mlq4/CHAOS_Submission_Manual_new.pdf
        ret = []
        train_dir = self.dataset_root / 'Train_Sets'
        for modality in ['CT', 'MR']:
            for case_dir in (train_dir / modality).iterdir():
                if modality == 'CT':
                    ret.append(self._prepare(f'CT-{case_dir.name}', 'CT', case_dir / 'DICOM_anon', case_dir / 'Ground'))
                else:
                    ret.extend([
                        self._prepare(
                            f'MR-{case_dir.name}-T1DUAL-in',
                            'T1-DUAL (in-phase) MRI',
                            case_dir / 'T1DUAL' / 'DICOM_anon' / 'InPhase',
                            case_dir / 'T1DUAL' / 'Ground',
                        ),
                        self._prepare(
                            f'MR-{case_dir.name}-T2SPIR',
                            'T2-SPIR MRI',
                            case_dir / 'T2SPIR' / 'DICOM_anon',
                            case_dir / 'T2SPIR' / 'Ground',
                        ),
                    ])
        return ret
