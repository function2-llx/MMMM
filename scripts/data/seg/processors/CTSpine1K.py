from pathlib import Path

import cytoolz

from mmmm.data.defs import ORIGIN_SEG_DATA_ROOT
from ._base import DefaultImageLoaderMixin, DefaultMaskLoaderMixin, MultiClassDataPoint, Processor
from .VerSe import VerSeProcessor

class CTSpine1KProcessor(DefaultImageLoaderMixin, DefaultMaskLoaderMixin, Processor):
    name = 'CTSpine1K'
    # https://nipy.org/nibabel/nifti_images.html#the-fall-back-header-affine
    image_reader = 'itkreader'
    mask_reader = 'itkreader'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.path_mapping = {
            path.parents[-2].name: path
            for path in map(Path, (self.dataset_root / 'Path.csv').read_text().strip().splitlines())
        }

    def get_image_info(self, origin_key: str) -> tuple[Path, str]:
        if origin_key.startswith('HN'):
            dataset = 'HNSCC-3DCT-RT'
            path = ORIGIN_SEG_DATA_ROOT / 'HNSCC-3DCT-RT/download/HNSCC-3DCT-RT' / self.path_mapping[origin_key]
        elif origin_key.startswith('liver'):
            dataset = 'MSD_T10'
            path = cytoolz.first((ORIGIN_SEG_DATA_ROOT / 'MSD/Task03_Liver').glob(f'images*/{origin_key}.nii.gz'))
        elif origin_key.startswith('volume-covid19'):
            dataset = 'COVID-19'
            assert origin_key.endswith('_ct')
            origin_key = origin_key[:-len('_ct')]
            path = cytoolz.first((ORIGIN_SEG_DATA_ROOT / f'CT-COVID-19').glob(f'*/*/{origin_key}.nii.gz'))
        else:
            dataset = 'COLONOG'
            path = ORIGIN_SEG_DATA_ROOT / 'CT-COLONOGRAPHY/download/CT COLONOGRAPHY' / self.path_mapping[origin_key]
        return path, f'{dataset}-{origin_key}'

    def get_data_points(self):
        ret = []
        for label_path in self.dataset_root.glob(f'CTSpine1K/*/gt/*_seg.nii.gz'):
            origin_key = label_path.name[:-len('_seg.nii.gz')]
            img_path, key = self.get_image_info(origin_key)
            ret.append(
                MultiClassDataPoint(
                    key=key,
                    images={'CT': img_path},
                    label=label_path,
                    class_mapping=VerSeProcessor.class_mapping,
                ),
            )
        return ret
