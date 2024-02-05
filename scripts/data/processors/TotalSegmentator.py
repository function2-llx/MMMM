import numpy as np
import pandas as pd
import torch

from monai.data import MetaTensor
from monai import transforms as mt

from mmmm.data.defs import ORIGIN_DATA_ROOT

from .base import Default3DImageLoaderMixin, Binary3DMaskLoaderMixin, Processor, MultiLabelMultiFileDataPoint

class TotalSegmentatorProcessor(Default3DImageLoaderMixin, Binary3DMaskLoaderMixin, Processor):
    name = 'TotalSegmentator'
    max_workers = 8

    def get_orientation(self, images: MetaTensor):
        codes = ['RAS', 'ASR', 'SRA']
        diff = np.empty(len(codes))
        shape_diff = np.empty(len(codes), dtype=np.int32)
        dummy = MetaTensor(torch.empty(1, *images.shape[1:]), images.affine)
        for i, code in enumerate(codes):
            orientation = mt.Orientation(code)
            dummy_t: MetaTensor = orientation(dummy)
            diff[i] = abs(dummy_t.pixdim[1] - dummy_t.pixdim[2])
            shape_diff[i] = abs(dummy_t.shape[2] - dummy_t.shape[3])
        if shape_diff.min().item() == 0 and shape_diff.max().item() != 0:
            i = shape_diff.argmin().item()
            if diff[i] > diff.min():
                self.logger.info(f'{self.key}: min spacing diff not match isotropic in-plane shape')
            orientation = codes[i]
        elif diff.min() != diff.max():
            orientation = codes[diff.argmin()]
            self.logger.info(f'{self.key}: use min spacing diff')
        else:
            orientation = 'SRA'
            self.logger.info(f'{self.key}: fall back to {orientation}')
        return orientation

    @property
    def dataset_root(self):
        return super().dataset_root / 'Totalsegmentator_dataset_v201'

    def get_data_points(self):
        ret = []
        suffix = '.nii.gz'
        tax = pd.read_excel(ORIGIN_DATA_ROOT / 'seg-tax.xlsx')
        tax = tax[~tax['TS-v2 name'].isna()]
        tax.set_index('TS-v2 name', inplace=True)
        for case_dir in self.dataset_root.iterdir():
            if not case_dir.is_dir():
                continue
            data_point = MultiLabelMultiFileDataPoint(
                case_dir.name,
                {'CT': case_dir / 'ct.nii.gz'},
                {
                    tax.loc[path.name[:-len(suffix)], 'name']: path
                    for path in (case_dir / 'segmentations').iterdir()
                },
            )
            ret.append(data_point)
        return ret

    def process(self):
        meta = pd.read_csv(self.dataset_root / 'meta.csv', sep=';')
        meta.to_csv(self.dataset_root / 'meta-comma.csv', index=False)
        super().process()
