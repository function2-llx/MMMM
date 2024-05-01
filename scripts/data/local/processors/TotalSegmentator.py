from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import torch
import zstandard as zstd

from monai.data import MetaTensor
from monai import transforms as mt

from mmmm.data.defs import ORIGIN_DATA_ROOT
from ._base import DefaultImageLoaderMixin, DefaultMaskLoaderMixin, MultiLabelMultiFileDataPoint, Processor

class TotalSegmentatorProcessor(DefaultImageLoaderMixin, DefaultMaskLoaderMixin, Processor):
    name = 'TotalSegmentator'
    max_workers = 8

    def mask_loader(self, path: Path) -> MetaTensor:
        with open(path, 'rb') as f:
            data_zst = f.read()
        data = zstd.decompress(data_zst)
        with tempfile.NamedTemporaryFile(suffix='.nii', dir='/dev/shm') as f:
            f.write(data)
            return super().mask_loader(Path(f.name))

    def get_orientation(self, images: MetaTensor):
        codes = ['RAS', 'ASR', 'SRA']
        diff = np.empty(len(codes))
        shape_diff = np.empty(len(codes), dtype=np.int32)
        dummy = MetaTensor(torch.empty(1, *images.shape[1:], device=images.device), images.affine)
        for i, code in enumerate(codes):
            orientation = mt.Orientation(code)
            dummy_t: MetaTensor = orientation(dummy)
            diff[i] = abs(dummy_t.pixdim[1] - dummy_t.pixdim[2])
            shape_diff[i] = abs(dummy_t.shape[2] - dummy_t.shape[3])
        if shape_diff.min().item() == 0 and shape_diff.max().item() != 0:
            # select plane with edges of equal lengths
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
        tax = pd.read_excel(ORIGIN_DATA_ROOT / 'target-tax.xlsx', sheet_name='anatomy')
        tax = tax[~tax['TS-v2 key'].isna()]
        tax.set_index('TS-v2 key', inplace=True)
        for case_dir in self.dataset_root.iterdir():
            if not case_dir.is_dir():
                continue
            data_point = MultiLabelMultiFileDataPoint(
                key=case_dir.name,
                images={'CT': case_dir / 'ct.nii.gz'},
                masks=[
                    (tax.loc[path.name[:-len('.nii.zst')], 'name'], path)
                    for path in (case_dir / 'segmentations-zstd').glob('*.nii.zst')
                ],
            )
            ret.append(data_point)
        return ret

    def process(self, *args, **kwargs):
        meta = pd.read_csv(self.dataset_root / 'meta.csv', sep=';')
        meta.to_csv(self.dataset_root / 'meta-comma.csv', index=False)
        super().process(*args, **kwargs)
