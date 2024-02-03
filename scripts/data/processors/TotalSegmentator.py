import pandas as pd

from mmmm.data.defs import ORIGIN_DATA_ROOT
from .base import Default3DLoaderMixin, MultiLabelMultiFileDataPoint, Processor

class TotalSegmentatorProcessor(Default3DLoaderMixin, Processor):
    name = 'TotalSegmentator'
    orientation = 'SRA'

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
