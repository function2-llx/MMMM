import torch

from ._base import DefaultImageLoaderMixin, DefaultMaskLoaderMixin, MultiLabelMultiFileDataPoint, Processor

class SEGA2022Processor(DefaultImageLoaderMixin, DefaultMaskLoaderMixin, Processor):
    name = 'SEG.A.2023'
    orientation = 'SRA'
    image_reader = 'nrrdreader'
    mask_reader = 'nrrdreader'

    def get_data_points(self):
        ret = []
        for mask_path in self.dataset_root.glob('*/*/*.seg.nrrd'):
            case_dir = mask_path.parent
            key = case_dir.name
            ret.append(
                MultiLabelMultiFileDataPoint(
                    key=key,
                    images={'CT angiography': case_dir / f'{key}.nrrd'},
                    masks=[('aortic vessel tree', mask_path)],
                ),
            )
        return ret
