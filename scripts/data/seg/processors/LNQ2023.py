from ._base import Default3DImageLoaderMixin, Default3DMaskLoaderMixin, MultiLabelMultiFileDataPoint, Processor

class LNQ2023Processor(Default3DImageLoaderMixin, Default3DMaskLoaderMixin, Processor):
    name = 'LNQ2023'
    image_reader = 'nrrdreader'
    mask_reader = 'nrrdreader'
    orientation = 'SRA'

    def get_data_points(self):
        ret = []
        for mask_path in self.dataset_root.glob('train/lnq2023-train-*-seg.nrrd'):
            key = mask_path.stem[:-len('-seg')]
            ret.append(
                MultiLabelMultiFileDataPoint(
                    key=key,
                    images={'CT': mask_path.with_stem(f'{key}-ct')},
                    masks=[('mediastinal lymph node', mask_path)],
                ),
            )
        return ret
