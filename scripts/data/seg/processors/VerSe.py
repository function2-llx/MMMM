from ._base import Default3DImageLoaderMixin, Default3DMaskLoaderMixin, Processor, MultiClassDataPoint

class VerSeProcessor(Default3DImageLoaderMixin, Default3DMaskLoaderMixin, Processor):
    name = 'VerSe'
    orientation = 'SRA'

    def get_data_points(self):
        class_mapping = {
            **{
                i: f'C{i} vertebra'
                for i in range(1, 8)
            },
            **{
                i: f'T{i - 7} vertebra'
                for i in range(8, 20)
            },
            **{
                i: 'L1 vertebra'
                for i in range(20, 26)
            },
        }
        ret = []
        for label_path in self.dataset_root.glob('dataset-*/derivatives/*/*_seg-vert_msk.nii.gz'):
            case = label_path.parent.name
            key = label_path.name[:-len('_seg-vert_msk.nii.gz')]
            ret.append(
                MultiClassDataPoint(
                    key=key,
                    images={'CT': label_path.parents[2] / 'rawdata' / case / f'{key}_ct.nii.gz'},
                    label=label_path,
                    class_mapping=class_mapping,
                ),
            )

        return ret
