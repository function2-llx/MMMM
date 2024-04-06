from ._base import Default3DMaskLoaderMixin, Default3DImageLoaderMixin, MultiLabelMultiFileDataPoint, Processor

class ATM22Processor(Default3DImageLoaderMixin, Default3DMaskLoaderMixin, Processor):
    name = 'ATM22'
    orientation = 'SRA'

    def get_data_points(self):
        ret = []
        suffix = '.nii.gz'
        # NOTE: the website hosting EXACT'09 data seems to be down
        for image_path in self.dataset_root.glob(f'TrainBatch*/imagesTr/*{suffix}'):
            case = image_path.name[:-len(suffix)]
            if case == 'ATM_164_0000':
                # exclude this case following the official announcement
                continue
            label_path = image_path.parent.parent / 'labelsTr' / image_path.name
            data_point = MultiLabelMultiFileDataPoint(
                key=case,
                images={'CT': image_path},
                masks=[('airway', label_path)],
            )
            ret.append(data_point)
        return ret
