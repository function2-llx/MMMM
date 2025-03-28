from ._base import DefaultImageLoaderMixin, DefaultMaskLoaderMixin, MultiClassDataPoint, Processor

class CT_ORGProcessor(DefaultImageLoaderMixin, DefaultMaskLoaderMixin, Processor):
    name = 'CT-ORG'
    orientation = 'SRA'
    semantic_targets = {'kidney', 'lung'}

    def get_data_points(self):
        ret = []
        for i in range(140):
            class_mapping = {
                1: 'liver',
                2: 'bladder',
                4: 'kidney',
            }
            if i <= 20:
                # the test set is manually annotated
                class_mapping[3] = 'lung'
            ret.append(
                MultiClassDataPoint(
                    key=str(i),
                    images={'CT': self.dataset_root / f'volume-{i}.nii.gz'},
                    class_mapping=class_mapping,
                    label=self.dataset_root / f'labels-{i}.nii.gz',
                ),
            )
        return ret
