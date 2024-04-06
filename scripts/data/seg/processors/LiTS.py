from monai.data import MetaTensor
from ._base import Default3DImageLoaderMixin, Default3DMaskLoaderMixin, MultiClassDataPoint, Processor

class LiTSProcessor(Default3DImageLoaderMixin, Default3DMaskLoaderMixin, Processor):
    name = 'LiTS'
    orientation = 'SRA'

    def orient(self, images: MetaTensor, masks: MetaTensor) -> tuple[MetaTensor, MetaTensor]:
        if self.key in ['48', '49', '50', '51', '52']:
            # manually checked
            masks.affine = images.affine
        return super().orient(images, masks)

    def get_data_points(self):
        ret = []
        class_mapping = {
            1: 'liver',
            2: 'liver tumor',
        }
        for label_path in self.dataset_root.glob('Training Batch */segmentation-*.nii'):
            key = label_path.stem.split('-')[1]
            ret.append(
                MultiClassDataPoint(
                    key=key,
                    images={'CT': label_path.with_stem(f'volume-{key}')},
                    label=label_path,
                    class_mapping=class_mapping,
                ),
            )
        return ret
