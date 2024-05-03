from monai.data import MetaTensor

from ._base import DefaultImageLoaderMixin, DefaultMaskLoaderMixin, MultiClassDataPoint, Processor, SegDataPoint

# affine for these keys are manually checked
_checked_keys = {'48', '49', '50', '51', '52'}

class LiTSProcessor(DefaultImageLoaderMixin, DefaultMaskLoaderMixin, Processor):
    name = 'LiTS'
    orientation = 'SRA'
    semantic_targets = {'liver tumor'}

    def load_masks(self, data_point: SegDataPoint, images: MetaTensor):
        targets, masks = super().load_masks(data_point, images)
        if data_point.key in _checked_keys:
            masks.affine = images.affine
        return targets, masks

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
