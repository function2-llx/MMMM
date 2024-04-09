import torch

from ._base import DefaultMaskLoaderMixin, DefaultImageLoaderMixin, MultiLabelMultiFileDataPoint, Processor

class RibFracProcessor(DefaultImageLoaderMixin, DefaultMaskLoaderMixin, Processor):
    name = 'RibFrac'
    orientation = 'SRA'
    bbox_ignore_targets = {'rib fracture'}

    def _ensure_binary_mask(self, mask: torch.Tensor):
        mask[mask != 0] = 1
        return super()._ensure_binary_mask(mask)

    def get_data_points(self):
        ret = []
        for label_path in self.dataset_root.glob('Part*/RibFrac*-label.nii.gz'):
            key = label_path.name[:-len('-label.nii.gz')]
            image_path = label_path.with_name(f'{key}-image.nii.gz')
            ret.append(
                MultiLabelMultiFileDataPoint(
                    key=key,
                    images={'CT': image_path},
                    masks=[('rib fracture', label_path)],
                )
            )
        return ret
