import torch

from luolib.utils import get_cuda_device
from monai.data import MetaTensor

from mmmm.data.defs import ORIGIN_SEG_DATA_ROOT
from ._base import Default3DImageLoaderMixin, Default3DMaskLoaderMixin, MultiClassDataPoint, Processor

class PENGWINProcessor(Processor):
    @property
    def dataset_root(self):
        return ORIGIN_SEG_DATA_ROOT / 'PENGWIN'

    def load_masks(self, data_point: MultiClassDataPoint) -> tuple[MetaTensor, list[str]]:
        label: MetaTensor = self.mask_loader(data_point.label).to(device=get_cuda_device())
        masks: MetaTensor = label.new_empty((3, *label.shape[1:]), dtype=torch.bool)
        masks.affine = label.affine
        targets = ['sacrum', 'left hip bone', 'right hip bone']
        fragments = {}
        unique_labels = label.unique()
        for i, name in enumerate(targets):
            low, high = i * 10 + 1, (i + 1) * 10
            masks[i] = (low <= label) & (label <= high)
            fragments[name] = ((low <= unique_labels) & (unique_labels <= high)).sum().item()
        data_point.extra = {'fragments': fragments}
        return masks, targets

class PENGWINT1Processor(Default3DImageLoaderMixin, Default3DMaskLoaderMixin, PENGWINProcessor):
    name = 'PENGWIN-T1'
    orientation = 'SRA'

    def get_data_points(self):
        ret = []
        for image_path in self.dataset_root.glob('PENGWIN_CT_train_images_part*/*.mha'):
            case = image_path.stem
            ret.append(
                MultiClassDataPoint(
                    key=f'T1-{case}',
                    images={'CT': image_path},
                    label=self.dataset_root / f'PENGWIN_CT_train_labels/{case}.mha',
                    class_mapping={},
                ),
            )
        return ret
