import torch

from luolib.utils import get_cuda_device
from monai.data import MetaTensor

from mmmm.data.defs import ORIGIN_SEG_DATA_ROOT
from ._base import DefaultImageLoaderMixin, DefaultMaskLoaderMixin, MultiClassDataPoint, Processor

class PENGWINProcessor(DefaultImageLoaderMixin, DefaultMaskLoaderMixin, Processor):
    mask_dtype = torch.int32

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.targets = ['sacrum', 'left hip bone', 'right hip bone']

    @property
    def dataset_root(self):
        return ORIGIN_SEG_DATA_ROOT / 'PENGWIN'

    def load_annotations(self, data_point: MultiClassDataPoint, *args, **kwargs):
        label: MetaTensor = self.mask_loader(data_point.label).to(device=get_cuda_device())

        # masks: MetaTensor = label.new_empty((3, *label.shape[1:]), dtype=torch.bool)
        targets = []
        neg_targets = []
        masks = []
        for i, name in enumerate(self.targets):
            is_neg = True
            for j in range(i * 10 + 1, (i + 1) * 10):
                mask = (label >> j & 1).bool()
                # reduction for bitwise operation is not yet supported: https://github.com/pytorch/pytorch/issues/35641
                if mask.any():
                    is_neg = False
                    targets.append(name)
                    masks.append(mask)
            if is_neg:
                neg_targets.append(name)
        masks = torch.stack(masks)
        masks.affine = label.affine
        return self.targets, set(neg_targets), masks, None

class PENGWINT1Processor(PENGWINProcessor):
    name = 'PENGWIN-T1'
    orientation = 'SRA'

    @property
    def dataset_root(self):
        return super().dataset_root / 'Task-1'

    def get_data_points(self):
        ret = []
        for image_path in self.dataset_root.glob('PENGWIN_CT_train_images_part*/*.mha'):
            key = image_path.stem
            ret.append(
                MultiClassDataPoint(
                    key=key,
                    images={'CT': image_path},
                    label=self.dataset_root / f'PENGWIN_CT_train_labels/{key}.mha',
                    class_mapping={},
                ),
            )
        return ret

class PENGWINT2Processor(PENGWINProcessor):
    name = 'PENGWIN-T2'
    image_reader = 'pilreader'
    assert_gray_scale = True
    mask_reader = 'pilreader'

    @property
    def dataset_root(self):
        return super().dataset_root / 'Task-2'

    def get_data_points(self):
        ret = []
        for image_path in (self.dataset_root / 'train').glob('input/images/x-ray/*.tif'):
            key = image_path.stem
            label_path = image_path.parents[3] / f'output/images/x-ray/{key}.tif'
            ret.append(
                MultiClassDataPoint(
                    key=key,
                    images={'X-ray': image_path},
                    label=label_path,
                    class_mapping={},
                ),
            )
        return ret
