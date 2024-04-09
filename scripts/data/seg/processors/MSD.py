from pathlib import Path

import torch

from luolib.utils import get_cuda_device
from monai.data import MetaTensor
from ._base import DataPoint, DefaultImageLoaderMixin, DefaultMaskLoaderMixin, Processor, MultiClassDataPoint

class MSDProcessor(DefaultImageLoaderMixin, DefaultMaskLoaderMixin, Processor):
    class_mapping: dict[int, str]

    @property
    def output_name(self):
        return self.name.replace('/', '-')

    def get_modalities(self) -> list[str]:
        import json
        meta = json.loads(Path(self.dataset_root / 'dataset.json').read_bytes())
        modality = meta['modality']
        return [modality[str(i)] for i in range(len(modality))]

    def load_images(self, data_point: DataPoint) -> tuple[list[str], MetaTensor, bool]:
        modalities = self.get_modalities()
        images, _ = self.image_loader(data_point.images['_all'])
        return modalities, images.to(device=get_cuda_device()), False

    def get_data_points(self) -> list[DataPoint]:
        ret = []
        for label_path in self.dataset_root.glob('labelsTr/*.nii.gz'):
            key = label_path.name[:-len('.nii.gz')]
            ret.append(
                MultiClassDataPoint(
                    key=key,
                    images={'_all': label_path.parents[1] / 'imagesTr' / f'{key}.nii.gz'},
                    label=label_path,
                    class_mapping=self.class_mapping,
                )
            )
        return ret

class MSDHeartProcessor(MSDProcessor):
    name = 'MSD/Task02_Heart'
    orientation = 'SRA'
    class_mapping = {1: 'left atrium'}

class MSDLiverProcessor(MSDProcessor):
    name = 'MSD/Task03_Liver'
    orientation = 'SRA'
    class_mapping = {1: 'liver', 2: 'liver tumor'}

class MSDHippocampusProcessor(MSDProcessor):
    name = 'MSD/Task04_Hippocampus'
    class_mapping = {
        1: 'anterior hippocampus',
        2: 'posterior hippocampus',
    }

    def get_modalities(self) -> list[str]:
        # according to https://arxiv.org/pdf/1902.09063.pdf Methods.Datasets.Task04_Hippocampus
        return ['T1 MRI']

class MSDProstateProcessor(MSDProcessor):
    name = 'MSD/Task05_Prostate'
    orientation = 'SRA'
    class_mapping = {
        1: 'peripheral zone of prostate',
        2: 'transition zone of prostate',
    }

    def get_modalities(self) -> list[str]:
        return [f'{m} MRI' for m in super().get_modalities()]

class MSDLungProcessor(MSDProcessor):
    name = 'MSD/Task06_Lung'
    orientation = 'SRA'
    class_mapping = {1: 'lung cancer'}

class MSDPancreasProcessor(MSDProcessor):
    name = 'MSD/Task07_Pancreas'
    orientation = 'SRA'
    class_mapping = {1: 'pancreas', 2: 'pancreatic cancer'}

class MSDHepaticVesselProcessor(MSDProcessor):
    name = 'MSD/Task08_HepaticVessel'
    orientation = 'SRA'
    class_mapping = {1: 'hepatic vessel', 2: 'liver tumor'}

class MSDSpleenProcessor(MSDProcessor):
    name = 'MSD/Task09_Spleen'
    orientation = 'SRA'
    class_mapping = {1: 'spleen'}

class MSDColonProcessor(MSDProcessor):
    name = 'MSD/Task10_Colon'
    class_mapping = {1: 'colon cancer'}
