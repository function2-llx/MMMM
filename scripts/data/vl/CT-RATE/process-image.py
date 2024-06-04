from dataclasses import dataclass
from logging import getLogger
from pathlib import Path

from jsonargparse import ArgumentParser
import numpy as np
import orjson
import torch
import torchvision.transforms.v2.functional as tvtf

from luolib.utils import save_pt_zst
from monai.data import MetaTensor
import monai.transforms as mt
from monai.utils import GridSampleMode
from scripts.data.local.processors._base import (
    DataPoint, DefaultImageLoaderMixin, Processor as _ProcessorBase,
    clip_intensity,
)

from mmmm.data.dataset.vl import VLDataPoint
from mmmm.data.defs import PROCESSED_VL_DATA_ROOT
from scripts.data.vl._utils import get_resize

class CT_RATEVLDataPoint(VLDataPoint):
    key: str

@dataclass(kw_only=True)
class CT_RATEDataPoint(DataPoint):
    study: CT_RATEVLDataPoint

class Processor(DefaultImageLoaderMixin, _ProcessorBase):
    def get_data_points(self):
        ret = []
        for split in ['train', 'validate']:
            data: list[CT_RATEVLDataPoint] = orjson.loads((PROCESSED_VL_DATA_ROOT / f'CT-RATE/{split}-raw.json').read_bytes())
            for study in data:
                ret.append(CT_RATEDataPoint(key=study['key'], study=study, images={}))
        return ret, None

    @property
    def case_data_root(self):
        return PROCESSED_VL_DATA_ROOT / f'CT-RATE/image'

    def normalize_image(self, images: MetaTensor, *args, **kwargs):
        zero_mask = images == 0
        images[zero_mask] = images[~zero_mask].min()
        return super().normalize_image(images, *args, **kwargs)

    def process_data_point(self, data_point: CT_RATEDataPoint, empty_cache: bool, raise_error: bool):
        self.key = key = data_point.key
        try:
            if empty_cache:
                self._check_cuda_cache()
            study = data_point.study
            save_dir = self.case_data_root / f'.{key}'
            save_dir.mkdir(exist_ok=True, parents=True)
            for image_path in study['image']:
                image_path = Path(image_path)
                image = self.image_loader(image_path).to(self.device, dtype=torch.float32)
                orient = mt.Orientation(self.get_orientation(image))
                image = orient(image).contiguous()
                # 2. clip intensity, and crop the images & masks
                cropper = clip_intensity(image)
                image = cropper(image)  # type: ignore
                # 3. compute resize (default: adapt to self.max_smaller_edge and self.min_aniso_ratio)
                resize_shape = get_resize(image.shape[1:])
                # 4.1. normalize and save images
                image, *_ = self.normalize_image(image, np.array(resize_shape), GridSampleMode.BILINEAR)
                save_pt_zst(
                    tvtf.to_dtype(image.as_tensor().cpu(), torch.uint8, scale=True),
                    save_dir / (image_path.name[:-len('.nii.gz')] + '.pt.zst'),
                )
            save_dir.rename(save_dir.with_name(key))
        except Exception as e:
            self.logger.error(key)
            self.logger.error(e)
            if raise_error:
                raise e
            else:
                import traceback
                self.logger.error(traceback.format_exc())

    def process(self, *args, **kwargs):
        super().process(*args, **kwargs)
        for split in ['train', 'validate']:
            data: list[CT_RATEVLDataPoint] = orjson.loads((PROCESSED_VL_DATA_ROOT / f'CT-RATE/{split}-raw.json').read_bytes())
            processed = []
            for study in data:
                key = study.pop('key')
                case_dir = self.case_data_root / key
                if not case_dir.exists():
                    continue
                for i, image_path in enumerate(study['image']):
                    image_path = Path(image_path)
                    processed_path = case_dir / (image_path.name[:-len('.nii.gz')] + '.pt.zst')
                    assert processed_path.exists()
                    study['image'][i] = str(processed_path)
                processed.append(study)
            (PROCESSED_VL_DATA_ROOT / f'CT-RATE/{split}.json').write_bytes(orjson.dumps(processed, option=orjson.OPT_INDENT_2))

def main():
    parser = ArgumentParser()
    parser.add_argument('--max_workers', type=int, default=8)
    args = parser.parse_args()
    processor = Processor(getLogger(), max_workers=args.max_workers, chunksize=1, override=True)
    processor.process(empty_cache=True)

if __name__ == '__main__':
    main()
