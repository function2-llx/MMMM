from pathlib import Path
import re

from mmmm.data.defs import ORIGIN_SEG_DATA_ROOT
from ._base import Default3DImageLoaderMixin, MultiClass3DMaskLoaderMixin, MultiClassDataPoint, Processor

class CTPelvic1KProcessor(Default3DImageLoaderMixin, MultiClass3DMaskLoaderMixin, Processor):
    name = 'CTPelvic1K'

    def get_image_info(self, origin_key: str, dataset_idx: int) -> tuple[dict[str, Path], str | None]:
        match dataset_idx:
            case 1:
                # BTCV-Abdomen
                data_dir = ORIGIN_SEG_DATA_ROOT / 'BTCV/Abdomen/RawData'
                if not (img_path := data_dir / 'Training' / 'img' / f'{origin_key}.nii.gz').exists():
                    img_path = data_dir / 'Testing' / 'img' / f'{origin_key}.nii.gz'
                return {'CT': img_path}, f'BTCV-Abdomen-{origin_key}'
            case 3:
                # MSD Task10_Colon
                data_dir = ORIGIN_SEG_DATA_ROOT / 'MSD/Task10_Colon'
                if not (img_path := data_dir / 'imagesTr' / f'{origin_key}.nii.gz').exists():
                    img_path = data_dir / 'imagesTs' / f'{origin_key}.nii.gz'
                return {'CT': img_path}, f'MSD-T10-{origin_key}'
            case _:
                return {}, None

    def get_data_points(self):
        # https://github.com/MIRACLE-Center/CTPelvic1K/blob/ebd422e00d4ca64c6a7e1d3e92a50b75d5e87af7/nnunet/dataset_conversion/JstPelvisSegmentation_5label.py#L108-L114
        class_mapping = {
            1: 'sacrum',
            2: 'right hip',
            3: 'left hip',
            4: 'lumbar vertebrae',
        }
        ret = []
        for dataset_idx in range(1, 8):
            mask_filename_pattern = re.compile(fr'dataset{dataset_idx}_(.+)_mask_4label.nii.gz')
            if dataset_idx <= 5:
                mask_dir = self.dataset_root / f'CTPelvic1K_dataset{dataset_idx}_mask_mappingback'
            else:
                mask_dir = self.dataset_root / f'CTPelvic1K_dataset{dataset_idx}_mask'
            for mask_path in mask_dir.iterdir():
                match = mask_filename_pattern.match(mask_path.name)
                if match is None:
                    continue
                origin_key = match.group(1)
                images, key = self.get_image_info(origin_key, dataset_idx)
                if key is None:
                    continue
                ret.append(
                    MultiClassDataPoint(
                        key=key,
                        images=images,
                        label=mask_path,
                        class_mapping=class_mapping,
                    ),
                )
        return ret
