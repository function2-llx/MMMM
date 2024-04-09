from pathlib import Path
import re

from mmmm.data.defs import ORIGIN_SEG_DATA_ROOT

from ._base import DefaultImageLoaderMixin, DefaultMaskLoaderMixin, MultiClassDataPoint, Processor

class CTPelvic1KProcessor(DefaultImageLoaderMixin, DefaultMaskLoaderMixin, Processor):
    name = 'CTPelvic1K'
    # https://nipy.org/nibabel/nifti_images.html#the-fall-back-header-affine
    image_reader = 'itkreader'
    mask_reader = 'itkreader'

    def get_image_info(self, origin_key: str, dataset_idx: int) -> tuple[Path | None, str]:
        match dataset_idx:
            case 1 | 5:
                # BTCV
                btcv_task = 'Abdomen' if dataset_idx == 1 else 'Cervix'
                data_dir = ORIGIN_SEG_DATA_ROOT / 'BTCV' / btcv_task / 'RawData'
                if btcv_task == 'Cervix':
                    origin_key = origin_key.replace('_', '-')
                sub_path = f'img/{origin_key}.nii.gz'
                if not (img_path := data_dir / 'Training' / sub_path).exists():
                    img_path = data_dir / 'Testing' / sub_path
                return img_path, f'BTCV-{btcv_task}-{origin_key}'
            case 2:
                # CT COLONOGRAPHY
                subject_id, series_idx, *_ = origin_key.split('_')
                data_dir = ORIGIN_SEG_DATA_ROOT / 'CT-COLONOGRAPHY/download/CT COLONOGRAPHY' / subject_id
                series_dirs = [*data_dir.glob(f'*/{series_idx}.*')]
                if len(series_dirs) != 1:
                    return None, ''
                return series_dirs[0], f'COLONOG-{origin_key}'
            case 3:
                # MSD Task10_Colon
                data_dir = ORIGIN_SEG_DATA_ROOT / 'MSD/Task10_Colon'
                if not (img_path := data_dir / 'imagesTr' / f'{origin_key}.nii.gz').exists():
                    img_path = data_dir / 'imagesTs' / f'{origin_key}.nii.gz'
                return img_path, f'MSD-T10-{origin_key}'
            case 4:
                # KiTS19
                data_dir = ORIGIN_SEG_DATA_ROOT / 'KiTS19/data'
                img_path = data_dir / f'{origin_key}/imaging.nii.gz'
                return img_path, f'KiTS19-{origin_key}'
            case 6 | 7:
                # CLINIC & CLINIC-metal
                img_path = self.dataset_root / f'CTPelvic1K_dataset{dataset_idx}_data/dataset{dataset_idx}_{origin_key}_data.nii.gz'
                return img_path, origin_key
            case _:
                return None, ''

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
            if dataset_idx <= 6:
                mask_filename_pattern = re.compile(fr'dataset{dataset_idx}_(.+)_mask_4label.nii.gz')
            else:
                # NT
                mask_filename_pattern = re.compile(fr'(.+)_mask_4label.nii.gz')
            if dataset_idx <= 5:
                mask_dir = self.dataset_root / f'CTPelvic1K_dataset{dataset_idx}_mask_mappingback'
            else:
                mask_dir = self.dataset_root / f'CTPelvic1K_dataset{dataset_idx}_mask'
            for mask_path in mask_dir.iterdir():
                match = mask_filename_pattern.match(mask_path.name)
                if match is None:
                    continue
                origin_key = match.group(1)
                img_path, key = self.get_image_info(origin_key, dataset_idx)
                if img_path is None:
                    continue
                ret.append(
                    MultiClassDataPoint(
                        key=key,
                        images={'CT': img_path},
                        label=mask_path,
                        class_mapping=class_mapping,
                    ),
                )
        return ret
