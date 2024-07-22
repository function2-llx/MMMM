from functools import partial
from pathlib import Path
import shutil

import einops
from jsonargparse import CLI
import nibabel as nib
import numpy as np
import orjson
import pandas as pd

from luolib.utils import process_map
from monai.data import orientation_ras_lps

from mmmm.data.defs import ORIGIN_VL_DATA_ROOT, PROCESSED_VG_DATA_ROOT

dataset_dir = ORIGIN_VL_DATA_ROOT / 'CT-RATE' / 'dataset'
vg_dataset_dir = PROCESSED_VG_DATA_ROOT / 'CT-RATE'
save_dir = vg_dataset_dir / 'image'

def from_df_str_array(array_str: str):
    assert array_str[0] == '[' and array_str[-1] == ']'
    return np.fromstring(array_str[1:-1], sep=', ')

def _np_cross(*args, **kwargs):
    # https://youtrack.jetbrains.com/issue/PY-63674/False-positive-This-code-is-unreachable-calling-a-method-with-an-override-having-NoReturn-or-Never
    return np.cross(*args, **kwargs)

def _get_affine(orientation: np.ndarray, position: np.ndarray, spacing_ij: np.ndarray, spacing_k: float):
    vj, vi = orientation[:3], orientation[3:]
    vk = _np_cross(vj, vi)
    spacing = np.array((*spacing_ij, spacing_k))
    affine = np.eye(4)
    affine[:3, :3] = np.stack((vi, vj, vk), axis=1) * spacing
    affine[3, :3] = position
    affine = orientation_ras_lps(affine)
    return affine

def process_volume(item: dict, meta: pd.DataFrame):
    volume_name: str = item['volume_name']
    image_path: Path = item['image_path']
    save_path: Path = item['save_path']
    image_sv: np.ndarray = nib.load(str(image_path)).get_fdata()
    volume_meta = meta.loc[volume_name]
    image_array = (image_sv * volume_meta['RescaleSlope'] + volume_meta['RescaleIntercept']).astype(np.int16)
    image_array = einops.rearrange(image_array, 'w h d -> h w d')
    affine = _get_affine(
        from_df_str_array(volume_meta['ImageOrientationPatient']),
        from_df_str_array(volume_meta['ImagePositionPatient']),
        from_df_str_array(volume_meta['XYSpacing']),
        volume_meta['ZSpacing'],
    )
    save_path.parent.mkdir(exist_ok=True, parents=True)
    tmp_save_path = save_path.with_name(f'.{save_path.name}')
    nib.save(nib.Nifti1Image(image_array, affine), tmp_save_path)
    shutil.move(tmp_save_path, save_path)

def process_split(split: str, max_workers: int):
    meta = pd.read_csv(dataset_dir / f'metadata/{split}_metadata.csv', index_col='VolumeName')
    data = []
    for item in orjson.loads((vg_dataset_dir / 'train.json').read_bytes()):
        for image_path in item['image']:
            volume_name = Path(image_path).name
            assert volume_name.endswith('.pt.zst')
            volume_name = volume_name[:-len('.pt.zst')] + '.nii.gz'
            case, study, scan = volume_name[:-len('.nii.gz')].rsplit('_', 2)
            volume_suffix = f'{split}/{case}/{case}_{study}/{volume_name}'
            save_path = save_dir / volume_suffix
            if save_path.exists():
                continue
            data.append({
                'volume_name': volume_name,
                'image_path': dataset_dir / volume_suffix,
                'save_path': save_path,
            })
    process_map(
        partial(process_volume, meta=meta),
        data,
        max_workers=max_workers,
    )

def main(max_workers: int = 8):
    process_split('train', max_workers)

if __name__ == '__main__':
    CLI(main)
