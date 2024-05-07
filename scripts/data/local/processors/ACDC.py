from pathlib import Path

from ._base import DefaultImageLoaderMixin, DefaultMaskLoaderMixin, Processor, MultiClassDataPoint

def parse_info(case_dir: Path):
    info = (case_dir / 'Info.cfg').read_text().strip()
    ret = {}
    for kv in info.splitlines():
        k, v = kv.strip().split(': ')
        ret[k] = v
    return ret

class ACDCProcessor(DefaultImageLoaderMixin, DefaultMaskLoaderMixin, Processor):
    name = 'ACDC'
    max_workers = 8

    def get_data_points(self):
        ret = []
        # ref: https://www.creatis.insa-lyon.fr/Challenge/acdc/evaluation.html
        class_mapping = {
            1: 'right ventricle cavity',
            2: 'myocardium',
            3: 'left ventricle cavity',
        }
        gt_suffix = '_gt.nii.gz'
        for case_dir in self.dataset_root.glob(f'*/patient*'):
            info = parse_info(case_dir)
            is_normal = info['Group'] == 'NOR'
            for label_path in case_dir.glob(f'*{gt_suffix}'):
                key = label_path.name[:-len(gt_suffix)]
                image_path = label_path.parent / f'{key}.nii.gz'
                data_point = MultiClassDataPoint(
                    key=key,
                    images={'cine MRI': image_path},
                    complete_anomaly=is_normal,
                    label=label_path,
                    class_mapping=class_mapping,
                )
                ret.append(data_point)
        return ret
