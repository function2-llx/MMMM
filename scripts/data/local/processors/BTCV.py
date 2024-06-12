from ._base import DefaultImageLoaderMixin, DefaultMaskLoaderMixin, MultiClassDataPoint, Processor

class BTCVProcessor(DefaultImageLoaderMixin, DefaultMaskLoaderMixin, Processor):
    class_mapping: dict[int, str]

    @property
    def output_name(self):
        return self.name.replace('/', '-')

    def get_data_points(self):
        ret = []
        data_dir = self.dataset_root / 'RawData' / 'Training'
        for path in (data_dir / 'img').glob('*.nii.gz'):
            match self.dataset_root.name:
                case 'Abdomen':
                    key = path.name[3:7]
                    label_path = data_dir / 'label' / f'label{key}.nii.gz'
                case 'Cervix':
                    key = path.name[:7]
                    label_path = data_dir / 'label' / f'{key}-Mask.nii.gz'
                case _:
                    raise ValueError
            modality = 'CT'
            ret.append(
                MultiClassDataPoint(
                    key=key,
                    images={modality: path},
                    label=label_path,
                    class_mapping=self.class_mapping,
                ),
            )
        return ret, None

class BTCVAbdomenProcessor(BTCVProcessor):
    name = 'BTCV/Abdomen'
    class_mapping = {
        1: "spleen",
        2: "right kidney",
        3: "left kidney",
        4: "gallbladder",
        5: "esophagus",
        6: "liver",
        7: "stomach",
        8: "aorta",
        9: "inferior vena cava",
        10: 'portal vein and splenic vein',
        11: "pancreas",
        12: "right adrenal gland",
        13: "left adrenal gland",
    }

class BTCVCervixProcessor(BTCVProcessor):
    name = 'BTCV/Cervix'
    class_mapping = {
        1: 'bladder',
        2: 'uterus',
        3: 'rectum',
        4: 'small intestine',
    }
