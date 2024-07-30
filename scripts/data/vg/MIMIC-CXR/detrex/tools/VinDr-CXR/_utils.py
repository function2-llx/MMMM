import os
from pathlib import Path

import torch
from torchvision.io import write_png, write_jpeg

local_labels = [
    'Aortic enlargement',
    'Atelectasis',
    'Calcification',
    'Cardiomegaly',
    'Clavicle fracture',
    'Consolidation',
    'Edema',
    'Emphysema',
    'Enlarged PA',
    'ILD',
    'Infiltration',
    'Lung cavity',
    'Lung cyst',
    'Lung Opacity',
    'Mediastinal shift',
    'Nodule/Mass',
    'Pleural effusion',
    'Pleural thickening',
    'Pneumothorax',
    'Pulmonary fibrosis',
    'Rib fracture',
]

src_dir = Path('data/origin/local/VinDr-CXR')
save_dir = Path(os.getenv('DETECTRON2_DATASETS', 'datasets')) / 'VinDr-CXR'

def _save_image(image: torch.Tensor, path: Path):
    tmp_save_path = path.with_name(f'.{path.name}')
    match path.suffix:
        case '.png':
            write_png(image.cpu(), str(tmp_save_path))
        case '.jpg' | '.jpeg':
            write_jpeg(image.cpu(), str(tmp_save_path))
    tmp_save_path.rename(path)
