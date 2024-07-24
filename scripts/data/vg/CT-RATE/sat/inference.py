import argparse
from pathlib import Path

import jsonargparse
import orjson
import torch
from lightning import Fabric
from torch.utils.data import DataLoader

from luolib.utils.misc import min_stem

from data.inference_dataset import Inference_Dataset, collate_fn
from evaluate.inference_engine import inference
from model.build_model import load_checkpoint
from model.maskformer import Maskformer
from model.text_encoder import Text_Encoder

def parse_args():
    parser = jsonargparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--max_queries", type=int, default=256)
    parser.add_argument("--sw_batch_size", type=int, default=2)

    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--text_encoder_checkpoint", type=str)
    parser.add_argument("--text_encoder", type=str, choices=['ours', 'medcpt', 'basebert'])
    # MaskFormer
    parser.add_argument("--vision_backbone", type=str, help='UNETs UMamba or SwinUNETR')
    parser.add_argument(
        "--patch_size",
        type=tuple[int, int, int],
        default=(32, 32, 32),
        help='patch size on h w and d'
    )
    parser.add_argument('--range', type=tuple[int | None, int | None], default=(None, None))
    args = parser.parse_args()
    return args

def build_maskformer(args):
    model = Maskformer(args.vision_backbone, args.crop_size, args.patch_size, args.deep_supervision)

    def get_parameter_number(model):
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    # if is_master():
    print(f"** MODEL ** {get_parameter_number(model)['Total'] / 1e6}M parameters")

    return model

vl_dataset_dir = Path('data/processed/vision-language/CT-RATE')
vg_dataset_dir = Path('data/processed/visual-grounding/CT-RATE')
supported_classes: set[str] = set(orjson.loads((Path(__file__).parent / 'supported-classes.json').read_bytes())['CT'])

def parse_supported_targets(tags: list[dict]) -> list[str]:
    correction = {
        'main bronchus': 'bronchie',
        'bronchi': 'bronchie',
        'bronchus': 'bronchie',

        'thyroid': 'thyroid gland',
        'thyroid lobe': 'thyroid gland',
        'left thyroid lobe': 'left thyroid',
        'left thyroid gland': 'left thyroid',
        'right thyroid lobe': 'right thyroid',
        'right thyroid gland': 'right thyroid',

        'ventricle': 'heart ventricle',
        'atrium': 'heart atrium',
        'left atrium': 'left heart ventricle',
        'right atrium': 'right heart ventricle',
        'left ventricle': 'left heart ventricle',
        'right ventricle': 'right heart ventricle',

        'lung middle lobe': 'right lung middle lobe',
        'pleural effusion': 'lung effusion',
        'pulmonary nodule': 'lung nodule',
        'bladder': 'urinary bladder',
    }
    for location, num in [
        ('cervical', 7),
        ('thoracic', 12),
        ('lumbar', 6),
    ]:
        for i in range(1, num + 1):
            abbr = location[0]
            correction[f'{abbr.upper()}{i} vertebra'] = f'{location} vertebrae {i} ({abbr}{i})'

    targets = []
    for tag in tags:
        target = tag['target']
        target = correction.get(target, target)
        if target in supported_classes:
            targets.append(target)
    return targets

def main():
    args = parse_args()
    torch.set_float32_matmul_precision('medium')
    items = []
    split = 'train'
    data_list = orjson.loads((vg_dataset_dir / f'{split}.json').read_bytes())
    print(f'total: {len(data_list)}')
    data_list = data_list[slice(*args.range)]
    print(f'filtered len: {len(data_list)}')
    for item in data_list:
        targets = parse_supported_targets(item['tags'])
        if len(targets) == 0:
            continue
        for image_path in item['image']:
            volume_name = min_stem(Path(image_path))
            case, study, scan = volume_name.rsplit('_', 2)
            volume_suffix = f'{split}/{case}/{case}_{study}/{volume_name}'
            if (save_path := vg_dataset_dir / 'image' / f'{volume_suffix}_seg.pt.zst').exists():
                continue
            items.append({
                'image': str(vg_dataset_dir / 'image' / f'{volume_suffix}.nii.gz'),
                'save_path': save_path,
                'modality': 'ct',
                'dataset': 'CT-RATE',
                'label': targets,
            })

    test_set = Inference_Dataset(items, args.max_queries)
    test_loader = DataLoader(
        test_set,
        batch_size=1,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    fabric = Fabric(precision='16-mixed')
    torch.cuda.set_device(fabric.device)
    model = build_maskformer(args)
    # load knowledge encoder
    text_encoder = Text_Encoder(
        text_encoder=args.text_encoder,
        checkpoint=args.text_encoder_checkpoint,
        partial_load=False,
        open_bert_layer=12,
        open_modality_embed=False,
    )
    model, *_ = load_checkpoint(
        checkpoint=args.checkpoint,
        resume=False,
        partial_load=True,
        model=model,
    )
    model = fabric.setup(model)
    text_encoder = fabric.setup(text_encoder)
    test_loader = fabric.setup_dataloaders(test_loader)
    inference(model, text_encoder, test_loader, args.sw_batch_size)

if __name__ == '__main__':
    main()
