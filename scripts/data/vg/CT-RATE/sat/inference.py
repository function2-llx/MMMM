from pathlib import Path

import orjson
import torch
from lightning import Fabric
from torch.utils.data import DataLoader

from luolib.utils.misc import min_stem

from data.inference_dataset import Inference_Dataset, collate_fn
from evaluate.inference_engine import inference
from evaluate.params import parse_args
from model.build_model import load_checkpoint
from model.maskformer import Maskformer
from model.text_encoder import Text_Encoder

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

supported_classes = {
    "bronchie",
    "left lung upper lobe",
    "left lung lower lobe",
    "right lung lower lobe",
    "right lung middle lobe",
    "right lung upper lobe",
    "heart",
    "esophagus",
    "adrenal gland",
    "left adrenal gland",
    "right adrenal gland",
    "lung nodule",
    # TODO
}

def parse_targets(tags: list[dict]) -> list[str]:
    for tag in tags:
        if (target := tag['target']) in supported_classes)

def main(args):
    torch.set_float32_matmul_precision('medium')
    items = []
    split = 'train'
    for item in orjson.loads((vg_dataset_dir / f'{split}.json').read_bytes()):
        targets = list(set(target for tag in item['tags'] if (target := tag['target']) in supported_classes))
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

    test_set = Inference_Dataset(items, args.max_queries, args.batchsize_3d)
    # sampler = DistributedSampler(testset)
    test_loader = DataLoader(
        test_set,
        # sampler=sampler,
        batch_size=1,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    # sampler.set_epoch(0)
    fabric = Fabric(precision='16-mixed')
    torch.cuda.set_device(fabric.device)
    # set model (by default gpu
    model = build_maskformer(args)

    # load knowledge encoder
    text_encoder = Text_Encoder(
        text_encoder=args.text_encoder,
        checkpoint=args.text_encoder_checkpoint,
        partial_load=args.text_encoder_partial_load,
        open_bert_layer=12,
        open_modality_embed=False,
    )

    # load checkpoint if specified
    model, _, _ = load_checkpoint(
        checkpoint=args.checkpoint,
        resume=False,
        partial_load=args.partial_load,
        model=model,
    )
    # choose how to evaluate the checkpoint
    model = fabric.setup(model)
    text_encoder = fabric.setup(text_encoder)
    test_loader = fabric.setup_dataloaders(test_loader)
    inference(model, text_encoder, test_loader, args.batchsize_3d)

if __name__ == '__main__':
    # get configs
    args = parse_args()
    main(args)
