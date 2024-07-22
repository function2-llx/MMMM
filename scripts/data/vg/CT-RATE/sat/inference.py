from pathlib import Path

import torch
from lightning import Fabric
from torch.utils.data import DataLoader

from data.inference_dataset import Inference_Dataset, collate_fn
from evaluate.inference_engine import inference
from evaluate.params import parse_args
from model.build_model import load_checkpoint
from model.maskformer import Maskformer
from model.text_encoder import Text_Encoder

def build_maskformer(args):
    model = Maskformer(args.vision_backbone, args.crop_size, args.patch_size, args.deep_supervision)

    # model = model.to(device)
    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu_id], find_unused_parameters=True)

    def get_parameter_number(model):
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    # if is_master():
    print(f"** MODEL ** {get_parameter_number(model)['Total'] / 1e6}M parameters")

    return model

def main(args):
    Path(args.rcd_dir).mkdir(exist_ok=True, parents=True)
    print(f'Inference Results will be Saved to ** {args.rcd_dir} **')

    # dataset and loader
    test_set = Inference_Dataset(str(Path(__file__).parent / 'data.jsonl'), args.max_queries, args.batchsize_3d)
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
    inference(
        model,
        text_encoder,
        test_set,
        test_loader,
        args.rcd_dir,
        fabric,
    )

if __name__ == '__main__':
    # get configs
    args = parse_args()

    main(args)
