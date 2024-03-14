from dataclasses import dataclass
from pathlib import Path

import torch

from luolib.types import tuple3_t

from .modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer

@dataclass
class SamArgs:
    pos_embed_shape: tuple3_t[int]
    pt_in_channels: int | None = None
    pt_patch_size: tuple3_t[int] | None = None
    patch_size: tuple3_t[int] | int = 16
    pt_pos_embed_shape: tuple3_t[int] | None = None
    checkpoint: Path | None = None
    text_sim: bool = False

def build_sam_vit_3d(args: SamArgs) -> Sam:
    return _build_sam(
        embed_dim=768,
        encoder_num_layers=12,
        num_heads=12,
        args=args,
    )

def _build_sam(
    embed_dim: int,
    encoder_num_layers: int,
    num_heads: int,
    args: SamArgs,
) -> Sam:
    mlp_ratio = 4
    mlp_dim = embed_dim * mlp_ratio
    dropout_rate = 0.0

    sam = Sam(
        image_encoder=ImageEncoderViT(
            in_channels=3,
            pos_embed_shape=args.pos_embed_shape,
            patch_size=args.patch_size,
            hidden_size=embed_dim,
            mlp_dim=mlp_dim,
            num_layers=encoder_num_layers,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            pt_in_channels=args.pt_in_channels,
            pt_patch_size=args.pt_patch_size,
            pt_pos_embed_shape=args.pt_pos_embed_shape,
        ),
        prompt_encoder=PromptEncoder(embed_dim=embed_dim),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            text_sim=args.text_sim,
        ),
    )

    if args.checkpoint is not None:
        # load from SegVol checkpoint
        state_dict = torch.load(args.checkpoint, map_location='cpu')['model']
        prefix = 'module.'
        state_dict = {
            key[len(prefix):]: value
            for key, value in state_dict.items() if key.startswith(prefix) and not key.startswith(f'{prefix}text_encoder')
        }
        sam.load_state_dict(state_dict)

    return sam
