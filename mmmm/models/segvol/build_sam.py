from pathlib import Path

from jsonargparse import class_from_function
import torch

from luolib.types import tuple3_t

from .modeling import ImageEncoderViT, InstanceSam, MaskDecoder, PromptEncoder, TwoWayTransformer
from .modeling.sam import Sam

def _build_sam(
    *,
    embed_dim: int = 768,
    encoder_mlp_ratio: int = 4,
    encoder_num_layers: int = 12,
    num_heads: int = 12,
    dropout_rate: float = 0.0,
    patch_size: tuple3_t[int] | int,
    pos_embed_shape: tuple3_t[int],
    pt_in_channels: int | None = None,
    pt_patch_size: tuple3_t[int] | None = None,
    pt_pos_embed_shape: tuple3_t[int] | None = None,
    checkpoint: Path | None = None,
    state_dict_key: str | None = None,
    weight_prefix: str = '',
) -> Sam:
    encoder_mlp_dim = embed_dim * encoder_mlp_ratio
    sam = Sam(
        image_encoder=ImageEncoderViT(
            in_channels=3,
            pos_embed_shape=pos_embed_shape,
            patch_size=patch_size,
            hidden_size=embed_dim,
            mlp_dim=encoder_mlp_dim,
            num_layers=encoder_num_layers,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            pt_in_channels=pt_in_channels,
            pt_patch_size=pt_patch_size,
            pt_pos_embed_shape=pt_pos_embed_shape,
        ),
        prompt_encoder=PromptEncoder(embed_dim=embed_dim),
        mask_decoder=MaskDecoder(
            num_instances=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=embed_dim,
        ),
    )

    if checkpoint is not None:
        state_dict = torch.load(checkpoint)
        if state_dict_key is not None:
            state_dict = state_dict[state_dict_key]
        state_dict = {
            key[len(weight_prefix):]: value
            for key, value in state_dict.items() if key.startswith(weight_prefix) and not key.startswith(f'{weight_prefix}text_encoder')
        }
        missing_keys, unexpected_keys = sam.load_state_dict(state_dict, strict=False)
        print(f'load pre-trained SAM checkpoint from {checkpoint}')
        if missing_keys or unexpected_keys:
            print('missing:', missing_keys)
            print('unexpected:', unexpected_keys)

    return sam

def _build_instance_sam(
    *,
    embed_dim: int = 768,
    encoder_mlp_ratio: int = 4,
    encoder_num_layers: int = 12,
    num_heads: int = 12,
    dropout_rate: float = 0.0,
    patch_size: tuple3_t[int] | int,
    pos_embed_shape: tuple3_t[int],
    num_instances: int,
    pt_in_channels: int | None = None,
    pt_patch_size: tuple3_t[int] | None = None,
    pt_pos_embed_shape: tuple3_t[int] | None = None,
    checkpoint: Path | None = None,
    weight_prefix: str = '',
) -> InstanceSam:
    encoder_mlp_dim = embed_dim * encoder_mlp_ratio
    sam = InstanceSam(
        image_encoder=ImageEncoderViT(
            in_channels=3,
            pos_embed_shape=pos_embed_shape,
            patch_size=patch_size,
            hidden_size=embed_dim,
            mlp_dim=encoder_mlp_dim,
            num_layers=encoder_num_layers,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            pt_in_channels=pt_in_channels,
            pt_patch_size=pt_patch_size,
            pt_pos_embed_shape=pt_pos_embed_shape,
        ),
        prompt_encoder=PromptEncoder(embed_dim=embed_dim),
        mask_decoder=MaskDecoder(
            num_instances=num_instances,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=embed_dim,
        ),
    )

    if checkpoint is not None:
        state_dict = torch.load(checkpoint)
        state_dict = {
            key[len(weight_prefix):]: value
            for key, value in state_dict.items() if key.startswith(weight_prefix) and not key.startswith(f'{weight_prefix}text_encoder')
        }
        missing_keys, unexpected_keys = sam.load_state_dict(state_dict, strict=False)
        print(f'load pre-trained SAM checkpoint from {checkpoint}')
        if missing_keys or unexpected_keys:
            print('missing:', missing_keys)
            print('unexpected:', unexpected_keys)

    return sam

build_sam = class_from_function(_build_sam)
