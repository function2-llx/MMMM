from __future__ import annotations as _

from collections.abc import Sequence
from functools import partial
from typing import TYPE_CHECKING

import einops
import torch
import torch.nn as nn
if TYPE_CHECKING:
    import xformers.ops as xops
    from xformers.ops.fmha import BlockDiagonalMask

from luolib.models import spadop
from luolib.models.param import NoWeightDecayParameter
from luolib.models.utils import forward_gc
from luolib.types import param3_t, tuple3_t
from monai.networks.blocks import SABlock, TransformerBlock

from mmmm.utils import ParameterWrapper
from mmmm.models import resample

class PatchEmbeddingBlock(nn.Module):
    """
    modified from monai.networks.blocks.PatchEmbeddingBlock
    """

    def __init__(
        self,
        in_channels: int,
        patch_size: param3_t[int],
        pos_embed_shape: tuple3_t[int],
        hidden_size: int,
        num_heads: int,
        dropout_rate: float,
        pt_in_channels: int | None,
        pt_patch_size: tuple3_t[int] | None,
        pt_pos_embed_shape: tuple3_t[int] | None,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            patch_size: dimension of patch size.
            hidden_size: dimension of hidden layer.
            num_heads: number of attention heads.
            dropout_rate: fraction of the input units to drop.
        """
        super().__init__()
        if not (0 <= dropout_rate <= 1):
            raise ValueError(f"dropout_rate {dropout_rate} should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden size {hidden_size} should be divisible by num_heads {num_heads}.")

        self.proj = resample.Downsample(in_channels, hidden_size, patch_size)
        self.position_embeddings = ParameterWrapper(NoWeightDecayParameter(torch.zeros(1, hidden_size, *pos_embed_shape)))
        self.dropout = nn.Dropout(dropout_rate, inplace=True)

        # for loading SegVol
        self.pt_in_channels = pt_in_channels
        self.pt_patch_size = pt_patch_size
        self.pt_pos_embed_shape = pt_pos_embed_shape

    @property
    def in_channels(self):
        return self.proj.in_channels

    def forward(self, image_list: list[torch.Tensor], patch_size_list: list[tuple3_t[int]]):
        from xformers.ops.fmha import BlockDiagonalMask
        x_list, shape_list = [], []
        for image, patch_size in zip(image_list, patch_size_list):
            x = self.proj(image[None], patch_size)
            shape = x.shape[2:]
            pos_embed = spadop.resample(self.position_embeddings.weight, shape)
            x = einops.rearrange(x + pos_embed, '1 c ... -> 1 (...) c')
            x_list.append(x)
            shape_list.append(shape)
        attn_mask, x = BlockDiagonalMask.from_tensor_list(x_list)
        x = self.dropout(x)
        return x, shape_list, attn_mask

    def _load_from_state_dict(self, state_dict: dict[str, torch.Tensor], prefix: str, *args, **kwargs):
        if (proj_weight := state_dict.pop(f'{prefix}patch_embeddings.1.weight', None)) is not None and proj_weight.ndim == 2:
            # load from SegVol checkpoint
            p0, p1, p2 = self.pt_patch_size
            proj_weight = spadop.resample(
                einops.rearrange(
                    proj_weight,
                    'co (p0 p1 p2 ci) -> co ci p0 p1 p2', p0=p0, p1=p1, p2=p2, ci=self.pt_in_channels,
                ),
                self.proj.kernel_size,
                scale=True,
            )
            if self.pt_in_channels == 1 and self.in_channels != 1:
                proj_weight = einops.repeat(proj_weight, 'co 1 ... -> co ci ...', ci=self.in_channels) / self.in_channels
            state_dict[f'{prefix}proj.weight'] = proj_weight
            state_dict[f'{prefix}proj.bias'] = state_dict.pop(f'{prefix}patch_embeddings.1.bias')

            d, h, w = self.pt_pos_embed_shape
            pos_embed = spadop.resample(
                einops.rearrange(
                    state_dict[f'{prefix}position_embeddings'],
                    '1 (d h w) c -> 1 c d h w', d=d, h=h, w=w,
                ),
                self.position_embeddings.weight.shape[2:],
            )
            state_dict[f'{prefix}position_embeddings'] = pos_embed
        elif (
            (pos_embed := state_dict.get(f'{prefix}position_embeddings.weight')) is not None and
            pos_embed.shape[2:] != (_shape := self.position_embeddings.weight.shape[2:])
        ):
            # TODO: refactor here
            pos_embed = spadop.resample(pos_embed, _shape)
            state_dict[f'{prefix}position_embeddings.weight'] = pos_embed

        ParameterWrapper.wrap(self, state_dict, prefix)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

def _patch_TransformerBlock_forward(self: TransformerBlock, x: torch.Tensor, attn_mask: BlockDiagonalMask):
    x = x + self.attn(self.norm1(x), attn_mask)
    x = x + self.mlp(self.norm2(x))
    return x

def _patch_SABlock_forward(self: SABlock, x: torch.Tensor, attn_mask: BlockDiagonalMask):
    # noinspection PyShadowingNames
    import xformers.ops as xops
    qkv = self.qkv(x)
    qkv = einops.rearrange(qkv, 'n l (qkv h d) -> qkv n l h d', qkv=3, h=self.num_heads)
    q, k, v = qkv[0], qkv[1], qkv[2]
    out = xops.memory_efficient_attention(q, k, v, attn_mask, scale=self.scale)
    out = einops.rearrange(out, 'n l h d -> n l (h d)')
    out = self.out_proj(out)
    out = self.drop_output(out)
    return out

class ImageEncoderViT(nn.Module):
    blocks: Sequence[TransformerBlock] | nn.ModuleList

    """
    modified from monai.networks.nets.ViT
    """
    def __init__(
        self,
        in_channels: int,
        patch_size: param3_t[int],
        pos_embed_shape: tuple3_t[int],
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        dropout_rate: float = 0.0,
        qkv_bias: bool = False,
        pt_in_channels: int | None = None,
        pt_patch_size: tuple3_t[int] | None = None,
        pt_pos_embed_shape: tuple3_t[int] | None = None,
    ) -> None:
        """
        Args:
            in_channels (int): dimension of input channels.
            patch_size (Union[Sequence[int], int]): dimension of patch size.
            hidden_size (int, optional): dimension of hidden layer. Defaults to 768.
            mlp_dim (int, optional): dimension of feedforward layer. Defaults to 3072.
            num_layers (int, optional): number of transformer blocks. Defaults to 12.
            num_heads (int, optional): number of attention heads. Defaults to 12.
            dropout_rate (float, optional): fraction of the input units to drop. Defaults to 0.0.
            qkv_bias (bool, optional): apply bias to the qkv linear layer in self attention block. Defaults to False.
            save_attn (bool, optional): to make accessible the attention in self attention block. Defaults to False.
        """
        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            patch_size=patch_size,
            pos_embed_shape=pos_embed_shape,
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            pt_in_channels=pt_in_channels,
            pt_patch_size=pt_patch_size,
            pt_pos_embed_shape=pt_pos_embed_shape,
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=hidden_size,
                    mlp_dim=mlp_dim,
                    num_heads=num_heads,
                    dropout_rate=dropout_rate,
                    qkv_bias=qkv_bias,
                    save_attn=False,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_size)
        # monkey patch go brrrrr
        for block in self.blocks:
            block.forward = partial(_patch_TransformerBlock_forward, block)
            block.attn.forward = partial(_patch_SABlock_forward, block.attn)
        self.gradient_checkpointing = False
        self._gradient_checkpointing_func = None

    def forward(self, image: list[torch.Tensor], patch_size: list[tuple3_t[int]]):
        attn_mask: BlockDiagonalMask
        x, shape_list, attn_mask = self.patch_embedding(image, patch_size)
        for blk in self.blocks:
            x = forward_gc(
                blk,
                self.gradient_checkpointing, self._gradient_checkpointing_func,
                x, attn_mask,
            )
        x = self.norm(x)
        x_list = list(attn_mask.split(x))
        for i, (x, (d, h, w)) in enumerate(zip(x_list, shape_list)):
            x = einops.rearrange(x, '1 (d h w) c -> 1 c d h w', d=d, h=h, w=w).contiguous()
            x_list[i] = x
        return x_list
