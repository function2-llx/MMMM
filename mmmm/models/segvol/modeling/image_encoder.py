import einops
import torch
import torch.nn as nn

from luolib.models import spadop
from luolib.models.param import NoWeightDecayParameter
from luolib.models.utils import forward_gc
from luolib.types import param3_t, tuple3_t
from monai.networks.blocks import TransformerBlock

from mmmm.utils import ParameterWrapper

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

        self.proj = spadop.InputConv3D(
            in_channels=in_channels, out_channels=hidden_size, kernel_size=patch_size, stride=patch_size, adaptive=False,
        )

        self.position_embeddings = ParameterWrapper(NoWeightDecayParameter(torch.zeros(1, hidden_size, *pos_embed_shape)))
        self.dropout = nn.Dropout(dropout_rate, inplace=True)

        # for loading SegVol
        self.pt_in_channels = pt_in_channels
        self.pt_patch_size = pt_patch_size
        self.pt_pos_embed_shape = pt_pos_embed_shape

    def forward(self, x):
        x = self.proj(x)
        shape = x.shape[2:]
        embeddings = x + spadop.resample(self.position_embeddings.weight, shape)
        embeddings = einops.rearrange(embeddings, 'n c ... -> n (...) c').contiguous()
        embeddings = self.dropout(embeddings)
        return embeddings, shape

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
        ParameterWrapper.wrap(self, state_dict, prefix)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

class ImageEncoderViT(nn.Module):
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
        save_attn: bool = False,
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
                    save_attn=save_attn,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.gradient_checkpointing = False
        self._gradient_checkpointing_func = None

    def forward(self, x):
        x, (d, h, w) = self.patch_embedding(x)
        for blk in self.blocks:
            x = forward_gc(
                blk, self.gradient_checkpointing, self._gradient_checkpointing_func, x,
            )
        x = self.norm(x)
        x = einops.rearrange(x, 'n (d h w) c -> n c d h w', d=d, h=h, w=w).contiguous()
        return x
