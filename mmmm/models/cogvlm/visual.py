from argparse import Namespace

from einops import einops
import torch
from torch import nn
from torch.nn import functional as nnf
from transformers.activations import ACT2FN
import xformers.ops as xops

from luolib.models import spadop
from luolib.models.param import NoWeightDecayParameter
from luolib.types import tuple2_t

class PatchEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = spadop.Conv3d(
            config.in_channels,
            config.hidden_size,
            config.patch_size,
            config.patch_size,
            adaptive=False,
        )
        self.pretrained_pos_embed_shape: tuple2_t[int] = config.pretrained_pos_embed_shape
        self.cls_embedding = NoWeightDecayParameter(torch.zeros(1, config.hidden_size))
        self.cls_pos_embed = NoWeightDecayParameter(torch.zeros(1, config.hidden_size))
        self.pos_embed = NoWeightDecayParameter(torch.zeros(1, config.hidden_size, *config.pos_embed_shape))
        # self.position_embedding = nn.Embedding(config.num_positions, config.hidden_size)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        if (pos_embed := state_dict.get(f'{prefix}position_embedding.weight')) is not None:
            state_dict[f'{prefix}cls_pos_embed'], pos_embed = pos_embed[0:1], pos_embed[1:]
            h, w = self.pretrained_pos_embed_shape
            pos_embed = nnf.interpolate(
                einops.rearrange(pos_embed, '(h w) c -> 1 c h w', h=w, w=w),
                self.pos_embed.shape[-2:],
                mode='bicubic',
            )
            pos_embed = einops.repeat(pos_embed, '1 c h w -> 1 c d h w', d=self.pos_embed.shape[2])

            state_dict[f'{prefix}pos_embed'] = pos_embed

        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, images: "tensor(B, C, H, W)") -> "tensor(B, L, D)":
        x = self.proj(images)
        cls_token = (self.cls_embedding + self.cls_pos_embed).expand(x.shape[0], -1, -1)
        x += nnf.interpolate(self.pos_embed, x.shape[2:], mode='trilinear')
        x = einops.rearrange(x, 'n c ... -> n (...) c')
        x = torch.cat((cls_token, x), dim=1)
        return x


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        head_dim = config.hidden_size // config.num_heads
        self.scale = head_dim ** -0.5
        self.query_key_value = nn.Linear(config.hidden_size, config.hidden_size * 3)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.output_dropout = torch.nn.Dropout(config.dropout_prob)

    def forward(self, x: "tensor(B, L, D)") -> "tensor(B, L, D)":
        B, L, _ = x.shape
        qkv = self.query_key_value(x)
        qkv = qkv.reshape(B, L, 3, self.num_heads, -1).permute(2, 0, 1, 3, 4)  # 3, B, L, H, D
        q, k, v = qkv[0], qkv[1], qkv[2]

        out = xops.memory_efficient_attention(
            q, k, v, scale=self.scale,
        )
        output = self.dense(out.view(B, L, -1))
        output = self.output_dropout(output)
        return output

    def attention(self, q, k, v):
        attn_weights = torch.matmul(q * self.scale, k.transpose(-2, -1))
        attn_weights = attn_weights.softmax(dim=-1)
        output = torch.matmul(attn_weights, v)
        return output


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x)
        return x


class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = Attention(config)
        self.mlp = MLP(config)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        attention_input = hidden_states
        attention_output = self.input_layernorm(self.attention(attention_input))
        hidden_states = attention_input + attention_output
        mlp_input = hidden_states
        mlp_output = self.post_attention_layernorm(self.mlp(mlp_input))
        output = mlp_input + mlp_output
        return output


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([TransformerLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states):
        for layer_module in self.layers:
            hidden_states = layer_module(hidden_states)
        return hidden_states


class GLU(nn.Module):
    def __init__(self, config, in_features):
        super().__init__()
        self.linear_proj = nn.Linear(in_features, config.hidden_size, bias=False)
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.act1 = nn.GELU()
        self.act2 = nn.functional.silu
        self.dense_h_to_4h = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.dense_4h_to_h = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        x = self.linear_proj(x)
        x = self.act1(self.norm1(x))
        x = self.act2(self.gate_proj(x)) * self.dense_h_to_4h(x)
        x = self.dense_4h_to_h(x)
        return x


class EVA2CLIPModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        vision_config = Namespace(**config.vision_config)
        self.patch_embedding = PatchEmbedding(vision_config)
        self.transformer = Transformer(vision_config)
        self.linear_proj = GLU(config, in_features=vision_config.hidden_size)
        self.boi = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.eoi = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

    def forward(self, images: "tensor(B, C, H, W)") -> "tensor(B, L, D)":
        x = self.patch_embedding(images)
        x = self.transformer(x)
        x = x[:, 1:]
        x = self.linear_proj(x)
        boi = self.boi.expand(x.shape[0], -1, -1)
        eoi = self.eoi.expand(x.shape[0], -1, -1)
        x = torch.cat((boi, x, eoi), dim=1)
        return x
