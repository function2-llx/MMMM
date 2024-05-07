# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
import re
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Tuple, Type

import einops
import torch
from torch import nn
from torch.nn import functional as F

from luolib.models import spadop
from luolib.utils import channel_first, channel_last
from luolib.utils.misc import ceil_divide

from mmmm.models import resample
from .transformer import TwoWayTransformer

class LayerNormNd(nn.LayerNorm):
    def __init__(self, num_channels: int, contiguous: bool = True):
        super().__init__(num_channels)
        self.contiguous = contiguous

    def forward(self, x: torch.Tensor):
        x = channel_last(x)
        x = super().forward(x)
        x = channel_first(x)
        if self.contiguous:
            x = x.contiguous()
        return x

class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: TwoWayTransformer,
        num_instances: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        text_sim: bool = False,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_instances (int): the number of queries for the instance branch
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
          text_sim (bool): whether to use text similarity, disable by default
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_instances = num_instances
        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_instances + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            resample.Upsample(transformer_dim, transformer_dim // 4, cnt=0),
            # This is what SegVol originally used:
            #   nn.LayerNorm((transformer_dim // 4, int(self.feat_shape[0]), int(self.feat_shape[1]), int(self.feat_shape[2])))
            # Why? Perhaps they don't want to reshape the tensor here
            LayerNormNd(transformer_dim // 4),
            activation(),
            resample.Upsample(transformer_dim // 4, transformer_dim // 8, cnt=1),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for _ in range(self.num_mask_tokens)
            ]
        )
        # we don't do that here
        # self.iou_prediction_head = MLP(
        #     transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        # )

        self.txt_align_upscaled_embedding = nn.Linear(768, 96)
        self.text_sim = text_sim

    def _load_from_state_dict(self, state_dict: dict[str, torch.Tensor], prefix: str, *args, **kwargs):
        ln_prefix = f'{prefix}output_upscaling.1.'
        if (weight := state_dict.get(f'{ln_prefix}weight')) is not None and weight.ndim == 4:
            bias = state_dict.get(f'{ln_prefix}bias')
            state_dict[f'{ln_prefix}weight'] = einops.reduce(weight, 'c ... -> c', 'mean')
            state_dict[f'{ln_prefix}bias'] = einops.reduce(bias, 'c ... -> c', 'mean')
        pt_num_instances = 0
        pattern = re.compile(rf'{prefix}output_hypernetworks_mlps\.(\d+)\.layers\.0\.weight')
        for key in list(state_dict.keys()):
            if match := pattern.match(key):
                pt_num_instances = max(int(match.group(1)), pt_num_instances)
            if key.startswith(f'{prefix}iou_prediction_head'):
                state_dict.pop(key)
        hyper_prefix = f'{prefix}output_hypernetworks_mlps.'
        for i in range(1 + pt_num_instances, self.num_mask_tokens):
            ref = 1 + (i - 1) % pt_num_instances
            for key, _ in self.output_hypernetworks_mlps[i].named_parameters():
                state_dict[f'{hyper_prefix}{i}.{key}'] = state_dict[f'{hyper_prefix}{ref}.{key}']
        pt_mask_tokens_weight = state_dict[f'{prefix}mask_tokens.weight']
        mask_tokens_weight_pad = torch.cat([
            pt_mask_tokens_weight[0:1],
            einops.repeat(
                pt_mask_tokens_weight[1:],
                'n ... -> (q n) ...',
                q=ceil_divide(self.num_instances, pt_num_instances),
            ),
        ])
        state_dict[f'{prefix}mask_tokens.weight'] = mask_tokens_weight_pad[:self.num_mask_tokens]
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(
        self,
        image_embeddings: torch.Tensor,
        text_embedding: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        patch_size_z: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            - predicted masks logits
            - embeddings for each mask
        """
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        # output_tokens = einops.repeat(output_tokens, '... -> n ...', n=num_queries)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        # Expand per-image data in batch direction to be per-mask
        if image_embeddings.shape[0] != tokens.shape[0]:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            src = image_embeddings
        # src = einops.repeat(image_embeddings, '1 ... -> n ...', n=num_queries)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        # pos_src = einops.repeat(image_pe, '1 ... -> n ...', n=num_queries)
        b, c, h, w, d = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        mask_tokens_out = hs[:, 1:1 + self.num_mask_tokens, :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w, d)
        # src = einops.rearrange(src, 'n (d h w) c -> n c d h w')
        # upscaled_embedding = self.output_upscaling(src)
        # 对不起, 实在想不到更好看的写法了，下次还敢
        upscaled_embedding = src
        for i, module in enumerate(self.output_upscaling):
            if i % 3 == 0:
                upscaled_embedding = module(upscaled_embedding, patch_size_z=patch_size_z)
            else:
                upscaled_embedding = module(upscaled_embedding)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w, d = upscaled_embedding.shape
        # masks = (hyper_in @ upscaled_embedding.view(b, c, h * w * d)).view(b, -1, h, w, d)
        masks = einops.einsum(hyper_in, upscaled_embedding, 'n m c, n c ... -> n m ...')

        if self.text_sim and text_embedding is not None:
            text_embedding_down = self.txt_align_upscaled_embedding(text_embedding).unsqueeze(dim=1)
            upscaled_embedding = upscaled_embedding.view(b, c, h * w * d)
            sim = (text_embedding_down @ upscaled_embedding).view(b, -1, h, w, d)
            # sim = einops.einsum(text_embedding_down, upscaled_embedding, 'n 1 c, n c ... -> ')
            sim = sim.repeat(1, masks.shape[1], 1, 1, 1)
            masks = masks + sim

        return masks, mask_tokens_out

# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
