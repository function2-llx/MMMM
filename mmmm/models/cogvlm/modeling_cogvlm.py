"""largely copy from llama and adapt for cogvlm"""
from typing import Any, Dict, List, Literal, Optional, TYPE_CHECKING, Tuple, Union
import warnings

import einops
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils.logging import get_logger

from luolib.models.param import NoWeightDecayParameter
from luolib.models.utils import forward_gc
from luolib.types import tuple3_t

from mmmm.utils import apply_prefix, get_lora_modules_default
from mmmm.data.defs import CE_IGNORE_INDEX
from .configuration_cogvlm import CogVLMConfig
from .visual import EVA2CLIPModel
from ...data.utils import LANGUAGE_TOKEN_TYPE, VISION_TOKEN_TYPE

if TYPE_CHECKING:
    from transformers.utils import ModelOutput

logger = get_logger(__name__)

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = NoWeightDecayParameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

def get_expert_mask(
    token_type_ids: torch.LongTensor, padding_mask: torch.BoolTensor,
) -> tuple[torch.BoolTensor, torch.BoolTensor]:
    vision_token_mask = torch.zeros_like(token_type_ids, dtype=torch.bool)
    # I don't know why do authors of CogVLM use this. as a result, a token is masked as vision iff itself & the token after
    # it have the vision type, which make eoi token have language mask
    vision_token_mask[:, :-1] = (token_type_ids[:, :-1] == VISION_TOKEN_TYPE) & (token_type_ids[:, 1:] == VISION_TOKEN_TYPE)
    # vision_token_mask = token_type_ids == VISION_TOKEN_TYPE
    language_token_mask = ~vision_token_mask
    if token_type_ids.shape[1] > 1:
        vision_token_mask &= padding_mask
        language_token_mask &= padding_mask
    return vision_token_mask, language_token_mask

class VisionExpertMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.language_mlp = MLP(config)
        self.vision_mlp = MLP(config)

    def get_lora_modules(self, prefix: str):
        if self.config.lora_lang:
            return get_lora_modules_default(self, prefix, False)
        else:
            return get_lora_modules_default(
                self.vision_mlp, apply_prefix(prefix, 'vision_mlp'),
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        token_type_ids: torch.Tensor,
        padding_mask: torch.BoolTensor,
    ):
        vision_token_mask, language_token_mask = get_expert_mask(token_type_ids, padding_mask)
        output = torch.zeros_like(hidden_states)
        output[vision_token_mask] = self.vision_mlp(hidden_states[vision_token_mask])
        output[language_token_mask] = self.language_mlp(hidden_states[language_token_mask])
        return output

def _to_tensor_list(x: torch.Tensor, padding_mask: torch.BoolTensor):
    return [
        x[i:i + 1, mask]
        for i, mask in enumerate(padding_mask)
    ]

def attention_fn(
    query_layer: "torch.tensor(B, H, L, HD)",
    key_layer: "torch.tensor(B, H, L, HD)",
    value_layer: "torch.tensor(B, H, L, HD)",
    padding_mask: torch.BoolTensor,
    dropout_p: float = 0.,
):
    import xformers.ops as xops
    query_layer = einops.rearrange(query_layer, 'n h l d -> n l h d')
    key_layer = einops.rearrange(key_layer, 'n h l d -> n l h d')
    value_layer = einops.rearrange(value_layer, 'n h l d -> n l h d')
    if padding_mask.shape[1] == query_layer.shape[1]:
        from xformers.ops.fmha.attn_bias import BlockDiagonalCausalMask
        output = torch.zeros_like(query_layer)
        attn_bias, query_layer, key_layer, value_layer = BlockDiagonalCausalMask.from_tensor_lists_qkv(
            _to_tensor_list(query_layer, padding_mask),
            _to_tensor_list(key_layer, padding_mask),
            _to_tensor_list(value_layer, padding_mask),
        )
        output[padding_mask] = xops.memory_efficient_attention(
            query_layer, key_layer, value_layer, attn_bias, dropout_p,
        )
    else:
        # for generation, implement attention manually
        assert query_layer.shape[1] == 1
        query_layer *= query_layer.shape[-1] ** -0.5
        # avoid NaN
        key_layer[~padding_mask] = 0
        value_layer[~padding_mask] = 0
        attention_scores = einops.einsum(query_layer[:, 0], key_layer, 'n h d, n l h d -> n l h')
        attention_scores[~padding_mask] = -torch.inf
        attention_scores = attention_scores.softmax(dim=1, dtype=torch.float32).to(dtype=attention_scores.dtype)
        if dropout_p > 0:
            attention_scores = F.dropout(attention_scores, dropout_p)
        output = einops.einsum(value_layer, attention_scores, 'n l h d, n l h -> n h d')[:, None]
    return einops.rearrange(output, 'n l h d -> n h l d')


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = self._compute_inv_freq(device)
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len_cached = 0

    def _compute_inv_freq(self, device=None):
        return 1.0 / (
                self.base
                ** (torch.arange(0, self.dim, 2, device=device) / self.dim)
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()[:, None, :]
        self.sin_cached = emb.sin()[:, None, :]

    def forward(self, x, seq_len):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:seq_len, ...].to(dtype=x.dtype),
        )


def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=x1.ndim - 1)


def apply_rotary_pos_emb_index_bhs(q, k, cos, sin, position_id):
    # batch_size, num_head, seq_len, hidden_size
    cos, sin = F.embedding(position_id, cos.squeeze(1)).unsqueeze(1), \
        F.embedding(position_id, sin.squeeze(1)).unsqueeze(1)
    q, k = (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
    return q, k


class VisionExpertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings

        self.rotary_emb = RotaryEmbedding(self.head_dim)
        self.vision_expert_query_key_value = nn.Linear(self.hidden_size, self.hidden_size * 3, bias=False)
        self.vision_expert_dense = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.language_expert_query_key_value = nn.Linear(self.hidden_size, self.hidden_size * 3, bias=False)
        self.language_expert_dense = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def get_lora_modules(self, prefix: str):
        if self.config.lora_lang:
            return get_lora_modules_default(self, prefix, False)
        else:
            target_modules = [
                apply_prefix(prefix, 'vision_expert_query_key_value'),
                apply_prefix(prefix, 'vision_expert_dense'),
            ]
            modules_to_save = []
            return target_modules, modules_to_save

    def _transpose_for_scores(self, tensor):
        """Transpose a 3D tensor [B, L, H*HD] into a 4D tensor with size [B H L HD]."""
        new_tensor_shape = tensor.size()[:-1] + (self.num_heads, self.head_dim)
        tensor = tensor.view(*new_tensor_shape)
        return tensor.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_type_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        padding_mask: torch.BoolTensor,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        vision_token_mask, language_token_mask = get_expert_mask(token_type_ids, padding_mask)

        shape = list(hidden_states.shape)
        shape[-1] = shape[-1] * 3
        mixed_raw_layer = torch.zeros(shape, dtype=hidden_states.dtype, device=hidden_states.device)
        mixed_raw_layer[vision_token_mask] = self.vision_expert_query_key_value(hidden_states[vision_token_mask])
        mixed_raw_layer[language_token_mask] = self.language_expert_query_key_value(hidden_states[language_token_mask])

        query_states, key_states, value_states = torch.split(mixed_raw_layer, self.hidden_size, dim=-1)
        query_states = self._transpose_for_scores(query_states)  # B, H, L, HD
        key_states = self._transpose_for_scores(key_states)  # B, H, L, HD
        value_states = self._transpose_for_scores(value_states)  # B, H, L, HD

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=position_ids.max() + 1)
        query_states, key_states = apply_rotary_pos_emb_index_bhs(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        context_layer = attention_fn(
            query_states,
            key_states,
            value_states,
            padding_mask,
        )
        if context_layer.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {context_layer.size()}"
            )
        context_layer = context_layer.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)

        attn_output = torch.empty(context_layer.shape, dtype=hidden_states.dtype, device=hidden_states.device)
        attn_output[vision_token_mask] = self.vision_expert_dense(context_layer[vision_token_mask])
        attn_output[language_token_mask] = self.language_expert_dense(context_layer[language_token_mask])

        if output_attentions:
            warnings.warn("output_attentions is not implemented.")

        return attn_output, None, past_key_value

class CogVLMDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = VisionExpertAttention(config=config)
        self.mlp = VisionExpertMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_type_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        padding_mask: torch.BoolTensor,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        if hidden_states.shape[1] > 1:
            hidden_states = _mask_set(hidden_states, padding_mask, self.input_layernorm(hidden_states[padding_mask]))
        else:
            hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            padding_mask=padding_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        if hidden_states.shape[1] > 1:
            hidden_states = _mask_set(hidden_states, padding_mask, self.post_attention_layernorm(hidden_states[padding_mask]))
        else:
            hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, token_type_ids=token_type_ids, padding_mask=padding_mask)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs  # type: ignore


class CogVLMPreTrainedModel(PreTrainedModel):
    config_class = CogVLMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["CogVLMDecoderLayer", "TransformerLayer"]
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


def is_empty(images_list: Optional[List[List[torch.Tensor]]]):
    if images_list is None or len(images_list) == 0:
        return True
    for image_list in images_list:
        if len(image_list):
            return False
    return True


def build_position_ids(x: "torch.BoolTensor(B, L)", attention_mask: Optional["torch.BoolTensor(B, L)"] = None) -> "torch.LongTensor(B, L)":
    if attention_mask is not None:
        tmp = x.clone()
        tmp[~(attention_mask.bool())] = -1
    else:
        tmp = x.clone()
    # image boi eoi token as LANGUAGE_TOKEN_TYPE
    is_boi_eoi = torch.zeros_like(x, dtype=torch.bool)
    is_boi_eoi[:, 1:] |= (tmp[:, 1:] == VISION_TOKEN_TYPE) & (tmp[:, :-1] == LANGUAGE_TOKEN_TYPE)
    is_boi_eoi[:, 0] |= (tmp[:, 0] == VISION_TOKEN_TYPE)
    is_boi_eoi[:, :-1] |= (tmp[:, :-1] == VISION_TOKEN_TYPE) & (tmp[:, 1:] == LANGUAGE_TOKEN_TYPE)
    is_boi_eoi[:, -1] |= (tmp[:, -1] == VISION_TOKEN_TYPE)
    tmp[is_boi_eoi] = LANGUAGE_TOKEN_TYPE
    # final position ids
    y = torch.zeros_like(x, dtype=torch.long)
    y[:, 1:] = (tmp[:, 1:] == LANGUAGE_TOKEN_TYPE) | ((tmp[:, 1:] == VISION_TOKEN_TYPE) & (tmp[:, :-1] == LANGUAGE_TOKEN_TYPE))
    y = y.cumsum(dim=-1)
    return y

def _mask_set(x: torch.Tensor, m: torch.BoolTensor, y: torch.Tensor):
    x = x.clone()
    x[m] = y
    return x

class CogVLMModel(CogVLMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([CogVLMDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.vision = EVA2CLIPModel(config)

        self.gradient_checkpointing = False
        self._gradient_checkpointing_func = None
        # Initialize weights and apply final processing
        self.post_init()

    def get_lora_modules(self, prefix: str):
        # Let's fine-tune the whole embedding layer
        target_modules, modules_to_save = [], [apply_prefix(prefix, 'embed_tokens')]
        for name, child in self.named_children():
            if name == 'embed_tokens':
                continue
            c_target_modules, c_modules_to_save = get_lora_modules_default(child, apply_prefix(prefix, name))
            target_modules.extend(c_target_modules)
            modules_to_save.extend(c_modules_to_save)
        return target_modules, modules_to_save

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        *,
        image: list[torch.Tensor] | None = None,
        patch_size: list[tuple3_t[int]] | None = None,
        pool_size: list[tuple3_t[int]],
        token_type_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """take care of image_encode, token_type_ids, position_ids and (attention_mask = None is fine)"""
        if past_key_values is not None:
            pass  # generate mode with past_key_values. the image features are already mapped
        else:
            # not allow for inputs_embeds, because we want to process image feature
            assert input_ids is not None and inputs_embeds is None
            if image is not None:  # multi-modality
                assert token_type_ids is not None, f"multi-modality requires `token_type_ids`!"
                assert len(input_ids) == len(image), f"batch size mismatch: {len(input_ids)} {len(image)}"
                inputs_embeds = self.embed_tokens(input_ids)
                image_features_list: list[torch.Tensor] = self.vision(image, patch_size, pool_size)
                for i, image_features in enumerate(image_features_list):
                    inputs_embeds[i, 1:1 + image_features.shape[1]] = image_features[0]
                    inputs_embeds[i, 1:1 + image_features.shape[1]] = image_features[0]
            else:  # single-modality
                if token_type_ids is None:
                    token_type_ids = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device) * LANGUAGE_TOKEN_TYPE
                assert not (token_type_ids == VISION_TOKEN_TYPE).any(), f"unexpected vision tokens for single-modality: {(token_type_ids == VISION_TOKEN_TYPE).sum()}"
                inputs_embeds = self.embed_tokens(input_ids)

            # if position_ids is None:
            #     position_ids = build_position_ids(token_type_ids, attention_mask)
            input_ids = None

        return self.llm_forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    def llm_forward(
        self,
        input_ids: torch.LongTensor = None,
        token_type_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """largely copy from llama forward and adapt for cogvlm with `token_type_ids`"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        padding_mask = attention_mask.bool()
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            layer_outputs = forward_gc(
                decoder_layer,
                self.gradient_checkpointing,
                self._gradient_checkpointing_func,
                hidden_states,
                token_type_ids=token_type_ids,
                padding_mask=padding_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        if hidden_states.shape[1] > 1:
            hidden_states = _mask_set(hidden_states, padding_mask, self.norm(hidden_states[padding_mask]))
        else:
            hidden_states = self.norm(hidden_states)
        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

def _history_to_prompt(signal_type, history, query):
    if signal_type == 'base':
        return query
    elif signal_type == 'vqa':
        answer_format = 'Short answer:'
    elif signal_type == 'chat':
        answer_format = 'Answer:'
    else:
        assert False, f"Unknown signal type {signal_type}"

    prompt = ''
    for i, (old_query, response) in enumerate(history):
        prompt += 'Question: ' + old_query + " {} ".format(answer_format) + response + "\n"
    prompt += 'Question: {} {}'.format(query, answer_format)
    return prompt

def _sample_weighted_ce(logits: torch.FloatTensor, labels: torch.LongTensor, weight: torch.Tensor | None):
    """
    Args:
        weight: weight for each sample, not for class
    """
    logits = logits.view(-1, logits.shape[-1])
    labels = labels.view(-1)
    if weight is None:
        return F.cross_entropy(logits, labels)
    mask: torch.BoolTensor = labels != CE_IGNORE_INDEX
    ce = F.cross_entropy(logits, labels, reduction='none')
    # the life of weight:
    # 0. constructed as fp32 in DataModule
    # 1. converted to bf16 by precision plugin
    # 2. convert back to fp32 here
    # stupid and meaningless? yes
    ret = torch.dot(ce[mask], weight.float().view(-1)[mask]) / mask.sum()
    return ret

class CogVLMForCausalLM(CogVLMPreTrainedModel):
    _auto_class = "AutoModelForCausalLM"

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.model = CogVLMModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        *,
        image: list[torch.Tensor] = None,
        patch_size: list[tuple3_t[int]] = None,
        pool_size: list[tuple3_t[int]],
        token_type_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        weight: torch.Tensor | None = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        output: BaseModelOutputWithPast = self.model(
            input_ids,
            image=image,
            patch_size=patch_size,
            pool_size=pool_size,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )

        logits = self.lm_head(output.last_hidden_state).float()

        loss = None
        if labels is not None:
            # NOTE: the labels were already shifted in MMMMDataModule, not shifting again here
            loss = _sample_weighted_ce(logits, labels, weight)

        ret = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=output.past_key_values,
            hidden_states=output.hidden_states if output_hidden_states else None,
            attentions=output.attentions,
        )
        return ret if return_dict else ret.to_tuple()

    def _prepare_attention_mask_for_generation(
        self,
        inputs: torch.Tensor,
        pad_token_id: Optional[int],
        eos_token_id: Optional[Union[int, List[int]]],
    ) -> torch.LongTensor:
        return torch.ones(inputs.shape[:2], dtype=torch.long, device=inputs.device)  # type: ignore

    def prepare_inputs_for_generation(
        self,
        input_ids,
        token_type_ids,
        images=None,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        # build position_ids if needed
        position_ids = kwargs.get("position_ids", None)
        if position_ids is None:
            position_ids = torch.zeros_like(input_ids)
            for i in range(input_ids.shape[0]):
                _mask = attention_mask[i].bool()
                position_ids[i, _mask] = build_position_ids(token_type_ids[i, _mask][None])[0]

        if past_key_values:
            input_ids = input_ids[:, -1:]
            token_type_ids = token_type_ids[:, -1:]
            position_ids = position_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "images": images,
                "token_type_ids": token_type_ids,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "attention_mask": attention_mask,
                "use_cache": kwargs.get("use_cache"),
            }
        )
        return model_inputs

    def _update_model_kwargs_for_generation(
            self,
            outputs: "ModelOutput",
            model_kwargs: Dict[str, Any],
            is_encoder_decoder: bool = False,
            standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        token_type_ids = model_kwargs.pop('token_type_ids')
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder, standardize_cache_format,
        )
        new_token_type_ids = torch.ones(size=(token_type_ids.shape[0], 1), dtype=token_type_ids.dtype,
                                        device=token_type_ids.device) * LANGUAGE_TOKEN_TYPE
        model_kwargs["token_type_ids"] = torch.cat([token_type_ids, new_token_type_ids], dim=-1)
        return model_kwargs

    def _reorder_cache(self, past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past

    def build_conversation_input_ids(
        self,
        tokenizer: "PreTrainedTokenizer",
        *,
        query: str,
        history: Optional[List[Tuple[str, str]]] = None,
        images: Optional[List["PIL.Image"]] = None,
        template_version: Optional[Literal["base", "chat", "vqa"]] = None,
    ):
        image_size: int = self.config.vision_config['image_size']
        patch_size: int = self.config.vision_config['patch_size']
        template_version = template_version or self.config.template_version
        assert images is None or len(images) <= 1, f"not support multi images by now."
        history = history or []
        text = _history_to_prompt(template_version, history, query)

        input_ids = [tokenizer.bos_token_id]
        token_type_ids = [LANGUAGE_TOKEN_TYPE]
        if images is not None and len(images) == 1:
            # vision
            transform = transforms.Compose(
                [
                    transforms.Resize(
                        (image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                ]
            )
            images = [transform(images[0])]
            # language
            vision_token_num = (image_size // patch_size) * (image_size // patch_size) + 2
            input_ids += [tokenizer.pad_token_id] * vision_token_num
            token_type_ids += [VISION_TOKEN_TYPE] * vision_token_num
        text_ids = tokenizer.encode(text, add_special_tokens=False)

        input_ids += text_ids
        token_type_ids += [LANGUAGE_TOKEN_TYPE] * len(text_ids)
        attention_mask = [1] * len(input_ids)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'images': images,
        }
