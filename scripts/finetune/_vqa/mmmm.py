from typing import Callable

import einops
import torch
import torch.nn as nn
from peft import PeftModel
from transformers import AutoTokenizer

from _vqa._base import VQADataModule, VQATransform
from luolib.lightning import LightningModule
from mmmm.models.mmmm import from_pretrained, MMMMForCausalLM, MMMMTokenizer

LANGUAGE_TOKEN_TYPE = 0
VISION_TOKEN_TYPE = 1

class FinetuneMMMM(LightningModule):
    def __init__(self, *, model_path: str):
        super().__init__()

        adapter_path = "/data/MMMM/output/vlm-post/run-20240615_035211-cjw9e7jo/checkpoint/step=6000.ckpt/adapter"
        model, tokenizer = from_pretrained('conf/model.yaml', adapter_path, True)

        self.mmmm_model: MMMMForCausalLM = model.to('cuda')
        # self.target_modules, self.modules_to_save = self.get_lora_modules_default(self.mmmm_model)

        # pos_embed = self.mmmm_model.model.vision.patch_embedding.position_embedding.weight
        # cls_pos_embed, pos_embed = pos_embed[0:1], pos_embed[1:]
        # pos_embed = einops.rearrange(pos_embed, '(h w) c -> 1 c h w', h=35, w=35)
        # import torch.nn.functional as nnf
        # pos_embed = nnf.interpolate(pos_embed, (16, 16), mode='area')
        # pos_embed = torch.cat([cls_pos_embed, einops.rearrange(pos_embed, '1 c h w ->(h w) c')])
        # self.mmmm_model.model.vision.patch_embedding.position_embedding = nn.Embedding(
        #     *pos_embed.shape[:2], _weight=pos_embed,
        # )

    # def get_lora_modules_default(self, module: nn.Module, prefix: str = '', recursive: bool = True):
    #     target_modules, modules_to_save = [], []
    #
    #     def dfs(m: nn.Module, prefix: str):
    #         if isinstance(m, nn.Linear):
    #             target_modules.append(prefix)  # Linear layers as LoRA targets
    #         elif isinstance(m, nn.Embedding):
    #             modules_to_save.append(prefix)  # Embedding layers to save
    #
    #         for name, child in m.named_children():
    #             dfs(child, f"{prefix}.{name}" if prefix else name)
    #
    #     dfs(module, prefix)
    #     return target_modules, modules_to_save

    def training_step(self, batch, *args, **kwargs):
        input_ids = batch['vlm_inputs']['input_ids']
        num_vision_tokens = 12 * 12 + 2
        seq_len = input_ids.shape[1]
        token_type_ids = torch.full(
            (input_ids.shape[0], num_vision_tokens + seq_len), LANGUAGE_TOKEN_TYPE, device=self.device,
        )
        token_type_ids[:, 1:1 + num_vision_tokens] = VISION_TOKEN_TYPE
        new_input_ids = torch.zeros_like(token_type_ids)
        new_input_ids[token_type_ids == LANGUAGE_TOKEN_TYPE] = input_ids.view(-1)
        new_attn_mask = torch.ones_like(token_type_ids)
        new_attn_mask[token_type_ids == LANGUAGE_TOKEN_TYPE] = batch['vlm_inputs']['attention_mask'].view(-1)
        new_labels = torch.full_like(token_type_ids, -100)
        new_labels[token_type_ids == LANGUAGE_TOKEN_TYPE] = batch['vlm_inputs']['labels'].view(-1)
        batch_size = new_input_ids.shape[0]
        outputs = self.mmmm_model.forward(
            input_ids=new_input_ids,
            token_type_ids=token_type_ids,
            attention_mask=new_attn_mask,
            labels=new_labels,
            image=[x for x in batch['image'][:, :, None]],
            patch_size=[(1, 16, 16)] * batch_size,
            pool_size=[(1, 2, 2)] * batch_size,
        )
        loss = outputs.loss
        self.log('train/loss', loss)
        return loss

class MMMMVQADataModule(VQADataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_transform(self) -> Callable:
        self.tokenizer = MMMMTokenizer.build("lmsys/vicuna-7b-v1.5")
        return VQATransform(self.tokenizer, resize=self.resize, max_seq_len=None)
