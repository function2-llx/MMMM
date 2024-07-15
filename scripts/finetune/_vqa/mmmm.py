from pathlib import Path
from typing import Callable

import torch

from luolib.lightning import LightningModule
from luolib.lightning.peft import PeftMixin

from mmmm.models.mmmm import from_pretrained, MMMMForCausalLM, MMMMTokenizer

from _vqa._base import VQADataModule, VQATransform

LANGUAGE_TOKEN_TYPE = 0
VISION_TOKEN_TYPE = 1

class FinetuneMMMM(PeftMixin, LightningModule):
    def __init__(self, *, adapter_path: Path):
        super().__init__()

        model, tokenizer = from_pretrained('conf/model.yaml', adapter_path, True)

        self.mmmm_model: MMMMForCausalLM = model
        self.mmmm_model.gradient_checkpointing_enable({'use_reentrant': False})
        self.train()


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
