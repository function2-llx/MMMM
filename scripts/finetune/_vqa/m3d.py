from typing import Callable

import einops
import torch
import torch.nn as nn
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.v2 import functional as tvtf
from transformers import AutoModelForCausalLM, PreTrainedModel

from luolib.lightning import LightningModule
from scripts.finetune._utils import CE_IGNORE_INDEX
from scripts.finetune._vqa._base import VQATransform, VQADataModule


class FinetuneM3D(LightningModule):
    def __init__(self, *, model_path: str):
        super().__init__()
        dtype = torch.bfloat16
        self.m3d_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        self.target_modules, self.modules_to_save = self.get_lora_modules_default(self.m3d_model)
        self.m3d_model.gradient_checkpointing_enable({'use_reentrant': False})
        self.train()

    def get_lora_modules_default(self, module: nn.Module, prefix: str = '', recursive: bool = True):
        target_modules, modules_to_save = [], []

        def dfs(m: nn.Module, prefix: str):
            if isinstance(m, nn.Linear):
                target_modules.append(prefix)  # Linear layers as LoRA targets
            elif isinstance(m, nn.Embedding):
                modules_to_save.append(prefix)  # Embedding layers to save

            for name, child in m.named_children():
                dfs(child, f"{prefix}.{name}" if prefix else name)

        dfs(module, prefix)
        return target_modules, modules_to_save

    def training_step(self, batch, *args, **kwargs):
        outputs = self.m3d_model(
            input_ids=batch['vlm_inputs']['input_ids'],
            images=batch['image'],
            attention_mask=batch['vlm_inputs']['attention_mask'],
            labels=batch['vlm_inputs']['labels'],
        )
        loss = outputs.loss
        self.log('train/loss', loss)
        return loss

class M3DVQATransform(VQATransform):
    def __call__(self, data):
        dtype = torch.bfloat16
        image = read_image(data['image'], ImageReadMode.GRAY)
        image = tvtf.to_dtype(image, torch.float32, scale=True)
        image = tvtf.resize(image, self.resize)
        image = einops.repeat(image, '1 h w -> 1 d h w', d=32)
        image = image.to(dtype=dtype)

        proj_out_num = 256

        pairs = [(qa['question'], qa['answer']) for qa in data['vqa']]
        self.R.shuffle(pairs)
        tokenizer = self.tokenizer
        text_ids = []
        labels = []
        for i, (query, answer) in enumerate(pairs):
            prompt = f'Question: {query} Answer:'
            if i == 0:
                prompt = "<im_patch>" * proj_out_num + prompt
            prompt_ids = torch.tensor(tokenizer.encode(prompt, add_special_tokens=False))
            answer_ids = torch.tensor(tokenizer.encode(answer, add_special_tokens=False))
            text_ids.extend([prompt_ids, answer_ids])
            if i > 0:
                labels.extend([
                    torch.tensor([tokenizer.eos_token_id]),
                    torch.full((prompt_ids.shape[0] - 1,), CE_IGNORE_INDEX),
                    answer_ids,
                ])
            else:
                labels.extend(
                    [
                        torch.full((prompt_ids.shape[0],), CE_IGNORE_INDEX),
                        answer_ids,
                    ],
                )
        input_ids = torch.cat([
            torch.tensor([tokenizer.bos_token_id]),
            *text_ids,
            torch.tensor([tokenizer.eos_token_id]),
        ])
        labels = torch.cat([
            # the first item will be shifted
            torch.tensor([tokenizer.bos_token_id]),
            *labels,
            torch.tensor([tokenizer.eos_token_id]),
        ])
        return {
            'image': image,
            'vlm_inputs': {
                'input_ids': input_ids,
                'labels': labels,
                'attention_mask': torch.ones_like(input_ids),
            }
        }


class M3DVQADataModule(VQADataModule):
    def train_transform(self) -> Callable:
        return M3DVQATransform(self.tokenizer, self.resize, self.max_seq_len)
