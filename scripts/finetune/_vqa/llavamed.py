from typing import Callable

import einops
import torch
import torch.nn as nn
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.v2 import functional as tvtf

from llava.mm_utils import get_model_name_from_path
from llava.model.builder import load_pretrained_model
from luolib.lightning import LightningModule
from scripts.finetune._utils import CE_IGNORE_INDEX, intensity_norm_
from scripts.finetune._vqa._base import VQATransform, VQADataModule


class FinetuneLlavaMed(LightningModule):
    def __init__(self, *, model_path: str):
        super().__init__()
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path,
            None,
            model_name
        )
        self.llavaM_model = model
        self.target_modules, self.modules_to_save = self.get_lora_modules_default(self.llavaM_model)
        self.llavaM_model.gradient_checkpointing_enable({"use_reentrant": False})
        self.train()

        pos_embed = self.llavaM_model.model.vision_tower.vision_tower.vision_model.embeddings.position_embedding.weight
        cls_pos_embed, pos_embed = pos_embed[0:1], pos_embed[1:]
        pos_embed = einops.rearrange(pos_embed, '(h w) c -> 1 c h w', h=24, w=24)
        import torch.nn.functional as nnf
        pos_embed = nnf.interpolate(pos_embed, (16, 16), mode='area')
        pos_embed = torch.cat([cls_pos_embed, einops.rearrange(pos_embed, '1 c h w ->(h w) c')])
        self.llavaM_model.model.vision_tower.vision_tower.vision_model.embeddings.position_embedding = nn.Embedding(
            *pos_embed.shape[:2], _weight=pos_embed,
        )
        self.llavaM_model.model.vision_tower.vision_tower.vision_model.embeddings.position_ids = torch.arange(257).expand((1, -1))

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
        outputs = self.llavaM_model.forward(
            input_ids=batch['vlm_inputs']['input_ids'],
            images=batch['image'],
            attention_mask=batch['vlm_inputs']['attention_mask'],
            labels=batch['vlm_inputs']['labels'],
        )
        loss = outputs.loss
        self.log('train/loss', loss)
        return loss

class LLAVAMVQATransform(VQATransform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, data):
        image = read_image(data['image'], ImageReadMode.RGB)
        image = tvtf.to_dtype(image, torch.float32, scale=True)
        image = tvtf.resize(image, self.resize)
        intensity_norm_(image)

        pairs = [(qa['question'], qa['answer']) for qa in data['vqa']]
        self.R.shuffle(pairs)
        tokenizer = self.tokenizer
        text_ids = []
        labels = []
        for i, (query, answer) in enumerate(pairs):
            prompt = f'Question: {query} Answer:'
            if i == 0:
                prompt = '<image>\n' + prompt
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


class LLAVAMVQADataModule(VQADataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_transform(self) -> Callable:
        return LLAVAMVQATransform(self.tokenizer, resize=self.resize)