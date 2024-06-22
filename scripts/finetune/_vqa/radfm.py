from typing import Callable

import torch
import torch.nn as nn
from PIL import Image
from einops import rearrange, repeat
from torchvision import transforms

from RadFM.multimodality_model import MultiLLaMAForCausalLM
from luolib.lightning import LightningModule
from luolib.utils import load_pt_zst
from scripts.finetune._utils import CE_IGNORE_INDEX
from scripts.finetune._vqa._base import VQATransform, VQADataModule


class FinetuneRadFM(LightningModule):
    def __init__(self, *, model_path: str):
        super().__init__()
        self.radfm_model = MultiLLaMAForCausalLM(lang_model_path=model_path)
        checkpoint = torch.load(f'{model_path}/pytorch_model-bf16.bin', map_location='cpu')

        self.radfm_model.load_state_dict(checkpoint)

        self.radfm_model.to(torch.bfloat16)

        self.target_modules, self.modules_to_save = self.get_lora_modules_default(self.radfm_model)

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
        outputs = self.radfm_model(
            batch['vlm_inputs']['input_ids'],
            batch['image'],
            batch['vlm_inputs']['attention_mask'],
            batch['vlm_inputs']['labels'],
            batch['vlm_inputs']['loss_reweight'],
            None
        )
        loss = outputs['loss']
        self.log('train/loss', loss)
        return loss

class RadFMVQATransform(VQATransform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        special_tokens = {
            'additional_special_tokens': [f'<image{i}>' for i in range(32)] + ['<image>', '</image>']
        }
        self.tokenizer.add_special_tokens(special_tokens)
        self.tokenizer.pad_token_id = 0
        self.tokenizer.bos_token_id = 1
        self.tokenizer.eos_token_id = 2

    def __call__(self, data):
        if data['image'].endswith('.pt'):
            image = rearrange(torch.load(data['image']).float(), 'c d h w -> c h w d')
            image = (image - image.min()) / (image.max() - image.min())
        elif data['image'].endswith('.pt.zst'):
            image = rearrange(load_pt_zst(data['image']).float(), 'c d h w -> c h w d')
            image = (image - image.min()) / (image.max() - image.min())
        else:
            transform = transforms.ToTensor()
            image = Image.open(data['image']).convert('RGB')
            image = transform(image)

        target_d, max_d = 4, 4
        if len(image.shape) == 4:
            max_d = max(image.shape[3], max_d)
        for temp_d in range(4, 65, 4):
            if abs(temp_d - max_d) < abs(target_d - max_d):
                target_d = temp_d
        if len(image.shape) == 3:
            image = torch.nn.functional.interpolate(
                repeat(image, 'c h w -> 1 c h w 1'), size=(512, 512, target_d)
            )
        else:
            if image.shape[0] == 1:
                image = torch.nn.functional.interpolate(
                    repeat(image, '1 h w d -> 1 3 h w d'), size=(512, 512, target_d)
                )
            else:
                image = torch.nn.functional.interpolate(
                    repeat(image, 'c h w d -> 1 c h w d'), size=(512, 512, target_d)
                )

        pairs = [(qa['question'], qa['answer']) for qa in data['vqa']]
        self.R.shuffle(pairs)
        tokenizer = self.tokenizer
        text_ids = []
        labels = []
        for i, (query, answer) in enumerate(pairs):
            prompt = f'Question: {query} Answer:'
            if i == 0:
                prompt = '<image>' + ''.join([f'<image{i}>' for i in range(32)]) + '</image>' + prompt
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
                'loss_reweight': torch.ones_like(labels)
            }
        }


class RadFMVQADataModule(VQADataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_transform(self) -> Callable:
        return RadFMVQATransform(self.tokenizer, resize=self.resize, max_seq_len=None)