from typing import Callable

import torch
from PIL import Image
from einops import rearrange, repeat
from torchvision import transforms

from _rg._base import RGDataModule, RGTransform
from luolib.utils import load_pt_zst
from scripts.finetune._utils import CE_IGNORE_INDEX


class RadFMRGTransform(RGTransform):
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


        answer = data['processed_report']
        tokenizer = self.tokenizer
        text_ids = []
        labels = []

        prompt = '<image>' + ''.join([f'<image{i}>' for i in range(32)]) + '</image>' + 'Please write a radiology report for me:'
        prompt_ids = torch.tensor(tokenizer.encode(prompt, add_special_tokens=False))
        answer_ids = torch.tensor(tokenizer.encode(answer, add_special_tokens=False))
        text_ids.extend([prompt_ids, answer_ids])

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
        if self.max_seq_len is not None:
            input_ids = input_ids[:self.max_seq_len]
            labels = labels[:self.max_seq_len]
        return {
            'image': image,
            'vlm_inputs': {
                'input_ids': input_ids,
                'labels': labels,
                'attention_mask': torch.ones_like(input_ids),
                'loss_reweight': torch.ones_like(labels)
            }
        }


class RadFMRGDataModule(RGDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_transform(self) -> Callable:
        return RadFMRGTransform(self.tokenizer, resize=self.resize, dataset_name=self.dataset_name, max_seq_len=None)