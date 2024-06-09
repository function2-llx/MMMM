from typing import Callable

import torch
from torchvision.transforms.v2 import functional as tvtf

from luolib.utils import load_pt_zst
from scripts.finetune._rg._base import RGTransform, RGDataModule
from scripts.finetune._utils import CE_IGNORE_INDEX, intensity_norm_
from torchvision.io import read_image, ImageReadMode

class LLAVAMRGTransform(RGTransform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, data):
        if self.dataset_name == "MIMIC-CXR":
            image = load_pt_zst(data['image'])
            image = torch.squeeze(image, 1)
            image = image.expand(3, image.shape[1], image.shape[2])
        elif self.dataset_name == "OpenI":
            image = read_image(data['image'], ImageReadMode.RGB)
        image = tvtf.to_dtype(image, torch.float32, scale=True)
        image = tvtf.resize(image, self.resize)
        intensity_norm_(image)

        answer = data['processed_report']
        tokenizer = self.tokenizer
        text_ids = []
        labels = []

        prompt = f'<image>\nPlease write a radiology report for me:'
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
        return {
            'image': image,
            'vlm_inputs': {
                'input_ids': input_ids,
                'labels': labels,
                'attention_mask': torch.ones_like(input_ids),
            }
        }


class LLAVAMRGDataModule(RGDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_transform(self) -> Callable:
        return LLAVAMRGTransform(self.tokenizer, resize=self.resize, dataset_name=self.dataset_name)