from typing import Callable

import einops
import torch
import torch.nn.functional as F
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.v2 import functional as tvtf

from _rg._base import RGDataModule, RGTransform
from luolib.utils import load_pt_zst
from scripts.finetune._utils import CE_IGNORE_INDEX


class M3DRGTransform(RGTransform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, data):
        dtype = torch.bfloat16

        if self.dataset_name == "MIMIC-CXR" or self.dataset_name == "CT-RATE":
            image = load_pt_zst(data['image'])
            image = image.squeeze(0)
        elif self.dataset_name == "OpenI":
            image = read_image(data['image'], ImageReadMode.GRAY)
        image = tvtf.to_dtype(image, torch.float32, scale=True)
        image = tvtf.resize(image, self.resize)
        if self.dataset_name == "MIMIC-CXR" or self.dataset_name == "OpenI":
            image = einops.repeat(image, '1 h w -> 1 d h w', d=32)
        elif self.dataset_name == "CT-RATE":
            image = image.unsqueeze(0).unsqueeze(0)
            image = F.interpolate(image, size=(32, 256, 256), mode='trilinear', align_corners=False)
            image = image.squeeze(0)
        image = image.to(dtype=dtype)

        proj_out_num = 256

        answer = data['processed_report']
        tokenizer = self.tokenizer
        text_ids = []
        labels = []

        prompt = "<im_patch>" * proj_out_num + 'Please write a radiology report for me:'
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


class M3DRGDataModule(RGDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_transform(self) -> Callable:
        return M3DRGTransform(self.tokenizer, resize=self.resize, dataset_name=self.dataset_name)