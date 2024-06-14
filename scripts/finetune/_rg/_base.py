from typing import Callable

import torch
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.v2 import functional as tvtf
from transformers import PreTrainedTokenizer

from scripts.finetune._utils import intensity_norm_, _pad_inputs, CE_IGNORE_INDEX
from luolib.data.utils import list_data_collate
from luolib.datamodule import ExpDataModuleBase
from luolib.types import tuple2_t
from mmmm.data.dataset.vl import get_vl_data_list
from mmmm.data.defs import Split
from monai import transforms as mt

class RGTransform(mt.Randomizable):
    def __init__(self, tokenizer: PreTrainedTokenizer, resize: tuple2_t[int], dataset_name: str):
        super().__init__()
        self.tokenizer = tokenizer
        self.resize = resize
        self.dataset_name = dataset_name

    def __call__(self, data):
        image = read_image(data['image'], ImageReadMode.RGB)
        image = tvtf.to_dtype(image, torch.float32, scale=True)
        image = tvtf.resize(image, self.resize)
        intensity_norm_(image)

        answer = data['processed_report']
        tokenizer = self.tokenizer
        text_ids = []
        labels = []

        prompt = f'Please write a radiology report for me:'
        prompt_ids = torch.tensor(tokenizer.encode(prompt, add_special_tokens=False))
        answer_ids = torch.tensor(tokenizer.encode(answer, add_special_tokens=False))
        text_ids.extend([prompt_ids, answer_ids])

        labels.extend(
            [
                torch.full((prompt_ids.shape[0], ), CE_IGNORE_INDEX),
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


class RGDataModule(ExpDataModuleBase):
    tokenizer: PreTrainedTokenizer

    def __init__(self, *, dataset_name: str, resize: tuple2_t[int], **kwargs):
        super().__init__(**kwargs)
        self.dataset_name = dataset_name
        self.resize = resize

    def train_data(self):
        data_list = get_vl_data_list(self.dataset_name, Split.TRAIN)
        new_data_list = []
        if self.dataset_name == "MIMIC-CXR":
            plane_list = ['AP', 'PA']
        elif self.dataset_name == "OpenI":
            plane_list = ['frontal']
        elif self.dataset_name == "CT-RATE":
            for data in data_list:
                image_idx = 0
                data['image'] = data['image'][image_idx]
                new_data_list.append(data)
            return new_data_list
        for data in data_list:
            in_plane_list = False
            for plane in plane_list:
                if plane in data['plane']:
                    in_plane_list = True
            if not in_plane_list:
                continue
            image_idx = 10
            for plane in plane_list:
                if plane in data['plane']:
                    image_idx = min(image_idx, data['plane'].index(plane))
            data['image'] = data['image'][image_idx]
            new_data_list.append(data)
        return new_data_list

    def train_transform(self) -> Callable:
        return RGTransform(self.tokenizer, self.resize,  self.dataset_name)

    def _collate_fn(self, batch: list[dict]):
        batch_vlm_inputs: list[dict] = []
        for x in batch:
            batch_vlm_inputs.append(x.pop('vlm_inputs'))
        ret = {
            **list_data_collate(batch),
            'vlm_inputs': _pad_inputs(batch_vlm_inputs, self.tokenizer.pad_token_id),
        }
        return ret

    def get_train_collate_fn(self):
        return self._collate_fn
