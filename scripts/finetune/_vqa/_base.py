from typing import Callable

import torch
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.v2 import functional as tvtf
from transformers import PreTrainedTokenizer

from luolib.data.utils import list_data_collate
from luolib.datamodule import ExpDataModuleBase
from luolib.types import tuple2_t
from monai import transforms as mt

from mmmm.data.dataset.vl import get_vl_data_list
from mmmm.data.defs import Split
from _utils import intensity_norm_, _pad_inputs, CE_IGNORE_INDEX

class VQATransform(mt.Randomizable):
    def __init__(self, tokenizer: PreTrainedTokenizer, resize: tuple2_t[int], max_seq_len: int | None):
        super().__init__()
        self.tokenizer = tokenizer
        self.resize = resize
        self.max_seq_len = max_seq_len

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
            prompt_ids = torch.tensor(tokenizer.encode(prompt, add_special_tokens=False))
            answer_ids = torch.tensor(tokenizer.encode(answer, add_special_tokens=False))
            text_ids.extend([prompt_ids, answer_ids])
            if i > 0:
                labels.extend([
                    torch.tensor([tokenizer.eos_token_id]),
                    torch.full((prompt_ids.shape[0] - 1, ), CE_IGNORE_INDEX),
                    answer_ids,
                ])
            else:
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
        if self.max_seq_len is not None:
            input_ids = input_ids[:self.max_seq_len]
            labels = labels[:self.max_seq_len]
        return {
            'image': image,
            'vlm_inputs': {
                'input_ids': input_ids,
                'labels': labels,
                'attention_mask': torch.ones_like(input_ids),
            },
        }


class VQADataModule(ExpDataModuleBase):
    tokenizer: PreTrainedTokenizer

    def __init__(self, *, dataset_name: str, resize: tuple2_t[int], max_seq_len: int | None = None, **kwargs):
        super().__init__(**kwargs)
        self.dataset_name = dataset_name
        self.resize = resize
        self.max_seq_len = max_seq_len

    def train_data(self):
        data_list = get_vl_data_list(self.dataset_name, Split.TRAIN)
        for data in data_list:
            assert len(data['image']) == 1
            data['image'] = data['image'][0]
        return data_list

    def train_transform(self) -> Callable:
        return VQATransform(self.tokenizer, self.resize, self.max_seq_len)

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
