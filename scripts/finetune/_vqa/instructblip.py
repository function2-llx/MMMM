from typing import Callable

import torch
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.v2 import functional as tvtf
from transformers import InstructBlipForConditionalGeneration, PreTrainedTokenizer, AutoTokenizer

from _utils import intensity_norm_, _pad_inputs, CE_IGNORE_INDEX
from _vqa._base import VQATransform, VQADataModule
from luolib.data.utils import list_data_collate

from luolib.lightning import LightningModule


class FinetuneInstructBlip(LightningModule):
    def __init__(self, *, model_path: str):
        super().__init__()
        self.instructblip_model: InstructBlipForConditionalGeneration = InstructBlipForConditionalGeneration.from_pretrained(model_path)
        for name, param in self.instructblip_model.named_parameters():
            if 'qformer' not in name:
                param.requires_grad = False

    def training_step(self, batch: dict, *args, **kwargs):
        from transformers.models.instructblip.modeling_instructblip import InstructBlipForConditionalGenerationModelOutput
        outputs: InstructBlipForConditionalGenerationModelOutput = self.instructblip_model.forward(
            pixel_values=batch['image'],
            **{
                f'qformer_{k}': v
                for k, v in batch['qformer_inputs'].items()
            },
            **batch['vlm_inputs'],
            return_dict=True,
        )
        loss = outputs.loss
        self.log('train/loss', loss)
        return loss


class IBLIPVQATransform(VQATransform):
    def __init__(self, *args, qformer_tokenizer: PreTrainedTokenizer, **kwargs):
        super().__init__(*args, **kwargs)
        self.qformer_tokenizer = qformer_tokenizer

    def __call__(self, data):
        image = read_image(data['image'], ImageReadMode.RGB)
        image = tvtf.to_dtype(image, torch.float32, scale=True)
        image = tvtf.resize(image, self.resize)
        intensity_norm_(image)

        pairs = [(qa['question'], qa['answer']) for qa in data['vqa']]
        self.R.shuffle(pairs)
        tokenizer = self.tokenizer
        query, answer = pairs[0]
        prompt = f'Question: {query} Answer:'
        prompt_ids = torch.tensor(tokenizer.encode(prompt, add_special_tokens=False))
        answer_ids = torch.tensor(tokenizer.encode(answer, add_special_tokens=False))
        input_ids = torch.cat([
            torch.tensor([tokenizer.bos_token_id]),
            prompt_ids,
            answer_ids,
            torch.tensor([tokenizer.eos_token_id]),
        ])
        labels = torch.cat([
            torch.tensor([tokenizer.bos_token_id]),
            torch.full((prompt_ids.shape[0], ), CE_IGNORE_INDEX),
            answer_ids,
            torch.tensor([tokenizer.eos_token_id]),
        ])

        qformer_input_ids = torch.tensor(self.qformer_tokenizer.encode(prompt, add_special_tokens=False))  # TODO: check add_special_tokens?
        return {
            'image': image,
            'vlm_inputs': {
                'input_ids': input_ids,
                'labels': labels,
                'attention_mask': torch.ones_like(input_ids),
            },
            'qformer_inputs': {
                'input_ids': qformer_input_ids,
                'attention_mask': torch.ones_like(qformer_input_ids),
            }
        }


class IBLIPVQADataModule(VQADataModule):
    def __init__(self, *args, qformer_tokenizer: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.qformer_tokenizer = AutoTokenizer.from_pretrained(qformer_tokenizer)

    def train_transform(self) -> Callable:
        return IBLIPVQATransform(self.tokenizer, qformer_tokenizer=self.qformer_tokenizer, resize=self.resize)

    def _collate_fn(self, batch: list[dict]):
        dict_keys = ['vlm_inputs', 'qformer_inputs']
        dict_data = {key: [] for key in dict_keys}
        for x in batch:
            for key in dict_keys:
                dict_data[key].append(x.pop(key))
        ret = {
            **list_data_collate(batch),
            'vlm_inputs': _pad_inputs(dict_data['vlm_inputs'], self.tokenizer.pad_token_id),
            'qformer_inputs': _pad_inputs(dict_data['qformer_inputs'], self.qformer_tokenizer.pad_token_id),
        }
        return ret
