from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import h5py
import numpy as np
from numpy import typing as npt
import pandas as pd
import torch
from torch.types import Device
from transformers import PreTrainedTokenizer

from luolib.datamodule import ExpDataModuleBase
from luolib.types import tuple3_t
from monai import transforms as mt

from mmmm.data.dataset import IGNORE_INDEX
from mmmm.data.defs import Meta, PROCESSED_SEG_DATA_ROOT, encode_patch_size
from mmmm.model.cogvlm.modeling_cogvlm import LANGUAGE_TOKEN_TYPE, VISION_TOKEN_TYPE
from mmmm.model.tokenizer import MMMMTokenizer

__all__ = [
    'MMMMDataModule',
]

class SamplePatch(mt.RandomizableTransform):
    def __init__(
        self,
        patch_size: tuple3_t[int],
        num_pos: int,
        num_neg: int,
        tokenizer: MMMMTokenizer,
        force_fg_ratio: float = 1 / 3,
        device: Device = 'cpu',
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_pos = num_pos
        self.num_neg = num_neg
        self.tokenizer = tokenizer
        self.force_fg_ratio = force_fg_ratio
        self.device = device
        self.pad = mt.SpatialPadD(['image', 'masks'], patch_size, lazy=True)

    def gen_conversation(self, modality: str, classes: list[str], pos_classes: list[str], neg_classes: list[str]):
        def _convert_list(names: Iterable[str], mask: bool):
            # FIXME: do not use special tokens explicitly in text
            if mask:
                names = map(lambda name: f'{tokenizer.mask_open} {name} {tokenizer.mask_close}', names)
            return ', '.join(names)

        tokenizer = self.tokenizer
        assert len(classes) > 0
        neg_mask = self.R.uniform() < 0.9
        if neg_mask:
            prompt = f"For the given {modality} image, find the following objects, and output segmentation masks for the found objects: {_convert_list(classes, False)}. "
        else:
            prompt = f'For the given {modality} image, output the segmentation masks for the following objects: {_convert_list(classes, False)}.'
        if len(pos_classes) > 0:
            if len(neg_classes) > 0:
                response = f'The following objects are found: {_convert_list(pos_classes, True)}. ' + \
                           f'The following objects are not found: {_convert_list(neg_classes, neg_mask)}. '
            else:
                response = f'All of the requested objects are found: {_convert_list(pos_classes, True)}. '
        else:
            if neg_mask:
                response = f'None of the requested objects are found: {_convert_list(classes, True)}. '
            else:
                response = 'None of the requested objects are found. '
        return [(prompt, response)]

    def prepare_vlm_inputs(self, conversation: list[tuple[str, str]]):
        # TODO: refactor this function to support various VLM formats
        tokenizer = self.tokenizer
        # template: CogVLM `chat_old_history_to_prompt`
        # just for viewing, don't tokenize it directly
        text = '\n'.join(
            f'{tokenizer.inst_open} {query} {tokenizer.inst_close} {answer}'
            for query, answer in conversation
        )
        dtype = torch.long
        text_ids = []
        labels = []
        for query, answer in conversation:
            prompt = f'{tokenizer.inst_open} {query} {tokenizer.inst_close}'
            prompt_ids = torch.tensor(tokenizer.encode(prompt, add_special_tokens=False))
            answer_ids = torch.tensor(tokenizer.encode(answer, add_special_tokens=False))
            text_ids.append(torch.cat([prompt_ids, answer_ids]))
            labels.append(
                torch.cat([
                    torch.full((prompt_ids.shape[0] - 1, ), IGNORE_INDEX, dtype=dtype),
                    answer_ids,
                    torch.tensor([tokenizer.eos_token_id]),
                ]),
            )
        text_ids = torch.cat(text_ids)
        labels = torch.cat(labels)

        # text_ids = torch.tensor(tokenizer.encode(text, add_special_tokens=False))
        # TODO: dynamically adjust patch size according to image spacing
        num_vision_tokens = np.prod([s // 16 for s in self.patch_size]).item() + 2  # including boi and eoi
        input_ids = torch.cat([
            torch.tensor([tokenizer.bos_token_id]),
            torch.full((num_vision_tokens, ), 0, dtype=dtype),
            text_ids,
        ])
        token_type_ids = torch.cat([
            torch.tensor([LANGUAGE_TOKEN_TYPE]),
            torch.full((num_vision_tokens, ), VISION_TOKEN_TYPE, dtype=dtype),
            torch.full(text_ids.shape, LANGUAGE_TOKEN_TYPE, dtype=dtype),
        ])
        position_ids = torch.cat([
            torch.tensor([0, 1]),  # bos and boi
            torch.full((num_vision_tokens - 2, ), 2, dtype=dtype),
            torch.tensor([3]),  # eoi
            torch.arange(4, 4 + text_ids.shape[0]),
        ])
        labels = torch.cat([torch.full((1 + num_vision_tokens, ), IGNORE_INDEX, dtype=dtype), labels])
        return {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'position_ids': position_ids,
            'lm_labels': labels,
        }

    def __call__(self, data: dict):
        dataset_dir: Path = data['dataset_dir']
        key: str = data['key']
        meta: Meta = data['meta']
        patch_dir = dataset_dir / 'patch' / encode_patch_size(self.patch_size) / key
        # all positive classes on the whole image
        all_pos_classes = meta['positive_classes']
        positive_mask: npt.NDArray[np.bool_] = np.load(patch_dir / 'positive_mask.npy', 'r')

        # sample patch position
        position: npt.NDArray[np.int16]
        if self.R.uniform() < self.force_fg_ratio:
            # foreground oversampling
            c = self.R.randint(len(all_pos_classes))
            with h5py.File(patch_dir / 'class_positions.h5') as f:
                positions: h5py.Dataset = f[str(c)]
                position = positions[self.R.randint(positions.shape[0])]
        else:
            # sample a random patch position
            c = None
            position = np.array([self.R.randint(s) for s in positive_mask.shape[:-1]], dtype=np.int16)
        positive_mask = np.array(positive_mask[*position, :])

        # sample negative classes
        neg_class_ids, = (~positive_mask).nonzero()
        neg_classes = [all_pos_classes[i] for i in neg_class_ids] + meta['negative_classes']
        neg_class_ids = self.R.choice(min(len(neg_classes), self.num_neg), self.num_neg, replace=False)
        neg_classes = [neg_classes[i] for i in neg_class_ids]

        # sample positive classes
        if c is not None:
            positive_mask[c] = False
        pos_class_ids, = positive_mask.nonzero()
        num_pos = self.num_pos - (c is not None)
        pos_class_ids = self.R.choice(pos_class_ids, min(pos_class_ids.shape[0], num_pos), replace=False)
        if c is not None:
            pos_class_ids = np.insert(pos_class_ids, self.R.randint(pos_class_ids.shape[0] + 1), c)
        pos_classes = [all_pos_classes[i] for i in pos_class_ids]

        # merge positive and negative classes
        pos_class_mask = torch.zeros(len(pos_classes) + len(neg_classes), dtype=torch.bool)
        pos_class_mask[self.R.choice(pos_class_mask.shape[0], len(pos_classes), replace=False)] = True
        pos_it, neg_it = map(iter, [pos_classes, neg_classes])
        classes = [
            next(pos_it) if m else next(neg_it)
            for m in pos_class_mask
        ]

        # construct image & masks patch
        data_dir = dataset_dir / 'data' / key
        modalities = meta['modalities']
        modality_id = self.R.randint(len(modalities))
        patch_slice = [
            slice(p, p + s)
            for p, s in zip(position, self.patch_size)
        ]
        image = np.load(data_dir / 'images.npy', 'r')[modality_id:modality_id + 1, *patch_slice]
        image = torch.as_tensor(np.array(image), device=self.device)
        masks = np.load(data_dir / 'masks.npy', 'r')[:, *patch_slice]
        masks = masks[pos_class_ids]
        masks = torch.as_tensor(np.array(masks), device=self.device)

        # prepare sample output
        modality = modalities[modality_id]
        conversation = self.gen_conversation(modality, classes, pos_classes, neg_classes)
        vlm_inputs = self.prepare_vlm_inputs(conversation)
        data = {
            'image': image,
            'modality': modality,
            'masks': masks,
            # FIXME: https://github.com/pytorch/pytorch/issues/13246
            'requested_classes': classes,
            'pos_class_mask': pos_class_mask,
            **vlm_inputs,
        }
        return self.pad(data)

@dataclass
class TransConf:
    patch_size: tuple3_t[int] = (96, 224, 224)
    num_pos: int = 10
    num_neg: int = 10

class MMMMDataModule(ExpDataModuleBase):
    def __init__(
        self,
        trans: TransConf,
        tokenizer: PreTrainedTokenizer,
        data_root: Path = PROCESSED_SEG_DATA_ROOT,
        *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.data_root = data_root
        self._train_data = []
        self.trans_conf = trans
        for dataset_dir in data_root.iterdir():
            dataset_meta: pd.DataFrame = pd.read_pickle(dataset_dir / 'meta.pkl')
            self._train_data.extend([
                {
                    'dataset_dir': dataset_dir,
                    'key': key,
                    'meta': meta,
                }
                for key, meta in dataset_meta.iterrows()
            ])

    def train_data(self) -> Sequence:
        return self._train_data

    def train_transform(self) -> Callable:
        conf = self.trans_conf
        return SamplePatch(conf.patch_size, conf.num_pos, conf.num_neg)

    def get_train_collate_fn(self):
        from luolib.data.utils import list_data_collate

        def collate_fn(batch: list[dict]):
            list_data = {key: [] for key in ['requested_classes', 'pos_class_mask', 'masks']}
            for x in batch:
                for key in list_data:
                    list_data[key].append(x.pop(key))
            ret = list_data_collate(batch)
            ret.update(list_data)
            return ret

        return collate_fn
