from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import h5py
import numpy as np
from numpy import typing as npt
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.types import Device

from luolib.datamodule import ExpDataModuleBase
from luolib.types import tuple3_t
from luolib.utils.misc import ensure_rgb
from monai import transforms as mt
from monai.data import MetaTensor

from mmmm.models import MMMMTokenizer
from mmmm.models.cogvlm import LANGUAGE_TOKEN_TYPE, VISION_TOKEN_TYPE
from .defs import Meta, PROCESSED_SEG_DATA_ROOT, encode_patch_size

__all__ = [
    'MMMMDataModule',
    'CE_IGNORE_INDEX',
]

CE_IGNORE_INDEX = -100

class SamplePatch(mt.RandomizableTransform):
    def __init__(
        self,
        patch_size: tuple3_t[int],
        num_pos: int,
        num_neg: int,
        tokenizer: MMMMTokenizer,
        force_fg_ratio: float = 2 / 3,
        device: Device = 'cpu',
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_pos = num_pos
        self.num_neg = num_neg
        self.tokenizer = tokenizer
        self.force_fg_ratio = force_fg_ratio
        self.device = device
        self.pad_image = mt.SpatialPadD(['image', 'masks'], patch_size, lazy=True)

    def gen_conversation(self, modality: str, pos_classes: list[str], neg_classes: list[str]):
        def _convert_list(names: Iterable[str], mask: bool):
            # FIXME: do not use special tokens explicitly in text
            if mask:
                names = map(lambda name: f'{tokenizer.bop_token} {name} {tokenizer.eop_token} {tokenizer.seg_token}', names)
            return ', '.join(names)
        # copy the input list because the shuffling is in-place
        pos_classes = list(pos_classes)
        self.R.shuffle(pos_classes)
        neg_classes = list(neg_classes)
        self.R.shuffle(neg_classes)

        # merge positive and negative classes with random order without shuffling
        pos_class_mask = torch.zeros(len(pos_classes) + len(neg_classes), dtype=torch.bool)
        pos_class_mask[self.R.choice(pos_class_mask.shape[0], len(pos_classes), replace=False)] = True
        pos_it, neg_it = map(iter, [pos_classes, neg_classes])
        classes = [
            next(pos_it) if m else next(neg_it)
            for m in pos_class_mask
        ]

        tokenizer = self.tokenizer
        assert len(classes) > 0
        p_use_neg_mask = 0.9
        neg_mask = self.R.uniform() < p_use_neg_mask
        if neg_mask:
            prompt = f'For the given {modality} image, output the segmentation masks for the following objects: {_convert_list(classes, False)}.'
        else:
            prompt = f"For the given {modality} image, find the following objects, and output segmentation masks for the found objects: {_convert_list(classes, False)}. "
        if len(pos_classes) > 0:
            if len(neg_classes) > 0:
                response = f'The following objects are found: {_convert_list(pos_classes, True)}. ' + \
                           f'The following objects are not found: {_convert_list(neg_classes, neg_mask)}. '
                mask_classes = pos_classes + (neg_classes if neg_mask else [])
            else:
                response = f'All of the requested objects are found: {_convert_list(classes, True)}. '
                mask_classes = classes
        else:
            if neg_mask:
                response = f'None of the requested objects are found: {_convert_list(classes, True)}. '
                mask_classes = classes
            else:
                response = 'None of the requested objects are found. '
                mask_classes = []
        # mask_classes: the list of classes that with masks, following the order occurring in the conversation
        return [(prompt, response)], mask_classes

    def prepare_vlm_inputs(self, conversation: list[tuple[str, str]]):
        # TODO: refactor this function to support various VLM formats
        tokenizer = self.tokenizer
        # template: CogVLM `chat_old_history_to_prompt`
        # just for viewing, don't tokenize it directly
        user_start = 'Answer:'
        sys_start = 'Question:'
        text = '\n'.join(
            f'{user_start} {query} {sys_start} {answer}'
            for query, answer in conversation
        )
        dtype = torch.long
        text_ids = []
        lm_targets = []
        for query, answer in conversation:
            prompt = f'{user_start} {query} {sys_start}'
            prompt_ids = torch.tensor(tokenizer.encode(prompt, add_special_tokens=False))
            answer_ids = torch.tensor(tokenizer.encode(answer, add_special_tokens=False))
            text_ids.append(torch.cat([prompt_ids, answer_ids]))
            lm_targets.append(
                torch.cat([
                    torch.full((prompt_ids.shape[0] - 1, ), CE_IGNORE_INDEX),
                    answer_ids,
                    torch.tensor([tokenizer.eos_token_id]),
                ]),
            )
        text_ids = torch.cat(text_ids)
        lm_targets = torch.cat(lm_targets)

        # text_ids = torch.tensor(tokenizer.encode(text, add_special_tokens=False))
        # TODO: dynamically adjust patch size according to image spacing
        num_vision_tokens = np.prod([s // 16 for s in self.patch_size]).item() + 2  # including boi and eoi
        input_ids = torch.cat([
            torch.tensor([tokenizer.bos_token_id]),
            torch.full((num_vision_tokens, ), 0),
            text_ids,
        ])
        token_type_ids = torch.cat([
            torch.tensor([LANGUAGE_TOKEN_TYPE]),
            torch.full((num_vision_tokens, ), VISION_TOKEN_TYPE),
            torch.full(text_ids.shape, LANGUAGE_TOKEN_TYPE),
        ])
        position_ids = torch.cat([
            torch.tensor([0, 1]),  # bos and boi
            torch.full((num_vision_tokens - 2, ), 2),
            torch.tensor([3]),  # eoi
            torch.arange(4, 4 + text_ids.shape[0]),
        ])
        attention_mask = torch.ones(input_ids.shape, dtype=dtype)
        lm_targets = torch.cat([torch.full((1 + num_vision_tokens, ), CE_IGNORE_INDEX), lm_targets])
        return {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'position_ids': position_ids,
            'attention_mask': attention_mask,
            'lm_targets': lm_targets,
        }, text

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
        neg_class_ids = self.R.choice(len(neg_classes), min(len(neg_classes), self.num_neg), replace=False)
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
        pos_masks = np.load(data_dir / 'masks.npy', 'r')[:, *patch_slice]
        pos_masks = pos_masks[pos_class_ids]
        pos_masks = torch.as_tensor(np.array(pos_masks), device=self.device)

        # prepare sample output
        modality = modalities[modality_id]
        conversation, mask_classes = self.gen_conversation(modality, pos_classes, neg_classes)
        masks = pos_masks.new_zeros((len(mask_classes), *pos_masks.shape[1:]))
        pos_class_to_idx = {name: i for i, name in enumerate(pos_classes)}
        for i, name in enumerate(mask_classes):
            if (pos_idx := pos_class_to_idx.get(name, -1)) != -1:
                masks[i] = pos_masks[pos_idx]

        vlm_inputs, conversation_text = self.prepare_vlm_inputs(conversation)
        data = {
            'key': key,
            'image': image,
            'modality': modality,
            'masks': masks,
            'vlm_inputs': vlm_inputs,
        }
        return self.pad_image(data)

class InputTransformD(mt.Transform):
    def __call__(self, data: dict):
        data = dict(data)
        img: MetaTensor = data['image']
        data['image'], _ = ensure_rgb(img.as_tensor())
        masks: MetaTensor = data['masks']
        data['masks'] = masks.as_tensor().round().bool()
        return data

@dataclass
class TransConf:
    patch_size: tuple3_t[int] = (96, 224, 224)
    num_pos: int = 48
    num_neg: int = 4

class MMMMDataModule(ExpDataModuleBase):
    def __init__(
        self,
        trans: TransConf,
        tokenizer: MMMMTokenizer,
        data_root: Path = PROCESSED_SEG_DATA_ROOT,
        *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.data_root = data_root
        self._train_data = []
        self.trans_conf = trans
        for dataset_dir in data_root.iterdir():
            if dataset_dir.name != 'AMOS22':
                continue
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
        # FIXME use a list as dataset with Python's multiprocessing can cause "memory leak": https://github.com/pytorch/pytorch/issues/13246
        return self._train_data

    def train_transform(self) -> Callable:
        conf = self.trans_conf
        return mt.Compose([
            SamplePatch(conf.patch_size, conf.num_pos, conf.num_neg, self.tokenizer),
            InputTransformD(),
        ])

    def get_train_collate_fn(self):
        from luolib.data.utils import list_data_collate

        def collate_fn(batch: list[dict]):
            list_data = {key: [] for key in ['key', 'masks', 'modality']}
            batch_vlm_inputs: list[dict] = []
            for x in batch:
                for key, data in list_data.items():
                    data.append(x.pop(key))
                batch_vlm_inputs.append(x.pop('vlm_inputs'))
            ret = {
                **list_data_collate(batch),
                **list_data,
                'vlm_inputs': {
                    key: pad_sequence(
                        [x[key] for x in batch_vlm_inputs],
                        batch_first=True,
                        padding_value=CE_IGNORE_INDEX if key == 'lm_targets' else 0,
                    )
                    for key in batch_vlm_inputs[0].keys()
                }
            }
            return ret

        return collate_fn
