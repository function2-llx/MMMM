from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache
import json
from pathlib import Path
from typing import Callable, List, Sequence, Iterator, Optional
import random

from einops import repeat
import h5py
from lightning.fabric.utilities.distributed import DistributedSamplerWrapper
import numpy as np
from numpy import typing as npt
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import ConcatDataset, DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence
from torch.types import Device
import torchvision.transforms.v2 as tvt

from luolib.data.utils import list_data_collate
from luolib.datamodule import ExpDataModuleBase
from luolib.types import param3_t, tuple3_t
from luolib.utils.misc import ensure_rgb
from monai import transforms as mt
from monai.data import MetaTensor
from monai.utils import ensure_tuple_rep

from mmmm.models import MMMMTokenizer
from mmmm.models.cogvlm import LANGUAGE_TOKEN_TYPE, VISION_TOKEN_TYPE
from .defs import Sparse, PROCESSED_SEG_DATA_ROOT, PROCESSED_VL_DATA_ROOT, encode_patch_size

__all__ = [
    'MMMMDataModule',
    'CE_IGNORE_INDEX',
]

CE_IGNORE_INDEX = -100

def mmmm_collate_fn(batch: list[dict]):
    list_data = {key: [] for key in ['mask_classes', 'masks']}
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

def gen_seg_conversation(
    modality: str,
    pos_classes: list[str],
    neg_classes: list[str],
    tokenizer: MMMMTokenizer,
    R: np.random.RandomState | int,
    p_use_neg_mask: float = 1.,
    inference: bool = False,
):
    def _convert_list(names: Iterable[str], mask: bool):
        # FIXME: do not use special tokens explicitly in text
        if mask:
            names = map(tokenizer.wrap_name, names)
        return ', '.join(names)

    if isinstance(R, int):
        R = np.random.RandomState(R)

    # copy the input list because the shuffling is in-place
    pos_classes = list(pos_classes)
    R.shuffle(pos_classes)
    neg_classes = list(neg_classes)
    R.shuffle(neg_classes)

    # merge positive and negative classes with random order without shuffling
    pos_class_mask = torch.zeros(len(pos_classes) + len(neg_classes), dtype=torch.bool)
    pos_class_mask[R.choice(pos_class_mask.shape[0], len(pos_classes), replace=False)] = True
    pos_it, neg_it = map(iter, [pos_classes, neg_classes])
    classes = [
        next(pos_it) if m else next(neg_it)
        for m in pos_class_mask
    ]

    assert len(classes) > 0
    neg_mask = R.uniform() < p_use_neg_mask
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
    if inference:
        # TODO: refactor this function
        response = ''
    return [(prompt, response)], mask_classes

def prepare_vlm_inputs(
    conversation: list[tuple[str, str]],
    tokenizer: MMMMTokenizer,
    num_image_tokens: int,
    inference: bool = False,
):
    """
    Args:
        conversation:
        tokenizer:
        num_image_tokens: the number of tokens corresponding to image patches (does not include special tokens)
        inference:
    Returns:

    """
    # TODO: refactor this function to support various VLM formats
    # template: CogVLM `chat_old_history_to_prompt`
    # just for viewing, don't tokenize it directly
    user_start = 'Question:'
    sys_start = 'Answer:'
    text = '\n'.join(
        f'{user_start} {query} {sys_start} {answer}'
        for query, answer in conversation
    )
    dtype = torch.long
    text_ids = []
    # the last response is empty iff inference
    assert inference ^ (conversation[-1][1] == '')
    if not inference:
        lm_targets = []
    for i, (query, answer) in enumerate(conversation):
        prompt = f'{user_start} {query} {sys_start}'
        prompt_ids = torch.tensor(tokenizer.encode(prompt, add_special_tokens=False))

        if answer == '':
            assert i == len(conversation) - 1 and inference
            text_ids.append(prompt_ids)
        else:
            answer_ids = torch.tensor(tokenizer.encode(answer, add_special_tokens=False))
            text_ids.append(torch.cat([prompt_ids, answer_ids]))
            if not inference:
                lm_targets.append(
                    torch.cat([
                        torch.full((prompt_ids.shape[0] - 1, ), CE_IGNORE_INDEX),
                        answer_ids,
                        torch.tensor([tokenizer.eos_token_id]),
                    ]),
                )
    text_ids = torch.cat(text_ids)
    if not inference:
        lm_targets = torch.cat(lm_targets)

    # text_ids = torch.tensor(tokenizer.encode(text, add_special_tokens=False))
    # TODO: dynamically adjust patch size according to image spacing
    # num_vision_tokens = np.prod([s // ps for s, ps in zip(image_size, vit_patch_size)]).item() + 2  # including boi and eoi
    num_image_tokens += 2  # including boi and eoi
    input_ids = torch.cat([
        torch.tensor([tokenizer.bos_token_id]),
        torch.full((num_image_tokens,), 0),
        text_ids,
    ])
    image_features_mask = torch.zeros(input_ids.shape[0], dtype=torch.bool)
    image_features_mask[1:1 + num_image_tokens] = True
    token_type_ids = torch.cat([
        torch.tensor([LANGUAGE_TOKEN_TYPE]),
        torch.full((num_image_tokens,), VISION_TOKEN_TYPE),
        # all new tokens will be processed by VE
        # torch.where(text_ids < tokenizer.base_vocab_size, LANGUAGE_TOKEN_TYPE, VISION_TOKEN_TYPE),
        torch.full((text_ids.shape[0], ), LANGUAGE_TOKEN_TYPE),
    ])
    position_ids = torch.cat([
        torch.tensor([0, 1]),  # bos and boi
        torch.full((num_image_tokens - 2,), 2),
        torch.tensor([3]),  # eoi
        torch.arange(4, 4 + text_ids.shape[0]),
    ])
    attention_mask = torch.ones(input_ids.shape, dtype=dtype)
    if not inference:
        lm_targets = torch.cat([torch.full((1 + num_image_tokens,), CE_IGNORE_INDEX), lm_targets])
    inputs = {
        'input_ids': input_ids,
        'image_features_mask': image_features_mask,
        'token_type_ids': token_type_ids,
        'position_ids': position_ids,
        'attention_mask': attention_mask,
    }
    if not inference:
        inputs['lm_targets'] = lm_targets
    return inputs, text

class SamplePatch(mt.RandomizableTransform):
    def __init__(
        self,
        patch_size: tuple3_t[int],
        vit_patch_size: param3_t[int],
        num_pos: int,
        num_neg: int,
        tokenizer: MMMMTokenizer,
        force_fg_ratio: float = 1 / 3,
        device: Device = 'cpu',
        inference: bool = False,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_pos = num_pos
        self.num_neg = num_neg
        self.tokenizer = tokenizer
        self.force_fg_ratio = force_fg_ratio
        self.device = device
        self.vit_patch_size: tuple3_t[int] = ensure_tuple_rep(vit_patch_size, 3)
        self.inference = inference

    def __call__(self, data: dict):
        dataset_dir: Path = data['dataset_dir']
        key: str = data['key']
        sparse: Sparse = pd.read_pickle(dataset_dir / 'data' / key / 'sparse.pkl')
        # TODO: adapt to sparse
        # all positive classes on the whole image
        image_pos_classes = sparse.anatomy.pos['positive_classes']
        patches_class_mask: npt.NDArray[np.bool_] = np.load(patch_dir / 'patches_class_mask.npy', 'r')

        # sample patch position
        position: npt.NDArray[np.int16]
        if self.R.uniform() < self.force_fg_ratio:
            # foreground oversampling
            c = self.R.randint(len(image_pos_classes))
            with h5py.File(patch_dir / 'class_positions.h5') as f:
                positions: h5py.Dataset = f[str(c)]
                position = positions[self.R.randint(positions.shape[0])]
        else:
            # sample a random patch position
            c = None
            position = np.array([self.R.randint(s) for s in patches_class_mask.shape[:-1]], dtype=np.int16)
        pos_class_mask = np.array(patches_class_mask[*position, :])

        # sample negative classes
        neg_class_ids, = (~pos_class_mask).nonzero()
        # all negative classes for this patch:
        # - positive classes in the whole image but not in this patch
        # - negative classes for the whole image
        neg_classes: list[str] = [image_pos_classes[i] for i in neg_class_ids] + meta['negative_classes']
        neg_classes = self.R.choice(neg_classes, min(len(neg_classes), self.num_neg), replace=False).tolist()

        # sample positive classes
        if c is not None:
            pos_class_mask[c] = False
        pos_class_ids, = pos_class_mask.nonzero()
        num_pos = self.num_pos - (c is not None)
        pos_class_ids = self.R.choice(pos_class_ids, min(pos_class_ids.shape[0], num_pos), replace=False)
        if c is not None:
            pos_class_ids = np.insert(pos_class_ids, 0, c)
        pos_classes: list[str] = [image_pos_classes[i] for i in pos_class_ids]

        # construct image & masks patch
        data_dir = dataset_dir / 'data' / key
        modalities = meta['modalities']
        modality_id = self.R.randint(len(modalities))
        patch_slice = [slice(p, p + s) for p, s in zip(position, self.patch_size)]
        # TODO: support RGB
        image = np.load(data_dir / 'images.npy', 'r')[modality_id:modality_id + 1, *patch_slice]
        image = torch.as_tensor(np.array(image), device=self.device)
        pos_masks = np.load(data_dir / 'masks.npy', 'r')[:, *patch_slice]
        pos_masks = pos_masks[pos_class_ids]
        pos_masks = torch.as_tensor(np.array(pos_masks), device=self.device)

        # prepare sample output
        modality = modalities[modality_id]
        conversation, mask_classes = gen_seg_conversation(
            modality, pos_classes, neg_classes, self.tokenizer, self.R, inference=self.inference,
        )
        masks = pos_masks.new_zeros((len(mask_classes), *pos_masks.shape[1:]))
        pos_class_to_idx = {name: i for i, name in enumerate(pos_classes)}
        for i, name in enumerate(mask_classes):
            if (pos_idx := pos_class_to_idx.get(name, -1)) != -1:
                masks[i] = pos_masks[pos_idx]

        vlm_inputs, conversation_text = prepare_vlm_inputs(
            conversation, self.tokenizer, self.patch_size, self.vit_patch_size, self.inference,
        )
        if np.less(image.shape[1:], self.patch_size).any():
            mean, std = meta['mean'], meta['std']
            padder = mt.SpatialPad(self.patch_size)
            image = torch.cat([
                padder(image[i:i + 1], value=-mean[i] / std[i])
                for i in range(image.shape[0])
            ])
            masks = padder(masks)
        data = {
            'image': image,
            'spacing': sparse.spacing,
            'modality': modality,
            'masks': masks,
            'mask_classes': mask_classes,
            'vlm_inputs': vlm_inputs,
        }
        return data

# class FullImageTransform(mt.Transform):
#     # TODO: merge these two transforms
#     # this class may include randomized procedure, but its output can be cached
#
#     def __init__(
#         self,
#         patch_size: tuple3_t[int],
#         vit_patch_size: param3_t[int],
#         tokenizer: MMMMTokenizer,
#         device: Device = 'cpu',
#     ):
#         super().__init__()
#         self.patch_size = patch_size
#         self.tokenizer = tokenizer
#         self.device = device
#         self.vit_patch_size: tuple3_t[int] = ensure_tuple_rep(vit_patch_size, 3)
#
#     def __call__(self, data: dict):
#         dataset_dir: Path = data['dataset_dir']
#         key: str = data['key']
#         meta: Meta = data['meta']
#         patch_dir = dataset_dir / 'patch' / encode_patch_size(self.patch_size) / key
#         # all positive classes on the whole image
#         assert len(meta['negative_classes']) == 0
#         pos_classes = meta['positive_classes']
#
#         # construct image & masks patch
#         data_dir = dataset_dir / 'data' / key
#         modalities = meta['modalities']
#         assert len(modalities) == 1
#         modality = modalities[0]
#         image = torch.as_tensor(np.load(data_dir / 'images.npy'), device=self.device)
#         masks = torch.as_tensor(np.load(data_dir / 'masks.npy'), device=self.device)
#         # prepare sample output
#         conversation, mask_classes = gen_conversation(
#             modality, pos_classes, [], self.tokenizer, 42, 1.,
#         )
#         pos_class_to_idx = {name: i for i, name in enumerate(pos_classes)}
#         mask_perm = [
#             pos_class_to_idx[name]
#             for i, name in enumerate(mask_classes)
#         ]
#         masks = masks[mask_perm]
#
#         vlm_inputs, conversation_text = prepare_vlm_inputs(
#             conversation, self.tokenizer, self.patch_size, self.vit_patch_size,
#         )
#         if np.less(image.shape[1:], self.patch_size).any():
#             mean, std = meta['mean'], meta['std']
#             padder = mt.SpatialPad(self.patch_size)
#             image = torch.cat([
#                 padder(image[i:i + 1], value=-mean[i] / std[i])
#                 for i in range(image.shape[0])
#             ])
#             masks = padder(masks)
#         data = {
#             'key': key,
#             'image': image,
#             'modality': modality,
#             'masks': masks,
#             'mask_classes': mask_classes,
#             'vlm_inputs': vlm_inputs,
#         }
#         return data

class InputTransformD(mt.Transform):
    def __call__(self, data: dict):
        data = dict(data)
        img = data['image']
        if isinstance(img, MetaTensor):
            img = img.as_tensor()
        data['image'], _ = ensure_rgb(img)
        masks = data['masks']
        if isinstance(masks, MetaTensor):
            masks = masks.as_tensor()
        if masks.dtype != torch.bool:
            masks = masks.round().bool()
        data['masks'] = masks
        return data
    
class SegTransform(mt.Transform):
    def __init__(
        self,
        patch_size: tuple3_t[int],
        vit_patch_size: param3_t[int],
        num_pos: int,
        num_neg: int,
        tokenizer: MMMMTokenizer,
    ):
        super().__init__()
        self.transforms = mt.Compose([
            SamplePatch(
                patch_size,
                vit_patch_size,
                num_pos,
                num_neg,
                tokenizer,
            ),
            InputTransformD(),
        ])

    def __call__(self, data: dict):
        return self.transforms(data)

class VLTransform(mt.Transform):
    def __init__(
        self,
        vit_patch_size: param3_t[int],
        tokenizer: MMMMTokenizer,
        inference: bool = False
    ):
        super().__init__()
        self.vit_patch_size: tuple3_t[int] = ensure_tuple_rep(vit_patch_size, 3)
        self.tokenizer = tokenizer
        self.inference = inference

    def __call__(self, data: dict):
        image_dir: Path = data['dataset_dir'] / 'images' / data['image']
        image = Image.open(image_dir)
        image = tvt.functional.to_tensor(image)
        if min(image.shape[1:]) > 512:
            image = tvt.functional.resize(image, 512)
        image = tvt.pad(image, (0, 0, torch.ceil(image.shape[2] / 16) * 16, torch.ceil(image.shape[1] / 16) * 16))
        image = repeat(image, 'c h w -> c 1 h w')
        conversation = [(data['question'], data['answer'])]
        vlm_inputs, conversation_text = prepare_vlm_inputs(
            conversation, self.tokenizer, self.patch_size, self.vit_patch_size, self.inference,
        )
        data = {
            'image': image,
            'masks': None,
            'mask_classes': 0,
            'vlm_inputs': vlm_inputs,
        }
        return data

@dataclass
class TransConf:
    patch_size: tuple3_t[int]
    vit_patch_size: param3_t[int]
    num_pos: int = 20  # I've encountered cases setting this to larger than 48 causing NCCL timeout
    num_neg: int = 20

class SegDataModule(ExpDataModuleBase):
    def __init__(
        self,
        trans: TransConf,
        tokenizer: MMMMTokenizer,
        data_root: Path = PROCESSED_SEG_DATA_ROOT,
        datasets: List[str] = ['AMOS22', 'AMOS22-debug'],
        *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.data_root = data_root
        self.datasets = datasets
        self.trans_conf = trans

    @cache
    def fit_split(self):
        # FIXME: use a list as dataset with Python's multiprocessing can cause "memory leak": https://github.com/pytorch/pytorch/issues/13246
        all_data = []
        for dataset in self.datasets:
            # NOTE: experiment on AMOS22 only now
            if dataset != 'AMOS22-debug':
                continue
            dataset_dir = self.data_root / dataset
            dataset_meta: pd.DataFrame = pd.read_pickle(dataset_dir / 'meta.pkl')
            all_data.extend([
                {
                    'dataset_dir': dataset_dir,
                    'key': key,
                    'meta': meta,
                }
                for key, meta in dataset_meta.iterrows()
            ])
        np.random.RandomState(42).shuffle(all_data)
        max_val_num: int = 0
        train_data, val_data = [], []
        all_classes = set()
        for item in all_data:
            # select fully labeled samples for validation
            meta: Meta = item['meta']
            all_classes |= set(meta['positive_classes'])
            all_classes |= set(meta['negative_classes'])
            if len(val_data) < max_val_num and len(meta['negative_classes']) == 0:
                val_data.append(item)
            else:
                train_data.append(item)
        self.tokenizer.build_classes_index(all_classes)
        return train_data, val_data

    def train_data(self) -> Sequence:
        train, _val = self.fit_split()
        return train

    def val_data(self) -> Sequence:
        _train, val = self.fit_split()
        return val

    def train_transform(self) -> Callable:
        conf = self.trans_conf
        return SegTransform(
            conf.patch_size,
            conf.vit_patch_size,
            conf.num_pose,
            conf.num_neg,
            self.tokenizer,
        )

    # def val_transform(self) -> Callable:
    #     conf = self.trans_conf
    #     return mt.Compose(
    #         [
    #             FullImageTransform(conf.patch_size, conf.vit_patch_size, self.tokenizer),
    #             InputTransformD(),
    #         ]
    #     )

    def get_train_collate_fn(self):
        return mmmm_collate_fn

class VQADataModule(ExpDataModuleBase):
    def __init__(
        self,
        trans: TransConf,
        tokenizer: MMMMTokenizer,
        data_root: Path = PROCESSED_VL_DATA_ROOT,
        datasets: List[str] = ['Slake', 'VQA-Med'],
        *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.data_root = data_root
        self.datasets = datasets
        self.trans_conf = trans

    def train_data(self) -> Sequence:
        train_data = []
        for dataset in self.datasets:
            with open(self.data_root / dataset / 'train.json') as f:
                data = json.load(f)
            if dataset == 'Radiopaedia':
                train_data.extend([
                    {
                        'dataset_dir': self.data_root / dataset,
                        'image': random.choice(item['image']),
                        **random.choice(item['qa_list']),
                    }
                    for item in data
                
                ])
            else:
                train_data.extend([
                    {
                        'dataset_dir': self.data_root / dataset,
                        'image': item['image'],
                        'question': item['question'],
                        'answer': item['answer'],
                    }
                    for item in data
                ])
        return train_data

    def val_data(self) -> Sequence:
        val_data = []
        for dataset in self.datasets:
            with open(self.data_root / dataset / 'validate.json') as f:
                data = json.load(f)
            val_data.extend([
                {
                    'dataset_dir': self.data_root / dataset,
                    'image': item['image'],
                    'question': item['question'],
                    'answer': item['answer'],
                }
                for item in data
            ])

        return val_data
    
    def train_transform(self) -> Callable:
        conf = self.trans_conf
        return VLTransform(
            conf.vit_patch_size,
            self.tokenizer,
        )

    def get_train_collate_fn(self):
        return mmmm_collate_fn
    
class CapDataModule(ExpDataModuleBase):
    def __init__(
        self,
        trans: TransConf,
        tokenizer: MMMMTokenizer,
        data_root: Path = PROCESSED_VL_DATA_ROOT,
        datasets: List[str] = ['Slake', 'VQA-Med'],
        *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.data_root = data_root
        self.datasets = datasets
        self.trans_conf = trans
        self.prompts = [
            '',
        ]

    def train_data(self) -> Sequence:
        train_data = []
        for dataset in self.datasets:
            with open(self.data_root / dataset / 'train.json') as f:
                data = json.load(f)
            if dataset == 'Radiopaedia':
                train_data.extend([
                    {
                        'dataset_dir': self.data_root / dataset,
                        'image': random.choice(item['image']),
                        'question': random.choice(self.prompts),
                        'answer': item['caption'],
                    }
                    for item in data
                
                ])
            else:
                train_data.extend([
                    {
                        'dataset_dir': self.data_root / dataset,
                        'image': item['image'],
                        'question': random.choice(self.prompts),
                        'answer': item['caption'],
                    }
                    for item in data
                ])
        return train_data

    def val_data(self) -> Sequence:
        val_data = []
        for dataset in self.datasets:
            with open(self.data_root / dataset / 'validate.json') as f:
                data = json.load(f)
            val_data.extend([
                {
                    'dataset_dir': self.data_root / dataset,
                    'image': item['image'],
                    'question': random.choice(self.prompts),
                    'answer': item['caption'],
                }
                for item in data
            ])

        return val_data
    
    def train_transform(self) -> Callable:
        conf = self.trans_conf
        return VLTransform(
            conf.vit_patch_size,
            self.tokenizer,
        )

    def get_train_collate_fn(self):
        return mmmm_collate_fn

class MMMMRandomSampler(Sampler):
    def __init__(self, data_source: ConcatDataset, num_samples: int, sample_rates: np.ndarray[np.float64]):
        self.num_samples = num_samples
        self.sample_rates = sample_rates
        self.len_cumsum = np.cumsum([len(d) for d in data_source.datasets])
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    def __iter__(self) -> Iterator[int]:
        datasets = np.random.choice(np.arange(self.data_source.datasets), self.num_samples, p=self.sample_rates)
        for dataset in datasets:
            low = self.len_cumsum[dataset - 1] if dataset > 0 else 0
            high = self.len_cumsum[dataset]
            yield torch.randint(low=low, high=high, size=(1,), generator=self.generator).item()

# Implemented as composition of datamodules
class MMMMDataModule(ExpDataModuleBase):
    def __init__(
        self,
        trans: TransConf,
        tokenizer: MMMMTokenizer,
        samples_per_epoch: Optional[int],
        seg_data_root: str = PROCESSED_SEG_DATA_ROOT,
        vl_data_root: str = PROCESSED_VL_DATA_ROOT,
        seg_data: List[str] = ['AMOS22', 'AMOS22-debug'],
        vqa_data: List[str] = ['Slake', 'VQA-Med', 'Radiopaedia'],
        cap_data: List[str] = ['PMC-OA', 'Radiopaedia'],
        datamodules: List[str] = ['seg', 'vqa', 'cap'],
        sample_rates: List[int] = [1, 1],
        *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.datamodules = []
        sample_rates = np.array(sample_rates)
        self.samples_per_epoch = samples_per_epoch
        self.sample_rates = sample_rates / np.sum(sample_rates)
        if 'seg' in datamodules:
            self.datamodules.append(
                SegDataModule(trans, tokenizer, seg_data_root, seg_data)
            )
        if 'vqa' in datamodules:
            self.datamodules.append(
                VQADataModule(trans, tokenizer, vl_data_root, vqa_data)
            )
        if 'cap' in datamodules:
            self.datamodules.append(
                CapDataModule(trans, tokenizer, vl_data_root, cap_data)
            )

    def train_dataset(self) -> Sequence:
        return ConcatDataset([d.train_dataset() for d in self.datamodules])
    
    def val_dataset(self) -> Sequence:
        return ConcatDataset([d.val_dataset() for d in self.datamodules])

    def train_dataloader(self):
        dataset = self.train_dataset()
        conf = self.dataloader_conf
        assert conf.train_batch_size is not None and conf.num_batches is not None
        if self.samples_per_epoch is None:
            self.samples_per_epoch = conf.num_batches * conf.train_batch_size * self.world_size
        sampler = MMMMRandomSampler(
            data_source=dataset,
            num_samples=self.samples_per_epoch,
            sample_rates=self.sample_rates,
        )
        if self.world_size > 1:
            # TODO: make this lazy (_DatasetSamplerWrapper). currently, it will consume the whole sampler at once
            sampler = DistributedSamplerWrapper(
                sampler,
                num_replicas=self.world_size,
                rank=self.trainer.global_rank,
                shuffle=False,
            )
        return DataLoader(
            dataset,
            batch_size=conf.train_batch_size,
            sampler=sampler,
            num_workers=conf.num_workers,
            pin_memory=conf.pin_memory,
            prefetch_factor=conf.prefetch_factor,
            persistent_workers=conf.persistent_workers,
            collate_fn=self.get_train_collate_fn(),
        )

    def get_train_collate_fn(self):
        return mmmm_collate_fn