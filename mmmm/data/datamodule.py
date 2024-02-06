from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import h5py
import numpy as np
from numpy import typing as npt
import pandas as pd
import torch
from torch.types import Device

from luolib.datamodule import ExpDataModuleBase
from luolib.types import tuple3_t
from monai import transforms as mt

from mmmm.data.defs import Meta, PROCESSED_SEG_DATA_ROOT, encode_patch_size

__all__ = [
    'MMMMDataModule',
]

@dataclass
class TransConf:
    patch_size: tuple3_t[int] = (96, 224, 224)
    num_pos: int = 10
    num_neg: int = 10

class SamplePatch(mt.RandomizableTransform):
    def __init__(
        self,
        patch_size: tuple3_t[int],
        num_pos: int,
        num_neg: int,
        force_fg_ratio: float = 1 / 3,
        device: Device = 'cpu',
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_pos = num_pos
        self.num_neg = num_neg
        self.force_fg_ratio = force_fg_ratio
        self.device = device
        self.pad = mt.SpatialPadD(['image', 'masks'], patch_size, lazy=True)

    def gen_conversation(self, modality: str, classes: list[str], pos_classes: list[str], neg_classes: list[str]):
        assert len(classes) > 0
        prompt = f"Find and output the segmentation masks on the given {modality} image for the following objects: {', '.join(classes)}."
        if len(pos_classes) > 0:
            response = f"The following requested objects are found and segmented: {', '.join(map(lambda name: f'<SEG> {name} </SEG>', pos_classes))}"
            if len(neg_classes) == 0:
                response += '.'
            else:
                response += f", while the following requested objects are not found: {', '.join(neg_classes)}."
        else:
            response = 'None of the requested object is found.'
        return prompt, response

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
        modality = modalities[modality_id]
        prompt, response = self.gen_conversation(modality, classes, pos_classes, neg_classes)
        data = {
            'image': image,
            'modality': modality,
            'masks': masks,
            # FIXME: https://github.com/pytorch/pytorch/issues/13246
            'requested_classes': classes,
            'pos_class_mask': pos_class_mask,
            'prompt': prompt,
            'response': response,
        }
        return self.pad(data)

class MMMMDataModule(ExpDataModuleBase):
    def __init__(
        self,
        trans: TransConf,
        data_root: Path = PROCESSED_SEG_DATA_ROOT,
        *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)
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
