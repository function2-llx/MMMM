from pathlib import Path

import h5py
from lightning import LightningDataModule
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
import numpy as np
from numpy import typing as npt
from torch.utils.data import Dataset as TorchDataset

from luolib.types import tuple3_t
from monai import transforms as mt

from mmmm.data.defs import Meta, encode_patch_size

class MMMMDataset(TorchDataset):
    pass

class SamplePatch(mt.RandomizableTransform):
    def __init__(self, num_pos: int, num_neg: int, patch_size: tuple3_t[int]):
        super().__init__()
        self.patch_size = patch_size
        self.num_pos = num_pos
        self.num_neg = num_neg

    def __call__(self, data: dict):
        dataset_dir: Path = data['dataset_dir']
        key: str = data['key']
        meta: Meta = data['meta']
        patch_dir = dataset_dir / 'patch' / encode_patch_size(self.patch_size) / key
        all_classes = meta['positive_classes']
        positive_mask: npt.NDArray[np.bool_] = np.load(patch_dir / 'positive_mask.npy', 'r')

        # sample patch position
        position: npt.NDArray[np.int16]
        if self.R.uniform() < 1 / 3:
            # foreground oversampling
            c = self.R.randint(len(all_classes))
            with h5py.File(patch_dir / 'class_to_patch.h5') as f:
                positions: h5py.Dataset = f[str(c)]
                position = positions[self.R.randint(positions.shape[0])]
        else:
            # sample a random patch position
            c = None
            position = np.array([self.R.randint(s) for s in positive_mask.shape[1:]], dtype=np.int16)
        positive_mask = positive_mask[*position, :]

        # sample negative classes
        neg_class_ids, = (~positive_mask).nonzero()
        neg_classes = [all_classes[i] for i in neg_class_ids] + meta['negative_classes']
        neg_class_ids = self.R.choice(min(len(neg_classes), self.num_neg), self.num_neg, replace=False)
        neg_classes = [neg_classes[i] for i in neg_class_ids]

        # sample positive classes
        if c is not None:
            positive_mask[c] = False
        class_ids, = positive_mask.nonzero()
        num_pos = self.num_pos - (c is not None)
        class_ids = self.R.choice(class_ids, min(class_ids.shape[0], num_pos), replace=False)
        if c is not None:
            class_ids = np.insert(class_ids, self.R.randint(class_ids.shape[0] + 1), c)
        classes = [all_classes[i] for i in class_ids]

        # TODO: crop image & masks patch
        data_dir = dataset_dir / 'data' / key
        modalities = meta['modalities']
        modality_id = self.R.randint(len(modalities))
        image = np.load(data_dir / 'images.npy', 'r')[modality_id:modality_id + 1]
        masks = np.load(data_dir / 'masks.npy', 'r')[class_ids]
        return {
            'image': image,
            'modality': modalities[modality_id],
            'masks': masks,
            'classes': classes,
            'neg_classes': neg_classes
        }

class MMMMDataModule(LightningDataModule):
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        pass
