from dataclasses import dataclass
from typing import Literal

import torch
from torch.utils.data import Dataset

from monai.transforms import apply_transform

from mmmm.models import MMMMTokenizer
from ..defs import split_t
from .seg import SegTransConf, get_seg_data_list, get_seg_transform
from .vl import VLTransform, get_vl_data_list

@dataclass(kw_only=True)
class DatasetSpec:
    """
    Attributes:
        weight: scale factor of the dataset weight
    """
    name: str
    type: Literal['seg', 'vl']
    weight: float = 1.

    def get_data_list(self, split: Literal['train', 'validate', 'test']) -> list:
        match self.type:
            case 'seg':
                return get_seg_data_list(self.name, split)
            case 'vl':
                return get_vl_data_list(self.name, split)
            case _:
                raise ValueError

def _is_power_of_2(x: int):
    return x & (x - 1) == 0

@dataclass(kw_only=True)
class DatasetConf:
    datasets: list[DatasetSpec]
    max_vision_tokens: int
    base_vit_patch_size_z: int = 16
    vit_patch_size_xy: int = 16
    seg_trans: SegTransConf

    def __post_init__(self):
        assert _is_power_of_2(self.vit_patch_size_xy)
        assert _is_power_of_2(self.base_vit_patch_size_z)

class MMMMDataset(Dataset):
    def __init__(self, conf: DatasetConf, split: split_t, tokenizer: MMMMTokenizer):
        super().__init__()
        self.conf = conf
        self.data_lists = [
            dataset.get_data_list(split)
            for dataset in conf.datasets
        ]
        self.transforms = {
            'seg': get_seg_transform(conf, tokenizer, False),
            'vl': VLTransform(conf.base_vit_patch_size, tokenizer, False),
        }

    @property
    def dataset_weights(self):
        weights = torch.tensor([dataset.weight * len(data_list) for dataset, data_list in self.data_lists.values()])
        weights /= weights.max()
        return weights

    def __getitem__(self, index: tuple[int, int]):
        dataset_idx, sub_idx = index
        dataset = self.conf.datasets[dataset_idx]
        data_list = self.data_lists[dataset_idx]
        data = data_list[sub_idx]
        return apply_transform(self.transforms[dataset.type], data)
