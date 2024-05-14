from dataclasses import dataclass
from typing import Literal

import torch
from torch.utils.data import Dataset

from monai.transforms import apply_transform

from mmmm.tokenizer import MMMMTokenizer
from ..defs import Split
from .local import LocalTransConf, get_local_data_list, get_local_transform
from .vl import VLTransConf, VLTransform, get_vl_data_list

@dataclass(kw_only=True)
class DatasetSpec:
    """
    Attributes:
        weight: scale factor of the dataset weight
    """
    name: str
    type: Literal['local', 'vl']
    weight: float = 1.

    def get_data_list(self, split: Split) -> list[dict]:
        match self.type:
            case 'local':
                return get_local_data_list(self.name, split)
            case 'vl':
                return get_vl_data_list(self.name, split)
            case _:
                raise ValueError

def _is_power_of_2(x: int):
    return x & (x - 1) == 0

@dataclass(kw_only=True)
class DatasetConf:
    datasets: list[DatasetSpec]
    base_vit_patch_size_z: int
    vit_patch_size_xy: int
    pool_size_xy: int
    base_pool_size_z: int
    local_trans: LocalTransConf | None = None
    vl_trans: VLTransConf | None = None
    max_seq_len: int | None = None
    bop_weight: float = 1.

    @property
    def base_stride_z(self):
        return self.base_vit_patch_size_z * self.base_pool_size_z

    @property
    def stride_xy(self):
        return self.vit_patch_size_xy * self.pool_size_xy

    def __post_init__(self):
        assert _is_power_of_2(self.vit_patch_size_xy)
        assert _is_power_of_2(self.base_vit_patch_size_z)
        assert _is_power_of_2(self.pool_size_xy)
        assert _is_power_of_2(self.base_pool_size_z)

class MMMMDataset(Dataset):
    def __init__(
        self,
        conf: DatasetConf,
        split: Split,
        tokenizer: MMMMTokenizer,
        inference: bool,
    ):
        super().__init__()
        self.conf = conf
        self.data_lists = [
            dataset.get_data_list(split)
            for dataset in conf.datasets
        ]
        # NOTE: use attributes instead of storing in a dict to make MONAI's set_rnd work
        if conf.local_trans is not None:
            self.local_transform = get_local_transform(conf, tokenizer, inference)
        if conf.vl_trans is not None:
            self.vl_transform = VLTransform(conf, tokenizer, inference)

    @property
    def dataset_weights(self):
        weights = torch.tensor(
            [
                dataset.weight * len(data_list)
                for dataset, data_list in zip(self.conf.datasets, self.data_lists)
            ],
            dtype=torch.float64,
        )
        return weights

    def __getitem__(self, index: tuple[int, int]):
        dataset_idx, sub_idx = index
        dataset = self.conf.datasets[dataset_idx]
        data_list = self.data_lists[dataset_idx]
        data = data_list[sub_idx]
        return apply_transform(getattr(self, f'{dataset.type}_transform'), data)
