from typing import Iterator

from lightning.fabric.utilities.distributed import DistributedSamplerWrapper
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Sampler

from luolib.data.utils import list_data_collate
from luolib.datamodule import ExpDataModuleBase

from mmmm.tokenizer import MMMMTokenizer
from .dataset import MMMMDataset, DatasetConf

__all__ = [
    'MMMMDataModule',
]

from .defs import CE_IGNORE_INDEX, DataPoint

def _collate_fn(batch: list[DataPoint]):
    # maybe TODO: can we follow the type annotation of Batch?
    list_keys = ['image', 'grounding_image', 'patch_size', 'mask', 'mask_index', 'bbox', 'bbox_index']
    list_data = {key: [] for key in list_keys}
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

class MMMMRandomSampler(Sampler):
    def __init__(self, dataset: MMMMDataset, num_samples: int, seed: int = 42):
        super().__init__()
        self.dataset = dataset
        self.dataset_weights = dataset.dataset_weights
        self.num_samples = num_samples
        self.G = torch.Generator()
        self.G.manual_seed(seed)

    def __iter__(self) -> Iterator[tuple[int, int]]:
        for dataset_idx in torch.multinomial(
            self.dataset_weights, self.num_samples, True, generator=self.G,
        ):
            sub_idx = torch.randint(len(self.dataset.data_lists[dataset_idx]), size=(1, ), generator=self.G)
            yield dataset_idx.item(), sub_idx.item()

    def __len__(self):
        return self.num_samples

class MMMMDataModule(ExpDataModuleBase):
    def __init__(
        self,
        dataset: DatasetConf,
        tokenizer: MMMMTokenizer,
        *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dataset_conf = dataset
        self.tokenizer = tokenizer
        assert len(set([d.name for d in dataset.datasets])) == len(dataset.datasets), 'duplicated dataset'

    def train_dataloader(self):
        dataset = MMMMDataset(self.dataset_conf, 'train', self.tokenizer)
        conf = self.dataloader_conf
        assert conf.train_batch_size is not None and conf.num_batches is not None
        sampler = MMMMRandomSampler(dataset, conf.num_batches * conf.train_batch_size * self.world_size)
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
        return _collate_fn
