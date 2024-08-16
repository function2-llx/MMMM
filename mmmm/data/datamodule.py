from typing import Iterator

from lightning.fabric.utilities.distributed import DistributedSamplerWrapper
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Sampler

from luolib.datamodule import ExpDataModuleBase
from monai.data import DataLoader

from mmmm.tokenizer import MMMMTokenizer
from .dataset import MMMMDataset, DatasetConf

__all__ = [
    'MMMMDataModule',
]

from .defs import Batch, CE_IGNORE_INDEX, Split

def _collate_fn(batch: list[dict]) -> Batch:
    vlm_inputs: list[dict | None] = []
    ret = {}
    for x in batch:
        vlm_inputs.append(x.pop('vlm_inputs'))
        for key, value in x.items():
            ret.setdefault(key, []).append(value)
    if any(x is None for x in vlm_inputs):
        assert all(x is None for x in vlm_inputs)
    else:
        ret['vlm_inputs'] = {
            key: pad_sequence(
                [x[key] for x in vlm_inputs],
                batch_first=True,
                padding_value=CE_IGNORE_INDEX if key == 'labels' else 0,
            )
            for key in vlm_inputs[0].keys()
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
        if (w := dataset.conf.mimic_cxr_neg_weight) is not None:
            assert 0 <= w <= 1
            for i, dataset_spec in enumerate(dataset.conf.datasets):
                if dataset_spec.name == 'MIMIC-CXR':
                    num_tot = len(dataset.data_lists[i])
                    neg_mask = torch.ones(num_tot, dtype=torch.bool)
                    for j, data in enumerate(dataset.data_lists[i]):
                        if len(data.get('anomaly_pos', [])) > 0:
                            neg_mask[j] = 0
                    num_neg = neg_mask.sum()
                    # sorry for an implementation like this
                    mimic_weight = torch.ones(num_tot, dtype=torch.float)
                    mimic_weight[neg_mask] *= (w * (num_tot - num_neg)) / ((1 - w) * num_neg)
                    self._mimic_weight = mimic_weight

    @property
    def num_datasets(self):
        return self.dataset_weights.shape[0]

    def __iter__(self) -> Iterator[tuple[int, int]]:
        cnt = torch.zeros(self.num_datasets, dtype=torch.int64)
        buffer = [torch.empty(0, dtype=torch.int64) for _ in range(self.num_datasets)]
        for dataset_idx in torch.multinomial(
            self.dataset_weights, self.num_samples, True, generator=self.G,
        ):
            if cnt[dataset_idx] == buffer[dataset_idx].shape[0]:
                if self.dataset.conf.datasets[dataset_idx].name == 'MIMIC-CXR':
                    buffer[dataset_idx] = torch.multinomial(self._mimic_weight, 131072, replacement=True, generator=self.G)
                else:
                    buffer[dataset_idx] = torch.randperm(len(self.dataset.data_lists[dataset_idx]))
                cnt[dataset_idx] = 0
            sub_idx = buffer[dataset_idx][cnt[dataset_idx]]
            cnt[dataset_idx] += 1
            yield dataset_idx.item(), sub_idx.item()

    def __len__(self):
        return self.num_samples

class MMMMDataModule(ExpDataModuleBase):
    def __init__(
        self,
        dataset: DatasetConf,
        tokenizer: MMMMTokenizer | None,
        *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dataset_conf = dataset
        self.tokenizer = tokenizer
        assert len(set([d.name for d in dataset.datasets])) == len(dataset.datasets), 'duplicated dataset'

    def train_dataloader(self):
        dataset = MMMMDataset(self.dataset_conf, Split.TRAIN, self.tokenizer, inference=False)
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
