from einops import repeat
import json
from jsonargparse import CLI
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from cogvlm import cogvlm_collate_fn, cogvlm_setup, cogvlm_vl_evaluate
from instructblip import instructblip_collate_fn, setup_instructblip, instructblip_vl_evaluate
from llavamed import llavamed_collate_fn, setup_llavamed, llavamed_vl_evaluate
from llavanext import llavanext_collate_fn, setup_llavanext, llavanext_vl_evaluate
from m3d import m3d_collate_fn, setup_m3d, m3d_vl_evaluate
from medflamingo import medflamingo_collate_fn, setup_medflamingo, medflamingo_vl_evaluate
from mmmm.data.defs import PROCESSED_VL_DATA_ROOT
from radfm import radfm_collate_fn, setup_radfm, radfm_vl_evaluate
from utils import (
    dump_results,
    setup_seed,
    NLPMetrics,
)


class VQATestDataset(Dataset):
    def __init__(self, dataset: str):
        super().__init__()
        self.name = dataset
        with open(PROCESSED_VL_DATA_ROOT / dataset / 'test.json') as f:
            self.dataset = [
                {'image': image, **vqa}
                for x in json.load(f)
                for vqa in x['vqa']
                for image in x['image']
            ]

    def __getitem__(self, index: int):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


class Evaluator:
    def __init__(
        self,
        checkpoint: str,
        tokenizer: str,
        task: str,
        dataset: str,
        setting: str,
        seed: int = 233,
        num_workers: int = 8,
        output_dir: str = 'results/',
    ):
        self.checkpoint = checkpoint
        self.tokenizer = tokenizer
        self.task = task
        if task == 'vqa':
            self.dataset = VQATestDataset(dataset)
        self.setting = setting
        self.num_workers = num_workers
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = NLPMetrics()
        setup_seed(seed)

    def radfm(self):
        model, tokenizer = setup_radfm(self.checkpoint, self.tokenizer)

        dataloader = DataLoader(
            self.dataset,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=radfm_collate_fn,
        )

        results = radfm_vl_evaluate(model, tokenizer, dataloader, self.metrics)

        dump_results(
            results,
            self.output_dir,
            self.task,
            self.dataset.name,
            'radfm',
            self.setting,
        )

    def medflamingo(self):
        model, processor = setup_medflamingo(self.checkpoint, self.tokenizer)

        dataloader = DataLoader(
            self.dataset,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=medflamingo_collate_fn,
        )
        
        results = medflamingo_vl_evaluate(model, processor, dataloader, self.metrics)

        dump_results(
            results,
            self.output_dir,
            self.task,
            self.dataset.name,
            'medflamingo',
            self.setting,
        )

    def m3d(self):
        model, tokenizer = setup_m3d(self.checkpoint, self.tokenizer)

        dataloader = DataLoader(
            self.dataset,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=m3d_collate_fn,
        )

        results = m3d_vl_evaluate(model, tokenizer, dataloader, self.metrics)

        dump_results(
            results, self.output_dir, self.task, self.dataset.name, 'm3d', self.setting
        )

    def llavamed(self):
        model, tokenizer, processor, image_token_len = setup_llavamed(
            self.checkpoint, self.tokenizer
        )

        dataloader = DataLoader(
            self.dataset,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=llavamed_collate_fn,
        )

        results = llavamed_vl_evaluate(model, tokenizer, processor, image_token_len, dataloader, self.metrics)

        dump_results(
            results,
            self.output_dir,
            self.task,
            self.dataset.name,
            'llavamed',
            self.setting,
        )

    def cogvlm(self):
        model, tokenizer = cogvlm_setup(self.checkpoint, self.tokenizer)

        dataloader = DataLoader(
            self.dataset,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=cogvlm_collate_fn,
        )

        results = cogvlm_vl_evaluate(model, tokenizer, dataloader, self.metrics)

        dump_results(
            results,
            self.output_dir,
            self.task,
            self.dataset.name,
            'cogvlm',
            self.setting,
        )

    def llavanext(self):     
        model, processor = setup_llavanext(
            self.checkpoint, self.tokenizer
        )

        dataloader = DataLoader(
            self.dataset,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=llavanext_collate_fn,
        )

        results = llavanext_vl_evaluate(model, processor, dataloader, self.metrics)

        dump_results(
            results,
            self.output_dir,
            self.task,
            self.dataset.name,
            'llavanext',
            self.setting,
        )

    def instructblip(self):
        model, processor = setup_instructblip(
            self.checkpoint, self.tokenizer
        )

        dataloader = DataLoader(
            self.dataset,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=instructblip_collate_fn,
        )

        results = instructblip_vl_evaluate(model, processor, dataloader, self.metrics)

        dump_results(
            results,
            self.output_dir,
            self.task,
            self.dataset.name,
            'instructblip',
            self.setting,
        )


if __name__ == '__main__':
    CLI(Evaluator, as_positional=False)
