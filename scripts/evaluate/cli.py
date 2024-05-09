from jsonargparse import CLI
from pathlib import Path
from torch.utils.data import DataLoader

from cogvlm import cogvlm_collate_fn, setup_cogvlm, cogvlm_vl_evaluate
from instructblip import (
    instructblip_collate_fn,
    setup_instructblip,
    instructblip_vl_evaluate,
)
from llavamed import llavamed_collate_fn, setup_llavamed, llavamed_vl_evaluate
from llavanext import llavanext_collate_fn, setup_llavanext, llavanext_vl_evaluate
from m3d import m3d_collate_fn, setup_m3d, m3d_vl_evaluate
from medflamingo import (
    medflamingo_collate_fn,
    setup_medflamingo,
    medflamingo_vl_evaluate,
)
from radfm import radfm_collate_fn, setup_radfm, radfm_vl_evaluate
from utils import (
    dump_results,
    setup_seed,
    VQATestDataset,
    ReportTestDataset,
    GenericMetrics,
    LlamaMetrics,
    CXRMetrics,
)

class Evaluator:
    def __init__(
        self,
        model: str,
        task: str,
        dataset: str,
        setting: str,
        seed: int = 233,
        output_dir: str = 'results/',
    ):
        self.model = model
        self.task = task
        self.dataset = dataset
        self.setting = setting
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        setup_seed(seed)

    def predict(
            self, 
            checkpoint: str,
            tokenizer: str,
        ):
        if self.task == 'vqa':
            dataset = VQATestDataset(self.dataset)
        elif self.task == 'report':
            dataset = ReportTestDataset(self.dataset)

        if self.model == 'radfm':
            setup_fn = setup_radfm
            collate_fn = radfm_collate_fn
            evaluate_fn = radfm_vl_evaluate
        elif self.model == 'medflamingo':
            setup_fn = setup_medflamingo
            collate_fn = medflamingo_collate_fn
            evaluate_fn = medflamingo_vl_evaluate
        elif self.model == 'm3d':
            setup_fn = setup_m3d
            collate_fn = m3d_collate_fn
            evaluate_fn = m3d_vl_evaluate
        elif self.model == 'llavamed':
            setup_fn = setup_llavamed
            collate_fn = llavamed_collate_fn
            evaluate_fn = llavamed_vl_evaluate
        elif self.model == 'cogvlm':
            setup_fn = setup_cogvlm
            collate_fn = cogvlm_collate_fn
            evaluate_fn = cogvlm_vl_evaluate
        elif self.model == 'llavanext':
            setup_fn = setup_llavanext
            collate_fn = llavanext_collate_fn
            evaluate_fn = llavanext_vl_evaluate
        elif self.model == 'instructblip':
            setup_fn = setup_instructblip
            collate_fn = instructblip_collate_fn
            evaluate_fn = instructblip_vl_evaluate

        packed = setup_fn(checkpoint, tokenizer)

        dataloader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=8,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        results = evaluate_fn(*packed, dataloader)

        dump_results(
            results,
            self.output_dir,
            self.task,
            self.dataset,
            self.model,
            self.setting,
        )

    def evaluate(self, metrics: str):
        if metrics == 'generic':
            metrics = GenericMetrics()
        elif metrics == 'llama':
            metrics = LlamaMetrics()
        elif metrics == 'cxr':
            metrics = CXRMetrics()
        
        metrics.process(self.output_dir / f'{self.task}_{self.dataset}_{self.model}_{self.setting}')


if __name__ == '__main__':
    CLI(Evaluator, as_positional=False)
