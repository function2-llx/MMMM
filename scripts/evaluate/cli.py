from functools import partial
from typing import Optional
from jsonargparse import CLI
from pathlib import Path
from torch.utils.data import DataLoader

from models.mmmm import (
    MMMMTransform,
    setup_mmmm,
    mmmm_vl_evaluate,
)
from models.cogvlm import CogVLMTransform, setup_cogvlm, cogvlm_vl_evaluate
from models.instructblip import (
    InstructBlipTransform,
    setup_instructblip,
    instructblip_vl_evaluate,
)
from models.llavamed import LlavaMedTransform, setup_llavamed, llavamed_vl_evaluate
from models.llavanext import LlavaNextTransform, setup_llavanext, llavanext_vl_evaluate
from models.m3d import M3DTransform, setup_m3d, m3d_vl_evaluate
from models.radfm import RadFMTransform, setup_radfm, radfm_vl_evaluate
from utils import (
    collate_fn,
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
        tokenizer: Optional[str] = None,
    ):

        if self.model == 'mmmm':
            packed = setup_mmmm(checkpoint, tokenizer)
            transform = MMMMTransform(packed[1])
            evaluate_fn = mmmm_vl_evaluate
        if self.model == 'radfm':
            packed = setup_radfm(checkpoint, tokenizer)
            transform = RadFMTransform(packed[1])
            evaluate_fn = radfm_vl_evaluate
        elif self.model == 'm3d':
            packed = setup_m3d(checkpoint, tokenizer)
            transform = M3DTransform(packed[1])
            evaluate_fn = m3d_vl_evaluate
        elif self.model == 'llavamed':
            packed = setup_llavamed(checkpoint, tokenizer)
            transform = LlavaMedTransform(packed[0].config, *packed[1:])
            evaluate_fn = llavamed_vl_evaluate
        elif self.model == 'cogvlm':
            packed = setup_cogvlm(checkpoint, tokenizer)
            transform = CogVLMTransform(
                packed[0].build_conversation_input_ids, packed[1]
            )
            evaluate_fn = cogvlm_vl_evaluate
        elif self.model == 'llavanext':
            packed = setup_llavanext(checkpoint, tokenizer)
            transform = LlavaNextTransform(packed[1])
            evaluate_fn = llavanext_vl_evaluate
        elif self.model == 'instructblip':
            packed = setup_instructblip(checkpoint, tokenizer)
            transform = InstructBlipTransform(packed[1])
            evaluate_fn = instructblip_vl_evaluate

        if self.task == 'vqa':
            dataset = VQATestDataset(self.dataset, transform)
        elif self.task == 'report':
            dataset = ReportTestDataset(self.dataset, transform)

        dataloader = DataLoader(
            dataset,
            batch_size=1,
            collate_fn=collate_fn,
            num_workers=16,
            pin_memory=True,
        )

        results = evaluate_fn(
            *packed,
            dataloader,
            f'{self.output_dir}/{self.task}_{self.dataset}_{self.model}_{self.setting}.csv'
        )

    def evaluate(self, metrics: str):
        if metrics == 'generic':
            metrics = GenericMetrics()
        elif metrics == 'llama':
            metrics = LlamaMetrics()
        elif metrics == 'cxr':
            metrics = CXRMetrics()

        metrics.process(
            self.output_dir / f'{self.task}_{self.dataset}_{self.model}_{self.setting}'
        )


if __name__ == '__main__':
    CLI(Evaluator, as_positional=False)