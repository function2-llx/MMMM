from functools import partial
from pathlib import Path
from typing import Optional

from jsonargparse import CLI
from torch.utils.data import DataLoader

from utils import (
    collate_fn,
    setup_seed,
    VQATestDataset,
    ReportTestDataset,
    GenericMetrics,
    LlamaMetrics,
    CXRMetrics,
    CTMetrics
)


class Evaluator:
    def __init__(
        self,
        model: str,
        task: str,
        dataset: str,
        setting: str,
        seed: int = 233,
        output_dir: str = '/data/MMMM/results/',
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
        checkpoint: Optional[str] = None,
        adapter: Optional[str] = None,
        tokenizer: Optional[str] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ):

        if self.model == 'mmmm':
            from models.mmmm import (
                MMMMTransform,
                setup_mmmm,
                mmmm_vl_evaluate,
            )
            packed = setup_mmmm(adapter)
            transform = MMMMTransform(packed[1], self.task)
            evaluate_fn = mmmm_vl_evaluate
        if self.model == 'radfm':
            from models.radfm import RadFMTransform, setup_radfm, radfm_vl_evaluate
            packed = setup_radfm(checkpoint, adapter, tokenizer)
            transform = RadFMTransform(packed[1])
            evaluate_fn = radfm_vl_evaluate
        elif self.model == 'm3d':
            from models.m3d import M3DTransform, setup_m3d, m3d_vl_evaluate
            packed = setup_m3d(checkpoint, adapter, tokenizer)
            transform = M3DTransform(packed[1])
            evaluate_fn = m3d_vl_evaluate
        elif self.model == 'llavamed':
            from models.llavamed import LlavaMedTransform, setup_llavamed, llavamed_vl_evaluate
            packed = setup_llavamed(checkpoint, adapter, tokenizer)
            transform = LlavaMedTransform(packed[0].config, *packed[1:], self.task, self.setting)
            evaluate_fn = llavamed_vl_evaluate
        elif self.model == 'cogvlm':
            from models.cogvlm import (
                CogVLMTransform,
                setup_cogvlm,
                cogvlm_vl_evaluate,
            )
            packed = setup_cogvlm(checkpoint, adapter, tokenizer)
            transform = CogVLMTransform(
                packed[0].build_conversation_input_ids, packed[1]
            )
            evaluate_fn = cogvlm_vl_evaluate
        elif self.model == 'llavanext':
            from models.llavanext import (
                LlavaNextTransform,
                setup_llavanext,
                llavanext_vl_evaluate,
                build_conversation_input_ids,
            )
            packed = setup_llavanext(checkpoint, adapter, tokenizer)
            transform = LlavaNextTransform(
                (
                    partial(build_conversation_input_ids, packed[1].tokenizer)
                    if self.setting == 'finetuned'
                    else packed[1]
                ),
                self.task,
            )
            evaluate_fn = llavanext_vl_evaluate
        elif self.model == 'instructblip':
            from models.instructblip import (
                InstructBlipTransform,
                setup_instructblip,
                instructblip_vl_evaluate,
            )
            packed = setup_instructblip(checkpoint, tokenizer)
            transform = InstructBlipTransform(
                packed[1], self.setting,
            )
            evaluate_fn = instructblip_vl_evaluate
        elif self.model == 'r2gengpt':
            from models.r2gengpt import R2GenGPTTransform, setup_r2gengpt, r2gengpt_vl_evaluate
            packed = setup_r2gengpt(adapter)
            transform = R2GenGPTTransform(packed[1])
            evaluate_fn = r2gengpt_vl_evaluate

        if self.task == 'vqa':
            dataset = VQATestDataset(self.dataset, transform, start, end)
        elif self.task == 'report':
            dataset = ReportTestDataset(self.dataset, transform, start, end)

        dataloader = DataLoader(
            dataset,
            batch_size=1,
            collate_fn=collate_fn,
            num_workers=16,
        )

        evaluate_fn(
            *packed,
            dataloader,
            f'{self.output_dir}/{self.task}_{self.dataset}_{self.model}_{self.setting}.csv',
        )

    def evaluate(self, metrics: str):
        if metrics == 'generic':
            metrics = GenericMetrics()
        elif metrics == 'llama':
            metrics = LlamaMetrics()
        elif metrics == 'cxr':
            metrics = CXRMetrics()
        elif metrics == 'ct':
            metrics = CTMetrics()
        metrics.process(
            self.output_dir / f'{self.task}_{self.dataset}_{self.model}_{self.setting}',
        )


if __name__ == '__main__':
    CLI(Evaluator, as_positional=False)
