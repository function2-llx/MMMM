import evaluate
import json
from jsonargparse import CLI
import pandas as pd
from pathlib import Path
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import LlamaTokenizer

from mmmm.data.defs import PROCESSED_VL_DATA_ROOT
from utils import setup_seed, radfm_collate_fn, NLPMetrics


class VQATestDataset(Dataset):
    def __init__(self, dataset: str):
        super().__init__()
        with open(PROCESSED_VL_DATA_ROOT / dataset / 'test.json') as f:
            self.dataset = [
                {'image': x['image'][0], **vqa}
                for x in json.load(f)
                for vqa in x['vqa']
            ]

    def __getitem__(self, index: int):
        return self.dataset[index]
    
    def __len__(self):
        return len(self.dataset)


class VQAEvaluator:
    def __init__(
        self,
        checkpoint: str,
        tokenizer: str,
        dataset: str,
        seed: int = 233,
        num_workers: int = 8,
        output_dir: str = 'results/',
    ):
        self.checkpoint = checkpoint
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.num_workers = num_workers
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = NLPMetrics()
        setup_seed(seed)

    def radfm(self):
        sys.path.append('third-party/RadFM/Quick_demo/Model')
        from RadFM.multimodality_model import MultiLLaMAForCausalLM

        model = MultiLLaMAForCausalLM(
            lang_model_path=self.tokenizer,
        )
        checkpoint = torch.load(self.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint)
        model = model.to('cuda')
        model.eval()
        dataset = VQATestDataset(self.dataset)
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=radfm_collate_fn,
        )
        results = []
        tokenizer = LlamaTokenizer.from_pretrained(self.tokenizer)
        special_tokens = {
            'additional_special_tokens': [f'<image{i}>' for i in range(32)]
            + ['<image>', '</image>']
        }
        tokenizer.add_special_tokens(special_tokens)
        tokenizer.pad_token_id = 0
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2

        for sample in tqdm(dataloader):
            language = tokenizer(
                sample['question'],
                max_length=2048,
                return_tensors='pt',
                padding=True,
                truncation=True,
            )['input_ids'].to('cuda')
            vision = sample['image'].to('cuda')
            prediction = tokenizer.decode(
                model.generate(language, vision)[0], skip_special_tokens=True
            )
            results.append(
                {
                    'question': sample['question'],
                    'answer': sample['answer'],
                    'prediction': prediction,
                    **self.metrics.compute(prediction, sample['answer'])
                },
            )

            print(sample['question'])
            print(sample['answer'])
            print(prediction)

        results = pd.DataFrame(results)
        results.to_csv(
            self.output_dir / f'{self.dataset}_radfm_zeroshot.csv'
        )
        with open(self.output_dir / f'{self.dataset}_radfm_zeroshot.json', 'w') as f:
            json.dump({
                'bleu': results['bleu'].mean(),
                'rouge': results['rouge'].mean(),
                'meteor': results['meteor'].mean(),
                'bertscore': results['bertscore'].mean(),
                'exact_match': results['exact_match'].mean(),
            }, f, indent=4)


if __name__ == '__main__':
    CLI(VQAEvaluator, as_positional=False)
