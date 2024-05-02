from einops import repeat
import json
from jsonargparse import CLI
import pandas as pd
from pathlib import Path
import random
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from llavamed import llavamed_collate_fn, setup_llavamed, KeywordsStoppingCriteria
from m3d import m3d_collate_fn, setup_m3d
from medflamingo import medflamingo_collate_fn, setup_medflamingo
from mmmm.data.defs import PROCESSED_VL_DATA_ROOT
from radfm import radfm_collate_fn, setup_radfm
from utils import (
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


class VLEvaluator:
    def __init__(
        self,
        checkpoint: str,
        tokenizer: str,
        task: str,
        dataset: str,
        seed: int = 233,
        num_workers: int = 8,
        output_dir: str = 'results/',
    ):
        self.checkpoint = checkpoint
        self.tokenizer = tokenizer
        self.task = task
        if task == 'vqa':
            self.dataset = VQATestDataset(dataset)
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
        results = []

        for sample in tqdm(dataloader):
            question_list = [False for _ in range(len(str(sample['question'])))]
            question = ''
            if random.random() < 0.5:
                position = 0
            else:
                position = len(question_list) - 1
            question_list[position] = True
            for i in range(len(question_list)):
                if question_list[i]:
                    question += (
                        '<image>'
                        + ''.join([f'<image{i}>' for i in range(32)])
                        + '</image>'
                        + sample['question'][i]
                    )
                else:
                    question += sample['question'][i]
            language = tokenizer(
                question,
                return_tensors='pt',
            )[
                'input_ids'
            ].to('cuda')
            vision = sample['image'].to('cuda')
            prediction = tokenizer.decode(
                model.generate(language, vision)[0], skip_special_tokens=True
            ).strip()

            results.append(
                {
                    'question': sample['question'],
                    'answer': sample['answer'],
                    'prediction': prediction,
                    **self.metrics.compute(prediction, sample['answer']),
                },
            )

            print(sample['question'])
            print(sample['answer'])
            print(prediction)

        results = pd.DataFrame(results)
        results.to_csv(
            self.output_dir / f'{self.task}_{self.dataset.name}_radfm_zeroshot.csv'
        )
        with open(
            self.output_dir / f'{self.task}_{self.dataset.name}_radfm_zeroshot.json',
            'w',
        ) as f:
            json.dump(
                {
                    'bleu': results['bleu'].mean(),
                    'rouge': results['rouge'].mean(),
                    'meteor': results['meteor'].mean(),
                    'bertscore': results['bertscore'].mean(),
                    'exact_match': results['exact_match'].mean(),
                },
                f,
                indent=4,
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
        results = []

        for sample in tqdm(dataloader):
            language = processor.encode_text(
                'You are a helpful medical assistant. You are being provided with images and a question about the image, please answer the question. <image>Question:'
                + sample['question']
                + ' Answer:'
            )
            vision = processor.preprocess_images([sample['image']])
            vision = repeat(vision, '1 c h w -> 1 1 1 c h w')
            prediction = (
                processor.tokenizer.decode(
                    model.generate(
                        vision_x=vision.to('cuda'),
                        lang_x=language['input_ids'].to('cuda'),
                        attention_mask=language['attention_mask'].to('cuda'),
                        max_new_tokens=50,
                    )[0]
                )
                .replace('<unk> ', '')
                .split('Answer:')[1]
                .strip()
                .split('\n')[0]
            )

            results.append(
                {
                    'question': sample['question'],
                    'answer': sample['answer'],
                    'prediction': prediction,
                    **self.metrics.compute(prediction, sample['answer']),
                },
            )

            print(sample['question'])
            print(sample['answer'])
            print(prediction)

        results = pd.DataFrame(results)
        results.to_csv(
            self.output_dir
            / f'{self.task}_{self.dataset.name}_medflamingo_zeroshot.csv'
        )
        with open(
            self.output_dir
            / f'{self.task}_{self.dataset.name}_medflamingo_zeroshot.json',
            'w',
        ) as f:
            json.dump(
                {
                    'bleu': results['bleu'].mean(),
                    'rouge': results['rouge'].mean(),
                    'meteor': results['meteor'].mean(),
                    'bertscore': results['bertscore'].mean(),
                    'exact_match': results['exact_match'].mean(),
                },
                f,
                indent=4,
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

        results = []

        for sample in tqdm(dataloader):
            language = tokenizer(
                '<im_patch>' * 256 + ' ' + sample['question'],
                return_tensors='pt',
            )['input_ids'].to('cuda')
            vision = sample['image'].to(device='cuda', dtype=torch.bfloat16)
            prediction = tokenizer.decode(
                model.generate(vision, language, max_new_tokens=256)[0],
                skip_special_tokens=True,
            ).strip()

            results.append(
                {
                    'question': sample['question'],
                    'answer': sample['answer'],
                    'prediction': prediction,
                    **self.metrics.compute(prediction, sample['answer']),
                },
            )

            print(sample['question'])
            print(sample['answer'])
            print(prediction)

        results = pd.DataFrame(results)
        results.to_csv(
            self.output_dir / f'{self.task}_{self.dataset.name}_m3d_zeroshot.csv'
        )
        with open(
            self.output_dir / f'{self.task}_{self.dataset.name}_m3d_zeroshot.json', 'w'
        ) as f:
            json.dump(
                {
                    'bleu': results['bleu'].mean(),
                    'rouge': results['rouge'].mean(),
                    'meteor': results['meteor'].mean(),
                    'bertscore': results['bertscore'].mean(),
                    'exact_match': results['exact_match'].mean(),
                },
                f,
                indent=4,
            )

    def llavamed(self):
        sys.path.append('third-party/LLaVA-Med')
        from llava.conversation import conv_templates

        model, tokenizer, image_processor, image_token_len = setup_llavamed(
            self.checkpoint, self.tokenizer
        )

        dataloader = DataLoader(
            self.dataset,
            batch_size=1,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=llavamed_collate_fn,
        )

        results = []

        for sample in tqdm(dataloader):
            question = sample['question']
            if getattr(model.config, 'mm_use_im_start_end', False):
                question = (
                    question
                    + '\n'
                    + '<im_start>'
                    + '<im_patch>' * image_token_len
                    + '<im_end>'
                )
            else:
                question = question + '\n' + '<im_patch>' * image_token_len
            conv = conv_templates['simple'].copy()
            conv.append_message(conv.roles[0], question)
            prompt = conv.get_prompt()
            language = torch.as_tensor(tokenizer([prompt]).input_ids).to('cuda')
            stopping_criteria = KeywordsStoppingCriteria(['###'], tokenizer, language)

            vision = image_processor.preprocess(sample['image'], return_tensors='pt')[
                'pixel_values'
            ][0]
            vision = repeat(vision, 'c h w -> 1 c h w').half().to('cuda')

            prediction = tokenizer.decode(
                model.generate(
                    language,
                    images=vision,
                    do_sample=True,
                    temperature=0.7,
                    stopping_criteria=[stopping_criteria],
                    max_new_tokens=1024,
                )[0, language.shape[1]:],
                skip_special_tokens=True,
            )

            try:
                index = prediction.index(conv.sep)
            except ValueError:
                prediction += conv.sep
                index = prediction.index(conv.sep)

            prediction = prediction[:index].split('Assistant: ')[1].strip()

            results.append(
                {
                    'question': sample['question'],
                    'answer': sample['answer'],
                    'prediction': prediction,
                    **self.metrics.compute(prediction, sample['answer']),
                },
            )

            print(sample['question'])
            print(sample['answer'])
            print(prediction)

        results = pd.DataFrame(results)
        results.to_csv(
            self.output_dir / f'{self.task}_{self.dataset.name}_llavamed_zeroshot.csv'
        )
        with open(
            self.output_dir / f'{self.task}_{self.dataset.name}_llavamed_zeroshot.json',
            'w',
        ) as f:
            json.dump(
                {
                    'bleu': results['bleu'].mean(),
                    'rouge': results['rouge'].mean(),
                    'meteor': results['meteor'].mean(),
                    'bertscore': results['bertscore'].mean(),
                    'exact_match': results['exact_match'].mean(),
                },
                f,
                indent=4,
            )


if __name__ == '__main__':
    CLI(VLEvaluator, as_positional=False)
