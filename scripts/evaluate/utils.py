import os
from pathlib import Path
import sys
from typing import OrderedDict
import evaluate
import json
import numpy as np
import pandas as pd
import pickle as pkl
from radgraph import F1RadGraph
import random
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BertTokenizer


from mmmm.data.defs import PROCESSED_VL_DATA_ROOT
from constants import (
    LLAMA3_PATH,
    LLAMA_SYSTEM_PROMPT,
    LLAMA_USER_PROMPT,
    CHEXBERT_PATH,
    RADCLIQ_PATH,
)


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dump_results(
    results: list[dict], output_dir, task: str, dataset: str, model: str, setting: str
):
    results = pd.DataFrame(results)
    results.to_csv(output_dir / f'{task}_{dataset}_{model}_{setting}.csv', index=False)


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


class ReportTestDataset(Dataset):
    def __init__(self, dataset: str):
        super().__init__()
        self.name = dataset
        with open(PROCESSED_VL_DATA_ROOT / dataset / 'test.json') as f:
            self.dataset = [
                {
                    'image': image,
                    'question': (
                            'Can you provide a radiology report for this medical image?'
                            if x.get('impression')
                            else 'Can you provide the findings for this medical image?'
                    ),
                    'answer': (
                        f'Findings: {x["findings"]}\nImpression: {x["impression"]}'
                        if x.get('impression')
                        else x['findings']
                    ),
                }
                for x in json.load(f)
                for image in x['image']
            ]

    def __getitem__(self, index: int):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


class GenericMetrics:
    def __init__(self):
        self.bleu = evaluate.load('bleu')
        self.rouge = evaluate.load('rouge')
        self.meteor = evaluate.load('meteor')
        self.bertscore = evaluate.load('bertscore')
        self.exact_match = evaluate.load('exact_match')

    def compute(self, prediction: str, reference: str):
        prediction = prediction.lower()
        reference = reference.lower()

        return {
            'bleu': (
                self.bleu.compute(
                    predictions=[prediction],
                    references=[[reference]],
                    max_order=1,
                )['bleu']
                if prediction.strip()
                else 0.0
            ),
            'rouge': self.rouge.compute(
                predictions=[prediction], references=[reference]
            )['rouge1'],
            'meteor': self.meteor.compute(
                predictions=[prediction], references=[reference]
            )['meteor'],
            'bertscore': self.bertscore.compute(
                predictions=[prediction],
                references=[reference],
                model_type='distilroberta-base',
            )['f1'][0],
            'exact_match': self.exact_match.compute(
                predictions=[prediction], references=[reference]
            )['exact_match'],
        }

    def process(self, run: Path):
        df = pd.read_csv(str(run) + '.csv')
        if os.path.exists(str(run) + '.json'):
            with open(str(run) + '.json', 'r') as f:
                summary = json.load(f)
        else:
            summary = {}

        results = {
            'bleu': [],
            'rouge': [],
            'meteor': [],
            'bertscore': [],
            'exact_match': [],
        }

        for _, row in tqdm(df.iterrows(), total=df.shape[0]):
            score = self.compute(
                str(row['prediction']) if pd.notna(row['prediction']) else '',
                str(row['answer']),
            )
            for key in score.keys():
                results[key].append(score[key])

        for key in results.keys():
            df[key] = results[key]
            summary[key] = sum(results[key]) / len(results[key])

        df.to_csv(str(run) + '.csv', index=False)
        with open(str(run) + '.json', 'w') as f:
            json.dump(summary, f, indent=4)


class LlamaMetrics:
    def __init__(self):
        from vllm import LLM

        self.llama = LLM(
            model=LLAMA3_PATH, dtype='bfloat16', gpu_memory_utilization=0.9, tensor_parallel_size=2
        )

    def process(self, run: Path):
        from vllm import SamplingParams

        df = pd.read_csv(str(run) + '.csv')
        if os.path.exists(str(run) + '.json'):
            with open(str(run) + '.json', 'r') as f:
                summary = json.load(f)
        else:
            summary = {}

        tokenizer = self.llama.get_tokenizer()
        conversations = [
            tokenizer.apply_chat_template(
                [
                    {'role': 'system', 'content': LLAMA_SYSTEM_PROMPT},
                    {
                        'role': 'user',
                        'content': LLAMA_USER_PROMPT.format(
                            question=str(row['question']),
                            answer=str(row['answer']),
                            prediction=(
                                str(row['prediction'])
                                if pd.notna(row['prediction'])
                                else ''
                            ),
                        ),
                    },
                ],
                tokenize=False,
            )
            for _, row in df.iterrows()
        ]

        sampling_params = SamplingParams(
            max_tokens=256,
            min_tokens=1,
            temperature=0.1,
            stop_token_ids=[
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids('<|eot_id|>'),
            ],
        )

        responses = self.llama.generate(
            prompts=conversations,
            sampling_params=sampling_params,
        )
        response_texts = []
        scores = []
        for i, response in tqdm(enumerate(responses)):
            if pd.isna(df['prediction'].iloc[i]) or not str(df['prediction'].iloc[i]).strip():
                response_texts.append('skipped')
                scores.append(0.0)
            else:
                while True:
                    try:
                        score = float(response.outputs[0].text.split('Rating: ')[1].strip().strip('.'))
                        response_texts.append(response.outputs[0].text)
                        scores.append(score)
                        break
                    except:
                        print(response.outputs[0].text)
                        response = self.llama.generate(
                            prompts=[conversations[i]],
                            sampling_params=sampling_params,
                        )[0]

        df['llama_responses'] = response_texts
        df['llama'] = scores
        summary['llama'] = sum(scores) / len(scores)

        df.to_csv(str(run) + '.csv', index=False)
        with open(str(run) + '.json', 'w') as f:
            json.dump(summary, f, indent=4)


class CXRMetrics:
    def setup_chexbert(self):
        sys.path.append('third-party/CXR-Report-Metric/CXRMetric/')
        sys.path.append('third-party/CXR-Report-Metric/CXRMetric/CheXbert/src')
        from CheXbert.src.models.bert_encoder import bert_encoder

        model = bert_encoder(False)

        checkpoint = torch.load(CHEXBERT_PATH, map_location='cpu')
        state_dict = OrderedDict()
        for key in checkpoint['model_state_dict']:
            state_dict[key[7:]] = checkpoint['model_state_dict'][key]
        model.load_state_dict(state_dict)

        model = model.to('cuda')
        model.eval()

        self.chexbert_model = model
        self.chexbert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def setup_radcliq(self):
        sys.path.append('third-party/CXR-Report-Metric/')
        sys.path.append('third-party/CXR-Report-Metric/CXRMetric/')
        from run_eval import CompositeMetric

        class RadCliQUnpickler(pkl.Unpickler):
            def find_class(self, module, name):
                if name == 'CompositeMetric':
                    return CompositeMetric
                return super().find_class(module, name)

        with open(RADCLIQ_PATH, 'rb') as f:
            self.radcliq = RadCliQUnpickler(f).load()

        self.bleu2 = evaluate.load('bleu')

    def __init__(self):
        self.setup_chexbert()
        self.radgraph = F1RadGraph(
            reward_level='partial', cuda=0, model_type='radgraph'
        )
        self.setup_radcliq()

    def compute_chexbert(self, prediction: str, reference: str):
        sys.path.append('third-party/CXR-Report-Metric/CXRMetric/')
        sys.path.append('third-party/CXR-Report-Metric/CXRMetric/CheXbert/src')
        from CheXbert.src.utils import generate_attention_masks

        with torch.inference_mode():
            prediction = self.chexbert_tokenizer.tokenize(prediction)
            if prediction:
                prediction = self.chexbert_tokenizer.encode_plus(prediction)[
                    'input_ids'
                ]
                if len(prediction) > 512:
                    prediction = prediction[:511] + [
                        self.chexbert_tokenizer.sep_token_id
                    ]
            else:
                prediction = [
                    self.chexbert_tokenizer.cls_token_id
                    + self.chexbert_tokenizer.sep_token_id
                ]
            prediction = torch.LongTensor(prediction).unsqueeze(0).to('cuda')
            attention_mask = torch.ones(1, prediction.shape[1]).to('cuda')
            prediction = (
                self.chexbert_model(prediction, attention_mask).squeeze(0).to('cpu')
            )

            reference = self.chexbert_tokenizer.tokenize(reference)
            if reference:
                reference = self.chexbert_tokenizer.encode_plus(reference)['input_ids']
                if len(reference) > 512:
                    reference = reference[:511] + [self.chexbert_tokenizer.sep_token_id]
            else:
                reference = [
                    self.chexbert_tokenizer.cls_token_id
                    + self.chexbert_tokenizer.sep_token_id
                ]
            reference = torch.LongTensor(reference).unsqueeze(0).to('cuda')
            attention_mask = torch.ones(1, reference.shape[1]).to('cuda')
            reference = (
                self.chexbert_model(reference, attention_mask).squeeze(0).to('cpu')
            )

        return (
            (prediction * reference).sum()
            / (torch.linalg.norm(prediction) * torch.linalg.norm(reference))
        ).item()

    @staticmethod
    def exact_entity_token_if_rel_exists_reward(
        hypothesis_annotation_list, reference_annotation_list
    ):
        candidates = []
        for annotation_list in [hypothesis_annotation_list, reference_annotation_list]:
            candidate = []
            for entity in annotation_list['entities'].values():
                if not entity['relations']:
                    candidate.append((entity['tokens'], entity['label']))
                if entity['relations']:
                    candidate.append((entity['tokens'], entity['label'], True))

            candidate = set(candidate)
            candidates.append(candidate)

        hypothesis_relation_token_list, reference_relation_token_list = candidates

        precision = (
            sum(
                [
                    1
                    for x in hypothesis_relation_token_list
                    if (x in reference_relation_token_list)
                ]
            )
            / len(hypothesis_relation_token_list)
            if len(hypothesis_relation_token_list) > 0
            else 0.0
        )
        recall = (
            sum(
                [
                    1
                    for x in reference_relation_token_list
                    if (x in hypothesis_relation_token_list)
                ]
            )
            / len(reference_relation_token_list)
            if len(reference_relation_token_list) > 0
            else 0.0
        )
        f1_score = (
            (2 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0.0
        )

        return f1_score

    def compute_radgraph(self, prediction: str, reference: str):
        _, _, hyp_annotation_lists, ref_annotation_lists = self.radgraph(
            hyps=[prediction], refs=[reference]
        )
        return CXRMetrics.exact_entity_token_if_rel_exists_reward(
            hyp_annotation_lists[0], ref_annotation_lists[0]
        )

    def compute_bleu2(self, prediction: str, reference: str):
        return (
            self.bleu2.compute(
                predictions=[prediction],
                references=[[reference]],
                max_order=2,
            )['bleu']
            if prediction.strip()
            else 0.0
        )

    def compute(self, prediction: str, reference: str):
        return {
            'chexbert': self.compute_chexbert(prediction, reference),
            'radgraph': self.compute_radgraph(prediction, reference),
            'bleu2': self.compute_bleu2(prediction, reference),
        }

    def process(self, run: Path):
        df = pd.read_csv(str(run) + '.csv')
        if os.path.exists(str(run) + '.json'):
            with open(str(run) + '.json', 'r') as f:
                summary = json.load(f)
        else:
            summary = {}

        results = {
            'chexbert': [],
            'radgraph': [],
            'bleu2': [],
        }

        for _, row in tqdm(df.iterrows(), total=df.shape[0]):
            score = self.compute(
                str(row['prediction']) if pd.notna(row['prediction']) else '',
                str(row['answer']),
            )
            for key in score.keys():
                results[key].append(score[key])

        for key in results.keys():
            df[key] = results[key]

        results['radcliq'] = self.radcliq.predict(
            np.array(df[['radgraph', 'bertscore', 'chexbert', 'bleu2']])
        )
        df['radcliq'] = results['radcliq']
        df = df.drop(columns=['bleu2'])

        for key in results.keys():
            summary[key] = sum(results[key]) / len(results[key])

        df.to_csv(str(run) + '.csv', index=False)
        with open(str(run) + '.json', 'w') as f:
            json.dump(summary, f, indent=4)
