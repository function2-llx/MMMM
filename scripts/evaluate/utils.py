import os
from pathlib import Path
import sys
from typing import OrderedDict
import evaluate
import json
import numpy as np
import pandas as pd
import random
import re
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import transformers
from transformers import BertTokenizer


from mmmm.data.defs import PROCESSED_VL_DATA_ROOT
from mmmm.data.dataset.vl import REPORT_PROMPTS, FINDINGS_PROMPT, COMPLETE_REFERRINGS


LLAMA3_PATH = '/data/llama3/Meta-Llama-3-8B-Instruct-hf'
CHEXBERT_PATH = '/data/chexbert/chexbert.pth'
RADGRAPH_PATH = '/data/MMMM/RadGraph/models/model_checkpoint'


LLAMA_SYSTEM_PROMPT = '''
You are an AI assistant specialized in medical topics. 
'''

LLAMA_USER_PROMPT = '''
You are given the question, ground truth and prediction of a medical visual question answering task. Your task is to evaluate the prediction based on the question and ground truth in terms of medical knowledge. You should consider various aspects, such as the correctness, completeness and relevance of the prediction. You should provide a final score from 0 to 10 to summarize the overall quality of the prediction. The output format is 'score: xx'. Do not output anything else other than the score.
question: {question}
ground truth: {answer}
prediction: {prediction}
'''


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dump_results(results: list[dict], output_dir, task: str, dataset: str, model: str, setting: str):
    results = pd.DataFrame(results)
    results.to_csv(
        output_dir / f'{task}_{dataset}_{model}_{setting}.csv'
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


class ReportTestDataset(Dataset):
    def __init__(self, dataset: str):
        super().__init__()
        self.name = dataset
        with open(PROCESSED_VL_DATA_ROOT / dataset / 'test.json') as f:
            self.dataset = [
                {
                    'image': image,
                    'question': (
                        random.choice(REPORT_PROMPTS).format(
                            random.choice(COMPLETE_REFERRINGS)
                        )
                        if x.get('impression')
                        else random.choice(FINDINGS_PROMPT).format(
                            random.choice(COMPLETE_REFERRINGS)
                        )
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
                model_type='microsoft/deberta-xlarge-mnli',
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

        for _, row in df.iterrows():
            score = self.compute(str(row['prediction']) if pd.notna(row['prediction']) else '', str(row['answer']))
            for key in results.keys():
                results[key].append(score[key])

        for key in results.keys():
            df[key] = results[key]
            summary[key] = sum(results[key]) / len(results[key])

        df.to_csv(str(run) + '.csv')
        with open(str(run) + '.json', 'w') as f:
            json.dump(summary, f, indent=4)


class LlamaMetrics:
    def __init__(self):
        self.llama = transformers.pipeline(
            'text-generation',
            model=LLAMA3_PATH,
            model_kwargs={'torch_dtype': torch.bfloat16},
            device_map='auto',
        )

    def compute(self, question: str, prediction: str, reference: str):
        if not prediction:
            return {
                'llama': 0
            }

        messages = [
            {'role': 'system', 'content': LLAMA_SYSTEM_PROMPT},
            {'role': 'user', 'content': LLAMA_USER_PROMPT.format(question=question, answer=reference, prediction=prediction)},
        ]

        prompt = self.llama.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        terminators = [
            self.llama.tokenizer.eos_token_id,
            self.llama.tokenizer.convert_tokens_to_ids('<|eot_id|>')
        ]

        while True:
            try:
                response = self.llama(
                    prompt,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.2,
                    eos_token_id=terminators,
                    pad_token_id=self.llama.tokenizer.eos_token_id,
                )[0]['generated_text'][len(prompt):]
                return {
                    'llama': int(response.split('score: ')[1].strip()),
                }
            except:
                continue

    def process(self, run: Path):
        df = pd.read_csv(str(run) + '.csv')
        if os.path.exists(str(run) + '.json'):
            with open(str(run) + '.json', 'r') as f:
                summary = json.load(f)
        else:
            summary = {}

        llama = []

        for _, row in tqdm(df.iterrows()):
            score = self.compute(row['question'], str(row['prediction']) if pd.notna(row['prediction']) else '', str(row['answer']))['llama']
            llama.append(score)

        df['llama'] = llama
        summary['llama'] = sum(llama) / len(llama)

        df.to_csv(str(run) + '.csv')
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

    def __init__(self):
        self.setup_chexbert()

    def compute_chexbert(self, prediction: str, reference: str):
        sys.path.append('third-party/CXR-Report-Metric/CXRMetric/')
        sys.path.append('third-party/CXR-Report-Metric/CXRMetric/CheXbert/src')
        from CheXbert.src.utils import generate_attention_masks

        with torch.inference_mode():
            prediction = self.chexbert_tokenizer.tokenize(prediction)
            if prediction:
                prediction = self.chexbert_tokenizer.encode_plus(prediction)['input_ids']
                if len(prediction) > 512:
                    prediction = prediction[:511] + [self.chexbert_tokenizer.sep_token_id]
            else:
                prediction = [self.chexbert_tokenizer.cls_token_id + self.chexbert_tokenizer.sep_token_id]
            prediction = torch.LongTensor(prediction).unsqueeze(0).to('cuda')
            attention_mask = torch.ones(1, prediction.shape[1]).to('cuda')
            prediction = self.chexbert_model(prediction, attention_mask).squeeze(0).to('cpu')

            reference = self.chexbert_tokenizer.tokenize(reference)
            if reference:
                reference = self.chexbert_tokenizer.encode_plus(reference)['input_ids']
                if len(reference) > 512:
                    reference = reference[:511] + [self.chexbert_tokenizer.sep_token_id]
            else:
                reference = [self.chexbert_tokenizer.cls_token_id + self.chexbert_tokenizer.sep_token_id]
            reference = torch.LongTensor(reference).unsqueeze(0).to('cuda')
            attention_mask = torch.ones(1, reference.shape[1]).to('cuda')
            reference = self.chexbert_model(reference, attention_mask).squeeze(0).to('cpu')

        return ((prediction * reference).sum() / (torch.linalg.norm(prediction) * torch.linalg.norm(reference))).item()


    def compute_radgraph(self, prediction: str, reference: str):
        pass
    
    def compute(self, prediction: str, reference: str):
        return {
            'chexbert': self.compute_chexbert(prediction, reference),
        }
    
    @staticmethod
    def preprocess_radgraph(run: Path):
        sys.path.append('third-party/CXR-Report-Metric/CXRMetric/radgraph_inference')
        from inference import get_entity

        if os.path.exists(str(run) + '_gt_radgraph.json') and os.path.exists(str(run) + '_pred_radgraph.json'):
            return
        df = pd.read_csv(str(run) + '.csv')
        gt = []
        for i, row in df.iterrows():
            sen = re.sub('(?<! )(?=[/,-,:,.,!?()])|(?<=[/,-,:,.,!?()])(?! )', r' ', row['answer']).split()
            gt.append({
                'doc_key': str(i),
                'sentences': [sen],
            })
        with open(str(run) + '_gt_radgraph.in.json', 'w') as f:
            json.dump(gt, f, indent=4)
        pred = []
        for i, row in df.iterrows():
            sen = re.sub('(?<! )(?=[/,-,:,.,!?()])|(?<=[/,-,:,.,!?()])(?! )', r' ', row['prediction']).split()
            pred.append({
                'doc_key': str(i),
                'sentences': [sen],
            })
        with open(str(run) + '_pred_radgraph.in.json', 'w') as f:
            json.dump(pred, f, indent=4)

        os.system(
            f"allennlp predict {RADGRAPH_PATH} {str(run) + '_gt_radgraph.in.json'} \
            --predictor dygie --include-package dygie \
            --use-dataset-reader \
            --output-file {str(run) + '_gt_radgraph.json'} \
            --cuda-device 0"
        )
        os.system(
            f"allennlp predict {RADGRAPH_PATH} {str(run) + '_pred_radgraph.in.json'} \
            --predictor dygie --include-package dygie \
            --use-dataset-reader \
            --output-file {str(run) + '_pred_radgraph.json'} \
            --cuda-device 0"
        )

        gt = {}
        data = []
        with open(str(run) + '_gt_radgraph.json', 'r') as f:
            for line in f:
                data.append(json.loads(line))

        for sample in data:
            item = {}

            item['text'] = ' '.join(sample['sentences'][0])
            ner = sample['predicted_ner'][0]
            relations = sample['predicted_relations'][0]
            sentences = sample['sentences'][0]
            sample['entities'] = get_entity(ner, relations, sentences)
            gt[sample['doc_key']] = item

        pred = {}
        data = []
        with open(str(run) + '_pred_radgraph.json', 'r') as f:
            for line in f:
                data.append(json.loads(line))
            
        for sample in data:
            item = {}

            item['text'] = ' '.join(sample['sentences'][0])
            ner = sample['predicted_ner'][0]
            relations = sample['predicted_relations'][0]
            sentences = sample['sentences'][0]
            sample['entities'] = get_entity(ner, relations, sentences)
            pred[sample['doc_key']] = item

    def process(self, run: Path):
        # self.preprocess_radgraph(run)
        df = pd.read_csv(str(run) + '.csv')
        if os.path.exists(str(run) + '.json'):
            with open(str(run) + '.json', 'r') as f:
                summary = json.load(f)
        else:
            summary = {}

        results = {
            'chexbert': [],
        }

        for _, row in tqdm(df.iterrows()):
            score = self.compute(str(row['prediction']) if pd.notna(row['prediction']) else '', str(row['answer']))
            for key in results.keys():
                results[key].append(score[key])

        for key in results.keys():
            df[key] = results[key]
            summary[key] = sum(results[key]) / len(results[key])

        df.to_csv(str(run) + '.csv')
        with open(str(run) + '.json', 'w') as f:
            json.dump(summary, f, indent=4)