import json
import os
import pickle as pkl
import random
import sys
from pathlib import Path

import evaluate
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from radgraph import F1RadGraph
from scipy.special import expit
from sklearn.metrics import f1_score
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, BertTokenizer

from constants import (
    LLAMA3_PATH,
    LLAMA_SYSTEM_PROMPT,
    LLAMA_FINETUNED_USER_PROMPT,
    CHEXBERT_PATH,
    NORMALIZER_PATH,
    COMPOSITE_METRIC_V0_PATH,
    COMPOSITE_METRIC_V1_PATH,
    CHEXPERT_CONDITIONS,
    RADBERT_CONDITIONS,
    CHEXPERT_5,
)
from mmmm.data.defs import ORIGIN_VL_DATA_ROOT, PROCESSED_VL_DATA_ROOT
from monai.transforms import apply_transform


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dump_results(results: list[dict], output: Path):
    results = pd.DataFrame(results)
    results.to_csv(output, index=False)


def combine_results(results: list[str], output: Path):
    combined = pd.concat([pd.read_csv(result) for result in results])
    combined.to_csv(output, index=False)


class VQATestDataset(Dataset):
    def __init__(self, dataset: str, transform, start, end):
        super().__init__()
        self.name = dataset
        self.transform = transform
        with open(PROCESSED_VL_DATA_ROOT / dataset / 'test.json') as f:
            self.dataset = [
                {'image': image, **vqa}
                for x in json.load(f)
                for vqa in x['vqa']
                for image in x['image']
            ][start:end]

    def __getitem__(self, index: int):
        return apply_transform(self.transform, self.dataset[index])

    def __len__(self):
        return len(self.dataset)


class ReportTestDataset(Dataset):
    def __init__(self, dataset: str, transform, start, end):
        super().__init__()
        self.name = dataset
        self.transform = transform
        with open(PROCESSED_VL_DATA_ROOT / dataset / 'test-processed.json') as f:
            self.dataset = [
                {
                    'image': image,
                    'question': 'Please write a radiology report for me:',
                    'answer': (x['processed_report']),
                }
                for x in json.load(f)
                for i, image in enumerate(x['image'])
                if (self.name != 'MIMIC-CXR' or (x['plane'][i] == 'AP' or x['plane'][i] == 'PA')) and (self.name != 'OpenI' or x['plane'][i] == 'frontal')
            ][start:end]

    def __getitem__(self, index: int):
        return apply_transform(self.transform, self.dataset[index])

    def __len__(self):
        return len(self.dataset)


def collate_fn(batch: list[dict]):
    return batch[0]


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
            'bleu1': (
                self.bleu.compute(
                    predictions=[prediction],
                    references=[[reference]],
                    max_order=1,
                )['bleu']
                if prediction.strip()
                else 0.0
            ),
            'bleu2': (
                self.bleu.compute(
                    predictions=[prediction],
                    references=[[reference]],
                    max_order=2,
                )['bleu']
                if prediction.strip()
                else 0.0
            ),
            'bleu4': (
                self.bleu.compute(
                    predictions=[prediction],
                    references=[[reference]],
                    max_order=4,
                )['bleu']
                if prediction.strip()
                else 0.0
            ),
            'rouge1': self.rouge.compute(
                predictions=[prediction], references=[reference]
            )['rouge1'],
            'rougeL': self.rouge.compute(
                predictions=[prediction], references=[reference]
            )['rougeL'],
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
            'bleu1': [],
            'bleu2': [],
            'bleu4': [],
            'rouge1': [],
            'rougeL': [],
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
            model=LLAMA3_PATH,
            dtype='bfloat16',
            gpu_memory_utilization=0.83,
            tensor_parallel_size=4,
            enable_prefix_caching=True,
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
        sampling_params = SamplingParams(
            max_tokens=1024,
            min_tokens=1,
            temperature=0.1,
            stop_token_ids=[
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids('<|eot_id|>'),
            ],
        )

        conversations = [
            tokenizer.apply_chat_template(
                [
                    {'role': 'system', 'content': LLAMA_SYSTEM_PROMPT},
                    {
                        'role': 'user',
                        'content': LLAMA_FINETUNED_USER_PROMPT.format(
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
                add_generation_prompt=True,
            )
            for _, row in df.iterrows()
        ]

        responses = self.llama.generate(
            prompts=conversations,
            sampling_params=sampling_params,
        )
        response_texts = []
        scores = []
        for i, response in tqdm(enumerate(responses)):
            retry = 0
            while True:
                try:
                    score = float(
                        response.outputs[0].text.split('Score: ')[1].strip().strip('.')
                    )
                    response_texts.append(response.outputs[0].text)
                    scores.append(score)
                    break
                except:
                    retry += 1
                    if retry > 3:
                        response_texts.append(response.outputs[0].text)
                        scores.append(0.0)
                        break
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

def compute_proportion(numerator: np.ndarray, denominator: np.ndarray) -> tuple[np.ndarray, float, float]:
    """
    multi-label format
    Args:
        numerator: (c, )
        denominator: (c, )
    Returns:
        - (c, ) proportion for each class
        - macro average
        - micro average
    """
    return numerator / denominator, (numerator / denominator).mean().item(), (numerator.sum() / denominator.sum()).item()

def false_negative_rate(ref: np.ndarray, pred: np.ndarray):
    ref = ref.astype(np.bool_)
    pred = pred.astype(np.bool_)
    fn = (ref & ~pred).sum(axis=0)
    n: npt.NDArray[np.int64] = (~pred).sum(axis=0)
    return compute_proportion(fn, n)

class CXRMetrics:
    def setup_chexbert(self):
        from CheXbert.models.bert_encoder import bert_encoder

        model = bert_encoder(False)

        checkpoint = torch.load(CHEXBERT_PATH, map_location='cpu')
        state_dict = dict()
        for key in checkpoint['model_state_dict']:
            state_dict[key[7:]] = checkpoint['model_state_dict'][key]
        model.load_state_dict(state_dict)

        model = model.to('cuda')
        model.eval()

        self.chexbert_model = model
        self.chexbert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def setup_radcliq(self):
        from CXRMetric.run_eval import CompositeMetric

        class RadCliQUnpickler(pkl.Unpickler):
            def find_class(self, module, name):
                if name == 'CompositeMetric':
                    return CompositeMetric
                return super().find_class(module, name)

        with open(NORMALIZER_PATH, 'rb') as f:
            self.normalizer = pkl.load(f)

        with open(COMPOSITE_METRIC_V0_PATH, 'rb') as f:
            self.radcliq_v0 = RadCliQUnpickler(f).load()

        with open(COMPOSITE_METRIC_V1_PATH, 'rb') as f:
            self.radcliq_v1 = RadCliQUnpickler(f).load()

        self.bleu2 = evaluate.load('bleu')

    def __init__(self):
        self.setup_chexbert()
        self.radgraph = F1RadGraph(
            reward_level='partial', cuda=0, model_type='radgraph'
        )
        self.setup_radcliq()

    def chexbert_tokenize(self, text: str):
        text = self.chexbert_tokenizer.tokenize(text)
        if text:
            text = self.chexbert_tokenizer.encode_plus(text)['input_ids']
            if len(text) > 512:
                text = text[:511] + [self.chexbert_tokenizer.sep_token_id]
        else:
            text = [
                self.chexbert_tokenizer.cls_token_id
                + self.chexbert_tokenizer.sep_token_id
            ]
        return torch.LongTensor(text).unsqueeze(0).to('cuda')

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
            'radgraph': self.compute_radgraph(prediction, reference),
            'bleu2': self.compute_bleu2(prediction, reference),
        }

    def compute_radcliq(self, df: pd.DataFrame):
        return self.radcliq_v0.predict(
            self.normalizer.transform(
                np.array(df[['radgraph', 'bertscore', 'chexbert', 'bleu2']])
            )
        ), self.radcliq_v1.predict(
            np.array(df[['radgraph', 'bertscore', 'chexbert', 'bleu2']])
        )

    def compute_chexbert(self, df: pd.DataFrame):
        similarities = []
        prediction_labels = []
        reference_labels = []
        for _, row in tqdm(df.iterrows(), total=df.shape[0], desc='computing CheXbert'):
            self.chexbert_model.logits = True
            prediction = self.chexbert_tokenize(
                str(row['prediction']) if pd.notna(row['prediction']) else ''
            )
            prediction_attention_mask = torch.ones(1, prediction.shape[1]).to('cuda')
            logits = self.chexbert_model(prediction, prediction_attention_mask)
            prediction_label = []
            for i in range(len(CHEXPERT_CONDITIONS)):
                prediction_label.append(torch.argmax(logits[i], dim=1).item())
            prediction_labels.append(prediction_label)

            reference = self.chexbert_tokenize(str(row['answer']))
            reference_attention_mask = torch.ones(1, reference.shape[1]).to('cuda')
            logits = self.chexbert_model(reference, reference_attention_mask)
            reference_label = []
            for i in range(len(CHEXPERT_CONDITIONS)):
                reference_label.append(torch.argmax(logits[i], dim=1).item())
            reference_labels.append(reference_label)

            self.chexbert_model.logits = False
            prediction = (
                self.chexbert_model(prediction, prediction_attention_mask)
                .squeeze(0)
                .to('cpu')
            )
            reference = (
                self.chexbert_model(reference, reference_attention_mask)
                .squeeze(0)
                .to('cpu')
            )
            similarities.append(
                (
                    (prediction * reference).sum()
                    / (torch.linalg.norm(prediction) * torch.linalg.norm(reference))
                ).item()
            )

        self.chexbert_model.logits = False

        prediction_labels = [
            [1 if x == 1 or x == 3 else 0 for x in y] for y in prediction_labels
        ]
        reference_labels = [
            [1 if x == 1 or x == 3 else 0 for x in y] for y in reference_labels
        ]

        return similarities, np.array(prediction_labels), np.array(reference_labels)

    def process(self, run: Path):
        df = pd.read_csv(str(run) + '.csv')
        if os.path.exists(str(run) + '.json'):
            with open(str(run) + '.json', 'r') as f:
                summary = json.load(f)
        else:
            summary = {}

        results = {
            'radgraph': [],
            'bleu2': [],
        }

        results['chexbert'], prediction_labels, reference_labels= (
            self.compute_chexbert(df)
        )
        f1s = f1_score(reference_labels, prediction_labels, average=None)

        summary['macro chexbert 14 f1'] = f1_score(reference_labels, prediction_labels, average='macro')
        summary['micro chexbert 14 f1'] = f1_score(reference_labels, prediction_labels, average='micro')
        fnr, summary['macro chexbert 14 fnr'], summary['micro chexbert 14 fnr'] = false_negative_rate(reference_labels, prediction_labels)

        for i, condition in enumerate(CHEXPERT_CONDITIONS):
            condition = condition.lower()
            df[condition + ' chexbert prediction'] = prediction_labels[:, i]
            df[condition + ' chexbert reference'] = reference_labels[:, i]
            summary[condition + ' chexbert f1'] = f1s[i]
            summary[condition + ' chexbert fnr'] = fnr[i]

        summary['macro chexbert 5 f1'] = f1_score(reference_labels[:, CHEXPERT_5], prediction_labels[:, CHEXPERT_5], average='macro')
        summary['micro chexbert 5 f1'] = f1_score(reference_labels[:, CHEXPERT_5], prediction_labels[:, CHEXPERT_5], average='micro')
        # yes, lazy me
        _, summary['macro chexbert 5 fnr'], summary['micro chexbert 5 fnr'] = false_negative_rate(
            reference_labels[:, CHEXPERT_5], prediction_labels[:, CHEXPERT_5],
        )

        for _, row in tqdm(df.iterrows(), total=df.shape[0], desc='computing CXRMetrics'):
            score = self.compute(
                str(row['prediction']) if pd.notna(row['prediction']) else '',
                str(row['answer']),
            )
            for key in score.keys():
                results[key].append(score[key])

        for key in results.keys():
            df[key] = results[key]

        results['radcliq-v0'], results['radcliq-v1'] = self.compute_radcliq(df)
        df['radcliq-v0'], df['radcliq-v1'] = (
            results['radcliq-v0'],
            results['radcliq-v1'],
        )

        for key in results.keys():
            summary[key] = sum(results[key]) / len(results[key])

        df.to_csv(str(run) + '.csv', index=False)
        with open(str(run) + '.json', 'w') as f:
            json.dump(summary, f, indent=4)


class CTMetrics:
    def __init__(self):
        sys.path.append('third-party/CT-CLIP/text_classifier')
        from classifier import RadBertClassifier

        radbert = RadBertClassifier(n_classes=len(RADBERT_CONDITIONS))
        radbert.load_state_dict(
            torch.load(
                ORIGIN_VL_DATA_ROOT / 'CT-RATE' / 'models' / 'RadBertClassifier.pth'
            ),
            strict=False,
        )
        radbert = radbert.to('cuda')
        radbert.eval()

        self.radbert = radbert
        self.tokenizer = AutoTokenizer.from_pretrained(
            'zzxslp/RadBERT-RoBERTa-4m', do_lower_case=True
        )

    def process(self, run: Path):
        sys.path.append('third-party/CT-CLIP/text_classifier')

        df = pd.read_csv(str(run) + '.csv')
        if os.path.exists(str(run) + '.json'):
            with open(str(run) + '.json', 'r') as f:
                summary = json.load(f)
        else:
            summary = {}

        pred_logits = np.zeros(len(RADBERT_CONDITIONS)).reshape(
            1, len(RADBERT_CONDITIONS)
        )

        for _, row in tqdm(df.iterrows(), total=df.shape[0], desc='computing RadBERT'):
            inputs = self.tokenizer(
                row['prediction'].replace('\n', ' '),
                return_tensors='pt',
                max_length=512,
                padding='max_length',
                truncation=True,
            )
            logits = self.radbert(
                input_ids=inputs['input_ids'].to('cuda'),
                attn_mask=inputs['attention_mask'].to('cuda'),
            )
            pred_logits = np.concatenate(
                (pred_logits, logits.detach().cpu().numpy()), axis=0
            )

        pred_logits = pred_logits[1:]
        prediction_labels = expit(pred_logits)

        prediction_labels[prediction_labels >= 0.5] = 1
        prediction_labels[prediction_labels < 0.5] = 0

        reference_labels = pd.read_csv(
            ORIGIN_VL_DATA_ROOT / 'CT-RATE' / 'dataset' / 'multi_abnormality_labels' / 'valid_predicted_labels.csv'
        ).set_index(['VolumeName'])[RADBERT_CONDITIONS]

        with open(PROCESSED_VL_DATA_ROOT / 'CT-RATE' / 'test-processed.json') as f:
            file_names = [
                image.split('/')[-1].replace('.pt.zst', '.nii.gz')
                for x in json.load(f)
                for i, image in enumerate(x['image'])
            ]
        reference_labels = reference_labels.loc[file_names].values

        f1s = f1_score(prediction_labels, reference_labels, average=None)
        summary['macro radbert f1'] = f1_score(prediction_labels, reference_labels, average='macro')
        summary['micro radbert f1'] = f1_score(prediction_labels, reference_labels, average='micro')
        fnr, summary['macro radbert fnr'], summary['micro radbert fnr'] = false_negative_rate(reference_labels, prediction_labels)

        for i, condition in enumerate(RADBERT_CONDITIONS):
            condition = condition.lower()
            df[condition + ' radbert prediction'] = prediction_labels[:, i]
            df[condition + ' radbert reference'] = reference_labels[:, i]
            summary[condition + ' radbert f1'] = f1s[i]
            summary[condition + ' radbert fnr'] = fnr[i]

        df.to_csv(str(run) + '.csv', index=False)
        with open(str(run) + '.json', 'w') as f:
            json.dump(summary, f, indent=4)
