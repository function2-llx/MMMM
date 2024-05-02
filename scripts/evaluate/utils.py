import evaluate
import json
import numpy as np
import pandas as pd
import random
import torch


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class NLPMetrics:
    def __init__(self):
        self.bleu = evaluate.load('bleu')
        self.rouge = evaluate.load('rouge')
        self.meteor = evaluate.load('meteor')
        self.bertscore = evaluate.load('bertscore')
        self.exact_match = evaluate.load('exact_match')

    def compute(self, prediction: str, reference: str):
        return {
            'bleu': (
                self.bleu.compute(
                    predictions=[prediction.lower()],
                    references=[[reference.lower()]],
                    max_order=1,
                )['bleu']
                if prediction.strip()
                else 0.0
            ),
            'rouge': self.rouge.compute(
                predictions=[prediction.lower()], references=[reference.lower()]
            )['rouge1'],
            'meteor': self.meteor.compute(
                predictions=[prediction.lower()], references=[reference.lower()]
            )['meteor'],
            'bertscore': self.bertscore.compute(
                predictions=[prediction],
                references=[reference],
                model_type='microsoft/deberta-xlarge-mnli',
            )['f1'][0],
            'exact_match': self.exact_match.compute(
                predictions=[prediction.lower()], references=[reference.lower()]
            )['exact_match'],
        }

    def process(self, csv: str, json_file: str):
        df = pd.read_csv(csv)
        with open(json_file, 'r') as f:
            summary = json.load(f)
        results = {
            'bleu': [],
            'rouge': [],
            'meteor': [],
            'bertscore': [],
            'exact_match': [],
        }

        for _, row in df.iterrows():
            score = self.compute(row['prediction'], row['answer'])
            for key in results.keys():
                results[key].append(score[key])

        for key in results.keys():
            df[key] = results[key]
            summary[key] = sum(results[key]) / len(results[key])

        df.to_csv(csv)
        with open(json_file, 'w') as f:
            json.dump(summary, f)
