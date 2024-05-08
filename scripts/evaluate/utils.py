import os
import evaluate
import json
import numpy as np
import pandas as pd
import random
import torch
from tqdm import tqdm
import transformers


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
    with open(
        output_dir / f'{task}_{dataset}_{model}_{setting}.json',
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


class NLPMetrics:
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

    def process(self, run: str):
        df = pd.read_csv(run + '.csv')
        with open(run + '.json', 'r') as f:
            summary = json.load(f)
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

        df.to_csv(run + '.csv')
        with open(run + '.json', 'w') as f:
            json.dump(summary, f, indent=4)


class LlamaMetric:
    def __init__(self):
        self.llama = transformers.pipeline(
            'text-generation',
            model='/data/llama3/Meta-Llama-3-70B-Instruct-hf',
            model_kwargs={'torch_dtype': torch.bfloat16},
            device_map='auto',
        )

    def compute(self, question: str, prediction: str, reference: str):
        if not prediction:
            return 0

        messages = [
            {"role": "system", "content": LLAMA_SYSTEM_PROMPT},
            {"role": "user", "content": LLAMA_USER_PROMPT.format(question=question, answer=reference, prediction=prediction)},
        ]

        prompt = self.llama.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        terminators = [
            self.llama.tokenizer.eos_token_id,
            self.llama.tokenizer.convert_tokens_to_ids("<|eot_id|>")
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
                print(response)
                return {
                    'llama': int(response.split('score: ')[1].strip()),
                }
            except:
                continue

    def process(self, run: str):
        df = pd.read_csv(run + '.csv')
        with open(run + '.json', 'r') as f:
            summary = json.load(f)
        llama = []

        for _, row in df.iterrows():
            score = self.compute(str(row['prediction']) if pd.notna(row['prediction']) else '', str(row['answer']))['llama']
            llama.append(score)

        df['llama'] = llama
        summary['llama'] = sum(llama) / len(llama)

        df.to_csv(run + '.csv')
        with open(run + '.json', 'w') as f:
            json.dump(summary, f, indent=4)


if __name__ == '__main__':
    nlp_metrics = NLPMetrics()
    for run in tqdm(os.listdir('results')):
        if run.endswith('.csv'):
            print(run)
            nlp_metrics.process('results/' + run.replace('.csv', ''))