import json
import pandas as pd
import torch
from tqdm import tqdm
import transformers

LLAMA_PROMPT = '''
You are an AI assistant specialized in medical topics. You are given the question, ground truth and prediction of a medical visual question answering task. Your task is to evaluate the prediction based on the question and ground truth in terms of medical knowledge. You should consider various aspects, such as the correctness, relevance, and completeness of the prediction. You should provide a final score from 0 to 10 to summarize the overall quality of the prediction. The output format is 'score: xx'. You should not output anything other than the score.
question: {question}
ground truth: {answer}
prediction: {prediction}
score:
'''


class LlamaMetric:
    def __init__(self):
        self.llama = transformers.pipeline(
            'text-generation',
            model='meta-llama/Meta-Llama-3-8B-Instruct',
            model_kwargs={'torch_dtype': torch.bfloat16},
            use_auth_token=True,
            device='cuda',
        )

    def compute(self, question: str, prediction: str, reference: str):
        return {
            'llama': int(
                self.llama(
                    LLAMA_PROMPT.format(
                        question=question, answer=reference, prediction=prediction
                    ),
                    max_length=128,
                    num_return_sequences=1,
                )[0]['generated_text'].split('score: ')[1]
            ),
        }

    def process(self, csv: str, json_file: str):
        df = pd.read_csv(csv)
        with open(json_file, 'r') as f:
            summary = json.load(f)
        llama = []

        for _, row in tqdm(df.iterrows()):
            score = self.compute(row['question'], row['prediction'], row['answer'])[
                'llama'
            ]
            llama.append(score)

        df['llama'] = llama
        summary['llama'] = sum(llama) / len(llama)

        df.to_csv(csv)
        with open(json_file, 'w') as f:
            json.dump(summary, f)
