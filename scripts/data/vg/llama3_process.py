import json
from tqdm import tqdm
from typing import List, Tuple
import transformers
import torch
import os

model_id = "/data/llama3/Meta-Llama-3-70B-Instruct-hf"


pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={
        "torch_dtype": torch.bfloat16, # original
    },
    device_map="auto",  # random output
)

system_prompt = """
You are an AI assistant specialized in radiology. Your task is to carefully analyze the provided radiology report and highlight key anatomical structures mentioned in the report by enclosing them with "<p>" and "</p>" tags. Avoid enclosing any targets with "<p>" and "</p>" tags explicitly stated as absent, negated, or otherwise indicated as not present or uncertain in the findings. E.g., in the context of "There is no pleural effusion", the "pleural effusion" should not be highlighted.
The anatomical structures include but are not limited to the following list: 
["atelectasis", "cardiomegaly", "edema", "emphysema", "cardiomediastinum", "fibrosis", "hernia", "pleural effusion", "pneumonia", "pneumothorax", "lung lesion", "pneumoperitoneum", "pneumomediastinum"].
"""

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

def llama3_generate(item_key):
    user_prompt = f"""
Here is the input text:
Input: {item_key}
Give the annotated text content. Do not generate other text.
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    prompt = pipeline.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    outputs = pipeline(
        prompt,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    return outputs[0]["generated_text"][len(prompt):]
            

def process(dataset: str, num_samples: Tuple[int] = None):
    os.makedirs('../test/' + dataset, exist_ok=True)
    if dataset in ['MIMIC-CXR']:
        key_list = ['findings', 'impression']
    
    for i, split in enumerate(['train', 'validate', 'test']):
        with open('./processed/vision-language/' +  dataset +  '/' + split + '.json', 'r') as f:
            data = json.load(f)

        if num_samples:
            data = data[:num_samples[i]]

        for item in tqdm(data):
            for key in key_list:
                response = llama3_generate(item[key])
                item[key] = response
                          

        with open('../test/' + dataset + '/' + split + '.json', 'w') as f:
            json.dump(data, f, indent=4)

def main():
   

    datasets = [
        ('MIMIC-CXR', (20, 20, 20)),
    ]

    for dataset, num_samples in datasets:
        print(dataset)
        process(dataset, num_samples)

if __name__ == '__main__':
    main()