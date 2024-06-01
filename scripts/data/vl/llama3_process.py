import json
import pandas as pd
from vllm import LLM, SamplingParams

prompt1 = '''
You are an AI assistant with expertise in radiology. You are given a radiology report. Your task is to process the report and remove contents that is impossible to be inferred solely from a single radiograph.
Specifically, you should:
1. Remove clinical meta information about the imaging planes and techniques, the patient's position or other aspects of the examination, like "AP and lateral views of the chest were provided", "evaluation is limited due to significant patient rotation to the right", "portable chest radiograph", "AP single view of the chest has been obtained with patient in sitting semi-upright position", "frontal images of the chest", "portable AP view of the chest", "on the lateral view", "is identified on both frontal and lateral views".
2. If such contents imply key findings, do paraphrase to retain the key information while performing the removals as requested. For example, "portable chest radiograph shows improved aeration at the right lung base" should be paraphrased to "aeration is seen at the right lung base" and "portable chest radiograph demonstrates a right pneumothorax" should be paraphrased to "a right pneumothorax is seen".
3. Avoid unnecessary removals and paraphrases. Modify the input as little as possible while meeting the above criteria.
Here is the input text for your task:
Input: {input}

Your output should be exactly the processed report. Do not output anything else.
'''

prompt2 = '''
You are an AI assistant with expertise in radiology. You are given a radiology report. Your task is to process the report and remove contents that is impossible to be inferred solely from a single radiograph.
Specifically, you should:
1. Remove comparison with prior examinations and description of interval changes, like "no significant change compared to the prior radiograph", "are similar to prior", "are again noted", "are compared to previous exam from ___", "since the prior radiograph", "there has been little interval change", "continues to be", "is re-demonstrated", "persistent", "unchanged", "as expected", "stable", "with possible slight decrease in", "perhaps somewhat decreased", "there is increased", "new", "previously", "known".
2. Remove the medical history of the patient and judements derived purely from it, like "the patient has had prior sternotomy and aortic valve repair", "is consistent with remote history of fracture", "which is compatible with provided clinical history of ILD", "the patient is status post median sternotomy, CABG, vascular stenting", "bilateral pleural catheters have been removed", "consistent with prior granulomatous disease", "the ETT has been removed", ""in view of history, a possibility of lymphangitic carcinomatosis also needs to be ruled out".
4. If such contents imply key findings, do paraphrase to retain the key information while performing the removals as requested. For example, "as compared to the prior radiograph performed yesterday morning, there has been slight interval improvement in extent of interstitial pulmonary edema" should be paraphrased to "there is interstitial pulmonary edema", "portable chest radiograph shows improved aeration at the right lung base" should be paraphrased to "there is aeration at the right lung base",　"relatively increased opacity projecting over the right lung base is seen" should be paraphrased to "opacity projecting over the right lung base is seen", and "the right lower lobe opacification has decreased substantially" should be paraphrased to "right lower lobe opacification are present".
5. If such contents only describe interval changes relative to prior and whether the abnormalities are currently present cannot be definitely inferred, remove them entirely. For example, "the mediastinal and hilar contours are relatively unchanged", "cardiac and mediastinal silhouettes are stable", "cardiomediastinal silhouette is unchanged" and "no new focal consolidation is seen" should be removed.
6. Avoid unnecessary removals and paraphrases. Modify the input as little as possible while meeting the above criteria.
Here is the input text for your task:
Input: {input}

Your output should be exactly the processed report. Do not output anything else.
'''


def process(model: LLM, sampling_params: SamplingParams, dataset: str, splits: list[str]):
    for split in splits:
        with open(f'/data/MMMM/data/processed/vision-language/{dataset}/{split}.json', 'r') as f:
            data = json.load(f)
        tokenizer = model.get_tokenizer()
        
        reports = [f'Findings: {x["findings"]}\nImpression: {x["impression"]}'
                        if x.get('impression')
                        else x['findings'] for x in data]
        prompts = [tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt1.format(input=x)}],
            tokenize=False,
        ) for x in reports]
        responses = model.generate(prompts, sampling_params)
        processed1 = [x.outputs[0].text for x in responses]
        prompts = [tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt2.format(input=x)}],
            tokenize=False,
        ) for x in processed1]
        responses = model.generate(prompts, sampling_params)
        processed2 = [x.outputs[0].text for x in responses]

        df = pd.DataFrame({'original': reports, 'processed1': processed1, 'processed2': processed2})
        df.to_csv(f'/data/MMMM/data/processed/vision-language/{dataset}/{split}-processed.csv', index=False)

        for i, x in enumerate(processed2):
            data[i]['processed_report'] = x

        with open(f'/data/MMMM/data/processed/vision-language/{dataset}/{split}-processed.json', 'w') as f:
            json.dump(data, f, indent=4)


def main():
    model = '/data/llama3/Meta-Llama-3-70B-Instruct-hf'

    sampling_params = SamplingParams(
        temperature=0.,
        max_tokens=8000,
        stop = ['<|eot_id|>'],
    )

    model = LLM(
        model=model,
        tensor_parallel_size=4,
        disable_custom_all_reduce=True,
        enable_prefix_caching=True,
    )
    process(model, sampling_params, 'MIMIC-CXR', ['test'])


if __name__ == '__main__':
    main()