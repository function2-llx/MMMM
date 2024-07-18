import json

import numpy as np
import pandas as pd
import torch
from vllm import LLM, SamplingParams

mimic_cxr_prompt_1 = '''
You are an AI assistant with expertise in radiology. You are given a radiology report. Your task is to process the report and remove contents that is impossible to be inferred solely from a single radiograph.
Specifically, you should:
1. Remove clinical meta information about the imaging planes and techniques and the patient's position, like "AP and lateral views of the chest were provided", "evaluation is limited due to significant patient rotation to the right", "portable chest radiograph", "AP single view of the chest has been obtained with patient in sitting semi-upright position", "frontal images of the chest", "portable AP view of the chest", "on the lateral view", "is identified on both frontal and lateral views".
2. If such contents imply key findings, do paraphrase to retain the key information while performing the removals as requested. For example, "portable chest radiograph shows improved aeration at the right lung base" should be paraphrased to "aeration is seen at the right lung base" and "portable chest radiograph demonstrates a right pneumothorax" should be paraphrased to "a right pneumothorax is seen".
3. Avoid unnecessary removals and paraphrases. Modify the input as little as possible while meeting the above criteria.
Here is the input text for your task:
Input: {input}

Your output should be exactly the processed report. Do not output anything else.
'''

mimic_cxr_prompt_2 = '''
You are an AI assistant with expertise in radiology. You are given a radiology report. Your task is to process the report and remove contents that is impossible to be inferred solely from a single radiograph.
Specifically, you should:
1. Remove comparison with prior examinations and description of interval changes, like "no significant change compared to the prior radiograph", "are similar to prior", "are again noted", "are compared to previous exam from ___", "since the prior radiograph", "there has been little interval change", "continues to be", "is re-demonstrated", "persistent", "unchanged", "as expected", "stable", "with possible slight decrease in", "perhaps somewhat decreased", "there is increased", "new", "previously", "known".
2. Remove the medical history of the patient and judements derived purely from it, like "the patient has had prior sternotomy and aortic valve repair", "is consistent with remote history of fracture", "which is compatible with provided clinical history of ILD", "the patient is status post median sternotomy, CABG, vascular stenting", "bilateral pleural catheters have been removed", "consistent with prior granulomatous disease", "the ETT has been removed", "in view of history, a possibility of lymphangitic carcinomatosis also needs to be ruled out".
3. If such contents imply key findings, do paraphrase to retain the key information while performing the removals as requested. For example, "as compared to the prior radiograph performed yesterday morning, there has been slight interval improvement in extent of interstitial pulmonary edema" should be paraphrased to "there is interstitial pulmonary edema", "portable chest radiograph shows improved aeration at the right lung base" should be paraphrased to "there is aeration at the right lung base",　"relatively increased opacity projecting over the right lung base is seen" should be paraphrased to "opacity projecting over the right lung base is seen", and "the right lower lobe opacification has decreased substantially" should be paraphrased to "right lower lobe opacification are present".
4. If such contents only describe interval changes relative to prior and whether the abnormalities are currently present cannot be definitely inferred, remove them entirely. For example, "the mediastinal and hilar contours are relatively unchanged", "cardiac and mediastinal silhouettes are stable", "cardiomediastinal silhouette is unchanged" and "no new focal consolidation is seen" should be removed.
5. Avoid unnecessary removals and paraphrases. Modify the input as little as possible while meeting the above criteria.
Here is the input text for your task:
Input: {input}
Your output should be exactly the processed report. Do not output anything else.
'''

ct_rate_prompt = '''
You are an AI assistant with expertise in radiology. You are given a radiology report. You should:
1. Remove comparison with prior examinations and description of interval changes, like "prior right rib fractures.", "newly developed", "newly emerged", "stable", "with the patient's previous examinations".
2. Remove the medical history of the patient, like "in the case with a history of perforation during dilatation due to achalasia", "previous pleura in a patient with a history of previous TB", "mentioned in the patient's clinical information may cause these findings".
3. Keep the rest of the report exactly the same without any modification.
Here is the input text for your task:
Input: {input}

Your output should be exactly the processed report. Do not output anything else.
'''

open_i_prompt = '''
You are an AI assistant with expertise in radiology. You are given a radiology report consisting of findings and impression. You should:
1. Remove clinical meta information about the imaging planes and techniques and the patient's position, like "2 images", "frontal and lateral views of the chest", "lateral views obscured by patient body habitus", "the patient is mildly rotated", "on the lateral view".
2. Remove comparison with prior examinations and description of interval changes, like "unchanged from prior examination", "increased from most recent prior exam", "is similar to the prior study", "stable", "prior granulomatous disease", "have increased in size and number", "stable changes of".
3. Remove the medical history of the patient and judgements derived purely from it, like "consistent with history of sarcoid", "for patient age", "patient is status post CABG", "the patient's known multiple myeloma", "critical result notification documented through Primordial".
4. If such contents imply key findings, do paraphrase to retain the key information while performing the removals as requested. For example, "compared with the prior study, there is mildly increased streaky airspace disease in the right lung base" should be paraphrased to "there is streaky airspace disease in the right lung base", and "no interval change in the appearance of the XXXX opacities in the bilateral lower lobes" should be paraphrased to "There are opacities in the bilateral lower lobes".
5. Remove information de-identified with "XXXX". If the de-identification cause irreversible information loss in the content, remove all affected contexts. For example, "XXXX are normal", "along the XXXX aspect of the sternum", "there is mild calcification of the transverse XXXX", "the heart and lungs have XXXX XXXX in the interval", and "cisualized XXXX of the chest XXXX are within normal limits", "normal XXXX", "XXXX consistent with mild interstitial edema", "and both XXXX", "stigmata of XXXX cell disease" should be entirely removed.
6. Keep the rest of the report exactly the same without any modification. Do not add any new contents.
Here is the input text for your task:
Input: {input}

Your output should be exactly the processed report. Do not output anything else.
'''

roco_prompt = '''
You are an AI assistant with expertise in radiology. You are given a caption of a radiological image. You should:
1. Remove the patient's personal information, like "a 26-year-old male patient".
2. Remove comparison with prior examinations and description of interval changes, like "comparing to prior studies", "in the previous CT", "previously noticed", "redemonstrated", "unchanged", "new".
3. Remove the medical history of the patient, like "with no previous history of disease", "previous liver surgery".
4. Remove references to figures and cases, like "in Figure 1", "for Case 2", but retain references to arrows.
5. Remove the date of the imaging study, like "taken five days after", "six months postoperative".
6. For the rest of the text that has no content to be removed, keep it exactly the same without any modification.
7. If you find the provided input text does not appear to be a caption of a radiological image, such as it does not mention any radiology-related concepts or terms, then your output should be exactly "The provided input text does not appear to be a caption of a radiological image.".
Here is the input text for your task: 
Input: {input}

Your output should be exactly the processed caption, or report that the input text does not appear to be a caption of a radiological image. Do not output anything else, such as other comments.
'''


def process_reports(model: LLM, sampling_params: SamplingParams, dataset: str, splits: list[str]):
    for split in splits:
        with open(f'/data/MMMM/data/processed/vision-language/{dataset}/{split}.json', 'r') as f:
            data = json.load(f)
        tokenizer = model.get_tokenizer()
        
        reports = [f'Findings: {x["findings"]}\nImpression: {x["impression"]}'
                        if x.get('impression')
                        else x['findings'] for x in data]
        
        if dataset == 'MIMIC-CXR':
            prompts = [tokenizer.apply_chat_template(
                [{'role': 'user', 'content': mimic_cxr_prompt_1.format(input=x)}],
                tokenize=False,
                add_generation_prompt=True,
            ) for x in reports]
            responses = model.generate(prompts, sampling_params)
            processed1 = [x.outputs[0].text for x in responses]
            prompts = [tokenizer.apply_chat_template(
                [{'role': 'user', 'content': mimic_cxr_prompt_2.format(input=x)}],
                tokenize=False,
                add_generation_prompt=True,
            ) for x in processed1]
            responses = model.generate(prompts, sampling_params)
            processed2 = [x.outputs[0].text for x in responses]

            df = pd.DataFrame({'original': reports, 'processed1': processed1, 'processed2': processed2})
            df.to_csv(f'/data/MMMM/data/processed/vision-language/{dataset}/{split}-processed.csv', index=False)

            for i, x in enumerate(processed2):
                data[i]['processed_report'] = x
        elif dataset == 'CT-RATE':
            filtered = [(i, x) for i, x in enumerate(reports) if any([y in x.lower() for y in ['prior', 'previous', 'new', 'stable', 'patient', 'history']])]
            prompts = [tokenizer.apply_chat_template(
                [{'role': 'user', 'content': ct_rate_prompt.format(input=x)}],
                tokenize=False,
                add_generation_prompt=True,
            ) for _, x in filtered]
            responses = model.generate(prompts, sampling_params)
            filtered_processed = [x.outputs[0].text for x in responses]
            processed = [x for x in reports]
            for i, x in filtered:
                processed[i] = filtered_processed.pop(0)

            df = pd.DataFrame({'original': reports, 'processed': processed})
            df.to_csv(f'/data/MMMM/data/processed/vision-language/{dataset}/{split}-processed.csv', index=False)

            for i, x in enumerate(processed):
                data[i]['processed_report'] = x
        elif dataset == 'OpenI':
            prompts = [tokenizer.apply_chat_template(
                [{'role': 'user', 'content': open_i_prompt.format(input=x)}],
                tokenize=False,
                add_generation_prompt=True,
            ) for x in reports]
            responses = model.generate(prompts, sampling_params)
            processed = [x.outputs[0].text for x in responses]

            df = pd.DataFrame({'original': reports, 'processed': processed})
            df.to_csv(f'/data/MMMM/data/processed/vision-language/{dataset}/{split}-processed.csv', index=False)

            for i, x in enumerate(processed):
                data[i]['processed_report'] = x

        with open(f'/data/MMMM/data/processed/vision-language/{dataset}/{split}-processed.json', 'w') as f:
            json.dump(data, f, indent=4)


def process_captions(model: LLM, sampling_params: SamplingParams, dataset: str, splits: list[str]):
    for split in splits:
        with open(f'/data/MMMM/data/processed/vision-language/{dataset}/{split}.json', 'r') as f:
            data = json.load(f)
        tokenizer = model.get_tokenizer()

        captions = [x['caption'] for x in data]
        prompts = [tokenizer.apply_chat_template(
            [{'role': 'user', 'content': roco_prompt.format(input=x)}],
            tokenize=False,
            add_generation_prompt=True,
        ) for x in captions]
        responses = model.generate(prompts, sampling_params)
        processed = [x.outputs[0].text for x in responses]

        df = pd.DataFrame({'original': captions, 'processed': processed})
        df.to_csv(f'/data/MMMM/data/processed/vision-language/{dataset}/{split}-processed.csv', index=False)

        for i, x in enumerate(processed):
            data[i]['processed_caption'] = x

        with open(f'/data/MMMM/data/processed/vision-language/{dataset}/{split}-processed.json', 'w') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)


def main():
    model = '/data/llama3/Meta-Llama-3-70B-Instruct-hf'

    sampling_params = SamplingParams(
        temperature=0.,
        max_tokens=8000,
        stop = ['<|eot_id|>'],
    )

    model = LLM(
        model=model,
        tensor_parallel_size=torch.cuda.device_count(),
        disable_custom_all_reduce=True,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.98,
    )

    process_captions(model, sampling_params, 'ROCOv2', ['train'])

if __name__ == '__main__':
    main()
