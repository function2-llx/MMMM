import json
from vllm import LLM, SamplingParams


prompt = '''
You are an AI assistant with expertise in radiology. You are given a radiology report. Your task is to process the report and remove contents that is impossible to be inferred solely from a single radiograph.
Specifically, you should:
1. Remove the medical history of the patient like "the patient has had prior sternotomy and aortic valve repair", "is consistent with remote history of fracture", "which is compatible with provided clinical history of ILD", "the patient is status post median sternotomy, CABG, vascular stenting".
2. Remove clinical meta information of the examination like "AP and lateral views of the chest were provided", "evaluation is limited due to significant patient rotation to the right".
3. Remove comparison with prior examinations like "no significant change compared to the prior radiograph", "are similar to prior", "are again noted", "are compared to previous exam from ___", "since the prior radiograph, there has been substantial increase", "there has been little interval change", "continues to be", "is re-demonstrated", "persistent".
4. If the contents implies positive findings, do paraphrase to retain the key information while performing the removals as above. For example, "As compared to the prior radiograph performed yesterday morning, there has been slight interval improvement in extent of interstitial pulmonary edema." should be paraphrased to "There is interstitial pulmonary edema."
Here is the input text for your task:
Input: {input}

Your output should modify the input as little as possible while meeting the above criteria. Do not output anything else other than the processed report.
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
            [{'role': 'user', 'content': prompt.format(input=x)}],
            tokenize=False,
        ) for x in reports]
        responses = model.generate(prompts[100:200], sampling_params)
        prompts = [tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt.format(input=reports[i])},
            {'role': 'assistant', 'content': x.outputs[0].text},
            {'role': 'user', 'content': 'Please provide a comprehensive analysis explaining each of your modifications.'}],
            tokenize=False,
        ) for i, x in enumerate(responses)]
        analyses = model.generate(prompts, sampling_params)
        for i, response in enumerate(responses):
            print('>>> ', reports[i])
            print('<<< ', response.outputs[0].text)
            print('Analysis: ', analyses[i].outputs[0].text)


def main():
    model_id = '/data/llama3/Meta-Llama-3-70B-Instruct-hf'
    # model_id = '/data/new_llm/Llama3-OpenBioLLM-70B'

    model_name = model_id.split('/')[-1]

    sampling_params = SamplingParams(
        temperature=0., 
        top_p=0.95,
        max_tokens=8000,
        stop = ['<|eot_id|>'],
    )

    model = LLM(
        model=model_id,
        tensor_parallel_size=4,
        disable_custom_all_reduce=True,
    )
    process(model, sampling_params, 'MIMIC-CXR', ['train'])


if __name__ == '__main__':
    main()