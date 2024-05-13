import json
from tqdm import tqdm
from typing import List, Tuple
from vllm import LLM, SamplingParams
import os
import sys

model_id = "/data/llama3/Meta-Llama-3-70B-Instruct-hf"
# model_id = "/data/new_llm/Llama3-OpenBioLLM-70B"

model_name = model_id.split("/")[-1]

sampling_params = SamplingParams(
    temperature=0., 
    top_p=0.5,
    max_tokens=256,
    stop = ["<|eot_id|>"],
)

llm = LLM(
    model=model_id,
    tensor_parallel_size=1,
    disable_custom_all_reduce=True,
)

system_prompt1 = """You are an AI assistant with expertise in radiology. Your main task is to meticulously review a provided radiology report and accurately identify key anatomical structures mentioned within it. To highlight these structures, enclose each relevant word or phrase with "<p>" and "</p>" tags. Be sure to apply these tags only to the specified anatomical structures, and avoid tagging any terms that are clearly described as absent, negated, or uncertain in the findings. For example, do not highlight terms in statements such as "There is no pleural effusion or pneumothorax" and "No pleural effusion, pneumothorax, or focal consolidation is present."

Focus on tagging these anatomical structures, among others:
["lung", "heart", "lobe", "atelectasis", "cardiomegaly", "edema", "emphysema", "cardiomediastinum", "fibrosis", "hernia", "pleural effusion", "pneumonia", "pneumothorax", "lung lesion", "pneumoperitoneum", "pneumomediastinum", "arterial wall calcification", "cardiomegaly", "pericardial effusion", "coronary artery wall calcification", "hiatal hernia", "lymphadenopathy", "emphysema", "atelectasis", "lung nodule", "lung opacity", "pulmonary fibrotic sequela", "pleural effusion", "mosaic attenuation pattern", "peribronchial thickening", "consolidation", "bronchiectasis", "interlobular septal thickening"].

Here is an example to guide you:
Example input:
Frontal and lateral views of the chest were obtained. Rounded calcified nodule in the region of the posterior right lung base is seen and represents calcified granuloma on CTs dating back to ___, likely secondary to prior granulomatous disease. Previously seen pretracheal lymph node conglomerate and right hilar lymph nodes are better seen/evaluated on CT. No focal consolidation is seen. There is no pleural effusion or pneumothorax. Cardiac and mediastinal silhouettes are stable with possible slight decrease in right paratracheal prominence.

Example output:
Frontal and lateral views of the chest were obtained. Rounded <p>calcified nodule</p> in the region of the posterior right <p>lung</p> base is seen and represents <p>calcified granuloma</p> on CTs dating back to ___, likely secondary to prior granulomatous disease. Previously seen <p>pretracheal lymph</p> <p>node conglomerate</p> and <p>right hilar lymph nodes</p> are better seen/evaluated on CT. No focal consolidation is seen. There is no pleural effusion or pneumothorax. <p>Cardiac</p> and <p>mediastinal silhouettes</p> are stable with possible slight decrease in right paratracheal prominence.
"""

system_prompt2 = """
You are an AI assistant with expertise in radiology. Your primary task is to process radiology reports and remove annotations from anatomical structures that are indicated as non-existent in the report text.
Retain the "<p>" and "</p>" tags around anatomical structures that are mentioned as being present or observed. 
Remove the tags from any structures that are described using terms like 'no', 'without', 'absent', 'not detected', 'not observed', 'grossly unremarkable', 'cannot be assessed', or any other negations indicating non-existence.

Here is an example to illustrate how you should perform your task:
Example input: 
Lateral view somewhat limited due to overlying motion artifact. The <p>lungs</p> are low in volume.  There is no focal airspace consolidation to suggest\n <p>pneumonia</p>.  A 1.2-cm <p>calcified granuloma</p> just below the medial aspect of the\n right <p>hemidiaphragm</p> is unchanged from prior study.  No <p>pleural effusions</p> or\n <p>pulmonary edema</p>. There is no <p>pneumothorax</p>.\n\n The inferior <p>sternotomy wire</p> is fractured but unchanged. Surgical clips and\n vascular markers in the <p>thorax</p> are related to prior CABG surgery.

Example output: 
Lateral view somewhat limited due to overlying motion artifact. The <p>lungs</p> are low in volume.  There is no focal airspace consolidation to suggest\n pneumonia.  A 1.2-cm <p>calcified granuloma</p> just below the medial aspect of the\n right <p>hemidiaphragm</p> is unchanged from prior study.  No pleural effusions or\n pulmonary edema. There is no pneumothorax.\n\n The inferior <p>sternotomy wire</p> is fractured but unchanged. Surgical clips and\n vascular markers in the <p>thorax</p> are related to prior CABG surgery.
"""

def llama3_user_prompt(item_key):
    user_prompt = f"""
Here is the input text for your task:
Input: {item_key}

Your output should be the complete original text with the appropriate annotations. Do not add any extraneous text.
"""
    return user_prompt
            

def process(dataset: str, num_samples: Tuple[int] = None, FIRST: bool = True):
    output_path = '/data/MMMM/data/processed/visual-grounding/' + dataset
    os.makedirs(output_path, exist_ok=True)
    if dataset in ['MIMIC-CXR']:
        key_list = ['findings']
    elif dataset in ['CT-RATE']:
        key_list = ['Findings_EN']
    if FIRST: 
        prefix = '/data/MMMM/data/processed/vision-language/'
    else:
        prefix = '/data/MMMM/data/processed/visual-grounding/'
    for i, split in enumerate(['validate', 'test', 'train']):
        with open(prefix + dataset +  '/' + split + '.json', 'r') as f:
            data = json.load(f)

        if num_samples:
            data = data[:num_samples[i]]
        
        prompts = []

        for item in data:
            for key in key_list:
                if not FIRST:
                    key = key + '_annotation1'
                user_prompt = llama3_user_prompt(item[key])
            
                if FIRST:
                    
                    prompts.append(system_prompt1 + user_prompt)
                else:
                    prompts.append(system_prompt2 + user_prompt)
            
        responses = llm.generate(prompts, sampling_params)

        for i, output in enumerate(responses):
            generated_text = output.outputs[0].text
            generated_text = generated_text.replace("Output:", '')
            generated_text = generated_text.replace("```", '')
            for key in key_list:
                if FIRST:
                    data[i][key+'_annotation1'] = generated_text
                else:
                    data[i][key+'_annotation2'] = generated_text
                if i % 1000 == 0:
                    with open(output_path + '/' + split + '.json', 'w') as f:
                        json.dump(data, f, indent=4)
        if len(prompts) > 0:
            with open(output_path + '/' + split + '.json', 'w') as f:
                json.dump(data, f, indent=4)

def main():
    datasets = [
        ('MIMIC-CXR', (10, 10, 10)),
        ('CT-RATE',   (10, 0, 10)),
    ]
    FIRST = sys.argv[1] == '1'
    for dataset, num_samples in datasets:
        print(dataset)
        process(dataset, num_samples, FIRST)
if __name__ == '__main__':
    main()