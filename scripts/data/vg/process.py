import google.generativeai as genai 
import json
from tqdm import tqdm
from typing import List, Tuple

from mmmm.data.defs import PROCESSED_VL_DATA_ROOT, PROCESSED_VG_DATA_ROOT

prompt_highlight = """
You are an AI assistant specialized in biomedical topics. Your task is to carefully analyze the provided biomedical text and highlight key anatomical structures and anomalies mentioned in the text and highlight them in the report by enclosing them with "<p>" and "</p>" tags.
Below are requirments:
1. The targets to be highlighted are limited to anatomical structures and anomalies. Anatomical structures and anomalies include organs, tissues, vessels, bones, etc. Anomalies include tumors, fractures, hemorrhages, etc. You should not highlight any other entities, such as symptoms, diseases, body systems or treatments.
2. Include anatomic modifiers essential for precise localization, such as "right", "left", "upper", "lower", "anterior", "posterior", etc., when highlighting anatomical structures. But you must not highlight them when they are not modifying any anatomical structures.
3. Avoid highlighting any targets explicitly stated as absent, negated, or otherwise indicated as not present in the findings. E.g., in the context of "no tumor is visible" and "lesion is not observed", the "tumor" and "lesion" should not be highlighted.
4. Do not highlight targets served as global descriptions and are unable to be spatially localized. E.g., you should not highlight "atelectasis".
5. If the very same target occurs multiple times, highlight the first occurance. E.g., in the context of "The <p>abdominal aorta</p> is observed. An <p>accessory hepatic artery</p> arises from the abdominal aorta.", the second "abdominal aorta" should not be highlighted as it is the same target as the first one. 
6. Different targets, even with the same name, should be hightlighted respectively. E.g., "A <p>lesion</p> is observed upperside, another <p>lesion</p> is observed on the right".
7. Do not highlight targets that are too coarse, ambiguous, or amorphous to be localized. E.g. you should not highlight "abdomen", "chest".
8. It is totally fine to miss a few targets but it is unacceptable to highlight any targets that do not meet the requirements. So please be cautious and conservative when highlighting the targets.
9. The output should be exactly the original text with additional tags, do not output any additional information. Even if no target is present in the text, the output should be the same as the input.
You must strictly follow the requirements above.

Here are some examples:

Input: The appendix is markedly distended by fluid-attenuation material and contains a single small gas locule. The appendix diameter is 2.7 cm and it is surrounded by a hazy mesoappendix with prominent regional lymph nodes. No intraperitoneal fluid.
The <p>appendix</p> is markedly distended by fluid-attenuation material and contains a single small <p>gas locule</p>. The appendix diameter is 2.7 cm and it is surrounded by a hazy <p>mesoappendix</p> with prominent <p>regional lymph nodes</p>. No intraperitoneal fluid.

Input: A hyper-dense well-defined solid mass is seen in the supra-sellar regions. Another hyper-dense mass with large calcification is seen in the pineal region. There is compression of the pineal mass over the posterior aspect of the third ventricle with proximal marked hydrocephalus showing trans-ependymal CSF leakage. No dilatation of the fourth ventricle.
A hyper-dense well-defined solid <p>mass</p> is seen in the supra-sellar regions. Another hyper-dense <p>mass</p> with large <p>calcification</p> is seen in the <p>pineal region</p>. There is compression of the pineal mass over the posterior aspect of the <p>third ventricle</p> with proximal marked hydrocephalus showing trans-ependymal CSF leakage. No dilatation of the fourth ventricle.
"""

instruction = """
Here is the input text:
Input:
"""

def process_text(llm: genai.GenerativeModel, text: str):
    while True:
        try:
            messages = [{'role': 'user', 'parts': [prompt_highlight + instruction + text]}]
            response = llm.generate_content(messages)
            print(response.text)
            return response.text
        except Exception as e:
            print(e)
            continue
            

def process(llm: genai.GenerativeModel, dataset: str, num_samples: Tuple[int] = None):
    (PROCESSED_VG_DATA_ROOT / dataset).mkdir(parents=True, exist_ok=True)
    if dataset in ['Slake', 'VQA-Med']:
        key = 'answer'
    elif dataset in ['Radiopaedia']:
        key = 'caption'
    for i, split in enumerate(['train', 'validate', 'test']):
        with open(PROCESSED_VL_DATA_ROOT / dataset / (split + '.json'), 'r') as f:
            data = json.load(f)

        if num_samples:
            data = data[:num_samples[i]]

        for item in tqdm(data):
            response = process_text(llm, item[key])
            item[key] = response
            if dataset == 'Radiopaedia':
                for qa in item['qa_list']:
                    response = process_text(llm, qa['answer'])
                    qa['answer'] = response              

        with open(PROCESSED_VG_DATA_ROOT / dataset / (split + '.json'), 'w') as f:
            json.dump(data, f, indent=4)

def main():
    with open("../google_api_key.txt", "r") as f:
        genai.configure(api_key=f.readline().strip(), transport="rest")

    llm = genai.GenerativeModel("gemini-pro")

    datasets = [
        ('Radiopaedia', (1200, 400, 400)),
        # ('Slake', None),
        # ('VQA-Med', None),
    ]

    for dataset, num_samples in datasets:
        print(dataset)
        process(llm, dataset, num_samples)

if __name__ == '__main__':
    main()