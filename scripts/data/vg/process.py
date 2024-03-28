import google.generativeai as genai 
import json
from tqdm import tqdm
from typing import List, Tuple

from mmmm.data.defs import PROCESSED_VL_DATA_ROOT, PROCESSED_VG_DATA_ROOT

prompt_highlight = """
You are an AI assistant specialized in biomedical topics. You will be provided with a biomedical text, while the corresponding medical images are unavailable to you. 
Your task is to highlight key anatomical structures and anomalies mentioned in the text strictly following the provided list of entities and highlight them in the report by enclosing them with "<p>" and "</p>" tags.
Below are requirments:
1. Include anatomic modifiers essential for precise localization, such as "right", "left", "upper", "lower", "anterior", "posterior", etc., when highlighting anatomical structures.
2. Avoid highlighting any targets explicitly stated as absent, negated, or otherwise indicated as not present in the findings. E.g., in the context of "no tumor is visible", "lesion is not observed", "No left pleural effusion", the "tumor", "lesion" and "left pleural effusion" should not be highlighted.
3. Do not highlight targets serving as global descriptions and are unable to be spatially localized. E.g., atelectasis.
4. If the very same target occurs multiple times, highlight the first occurance. E.g., in the context of "The <p>abdominal aorta</p> is observed. An <p>accessory hepatic artery</p> arises from the abdominal aorta.", the second "abdominal aorta" should not be highlighted as it is the same target as the first one. 
5. Different targets, even with the same name, should be hightlighted respectively. E.g., "A <p>lesion</p> is observed upperside, another <p>lesion</p> is observed on the right".
6. Do not highlight targets that do not represent anatomical structures or anomalies at all. E.g., in the context of "Nonobstructed <p>small bowel</p> right of midline." and "hypointense on T2, FLAIR, and GE centrally and hyperintense peripherally", dangling modifiers like "right", "centrally" and "peripherally" should not be highlighted. In the context of "<p>Retroperitoneal lymphadenopathies</p> are observed in the image.", the verb phrase "observed" should not be highlighted.
7. Do not highlight targets that are too coarse, ambiguous or amorphous to be localized. E.g., upper abdomen, chest, blood products.
8. The output should be exactly the original text with additional tags, do not output any additional information. Even if no target is present in the text, the output should be the same as the input.

Here are some examples:

Input: The appendix is markedly distended by fluid-attenuation material and contains a single small gas locule. The appendix diameter is 2.7 cm and it is surrounded by a hazy mesoappendix with prominent regional lymph nodes. No intraperitoneal fluid.
The <p>appendix</p> is markedly distended by fluid-attenuation material and contains a single small <p>gas locule</p>. The appendix diameter is 2.7 cm and it is surrounded by a hazy <p>mesoappendix</p> with prominent <p>regional lymph nodes</p>. No intraperitoneal fluid.

Input: A hyper-dense well-defined solid mass is seen in the supra-sellar regions. Another hyper-dense mass with large calcification is seen in the pineal region. There is compression of the pineal mass over the posterior aspect of the third ventricle with proximal marked hydrocephalus showing trans-ependymal CSF leakage. No dilatation of the fourth ventricle.
A hyper-dense well-defined solid <p>mass</p> is seen in the supra-sellar regions. Another hyper-dense <p>mass</p> with large <p>calcification</p> is seen in the <p>pineal region</p>. There is compression of the pineal mass over the posterior aspect of the <p>third ventricle</p> with proximal marked hydrocephalus showing trans-ependymal CSF leakage. No dilatation of the fourth ventricle.

Input: Left rectus sheath hematoma extending into the prevesical space. Pseudoaneurysm within left rectus sheath (with inferior component filling at venous phase). Ascites, irregular liver surface, extensive fatty change, and enlarged paraumbilical vein in keeping with cirrhosis.
Left <p>rectus sheath hematoma</p> extending into the <p>prevesical space</p>. <p>Pseudoaneurysm</p> within left rectus sheath (with <p>inferior component</p> filling at venous phase). <p>Ascites</p>, <p>irregular liver surface</p>, extensive <p>fatty change</p>, and <p>enlarged paraumbilical vein</p> in keeping with cirrhosis.

Here is the input text:
Input:
"""

def process_text(llm: genai.GenerativeModel, text: str):
    while True:
        try:
            response = llm.generate_content(prompt_highlight + text)
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