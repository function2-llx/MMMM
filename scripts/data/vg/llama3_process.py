import numpy as np
import orjson
import torch
from jsonargparse import ArgumentParser
from vllm import LLM, SamplingParams

from mmmm.data.defs import PROCESSED_VG_DATA_ROOT, PROCESSED_VL_DATA_ROOT

model_id = "/data/llama3/Meta-Llama-3-70B-Instruct-hf"
# model_id = "/data/new_llm/Llama3-OpenBioLLM-70B"

model_name = model_id.split("/")[-1]

sampling_params = SamplingParams(
    temperature=0.,
    max_tokens=1024,
    stop=["<|eot_id|>"],
)

llm: LLM

anatomy_lsit = [
    "aorta",
    'artery',
    'colon',
    'esophagus',
    'gallbladder',
    'atrium',
    'ventricle',
    'vena cava',
    'vein',
    'kidney',
    'liver',
    'lung',
    'lung lobe',
    'heart',
    'pancreas',
    'rib',
    'scapula',
    'spleen',
    'stomach',
    'trachea',
    'bladder',
    'vertebra',
    'thyroid',
    'adrenal gland',
]
anomaly_list = [
    'atelectasis',
    'cardiomegaly',
    'edema',
    'cardiomediastinum',
    'fibrosis',
    'hiatal hernia',
    'pleural effusion',
    'pneumothorax',
    'pneumoperitoneum',
    'pneumomediastinum',
    'arterial wall calcification',
    'pericardial effusion',
    'coronary artery wall calcification',
    'lymphadenopathy',
    'emphysema',
    'lung nodule',
    'lung opacity',
    'peribronchial thickening',
    'consolidation',
    'bronchiectasis',
    'interlobular septal thickening',
]

system_prompt1 = f"""You are an AI assistant with expertise in radiology. Your main task is to meticulously review a provided radiology report and accurately identify key anatomical structures and anomaly findings mentioned within it. Here are non-exclusive lists to be focused:
Anatomy: {', '.join(anatomy_lsit)}
Anomaly: {', '.join(anomaly_list)}
You should not highlight any other entities, such as symptoms, diseases, scan orientation, body systems or treatments, such as "AP/PA view", "chest", 
Below are requirements:
1. Include anatomic modifiers essential for precise localization when highlighting anatomical structures, such as "right", "left", "upper", "lower", "anterior", "posterior", "pulmonary", etc., when highlighting anatomical structures. But you must not highlight them when they are not modifying any anatomical structures.
2. Avoid highlighting any targets explicitly stated as absent, negated, or otherwise indicated as not present or uncertain in the findings. For example, do not highlight terms in statements such as "There is no pleural effusion or pneumothorax" and "No pleural effusion, pneumothorax, or focal consolidation is present."
3. Do not highlight targets that are too coarse, ambiguous, or amorphous to be spatially localized. E.g., you should not highlight "free fluid", "chest".
4. If the very same target occurs multiple times, highlight the first occurance. E.g., in the context of "The abdominal aorta is observed. An accessory hepatic artery arises from the abdominal aorta.", the second "abdominal aorta" should not be highlighted as it is the same target as the first one.
5. Different targets, even with the same name, should be highlighted respectively. E.g., in "A lesion is observed upperside, another lesion is observed on the right", both two "lesions" should be highlighted respectively. 
6. The output should be exactly the original text with additional tags, do not output any additional information. Even if no target is present in the text, the output should be the same as input.

Enclose each relevant phrase to be highlighted with "<p>" and "</p>" tags. Some examples:
Example input 1:
Trachea, both main bronchi are normal. The ascending aorta has a transverse diameter of 40 mm and is minimally enlarged. Heart size is within normal limits. There is no pericardial thickening or effusion. There are calcified lymph nodes smaller than 1 cm in the mediastinum and left pulmonary hilus. When examined in the lung parenchyma window; A linear fibrotic band is observed in the posterobasal segment of the lower lobe of the right lung. Within the sections, the density of stones with a diameter of 1 cm in the gallbladder draws attention.
Example output 1:
<p>Trachea</p>, both main <p>bronchi</p> are normal. The ascending <p>aorta</p> has a transverse diameter of 40 mm and is minimally enlarged. <p>Heart</p> size is within normal limits. There is no pericardial thickening or effusion. There are calcified <p>lymph nodes</p> smaller than 1 cm in the <p>mediastinum</p>. When examined in the lung parenchyma window; A <p>linear fibrotic</p> band is observed in the posterobasal segment of the <p>lower lobe of the right lung</p>. Within the sections, the density of <p>stones</p> with a diameter of 1 cm in the <p>gallbladder</p> draws attention.
Example input 2:
The lungs appear hyperexpanded suggestive of chronic obstructive pulmonary disease. A focal nodule is noted posterior to the sternum. Additionally, there is enlargement of the left main pulmonary artery.  Cardiac silhouette is normal. Bibasilar opacities are visualized likely representative of bronchiectasis and fibrosis.  Calcifications of the origin of the great vessels are noted.
Example output 2
The <p>lungs</p> appear hyperexpanded suggestive of chronic obstructive pulmonary disease. A focal nodule is noted posterior to the sternum. Additionally, there is enlargement of the left main pulmonary artery.  <p>Cardiac silhouette</p> is normal. <p>Bibasilar opacities</p> are visualized likely representative of bronchiectasis and fibrosis.  <p>Calcifications</p> of the origin of the great vessels are noted.
"""

system_prompt2 = """
You are an AI assistant with expertise in radiology. Your primary task is to process radiology reports and remove annotations from anatomical structures that are indicated as non-existent in the report text.
Retain the "<p>" and "</p>" tags around anatomical structures that are mentioned as being present or observed. 
Remove the tags from any structures that are described using terms like 'no', 'without', 'absent', 'not detected', 'not observed', 'grossly unremarkable', 'cannot be assessed', or any other negations indicating non-existence.

Here is an example to illustrate how you should perform your task:
Example input: 
Lateral view somewhat limited due to overlying motion artifact. The <p>lungs</p> are low in volume.  There is no focal airspace consolidation to suggest <p>pneumonia</p>.  A 1.2-cm <p>calcified granuloma</p> just below the medial aspect of the right <p>hemidiaphragm</p> is unchanged from prior study.  No <p>pleural effusions</p> or <p>pulmonary edema</p>. There is no <p>pneumothorax</p>. The inferior <p>sternotomy wire</p> is fractured but unchanged. Surgical clips and vascular markers in the <p>thorax</p> are related to prior CABG surgery.

Example output: 
Lateral view somewhat limited due to overlying motion artifact. The <p>lungs</p> are low in volume.  There is no focal airspace consolidation to suggest pneumonia.  A 1.2-cm <p>calcified granuloma</p> just below the medial aspect of the right <p>hemidiaphragm</p> is unchanged from prior study.  No pleural effusions or pulmonary edema. There is no pneumothorax. The inferior <p>sternotomy wire</p> is fractured but unchanged. Surgical clips and vascular markers in the <p>thorax</p> are related to prior CABG surgery.
"""

def llama3_user_prompt(text: str):
    user_prompt = f"""
Your input: {text}
Your output:
"""
    return user_prompt

def process(dataset: str, num_samples: tuple[int, int, int] = None, is_first: bool = True):
    output_dir = PROCESSED_VG_DATA_ROOT / dataset
    # llm.get_tokenizer()
    output_dir.mkdir(exist_ok=True, parents=True)
    src_dir = (PROCESSED_VL_DATA_ROOT if is_first else PROCESSED_VG_DATA_ROOT) / dataset
    for i, split in enumerate(['validate', 'test', 'train']):
        if not (data_path := src_dir / f'{split}-processed.json').exists():
            print(f'split file: "{data_path}" not found')
            continue
        data = orjson.loads(data_path.read_bytes())
        R = np.random.RandomState(42)
        R.shuffle(data)
        if num_samples[i] >= 0:
            data = data[:num_samples[i]]
        prompts = []
        for item in data:
            user_prompt = llama3_user_prompt(item['processed_report'])
            prompt = (system_prompt1 if is_first else system_prompt2) + '\n' + user_prompt
            prompts.append(prompt)
        responses = llm.generate(prompts, sampling_params)

        for j, output in enumerate(responses):
            generated_text = output.outputs[0].text
            # generated_text = generated_text.replace("Output: ", '')
            # generated_text = generated_text.replace("```", '')
            if is_first:
                data[j]['annotation'] = generated_text
            else:
                data[j]['annotation-filtered'] = generated_text
        (output_dir / f'{split}.json').write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))

def main():
    global llm

    llm = LLM(
        model=model_id,
        tensor_parallel_size=torch.cuda.device_count(),
        disable_custom_all_reduce=True,
        max_model_len=2048,
        enable_prefix_caching=True,
    )

    parser = ArgumentParser()
    parser.add_argument('--first', action='store_true')
    args = parser.parse_args()
    datasets = [
        # ('MIMIC-CXR', (10, 10, 10)),
        # ('CT-RATE', (10, 10, 10)),
        ('MIMIC-CXR', (-1, -1, 5000)),
        ('CT-RATE', (-1, -1, 2500)),
    ]
    for dataset, num_samples in datasets:
        print(dataset)
        process(dataset, num_samples, args.first)

if __name__ == '__main__':
    main()
