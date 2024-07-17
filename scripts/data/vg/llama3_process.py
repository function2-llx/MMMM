import re

import cytoolz
import numpy as np
import orjson
import torch
from transformers import PreTrainedTokenizerFast
from vllm import LLM, SamplingParams

from mmmm.data.defs import PROCESSED_VG_DATA_ROOT, PROCESSED_VL_DATA_ROOT

model_id = "/data/llama3/Meta-Llama-3-8B-Instruct-hf"
# model_id = "/data/llama3/Meta-Llama-3-70B-Instruct-hf"
# model_id = "/data/new_llm/Llama3-OpenBioLLM-70B"

# model_name = model_id.split("/")[-1]

llm = LLM(
    model=model_id,
    tensor_parallel_size=torch.cuda.device_count(),
    disable_custom_all_reduce=True,
    # max_model_len=8192,
    enable_prefix_caching=True,
)
tokenizer = llm.get_tokenizer()
sampling_params = SamplingParams(
    temperature=0.,
    max_tokens=1024,
    stop=['<|eot_id|>'],
)

llm: LLM
tokenizer: PreTrainedTokenizerFast
sampling_params: SamplingParams

anatomy_list = [
    '[left; right] adrenal gland',
    '[abdominal; thoracic] aorta',
    '[left; right] clavicle',
    'colon',
    'duodenum',
    'esophagus',
    '[left; right] femur',
    'gallbladder',
    '[left; right] atrium',
    '[left; right] ventricle',
    '[left; right] humerus',
    '[left; right] [common; main] [iliac; pulmonary; subclavian; carotid; brachiocephalic] artery',
    '[left; right] [common iliac; pulmonary; brachiocephalic] vein',
    '[inferior; superior] vena cava',
    '[left; right] kidney',
    'liver',
    '[left; right] lung [lower; middle; upper] lobe',
    'pancreas',
    '[left; right] [first; ...; twelfth] rib',
    '[left; right] scapula',
    'spleen',
    'stomach',
    'trachea',
    '[left; right] [main] bronchus',  # unable to segment, though
    'bladder',
    'heart',
    '[left; right] atrial appendage',
    '[left; right] lung',
    'thyroid',
]
anomaly_list = [
    'atelectasis',
    'cardiomegaly',
    'pulmonary consolidation',
    'pulmonary edema',
    'widened mediastinum',
    'rib fracture',
    'pulmonary fibrosis',
    'lung nodule',
    'pulmonary opacification',
    'pleural effusion',
    'pneumothorax',

    'pericardial effusion',
    'hiatal hernia',
    'lymphadenopathy',
    'pulmonary emphysema',
    'peribronchial thickening',
    'bronchiectasis',
    'interlobular septal thickening',
]


tag_system_prompt = f"""You are an AI assistant with expertise in radiology. Your main task is to meticulously review a provided sentence from a radiology report and accurately identify the specified anatomical structures and anomaly findings mentioned in the report.
The names of targets to be identified are primarily specified as follows:
- anatomy list (with optional anatomical modifiers): {'; '.join(anatomy_list)}
- anomaly list: {'; '.join(anomaly_list)}
For each phrase identified as a target, convert it to the following format (similar to a hyperlink in Markdown): [<phrase>](<target>), where "<phrase>" denotes the original text of the identified phrase, "<target>" denotes the name of the target that the phrase is identified as. 

Below are requirements:
1. Include anatomic modifiers essential for precise localization when highlighting anatomical structures, such as "right", "left", "upper", "lower", "anterior", "posterior", "pulmonary". But you must not include them when they are not modifying any anatomical structures.
2. Exclude any target explicitly stated as absent, negated, or otherwise indicated as not present or uncertain in the findings. For example, nothing should be included in the following negative statements:
  - There is no pleural effusion or pneumothorax
  - No pleural effusion, pneumothorax, or focal consolidation is present.
3. Do not include targets that are too coarse, ambiguous, or amorphous to be spatially localized. E.g., you should not highlight "free fluid", "chest".
4. The output should be exactly the original text with additional tags, do not output any additional information. Even if no target is present in the text, the output should be the same as input.
"""
tag_examples = {
    'CT-RATE': [
        (
            'Heart contour and size are normal. No pleural-pericardial effusion or thickening was detected. Trachea and both main bronchi are open. Minimal peribronchial thickness increase is observed. There are more prominent centriacinar emphysema and bulla-bleb formations in the upper lobes of both lungs. There are linear areas of atelectasis in both lungs and accompanying nonspecific ground-glass areas in the lower lobe posterior segments. There is a millimetric nonspecific nodule in the upper lobe of the left lung. No pathological increase in wall thickness was observed in the esophagus.',
            '[Heart](heart) contour and size are normal. No pleural-pericardial effusion or thickening was detected. [Trachea](trachea) and both [main bronchi](main bronchus) are open. Minimal [peribronchial thickness](peribronchial thickening) increase is observed. There are more prominent centriacinar [emphysema](pulmonary emphysema) and bulla-bleb formations in the [upper lobes of both lungs](lung upper lobe). There are linear areas of [atelectasis](atelectasis) in both [lungs](lung) and accompanying nonspecific [ground-glass areas](pulmonary opacification) in the lower lobe posterior segments. There is a millimetric nonspecific [nodule](lung nodule) in the [upper lobe of the left lung](left lung upper lobe). No pathological increase in wall thickness was observed in the [esophagus](esophagus).',
        ),
        (
            'Trachea, both main bronchi are open. Mediastinal main vascular structures, heart contour, size are normal. Thoracic aorta diameter is normal. Pericardial effusion-thickening was not observed. No enlarged lymph nodes in prevascular, pre-paratracheal, subcarinal or bilateral hilar-axillary pathological dimensions were detected. When examined in the lung parenchyma window; A calcific nodule with a diameter of 4 mm was observed in the paravertebral area in the superior lower lobe of the right lung. Upper abdominal organs included in the sections are normal. No space-occupying lesion was detected in the liver that entered the cross-sectional area. Bilateral adrenal glands were normal and no space-occupying lesion was detected.',
            '[Trachea](trachea), both [main bronchi](main bronchus) are open. Mediastinal main vascular structures, [heart](heart) contour, size are normal. [Thoracic aorta](thoracic aorta) diameter is normal. Pericardial effusion-thickening was not observed. No enlarged lymph nodes in prevascular, pre-paratracheal, subcarinal or bilateral hilar-axillary pathological dimensions were detected. When examined in the lung parenchyma window; A calcific [nodule](lung nodule) with a diameter of 4 mm was observed in the paravertebral area in the [superior lower lobe of the right lung](right lung upper lobe). Upper abdominal organs included in the sections are normal. No space-occupying lesion was detected in the [liver](liver) that entered the cross-sectional area. Bilateral [adrenal glands](adrenal gland) were normal and no space-occupying lesion was detected.',
        ),
    ],
    'MIMIC-CXR': [
        (
            'The lungs appear hyperexpanded suggestive of chronic obstructive pulmonary disease.',
            'The [lungs](lung) appear hyperexpanded suggestive of chronic obstructive pulmonary disease.',
        ),
        (
            'A focal nodule is noted posterior to the sternum.',
            'A [focal nodule](lung nodule) is noted posterior to the sternum.',
        ),
        (
            'Additionally, there is enlargement of the left main pulmonary artery.',
            'Additionally, there is enlargement of the [left main pulmonary artery](left main pulmonary artery).',
        ),
        (
            'Cardiac silhouette is normal. Bibasilar opacities are visualized likely representative of bronchiectasis and fibrosis.  Calcifications of the origin of the great vessels are noted.',
            'The <p>lungs</p> appear hyperexpanded suggestive of chronic obstructive pulmonary disease. A focal <p>nodule</p> is noted posterior to the <p>sternum</p>. Additionally, there is enlargement of the <p>left main pulmonary artery</p>.  <p>Cardiac silhouette</p> is normal. <p>Bibasilar opacities</p> are visualized likely representative of bronchiectasis and fibrosis.  <p>Calcifications</p> of the origin of the <p>great vessels</p> are noted.',
        ),
    ]
}

prompt_filter = {
    'system_prompt':
"""You are an AI assistant with expertise in radiology. You will be given with a preliminarily annotated radiology report, where the anatomical structures and anomaly findings mentioned in the report text are enclosed with the "<p>" and "</p>" tags. However, the targets that are mentioned as non-existent are not supposed to be annotated. Therefore, your primary task is to check each annotated entity and its context in the given report, remove the annotation tags of targets that are indicated as non-existent in the report text. For example, targets that are described with terms like 'no', 'without', 'absent', 'not detected', 'not observed', 'grossly unremarkable', 'cannot be assessed', or any other negations indicating non-existence. On the other hand, annotation tags of targets that are mentioned as being present or observed should still be retained. 

Your output should be exactly the same as the original text, except for annotations tags removed for targets that are mentioned to be absent. DO NOT output any additional information, such as your own comments. Also DO NOT add new annotation tags. Even if you find that there is no tags to be removed, the output should be the same as input.
""",
    'examples': [
        (
            'Lateral view somewhat limited due to overlying motion artifact. The <p>lungs</p> are low in volume.  There is no focal airspace <p>consolidation</p> to suggest <p>pneumonia</p>.  A 1.2-cm <p>calcified granuloma</p> just below the medial aspect of the right <p>hemidiaphragm</p> is unchanged from prior study.  No <p>pleural effusions</p> or <p>pulmonary edema</p>. There is no <p>pneumothorax</p>. The inferior <p>sternotomy wire</p> is fractured but unchanged. Surgical clips and vascular markers in the <p>thorax</p> are related to prior CABG surgery.',
            'Lateral view somewhat limited due to overlying motion artifact. The <p>lungs</p> are low in volume.  There is no focal airspace consolidation to suggest pneumonia.  A 1.2-cm <p>calcified granuloma</p> just below the medial aspect of the right <p>hemidiaphragm</p> is unchanged from prior study.  No pleural effusions or pulmonary edema. There is no pneumothorax. The inferior <p>sternotomy wire</p> is fractured but unchanged. Surgical clips and vascular markers in the <p>thorax</p> are related to prior CABG surgery.',
            'In the sentence "No <p>pleural effusions</p> or <p>pulmonary edema</p>", both "pleural effusions" and "pulmonary edema" are suggested to be absent according to the context, therefore their annotations should be removed.',
        )
    ]
}

# def llama3_user_prompt(text: str):
#     user_prompt = f"""
# Your input: {text}
# Your output:
# """
#     return user_prompt

def build_few_shot_conv(system_prompt: str, examples: list[tuple[str, str]], query: str):
    conv = [
        {
            'role': 'system',
            'content': system_prompt,
        },
        *cytoolz.concat(
            (
                {
                    'role': 'user',
                    'content': query_e,
                },
                {
                    'role': 'assistant',
                    'content': response_e,
                },
            )
            for query_e, response_e in examples
        ),
        {
            'role': 'user',
            'content': query,
        },
    ]
    return tokenizer.apply_chat_template(conv, tokenize=False)

def process(dataset: str, split: str, num_samples: int):
    output_dir = PROCESSED_VG_DATA_ROOT / dataset
    output_dir.mkdir(exist_ok=True, parents=True)
    report_pattern = re.compile(r'Findings:(.*?)(?=Impression:)Impression:(.*)', re.DOTALL)
    data_path = PROCESSED_VL_DATA_ROOT / dataset / f'{split}-processed.json'
    if not data_path.exists():
        print(f'split file: "{data_path}" not found')
        return
    data: list[dict] = orjson.loads(data_path.read_bytes())
    R = np.random.RandomState(42)
    R.shuffle(data)
    if num_samples >= 0:
        data = data[:num_samples]
    prompts = []
    impressions = []
    for item in data:
        item.pop('findings')
        item.pop('impression')
        report = item['processed_report']
        match = report_pattern.match(report)
        findings, impression = match.group(1).strip(), match.group(2).strip()
        impressions.append(impression)
        prompts.append(
            build_few_shot_conv(tag_system_prompt, tag_examples[dataset], findings),
        )
    responses = llm.generate(prompts, sampling_params)
    for i, output in enumerate(responses):
        tagged_findings = output.outputs[0].text
        data[i]['tagged_report'] = f'Findings: {tagged_findings}\nImpression: {impressions[i]}'
    (output_dir / f'{split}.json').write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))

def main():
    for dataset, num_samples_dict in [
        # ('MIMIC-CXR', {'train': 10, 'test': 10}),
        ('CT-RATE', {'train': 10, 'test': 10}),
    ]:
        for split, num_samples in num_samples_dict.items():
            process(dataset, split, num_samples)

if __name__ == '__main__':
    main()
