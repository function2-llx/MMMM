import os
import re

import cytoolz
import numpy as np
import orjson
import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast
from vllm import LLM, SamplingParams

from mmmm.data.defs import PROCESSED_VG_DATA_ROOT, PROCESSED_VL_DATA_ROOT

# model_id = "/data/llama3/Meta-Llama-3-8B-Instruct-hf"
model_id = os.getenv('GRG_MODEL_ID', '/data/llama3/Meta-Llama-3-70B-Instruct-hf')
# model_id = "/data/new_llm/Llama3-OpenBioLLM-70B"

# model_name = model_id.split("/")[-1]

llm = LLM(
    model=model_id,
    tensor_parallel_size=torch.cuda.device_count(),
    disable_custom_all_reduce=True,
    max_model_len=4096,
    enable_prefix_caching=True,
    gpu_memory_utilization=0.95,
)
tokenizer = llm.get_tokenizer()
sampling_params = SamplingParams(
    temperature=0.,
    max_tokens=2048,
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
    'vascular calcification',  # good luck!
]


tag_system_prompt = f"""You are an AI assistant with expertise in radiology. Your main task is to meticulously review a provided radiology report and accurately identify the specified anatomical structures and anomaly findings mentioned in the report.
The names of targets to be identified are primarily specified as follows:
- anatomy list (with optional anatomical modifiers): {'; '.join(anatomy_list)}
- anomaly list: {'; '.join(anomaly_list)}
For each phrase identified as a target, convert it to the following format (similar to a hyperlink in Markdown): [<phrase>](<target>), where "<phrase>" denotes the original text of the identified phrase, "<target>" denotes the name of the target provided above that the phrase is identified as.

Below are requirements:
1. Include anatomic modifiers essential for precise localization when highlighting anatomical structures, such as "right", "left", "upper", "lower", "anterior", "posterior", "pulmonary". But you must not include them when they are not modifying any anatomical structures.
2. Exclude any target explicitly stated as absent, negated, or otherwise indicated as not present or uncertain in the findings. For example, nothing should be included in the following negative statements:
  - There is no pleural effusion or pneumothorax
  - No pleural effusion, pneumothorax, or focal consolidation is present.
3. Do not include targets that are too coarse, ambiguous, or amorphous to be spatially localized, such as "free fluid", "chest", "abdomen".
4. The output should be exactly the original text extended with additional tags. Do not alter the input, or output any additional information. Even if no target is identified in the text, the output should be the same as input.
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
        (
            'No occlusive pathology was observed in the lumen. Although the mediastinum cannot be evaluated optimally in non-contrast examination; The mediastinal main vascular structures are normal in heart contour and size. Pericardial effusion-thickening was not observed. Thoracic esophageal calibration was normal and no significant tumoral wall thickening was detected. Sliding type hiatal hernia is observed at the lower end of the esophagus. Focal patchy ground glass densities and interlobar and interlobular septal thickenings are observed in the right lung lower lobe anterobasal and left lung lower lobe anteromediobasal segment. Correlation with clinical and laboratory is recommended for atypical pneumonia. A nonspecific subpleural millimetric nodule was observed in the left superior lingular segment in both lungs. Liver, gall bladder, spleen, pancreas, both adrenal glands, and both kidneys are normal. Intraabdominal free fluid-collection was not observed. Bone structures in the study area are natural. Vertebral corpus heights are preserved.',
            'No occlusive pathology was observed in the lumen. Although the mediastinum cannot be evaluated optimally in non-contrast examination; The mediastinal main vascular structures are normal in [heart](heart) contour and size. Pericardial effusion-thickening was not observed. Thoracic esophageal calibration was normal and no significant tumoral wall thickening was detected. Sliding type [hiatal hernia](hiatal hernia) is observed at the lower end of the [esophagus](esophagus). Focal patchy [ground glass densities](pulmonary opacification) and interlobar and [interlobular septal thickenings](interlobular septal thickening) are observed in the [right lung lower lobe](right lung lower lobe) anterobasal and [left lung lower lobe](left lung lower lobe) anteromediobasal segment. Correlation with clinical and laboratory is recommended for atypical pneumonia. A nonspecific subpleural millimetric [nodule](lung nodule) was observed in the left superior lingular segment in both [lungs](lung). [Liver](liver), [gall bladder](gallbladder), [spleen](spleen), [pancreas](pancreas), both [adrenal glands](adrenal gland), and both [kidneys](kidney) are normal. Intraabdominal free fluid-collection was not observed. Bone structures in the study area are natural. Vertebral corpus heights are preserved.',
        ),
    ],
    'MIMIC-CXR': [
        (
            'Lungs are fully expanded and clear without focal consolidation or suspicious pulmonary nodules.  No pleural effusions.  Mild cardiomegaly is present without pulmonary vascular congestion or pulmonary edema.  Descending thoracic aorta is tortuous.  Median sternotomy wires are well aligned and intact.',
            '[Lungs](lung) are fully expanded and clear without focal consolidation or suspicious pulmonary nodules.  No pleural effusions.  Mild [cardiomegaly](cardiomegaly) is present without pulmonary vascular congestion or pulmonary edema.  Descending [thoracic aorta](thoracic aorta) is tortuous.  Median sternotomy wires are well aligned and intact.',
        ),
        (
            'Airspace consolidation is noted within the left lower lobe compatible with pneumonia.  Right lung is clear.  Imaged osseous structures are intact. No free air below the right hemidiaphragm is seen.',
            'Airspace [consolidation](pulmonary consolidation) is noted within the [left lower lobe](left lung lower lobe) compatible with pneumonia.  [Right lung](right lung) is clear.  Imaged osseous structures are intact. No free air below the right hemidiaphragm is seen.',
        ),
        (
            'Moderate to severe cardiomegaly.  A hiatal hernia is present.  Minimal atelectasis is seen in the right base, and the lungs are otherwise clear.  No pneumothorax or pleural effusion is seen.',
            'Moderate to severe [cardiomegaly](cardiomegaly).  A [hiatal hernia](hiatal hernia) is present.  Minimal [atelectasis](atelectasis) is seen in the right base, and the [lungs](lung) are otherwise clear.  No pneumothorax or pleural effusion is seen.',
        ),
        (
            'Dual lumen right-sided central venous catheter seen with the tip in the upper right atrium. Mild prominence of interstitial markings is seen. No large effusion is present. The cardiomediastinal silhouette is within normal limits. No acute osseous abnormalities are identified.',
            'Dual lumen right-sided central venous catheter seen with the tip in the upper [right atrium](right atrium). Mild prominence of interstitial markings is seen. No large effusion is present. The cardiomediastinal silhouette is within normal limits. No acute osseous abnormalities are identified.',
        ),
        (
            'The lungs appear hyperexpanded. A focal nodule is noted posterior to the sternum. Additionally, there is enlargement of the left main pulmonary artery.  Cardiac silhouette is normal.  Bibasilar opacities are visualized.  Calcifications of the origin of the great vessels are noted.',
            'The [lungs](lung) appear hyperexpanded. A [focal nodule](lung nodule) is noted posterior to the sternum. Additionally, there is enlargement of the [left main pulmonary artery](left main pulmonary artery).  Cardiac silhouette is normal.  [Bibasilar opacities](pulmonary opacification) are visualized.  [Calcifications](vascular calcification) of the origin of the great vessels are noted.',
        ),
        (
            'The heart is normal.  The hilar and mediastinal contours are normal. There is a right sided pneumothorax.  The right hemidiaphragm is elevated.  The left lung is clear. Rib fractures are seen bilaterally.',
            'The [heart](heart) is normal.  The hilar and mediastinal contours are normal. There is a right sided [pneumothorax](pneumothorax).  The right hemidiaphragm is elevated.  The [left lung](left lung) is clear. [Rib fractures](rib fracture) are seen bilaterally.',
        ),
    ]
}

filter_system_prompt = """You are an AI assistant with expertise in radiology. You will be given with a preliminarily annotated radiology report. In the given report, some of the phrases of anatomical structures and anomaly findings are annotated with the following format: [<phrase>](<target>), where "<phrase>" denotes the original text of the annotated phrase, "<target>" denotes the standard name of the corresponding target.

However, targets that are mentioned to be non-existent in the report text may be wrongly included for annotating.  Therefore, your primary task is to check each annotated entity and its context in the given report, remove the annotation tags of phrases that are indicated as non-existent in the report text. For example, phrases that are described with terms like 'no', 'without', 'absent', 'not detected', 'not observed', 'grossly unremarkable', 'cannot be assessed', or any other negations indicating non-existence. To do the removal, for each annotation of "[<phrase>](<target>)" to be removed, convert it to "<phrase>". On the other hand, annotation tags of targets that are mentioned as being present or observed should still be retained.

Your output should be exactly the same as the original text, except for annotations tags removed for targets that are mentioned to be absent. DO NOT output any additional information, such as your own comments. Also DO NOT add new annotation tags. Even if you find that there is no tags to be removed, the output should be the same as input with all tags kept.
"""

filter_examples = {
    'MIMIC-CXR': [
        (
            'Left-sided pacemaker device is noted with leads terminating in the [right atrium](right atrium), [right ventricle](right ventricle), and coronary sinus.  The [heart](heart) size is mildly enlarged.  The [aortic knob](thoracic aorta) is calcified.  Mild [pulmonary edema](pulmonary edema) with perihilar haziness and vascular indistinctness is seen.  Focal opacities at [lung](lung) bases may reflect areas of [atelectasis](atelectasis) though infection cannot be excluded.  Small bilateral [pleural effusions](pleural effusion) may be present.  No [pneumothorax](pneumothorax) is identified.',
            'Left-sided pacemaker device is noted with leads terminating in the [right atrium](right atrium), [right ventricle](right ventricle), and coronary sinus.  The [heart](heart) size is mildly enlarged.  The [aortic knob](thoracic aorta) is calcified.  Mild [pulmonary edema](pulmonary edema) with perihilar haziness and vascular indistinctness is seen.  Focal opacities at [lung](lung) bases may reflect areas of [atelectasis](atelectasis) though infection cannot be excluded.  Small bilateral [pleural effusions](pleural effusion) may be present.  No pneumothorax is identified.',
        ),
        (
            'A endotracheal tube terminates 3.4 cm above the [carina](trachea).  There is an orogastric tube terminating within the [stomach](stomach).  A confluent right mid lower zone opacity with a central rounded lucency is seen, which may reflect cavitary lesion or abscess.  No underlying consolidations are present.  No [pneumothorax](pneumothorax) is seen.',
            'A endotracheal tube terminates 3.4 cm above the [carina](trachea).  There is an orogastric tube terminating within the [stomach](stomach).  A confluent right mid lower zone opacity with a central rounded lucency is seen, which may reflect cavitary lesion or abscess.  No underlying consolidations are present.  No pneumothorax is seen.',
        ),
        (
            'Mild-to-moderate [pulmonary edema](pulmonary edema) is present with interstitial markings.  Small bilateral [pleural effusions](pleural effusion) are not seen.  There is no focal consolidation or [pneumothorax](pneumothorax).  [Heart](heart) size is moderately enlarged.  A left chest wall Port-A-Cath terminates in the [RA](right atrium).  Vertebroplasties are seen.',
            'Mild-to-moderate [pulmonary edema](pulmonary edema) is present with interstitial markings.  Small bilateral pleural effusions are not seen.  There is no focal consolidation or pneumothorax.  [Heart](heart) size is moderately enlarged.  A left chest wall Port-A-Cath terminates in the [RA](right atrium).  Vertebroplasties are seen.',
        ),
        (
            'An endotracheal tube terminates 4.1 cm above the [carina](trachea).  An enteric tube terminates in the proximal [stomach](stomach) and could be advanced for ideal positioning.  Low [lung](lung) volumes. Minimal elevation of the right hemidiaphragm is present.  The [left lung](left lung) base is not visualized.  Increased opacity at the base of the [left lung](left lung) may reflect [atelectasis](atelectasis).  Mild [vascular congestion](pulmonary vascular congestion) with mild [pulmonary edema](pulmonary edema) is seen.  No [pneumothorax](pneumothorax) is seen.',
            'An endotracheal tube terminates 4.1 cm above the [carina](trachea).  An enteric tube terminates in the proximal [stomach](stomach) and could be advanced for ideal positioning.  Low [lung](lung) volumes. Minimal elevation of the right hemidiaphragm is present.  The left lung base is not visualized.  Increased opacity at the base of the [left lung](left lung) may reflect [atelectasis](atelectasis).  Mild [vascular congestion](pulmonary vascular congestion) with mild [pulmonary edema](pulmonary edema) is seen.  No pneumothorax is seen.',
        ),
    ],
    'CT-RATE': [
        (
            'CTO is normal. Calibration of mediastinal major vascular structures is natural. Calcific [atheroma plaques](vascular calcification) are observed in the coronary arteries. Thoracic [esophagus](esophagus) calibration was normal and no significant pathological wall thickening was detected. No lymph node with pathological size and configuration was detected in the mediastinum and hilar level. One or two lymph nodes, the largest of which is 13x10 mm in size, are observed at the right hilar level. When examined in the lung parenchyma window; [mosaic attenuation pattern](pulmonary opacification) is observed in both [lungs](lung) (small vessel disease?, small airway disease?). Mild sequelae changes are observed in both [lungs](lung). On this background, a [nodule](lung nodule) of approximately 5x3 mm in size is observed in the [right lung upper lobe](right lung upper lobe) caudal. No [pleural effusion](pleural effusion) [pneumothorax](pneumothorax) was detected. In the sections passing through the upper abdomen, a decrease in density consistent with hepatosteatosis is observed in the [liver](liver). Degenerative changes are observed in the bone structure entering the examination area.',
            'CTO is normal. Calibration of mediastinal major vascular structures is natural. Calcific [atheroma plaques](vascular calcification) are observed in the coronary arteries. Thoracic [esophagus](esophagus) calibration was normal and no significant pathological wall thickening was detected. No lymph node with pathological size and configuration was detected in the mediastinum and hilar level. One or two lymph nodes, the largest of which is 13x10 mm in size, are observed at the right hilar level. When examined in the lung parenchyma window; [mosaic attenuation pattern](pulmonary opacification) is observed in both [lungs](lung) (small vessel disease?, small airway disease?). Mild sequelae changes are observed in both [lungs](lung). On this background, a [nodule](lung nodule) of approximately 5x3 mm in size is observed in the [right lung upper lobe](right lung upper lobe) caudal. No pleural effusion pneumothorax was detected. In the sections passing through the upper abdomen, a decrease in density consistent with hepatosteatosis is observed in the [liver](liver). Degenerative changes are observed in the bone structure entering the examination area.'
        ),
        (
            'Mediastinal structures were evaluated as suboptimal since the examination was unenhanced. As far as can be seen; Calcified atherosclerotic changes were observed in the [coronary artery](coronary artery) wall. [Heart](heart) sizes are slightly increased. Minimal calcified atherosclerotic changes were observed in the wall of the [thoracic aorta](thoracic aorta). Pericardial thickening-effusion was not detected. [Trachea](trachea) and lumen of both [main bronchi](main bronchus) are open. No occlusive pathology was detected in the [trachea](trachea) and lumen of both [main bronchi](main bronchus). Thoracic [esophagus](esophagus) calibration was normal and no significant pathological wall thickening was detected. No lymph node was detected in mediastinal and bilateral hilar pathological size and appearance. When examined in the lung parenchyma window; Mild [emphysematous changes](pulmonary emphysema) were observed in both [lungs](lung). A mosaic attenuation pattern was observed in both [lungs](lung) (small airway disease?, small vessel disease?). A millimetric nonspecific parenchymal [nodule](lung nodule) was observed in the [left lung](left lung). Bilateral [pleural thickening-effusion](pleural effusion) was not observed. No significant pathology was detected in the upper abdominal sections that entered the examination area. Left-facing [scoliosis](scoliosis) was observed in the thoracic vertebrae. Mild degenerative changes were observed in bone structures.',
            'Mediastinal structures were evaluated as suboptimal since the examination was unenhanced. As far as can be seen; Calcified atherosclerotic changes were observed in the [coronary artery](coronary artery) wall. [Heart](heart) sizes are slightly increased. Minimal calcified atherosclerotic changes were observed in the wall of the [thoracic aorta](thoracic aorta). Pericardial thickening-effusion was not detected. [Trachea](trachea) and lumen of both [main bronchi](main bronchus) are open. No occlusive pathology was detected in the [trachea](trachea) and lumen of both [main bronchi](main bronchus). Thoracic [esophagus](esophagus) calibration was normal and no significant pathological wall thickening was detected. No lymph node was detected in mediastinal and bilateral hilar pathological size and appearance. When examined in the lung parenchyma window; Mild [emphysematous changes](pulmonary emphysema) were observed in both [lungs](lung). A mosaic attenuation pattern was observed in both [lungs](lung) (small airway disease?, small vessel disease?). A millimetric nonspecific parenchymal [nodule](lung nodule) was observed in the [left lung](left lung). Bilateral pleural thickening-effusion was not observed. No significant pathology was detected in the upper abdominal sections that entered the examination area. Left-facing [scoliosis](scoliosis) was observed in the thoracic vertebrae. Mild degenerative changes were observed in bone structures.',
        ),
        (
            'No occlusive pathology was observed in the lumen of the [trachea](trachea) and both [main bronchi](main bronchus). The mediastinum could not be evaluated optimally in the non-contrast examination. As far as can be seen; mediastinal main vascular structures, [heart](heart) contour, size are normal. Pericardial effusion-thickening was not observed. Millimetric calcific [atheroma plaques](vascular calcification) were observed in the [left coronary arteries](left coronary artery). Thoracic [esophagus](esophagus) calibration was normal and no significant pathological wall thickening was detected. No enlarged lymph nodes in prevascular, pre-paratracheal, subcarinal or bilateral hilar-axillary pathological dimensions were detected. When examined in the lung parenchyma window; Mosaic attenuation pattern was observed in both [lungs](lung), especially in the [lower lobes](lung lower lobe). Segmental-subsegmental [peribronchial thickening](peribronchial thickening) and luminal narrowing were observed in both [lungs](lung). Mosaic attenuation was found to be secondary to small airway stenosis. [Pleuroparenchymal fibroatelectasis](atelectasis) sequelae changes were observed in the [middle lobe of the right lung](right lung middle lobe) and the [inferior lingular segment of the left lung upper lobe](left lung upper lobe). No mass lesion, pneumonic infiltration or contusion area was observed in the lung parenchyma. When the upper abdominal organs included in the sections were evaluated; No space-occupying lesion was detected in the [liver](liver) that entered the cross-sectional area. The [gallbladder](gallbladder) was not observed (operated). Two [angiomyolipomas](angiomyolipoma) with 4.5 and 7.5 mm diameters were observed in the middle part of the [left kidney](left kidney). Bilateral [adrenal glands](adrenal gland) were normal and no space-occupying lesion was detected. Bone structures in the study area are natural. Vertebral corpus heights are preserved.',
            'No occlusive pathology was observed in the lumen of the [trachea](trachea) and both [main bronchi](main bronchus). The mediastinum could not be evaluated optimally in the non-contrast examination. As far as can be seen; mediastinal main vascular structures, [heart](heart) contour, size are normal. Pericardial effusion-thickening was not observed. Millimetric calcific [atheroma plaques](vascular calcification) were observed in the [left coronary arteries](left coronary artery). Thoracic [esophagus](esophagus) calibration was normal and no significant pathological wall thickening was detected. No enlarged lymph nodes in prevascular, pre-paratracheal, subcarinal or bilateral hilar-axillary pathological dimensions were detected. When examined in the lung parenchyma window; Mosaic attenuation pattern was observed in both [lungs](lung), especially in the [lower lobes](lung lower lobe). Segmental-subsegmental [peribronchial thickening](peribronchial thickening) and luminal narrowing were observed in both [lungs](lung). Mosaic attenuation was found to be secondary to small airway stenosis. [Pleuroparenchymal fibroatelectasis](atelectasis) sequelae changes were observed in the [middle lobe of the right lung](right lung middle lobe) and the [inferior lingular segment of the left lung upper lobe](left lung upper lobe). No mass lesion, pneumonic infiltration or contusion area was observed in the lung parenchyma. When the upper abdominal organs included in the sections were evaluated; No space-occupying lesion was detected in the [liver](liver) that entered the cross-sectional area. The gallbladder was not observed (operated). Two [angiomyolipomas](angiomyolipoma) with 4.5 and 7.5 mm diameters were observed in the middle part of the [left kidney](left kidney). Bilateral [adrenal glands](adrenal gland) were normal and no space-occupying lesion was detected. Bone structures in the study area are natural. Vertebral corpus heights are preserved.'
        ),
    ]
}

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

_target_pattern = re.compile(r'\[(.*?)\]\((.*?)\)')

def process(dataset: str, split: str, num_samples: int):
    print(dataset, split, num_samples)
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
    tag_prompts = []
    findings_list = []
    impressions = []
    for item in tqdm(data, 'building tag inputs'):
        item.pop('findings')
        item.pop('impression')
        report = item['processed_report']
        match = report_pattern.match(report)
        findings, impression = match.group(1).strip(), match.group(2).strip()
        findings_list.append(findings)
        impressions.append(impression)
        tag_prompts.append(
            build_few_shot_conv(tag_system_prompt, tag_examples[dataset], findings),
        )
    tag_responses = llm.generate(tag_prompts, sampling_params)

    filter_prompts = []
    for i, tag_response in enumerate(tqdm(tag_responses, 'building filter inputs')):
        tagged_findings = tag_response.outputs[0].text
        data[i]['tagged_findings'] = tagged_findings
        filter_prompts.append(
            build_few_shot_conv(filter_system_prompt, filter_examples[dataset], tagged_findings),
        )

    filter_responses = llm.generate(filter_prompts, sampling_params)
    for i, item in enumerate(tqdm(data, 'post processing')):
        findings_prefix = 'Findings: '
        prefix_offset = len(findings_prefix)
        tags = []
        filtered_tagged_findings = filter_responses[i].outputs[0].text
        # NOTE: the pattern is matched twice, once for `sub`, once for `finditer`
        for match in _target_pattern.finditer(filtered_tagged_findings):
            # do not strip `phrase` for safety
            phrase, target = match.group(1), match.group(2).strip()
            start = prefix_offset + match.start(1) - 1
            tags.append({
                'phrase': phrase,
                'target': target,
                'start': start,
                'end': start + len(phrase),
            })
        ref_findings = _target_pattern.sub(r'\1', filtered_tagged_findings)
        ref_report = f'{findings_prefix}{ref_findings}\nImpression: {impressions[i]}'
        item['ref_report'] = ref_report
        item['tags'] = tags
    output_dir = PROCESSED_VG_DATA_ROOT / dataset
    output_dir.mkdir(exist_ok=True, parents=True)
    (output_dir / f'{split}.json').write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))

def main():
    for dataset, num_samples_dict in [
        ('MIMIC-CXR', {'train': 500, 'test': 500}),
        ('CT-RATE', {'train': 500, 'test': 500}),
    ]:
        for split, num_samples in num_samples_dict.items():
            process(dataset, split, num_samples)

if __name__ == '__main__':
    main()
