from __future__ import annotations

import cytoolz
import numpy as np
import torch

from mmmm.data.dataset.misc import toss
from mmmm.data.defs import ConvTurn
from mmmm.data.target_tax import TargetClass
from mmmm.tokenizer import MMMMTokenizer

template = {
    'general-singular': [
        'Is {} included in the imaging data?',
        'Is {} included in the medical image analysis?',
        'Does the medical image show {} clearly?',
        'Can {} be detected in the medical image provided?',
        'Is there a clear depiction of {} in this medical image?',
        'Does this medical image contain {}?',
        'Is {} identifiable in this medical image?',
        'Does the medical image illustrate {}?',
        'Is {} evident in the medical image?',
        'Can we observe {} in this medical image?',
        'Is there evidence of {} in the medical image?',
        'Does the medical image include a representation of {}?',
        'Is {} featured in the medical image?',
        'Does the medical image capture {}?',
        'Is {} present in the medical image analysis?',
        'Can {} be seen in the medical image?',
        'Is {} detectable in this medical image?',
        'Does the image showcase {}?',
        'Is {} part of the image composition?',
        'Can {} be found in the medical image?',
        'Is {} observable in this medical image?',
        'Does the medical image contain visual information about {}?',
        'Is {} represented in the medical image?',
        'Does the image provide a view of {}?',
        'Is {} included in the visual analysis?',
        'Can {} be spotted in the medical image?',
        'Is {} visible in the detailed medical image?',
        'Does the medical image display {}?',
        'Is {} captured in the medical image?',
        'Can {} be identified from the medical image?',
        'Is {} clear in the medical image?',
        'Does the medical image reveal {}?',
        'Is {} noticeable in the medical image?',
        'Can {} be observed in this image?',
        'Is {} discernible in the medical image?',
        'Does the image show details of {}?',
        'Is {} detectable in the image analysis?',
        'Can {} be appreciated in the medical image?',
        'Is {} part of the displayed anatomy?',
        'Does the medical image provide insight into {}?',
        'Is {} evident from the medical image?',
        'Can {} be analyzed from this image?',
        'Is {} visible in this detailed image analysis?',
        'Does the medical image illustrate any details of {}?',
        'Is {} a component of the medical image?',
        'Can the presence of {} be confirmed in the image?',
        'Is {} included in the visual inspection of the image?',
        'Does this image provide a clear view of {}?',
        'Is {} analyzed in the medical image?',
        'Can {} be seen clearly in the medical image?',
    ],
    'general-plural': [
        'Are {} included in the imaging data?',
        'Are {} included in the medical image analysis?',
        'Does the medical image show {} clearly?',
        'Can {} be detected in the medical image provided?',
        'Are there clear depictions of {} in this medical image?',
        'Does the medical image contain {}?',
        'Are {} identifiable in this medical image?',
        'Does the medical image illustrate {}?',
        'Are {} evident in the medical image?',
        'Can we observe {} in the medical image?',
        'Is there evidence of {} in the medical image?',
        'Does the medical image include representations of {}?',
        'Are {} featured in the medical image?',
        'Does the medical image capture {}?',
        'Are {} present in the medical image analysis?',
        'Can {} be seen in the medical image?',
        'Are {} detectable in this medical image?',
        'Does the image showcase {}?',
        'Are {} part of the image composition?',
        'Can {} be found in the medical image?',
        'Are {} observable in the medical image?',
        'Does the medical image contain visual information about {}?',
        'Are {} represented in the medical image?',
        'Does the image provide a view of {}?',
        'Are {} included in the visual analysis?',
        'Can {} be spotted in the medical image?',
        'Are {} visible in the detailed medical image?',
        'Does the medical image display {}?',
        'Are {} captured in the medical image?',
        'Can {} be identified from the medical image?',
        'Are {} clear in the medical image?',
        'Does the medical image reveal {}?',
        'Are {} noticeable in the medical image?',
        'Can {} be observed in the image?',
        'Are {} discernible in the medical image?',
        'Does the image show details of {}?',
        'Are {} detectable in the image analysis?',
        'Can {} be appreciated in the medical image?',
        'Are {} part of the displayed anatomy in the image?',
        'Does the medical image provide insight into {}?',
        'Are {} evident from the medical image?',
        'Can {} be analyzed from the image?',
        'Are {} visible in this detailed image analysis?',
        'Does the medical image illustrate any details of {}?',
        'Are {} components of the medical image?',
        'Can the presence of {} be confirmed in the image?',
        'Are {} included in the visual inspection of the image?',
        'Does the image provide a clear view of {}?',
        'Are {} analyzed in the medical image?',
        'Can {} be seen clearly in the medical image?'
    ],
    'anomaly': [
        "What irregularities are present in this medical scan?",
        "What deviations from the normal can be seen in this medical image?",
        "Can you spot any abnormalities in this image from a medical examination?",
        "What peculiarities are visible in this medical imaging result?",
        "Are there any signs of pathology in this medical image?",
        "Can you detect any anomalies in this clinical image?",
        "What unusual aspects are noticeable in this medical imaging study?",
        "Does this medical image show any signs of disease or abnormality?",
        "What are the notable anomalies in this image from medical diagnostics?",
        "Can any pathological features be seen in this medical imaging?",
        "What abnormalities are detected in this diagnostic image?",
        "Are there any discrepancies in this medical image compared to a normal one?",
        "Does this image reveal any medical conditions?",
        "What are the key anomalies found in this medical imaging scan?",
        "Can you point out any unusual findings in this medical image?",
        "What anomalies can be detected in this image used for medical diagnosis?",
        "Is there anything abnormal detected in this medical examination image?",
        "Are there detectable abnormalities in this medical imaging?",
        "What unusual findings are present in this medical image?",
        "Can this medical scan show any pathological abnormalities?",
        "Are there any abnormal signs in this medical scan?",
        "What anomalies are observable in this image from a clinical examination?",
        "Can you identify any signs of abnormality in this medical imaging?",
        "Are there any findings that suggest a medical condition in this image?",
        "What does this medical image reveal about potential abnormalities?",
        "What could be considered abnormal in this clinical imaging?",
        "Are there signs of any medical issues in this image?",
        "What can be discerned as abnormal in this medical scan?",
        "What abnormalities can be observed in this clinical imaging study?",
        "Are there any irregular patterns or features in this medical image?",
        "Can this image indicate any potential medical concerns?",
        "What anomalies are discernible in this image from a medical check-up?",
        "Are there any medical issues visible in this diagnostic image?",
        "What can you identify as abnormal in this medical imaging?",
        "Are there any atypical observations in this medical scan?",
        "Can any pathological indications be observed in this medical image?",
        "What non-typical elements are present in this medical imaging?",
        "Can you spot any irregularities in this image from medical diagnostics?",
        "What anomalies are visible in this medical scan?",
        "Are there any unexpected conditions shown in this medical image?",
        "What atypical conditions can be identified in this medical scan?",
        "Can you observe any abnormalities in this medical diagnostic image?",
        "Are there signs of anomalies in this clinical imaging study?",
        "What does this medical imaging reveal about potential irregularities?",
        "Can this medical image show signs of a specific condition?",
        "What are the detectable anomalies in this diagnostic medical scan?",
        "What could potentially be abnormal in this clinical diagnostic image?",
        "Are there noticeable irregularities in this medical imaging result?",
        "What atypical results are visible in this medical image?",
        "Are there any concerning features in this medical diagnostic scan?",
        "What irregular features can you detect in this medical imaging?",
        "Can any pathology be observed in this clinical image?",
        "What unusual characteristics are visible in this medical scan?",
        "Are there observable signs of any condition in this medical image?",
        "Can any clinical anomalies be identified in this medical scan?",
        "What can you discern as abnormal in this image from a medical test?",
        "What signs of abnormality can be seen in this medical diagnostic scan?",
        "What deviations are detectable in this medical image?",
        "Are there any unusual aspects in this medical imaging analysis?",
        "Can this medical scan indicate any abnormalities?",
        "What are the signs of possible abnormalities in this medical image?",
        "Can you identify any atypical features in this medical scan?",
        "What does this clinical diagnostic image reveal about the patient's condition?",
        "Are there any noticeable irregularities in this medical image?",
        "What unusual findings does this medical scan show?",
        "Can any signs of disease be detected in this medical image?",
        "What can you point out as abnormal in this medical diagnostic image?",
        "Are there any abnormalities that stand out in this clinical scan?",
        "What unusual characteristics are found in this medical diagnostic study?",
        "Are there any signs of disease or irregularity in this medical scan?",
        "What can be identified as irregular in this image from a medical procedure?",
    ],
    'no-anomaly-answer': [
        "No anomaly is found.",
        "There are no anomalies detected.",
        "The image shows no signs of abnormalities.",
        "No abnormalities are present.",
        "The scan reveals no anomalies.",
        "No anomalies are detected.",
        "The medical scan indicates normal conditions with no anomalies.",
        "No signs of abnormality are found in the image.",
        "No irregularities are observed in the diagnostic image.",
        "There is no evidence of anomaly in the medical scan.",
        "No abnormalities or irregularities detected.",
        "The imaging study reveals no abnormal findings.",
        "No deviations from the normal are observed.",
        "The patient's image shows no signs of anomalies.",
        "Medical imaging is normal with no detected anomalies.",
        "There are no detectable abnormalities in the scan.",
        "No anomalies are visible in the medical imaging.",
        "The analysis confirms no abnormalities present.",
        "No signs of any abnormalities are present in this image.",
        "The scan is clear; no abnormalities found.",
        "The clinical scan shows no signs of irregularities.",
        "There are no abnormalities in the diagnostic images.",
        "The imaging does not reveal any anomalies.",
        "No anomalies or irregular findings are present.",
        "The scan results are normal with no evidence of anomalies.",
        "The medical image analysis shows no abnormalities.",
        "The medical review indicates no anomalies present.",
        "Diagnostic findings are normal; no abnormalities are seen.",
        "No anomalies are evident in the medical imaging results.",
        "There are no detectable signs of abnormalities in the scan.",
        "The medical image is normal, with no anomalies found.",
        "The image analysis shows a normal condition without anomalies.",
        "No anomalies are apparent in the medical scan.",
        "The diagnostic imaging is clear of any abnormalities.",
        "No medical anomalies are identified in the scan.",
        "The results show a normal scan with no abnormalities.",
        "No signs of any medical condition are detected.",
        "The clinical findings show no abnormalities.",
        "No pathological findings are observed in the image.",
        "The scan does not show any abnormal findings.",
        "No evidence of medical anomalies is found.",
        "There are no visible anomalies in the medical images.",
        "No unusual or pathological features are detected.",
        "The medical examination reveals no anomalies.",
        "No signs of disease or abnormalities are detected in the scan.",
        "The imaging study confirms a normal status with no anomalies.",
        "No abnormalities are detected in the clinical imaging.",
        "The scan shows a clear result with no anomalies found.",
        "No abnormalities are visible in the patient's diagnostic image.",
        "The medical analysis reveals no signs of abnormalities.",
        "No pathological changes are observed in the medical scan.",
        "The clinical image is free of any abnormal findings.",
        "No detectable abnormalities are present in the imaging.",
        "The medical imaging is normal, with no pathological signs.",
        "No abnormal medical conditions are identified in the image.",
        "The patient's medical imaging shows no signs of anomalies.",
        "No clinical abnormalities are found in the diagnostic imaging.",
        "The image is clear of any clinical anomalies.",
        "Medical diagnostic imaging shows no abnormalities.",
        "No anomalies or abnormal findings are evident in the scan.",
        "The medical imaging study reveals no anomalies.",
        "No signs of pathology are detected in the medical imaging.",
        "The scan results are clear, showing no abnormalities.",
        "No medical issues or abnormalities are observed in the image.",
        "The examination results show no abnormalities or anomalies.",
        "Diagnostic imaging is normal, with no unusual findings.",
    ],
}

def _join_list_natural(names: list[str]):
    if len(names) == 1:
        ret = names[0]
    elif len(names) == 2:
        ret = f'{names[0]} and {names[1]}'
    else:
        ret = ', '.join(names[:-1]) + f', and {names[-1]}'
    return ret

def _list_general_results(
    tokenizer: MMMMTokenizer,
    names: list[str],
    classes: list[str],
    pos_mask: torch.BoolTensor,
    *,
    wrap_pos: bool,
    wrap_neg: bool,
):
    ret = 'Results:'
    wrapped_classes = []
    for i, name in enumerate(names):
        pos = pos_mask[i]
        wrap = wrap_pos if pos else wrap_neg
        if wrap:
            ret += tokenizer.wrap_name(name, pos=pos)
            wrapped_classes.append(classes[i])
        else:
            ret += f' {name}'
        ret += ': ' + ('yes' if pos else 'no')
        ret += '.' if i + 1 == len(names) else ','
    # prepend a "Results" to avoid capitalization
    return ret, wrapped_classes

@cytoolz.curry
def sample_name(class_name: str, R: np.random.RandomState, target_tax: dict[str, TargetClass]):
    if (target := target_tax.get(class_name)) is None:
        return class_name
    else:
        return R.choice(target.synonyms)

GENERAL_LIST_DESC = 'List each request followed by "yes" or "no" to indicate its presence or absence.'

def gen_general_conv(
    pos_classes: list[str],
    neg_classes: list[str],
    grounding: bool,
    neg_grounding: bool,
    tokenizer: MMMMTokenizer,
    target_tax: dict[str, TargetClass],
    R: np.random.RandomState,
) -> tuple[list[ConvTurn], list[str]]:
    """
    Returns:
      - conversation
      - class names, following the order occurring in the conversation
    """
    if len(pos_classes) == 0 and len(neg_classes) == 0:
        return [], []
    # copy the input list because the shuffling is in-place
    pos_classes = list(pos_classes)
    R.shuffle(pos_classes)
    neg_classes = list(neg_classes)
    R.shuffle(neg_classes)
    # merge positive and negative classes with random order without shuffling
    pos_class_mask = torch.zeros(len(pos_classes) + len(neg_classes), dtype=torch.bool)
    pos_class_mask[R.choice(pos_class_mask.shape[0], len(pos_classes), replace=False)] = True
    pos_it, neg_it = map(iter, [pos_classes, neg_classes])
    classes = [
        next(pos_it) if m else next(neg_it)
        for m in pos_class_mask
    ]
    assert len(classes) > 0
    if len(classes) == 1:
        prompt_template = R.choice(template['general-singular'])
    else:
        prompt_template = R.choice(template['general-plural'])
    names = list(map(sample_name(R=R, target_tax=target_tax), classes))
    prompt = f'{prompt_template.format(_join_list_natural(names))} {GENERAL_LIST_DESC}'
    response, grounding_classes = _list_general_results(
        tokenizer, names, classes, pos_class_mask,
        wrap_pos=grounding, wrap_neg=neg_grounding,
    )
    return [ConvTurn(prompt, response)], grounding_classes

# NOTE: this description may be inconsistent with the prompt
ANOMALY_LIST_DESC = 'List each anomaly separated by commas.'

def gen_anomaly_detection_conv(
    anomaly_classes: list[str],
    grounding: bool,
    tokenizer: MMMMTokenizer,
    target_tax: dict[str, TargetClass],
    R: np.random.RandomState,
):
    prompt_template = R.choice(template['anomaly'])
    prompt = f'{prompt_template} {ANOMALY_LIST_DESC}'
    if len(anomaly_classes) == 0:
        response = R.choice(template['no-anomaly-answer'])
    else:
        names = list(map(sample_name(R=R, target_tax=target_tax), anomaly_classes))
        if grounding:
            results = ','.join(map(tokenizer.wrap_name(pos=True), names))
        else:
            results = ', '.join(names)
        response = 'Results: ' + results + '.'
    grounding_classes = list(anomaly_classes) if grounding else []
    return [ConvTurn(prompt, response)], grounding_classes

def gen_brats_conv(
    pos_classes: list[str],
    neg_classes: list[str],
    tokenizer: MMMMTokenizer,
    target_tax: dict[str, TargetClass],
    grounding: bool,
    neg_grounding: bool,
    R: np.random.RandomState,
):
    """
    Args:
        pos_classes: excluding glioma
    """
    ret = [
        gen_anomaly_detection_conv(
            ['glioma'],
            grounding,
            tokenizer,
            target_tax,
            R,
        ),
        gen_general_conv(
            pos_classes,
            neg_classes,
            grounding,
            neg_grounding,
            tokenizer,
            target_tax,
            R,
        ),
    ]
    conv, classes = zip(*ret)
    return list(cytoolz.concat(conv)), list(cytoolz.concat(classes))

def gen_anomaly_conv(
    pos_classes: list[str],
    neg_classes: list[str],
    complete_anomaly: bool,
    grounding: bool,
    neg_grounding: bool,
    tokenizer: MMMMTokenizer,
    target_tax: dict[str, TargetClass],
    dataset: str,
    R: np.random.RandomState,
):
    if dataset.startswith('BraTS2023'):
        if 'glioma' in pos_classes and toss(R, 0.9):
            pos_classes = list(pos_classes)
            pos_classes.remove('glioma')
            return gen_brats_conv(
                pos_classes,
                neg_classes,
                tokenizer,
                target_tax,
                grounding,
                neg_grounding,
                R,
            )
        else:
            return gen_general_conv(
                pos_classes,
                neg_classes,
                grounding,
                neg_grounding,
                tokenizer,
                target_tax,
                R,
            )

    if (len(pos_classes) == 0 and complete_anomaly) or (len(pos_classes) > 0 and toss(R, 0.8)):
        return gen_anomaly_detection_conv(
            pos_classes,
            grounding,
            tokenizer,
            target_tax,
            R,
        )

    return gen_general_conv(
        pos_classes,
        neg_classes,
        grounding,
        neg_grounding,
        tokenizer,
        target_tax,
        R,
    )
