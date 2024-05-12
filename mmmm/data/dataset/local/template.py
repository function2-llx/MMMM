from __future__ import annotations

import numpy as np
import torch

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
    ]
}

list_desc = 'List each request followed by "yes" or "no" to indicate its presence or absence.'

def _join_list_natural(names: list[str]):
    if len(names) == 1:
        ret = names[0]
    elif len(names) == 2:
        ret = f'{names[0]} and {names[1]}'
    else:
        ret = ', '.join(names[:-1]) + f', and {names[-1]}'
    return ret

def _list_results(
    tokenizer: MMMMTokenizer,
    names: list[str],
    classes: list[str],
    pos_mask: torch.BoolTensor,
    *,
    wrap_pos: bool,
    wrap_neg: bool,
):
    wrap = torch.empty(len(names), dtype=torch.bool)
    wrap[pos_mask] = wrap_pos
    wrap[~pos_mask] = wrap_neg
    ret = 'Results:'
    wrapped_classes = []
    for i, name in enumerate(names):
        if wrap[i]:
            ret += tokenizer.wrap_name(name, pos=pos_mask[i])
            wrapped_classes.append(classes[i])
        else:
            ret += f' {name}'
        ret += ': ' + ('yes' if pos_mask[i] else 'no')
        ret += '.' if i + 1 == len(names) else ','
    # prepend a "Results" to avoid capitalization
    return ret, wrapped_classes

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
    # replace with synonyms
    names = [
        class_name if (target := target_tax.get(class_name)) is None
        else R.choice(target.synonyms)
        for class_name in classes
    ]
    prompt = f'{prompt_template.format(_join_list_natural(names))} {list_desc}'
    response, grounding_classes = _list_results(
        tokenizer, names, classes, pos_class_mask,
        wrap_pos=grounding, wrap_neg=neg_grounding,
    )
    return [ConvTurn(prompt, response)], grounding_classes

def gen_anomaly_conv(
    pos_classes: list[str],
    neg_classes: list[str],
    _complete: bool,
    grounding: bool,
    neg_grounding: bool,
    tokenizer: MMMMTokenizer,
    target_tax: dict[str, TargetClass],
    dataset: str,
    R: np.random.RandomState | int,
):
    return gen_general_conv(
        pos_classes,
        neg_classes,
        grounding,
        neg_grounding,
        tokenizer,
        target_tax,
        R,
    )
