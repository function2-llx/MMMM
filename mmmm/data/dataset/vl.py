from __future__ import annotations as _

from dataclasses import dataclass
import json

import einops
import numpy as np
import torch
from torchvision.io import read_image
from torchvision.transforms import v2 as tvt

from luolib.utils.misc import ensure_rgb
from monai import transforms as mt
from monai.utils import convert_to_tensor

import mmmm.data.dataset._dataset as _dataset
from mmmm.tokenizer import MMMMTokenizer
from ..defs import ConvTurn, DataPoint, PROCESSED_VL_DATA_ROOT, split_t
from ..utils import prepare_vlm_inputs
from .misc import gen_modality_conversation, toss

CAPTION_PROMPTS = [
    'Describe the following image in detail.',
    'Provide a detailed description of the given image.',
    'Give an elaborate explanation of the image you see',
    'Share a comprehensive rundown of the presented image.',
    'Explain the various aspects of the image before you.',
    'Clarify the contents of the displayed image with great detail.',
    'Characterize the image using a well-detailed description.',
    'Break down the elements of the image in a detailed manner.',
    'Walk through the important details of the image.',
    'Portray the image with a rich, descriptive narrative.',
    'Narrate the contents of the image with precision.',
    'Illustrate the image through a descriptive explanation.',
    'Examine the image closely and share its details.',
    'Write an exhaustive depiction of the given image.',
    'What can you infer from this picture?',
    'Please caption this image.',
    'What are the key features of the image you see?',
    'What can you observe in this image?',
    'Please provide a detailed description of the image.',
    'What do you see in this image?',
    'Caption this image, highlighting its scientific or medical importance.',
    'What are the key features of the image you see?',
    'What can you observe in this image?',
    'Offer a thorough and descriptive summary of the image.',
]

REPORT_PROMPTS = [
    'Can you provide a report consists of findings and impression for this {}?',
    'Please report on this {} with findings and impression.',
    'Describe this {} with findings and impression.',
    'Please write a report consists of findings and impression for this {}.',
    'Please provide a report consists of findings and impression for this {}.',
    'Can you provide a summary consists of findings and impression of this {}?',
    'What are the findings and impression presented in this {}?',
    'Please write a report consists of findings and impression for this {}.',
    'Can you provide a description consists of findings and impression of this {}?',
    'Please report on this {} with finding and impression.',
    'Analyze this {} and provide a detailed report with both findings and impression.',
    'Examine the {} and construct a report that includes findings and impression.',
    'Based on your analysis, what would be an accurate report for this {}, including both findings and impression?',
    'What is the most appropriate report for this {}, detailing both the findings and impression?',
    'Can you provide a radiology report for this {}?',
    'Please diagnosis this {}.',
    'Please write a radiology report for this {}.',
    'Please generate a radiology report for this {}.',
    'Please provide a report for this {}.',
    'Please write a radiology report for this {}.',
    'Please generate a detailed radiology report for this {}, including a description of the findings and your impression.',
    'Evaluate the {} and generate a detailed report.',
    'Interpret this {} and produce a detailed diagnostic report.',
    'Review the {} and provide a thorough clinical report.',
    'Diagnose this {}.',
    'Report on this {}.',
    'Detail findings and impression from this {}.',
    'Offer a thorough analysis of the {}.',
    'Analyze the {} with findings and impression.',
    'Generate a detailed radiology report for the given {}.',
    'Please provide a detailed radiological interpretation of this {}.',
    'Construct a detailed report with diagnostic findings and clinical impressions for this {}.',
    'What is the diagnostic significance of this {}?',
    "What is the indication of this {}?",
    "Create a detailed report for this {}.",
    "Give a detailed radiology report for this {}.",
]

FINDINGS_PROMPT = [
    'What are the findings presented in this {}?'
]

COMPLETE_REFERRINGS = ['image', 'medical image', 'radiograph', 'scan', 'radiology image', 'radiology scan', 'medical scan']

PARTIAL_REFERRINGS = [' image', ' scan', ' radiograph']

def get_vl_data_list(name: str, split: split_t):
    dataset_dir = PROCESSED_VL_DATA_ROOT / name
    with open(dataset_dir / f'{split}.json') as f:
        info = json.load(f)
    return info

@dataclass
class VLTransConf:
    max_smaller_tokens_xy: int
    max_tokens_z: int

class VLTransform(mt.RandomizableTransform):
    def __init__(
        self,
        conf: _dataset.DatasetConf,
        tokenizer: MMMMTokenizer,
        inference: bool = False,
    ):
        super().__init__()
        self.conf = conf
        self.tokenizer = tokenizer
        self.inference = inference

    @property
    def max_smaller_edge(self) -> int:
        conf = self.conf
        return conf.vit_patch_size_xy * conf.vl_trans.max_smaller_tokens_xy

    def __call__(self, data: dict) -> DataPoint:
        conf = self.conf
        trans_conf = conf.vl_trans
        image_idx = self.R.randint(len(data['image']))
        image_path = data['image'][image_idx]
        if modalities := data.get('modality'):
            modality = modalities[image_idx]
        else:
            modality = None
        if image_path.endswith('.pt'):
            image = torch.load(image_path)
        else:
            image = read_image(image_path)
            image = einops.rearrange(image, 'c h w -> c 1 h w')
        image = tvt.functional.to_dtype(image, scale=True)
        smaller_edge = min(image.shape[2:])
        if image.shape[1] == 1:
            if smaller_edge > self.max_smaller_edge:
                # this might be redundant, but let's be careful
                image = einops.rearrange(image, 'c 1 h w -> c h w')
                image = tvt.functional.resize(image, self.max_smaller_edge)
                image = einops.rearrange(image, 'c h w -> c 1 h w')
            patch_size = (1, conf.vit_patch_size_xy, conf.vit_patch_size_xy)
        else:
            # TODO: adjust patch_size_z according to image
            patch_size_z = conf.base_vit_patch_size_z
            max_slices = patch_size_z * trans_conf.max_tokens_z
            new_shape = [min(image.shape[1], max_slices), *image.shape[2:]]
            if smaller_edge > self.max_smaller_edge:
                for i in (1, 2):
                    new_shape[i] = round(new_shape[i] * self.max_smaller_edge / smaller_edge)
            if tuple(new_shape) != image.shape[1:]:
                image = mt.Resize(new_shape)(image)
            patch_size = (patch_size_z, conf.vit_patch_size_xy, conf.vit_patch_size_xy)
        image = mt.DivisiblePad(patch_size)(image)
        image = convert_to_tensor(image)
        image, _ = ensure_rgb(image, contiguous=True)
        # TODO: intensity normalization
        referring: str = self.R.choice(COMPLETE_REFERRINGS)
        conversation = []
        if caption := data.get('caption'):
            conversation.append(ConvTurn(self.R.choice(CAPTION_PROMPTS), caption))
        if (findings := data.get('findings')) and (impression := data.get('impression')):
            conversation.append(
                ConvTurn(
                    self.R.choice(FINDINGS_PROMPT).format(referring),
                    findings,
                ),
            )
        elif findings := data.get('findings'):
            conversation.append(
                ConvTurn(
                    self.R.choice(REPORT_PROMPTS).format(referring),
                    f"Findings: {findings}\nImpression: {impression}",
                ),
            )
        if vqa := data.get('vqa'):
            conversation.extend([ConvTurn(qa['question'], qa['answer']) for qa in vqa])
        self.R.shuffle(conversation)
        if modality is not None and toss(self.R, 0.5):
            # prepend the modality conversation
            conversation = gen_modality_conversation(modality, self.R) + conversation
        vlm_inputs, conversation_text = prepare_vlm_inputs(
            conversation,
            self.tokenizer,
            (np.array(image.shape[1:]) // patch_size).prod().item(),
            self.inference,
            grounding=False,
        )
        data: DataPoint = {
            'image': image,
            'grounding_image': torch.zeros(3, *patch_size),
            'patch_size': patch_size,
            'vlm_inputs': vlm_inputs,
            'mask': torch.zeros(0, *patch_size, dtype=torch.bool),
            'mask_index': torch.empty(0, dtype=torch.bool),
            'bbox': torch.empty(0, 2, 3),
            'bbox_index': torch.zeros(0, dtype=torch.bool),
        }
        return data
