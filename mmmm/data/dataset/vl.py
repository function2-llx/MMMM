from __future__ import annotations as _

import json
import random

from PIL import Image
from einops import repeat
import torch
from torchvision.transforms import v2 as tvt

from luolib.types import tuple3_t
from monai import transforms as mt

from mmmm.models import MMMMTokenizer
import mmmm.data.dataset._dataset as _dataset
from ..defs import DataPoint, PROCESSED_VL_DATA_ROOT, split_t
from ..utils import prepare_vlm_inputs

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

COMPLETE_REFERRINGS = ['image', 'medical image', 'radiograph', 'scan', 'radiology image', 'radiology scan', 'medical scan']

PARTIAL_REFERRINGS = [' image', ' scan', ' radiograph']

def get_vl_data_list(name: str, split: split_t):
    dataset_dir = PROCESSED_VL_DATA_ROOT / name
    with open(dataset_dir / f'{split}.json') as f:
        info = json.load(f)
    return info

class VLTransform(mt.Transform):
    def __init__(
        self,
        conf: _dataset.DatasetConf,
        tokenizer: MMMMTokenizer,
        inference: bool = False
    ):
        super().__init__()
        self.conf = conf
        self.tokenizer = tokenizer
        self.inference = inference

    def __call__(self, data: dict) -> DataPoint:
        conf = self.conf
        image_path = random.choice(data['image'])
        if image_path.endswith('.pt'):
            image = torch.load(image_path)
        else:
            image = Image.open(image_path)
            image = tvt.functional.to_tensor(image)
        if len(image.shape) == 3:
            if min(image.shape[1:]) > 512:
                image = tvt.functional.resize(image, 512)
            image = repeat(image, 'c h w -> c 1 h w')
            vit_patch_size = (1, conf.vit_patch_size_xy, conf.vit_patch_size_xy)
        elif len(image.shape) == 4:
            if min(image.shape[1:]) > 512:
                slices = []
                for i in range(image.shape[3]):
                    slices.append(tvt.functional.resize(image[:, :, :, i], 512))
                image = torch.stack(slices, dim=3)
            # TODO: update vit_patch_size_z
            vit_patch_size = (conf.base_vit_patch_size_z, conf.vit_patch_size_xy, conf.vit_patch_size_xy)
        
        conversation = []
        if data.get('caption'):
            conversation.append((random.choice(CAPTION_PROMPTS), data['caption']))
        if data.get('findings') and data.get('impression'):
            referring = random.choice(COMPLETE_REFERRINGS)
            if data.get('modality'):
                referring = random.choice(referring, data['modality'] + random.choice(PARTIAL_REFERRINGS))
            conversation.append((random.choice(REPORT_PROMPTS).format(random.choice(referring)), 'Findings: ' + data['findings'] + ' Impression: ' + data['impression']))
        if data.get('vqa'):
            conversation.extend([(qa['question'], qa['answer']) for qa in data['vqa']])

        random.shuffle(conversation)
        vlm_inputs, conversation_text = prepare_vlm_inputs(
            conversation, self.tokenizer, self.patch_size, vit_patch_size, self.inference,
        )
        data: DataPoint = {
            'image': image,
            'grounding_image': None,
            'patch_size': vit_patch_size,
            'vlm_inputs': vlm_inputs,
            'mask': torch.empty(0, *image.shape[1:], dtype=torch.bool),
            'mask_index': torch.empty(0, dtype=torch.bool),
            'bbox': torch.empty(0, 2, 3),
            'bbox_index': torch.zeros(0, dtype=torch.bool),
        }
        return data
