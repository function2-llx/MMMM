import json
import random

from PIL import Image
from einops import repeat
import torch
from torchvision.transforms import v2 as tvt

from luolib.types import tuple3_t
from monai import transforms as mt

from mmmm.models import MMMMTokenizer
from ..defs import DataPoint, PROCESSED_VL_DATA_ROOT, split_t
from ..utils import prepare_vlm_inputs

CAPTION_PROMPTS = [
    'Describe the following image in detail.',
    'Provide a detailed description of the given image.',
    'Give an elaborate explanation of the image you see',
    'Share a comprehensive rundown of the presented image',
    'Offer a thorough analysis of the image',
    'Explain the various aspects of the image before you',
    'Clarify the contents of the displayed image with great detail',
    'Characterize the image using a well-detailed description',
    'Break down the elements of the image in a detailed manner',
    'Walk through the important details of the image',
    'Portray the image with a rich, descriptive narrative',
    'Narrate the contents of the image with precision',
    'Analyze the image in a comprehensive and detailed manner',
    'Illustrate the image through a descriptive explanation',
    'Examine the image closely and share its details',
    'Write an exhaustive depiction of the given image',
    'Summarize the visual content of the image.',
    'What can you infer from this picture?',
    'Please caption this image.',
    'Can you summarize the image presented?',
    'Describe this {}.',
]

REPORT_PROMPTS = [
    'Can you provide a report consists of findings and impression for this {}?',
    'Please report this {} with findings and impression.',
    'Describe this {} with findings and impression.',
    'Please write a report consists of findings and impression for this {}.',
    'Please provide a report consists of findings and impression for this {}.',
    'Can you provide a summary consists of findings and impression of this {}?',
    'What are the findings and impression presented in this {}?',
    'Please write a report consists of findings and impression for this {}.',
    'Can you provide a description consists of findings and impression of this {}?',
    'Please report this {} with finding and impression.',
    'Analyze this {} and provide a detailed report with both findings and impression.',
    'Examine the {} and construct a report that includes findings and impression.',
    'Based on your analysis, what would be an accurate report for this {}, including both findings and impression?',
    'What is the most appropriate report for this {}, detailing both the findings and impression?',
    'Can you provide a radiology report for this {}?',
    'Please report this {}.',
    'What is the medical significance of this {}?',
    'Can you provide a summary of this {}?',
    'Please write a radiology report for this {}.',
    'Please generate a radiology report for this {}.',
    'Please provide a report for this {}.',
    'Can you provide a brief summary of this {}?',
    'Please write a radiology report for this {}.',
    'Can you provide a report summary for this {}?'
]

COMPLETE_REFERRINGS = ['medical image', 'radiograph', 'scan', 'radiology image', 'radiology scan', 'medical scan']

PARTIAL_REFERRINGS = [' image', ' scan', ' radiograph']

def get_vl_data_list(name: str, split: split_t):
    dataset_dir = PROCESSED_VL_DATA_ROOT / name
    with open(dataset_dir / f'{split}.json') as f:
        info = json.load(f)
    return info

class VLTransform(mt.Transform):
    def __init__(
        self,
        base_vit_patch_size: tuple3_t[int],
        tokenizer: MMMMTokenizer,
        inference: bool = False
    ):
        super().__init__()
        self.vit_patch_size: tuple3_t[int] = base_vit_patch_size
        self.tokenizer = tokenizer
        self.inference = inference

    def __call__(self, data: dict) -> DataPoint:
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
            vit_patch_size = (1, *self.vit_patch_size[1:])
        elif len(image.shape) == 4:
            if min(image.shape[1:]) > 512:
                slices = []
                for i in range(image.shape[3]):
                    slices.append(tvt.functional.resize(image[:, :, :, i], 512))
                image = torch.stack(slices, dim=3)
            vit_patch_size = self.vit_patch_size
        
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
        data = {
            'image': image,
            'grounding_image': None,
            'vit_patch_size': vit_patch_size,
            'vlm_inputs': vlm_inputs,
            'mask': [],
            'bbox': [],
        }
        return data