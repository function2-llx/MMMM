from collections.abc import Callable, Sequence
import json
from pathlib import Path
import random

from PIL import Image
from einops import repeat
import torch
from torchvision.transforms import v2 as tvt

from luolib.datamodule import ExpDataModuleBase
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
    "Summarize the visual content of the image."
]

REPORT_PROMPTS = [
    'Can you provide a caption consists of finding and impression for this medical image?',
    'Please caption this medical scan with finding and impression.',
    'Describe this medical scan with finding and impression.',
    'Please write a caption consists of finding and impression for this image.',
    'Please provide a caption consists of finding and impression for this medical image.',
    'Can you provide a summary consists of finding and impression of this radiograph?',
    'What are the findings and impression presented in this medical scan?',
    'Please write a caption consists of finding and impression for this scan.',
    'Can you provide a description consists of finding and impression of this medical scan?',
    'Please caption this medical scan with finding and impression.',
    'Analyze this medical image and provide a detailed caption with both finding and impression.',
    'Examine the medical image and construct a caption that includes finding and impression.',
    'Based on your analysis, what would be an accurate caption for this medical image, including both finding and impression?',
    'What is the most appropriate caption for this medical scan, detailing both the finding and impression?',
    'Can you provide a radiology report for this medical image?',
    'Please report this medical scan.',
    'What is the medical significance of this image?',
    'What can you infer from this picture?',
    'Can you provide a quick summary of this image?',
    'Describe this medical scan.',
    'Please write a radiology report for this image.',
    'Can you summarize the images presented?',
    'Please generate a radiology report for this scan.',
    'Describe the regions of interest in this scan.',
    'Please provide a caption for this medical image.',
    'Can you provide a brief summary of this radiograph?',
    'Describe the structures involved in this medical image.',
    'What are the findings presented in this medical scan?',
    'Please write a radiology report for this scan.',
    'Please caption this medical scan.',
    'Can you provide a report summary for this medical scan?'
]

def get_vl_data_list(name: str, split: split_t):
    pass

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
        if 'pt' in data['image']:
            image = torch.load(data['image'])
        else:
            image = Image.open(data['image'])
            image = tvt.functional.to_tensor(image)
        if len(image.shape) == 3:
            if min(image.shape[1:]) > 512:
                image = tvt.functional.resize(image, 512)
            image = repeat(image, 'c h w -> c 1 h w')
        elif len(image.shape) == 4:
            if min(image.shape[1:]) > 512:
                slices = []
                for i in range(image.shape[3]):
                    slices.append(tvt.functional.resize(image[:, :, :, i], 512))
                image = torch.stack(slices, dim=3)
        conversation = [(data['question'], data['answer'])]
        vlm_inputs, conversation_text = prepare_vlm_inputs(
            conversation, self.tokenizer, self.patch_size, self.vit_patch_size, self.inference,
        )
        data = {
            'image': image,
            'masks': None,
            'mask_classes': 0,
            'vlm_inputs': vlm_inputs,
        }
        return data

class VQADataModule(ExpDataModuleBase):
    def __init__(
        self,
        trans,
        tokenizer: MMMMTokenizer,
        data_root: Path = PROCESSED_VL_DATA_ROOT,
        datasets: list[str] = ['Slake', 'VQA-Med'],
        *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.data_root = data_root
        self.datasets = datasets
        self.trans_conf = trans

    def train_data(self) -> Sequence:
        train_data = []
        for dataset in self.datasets:
            with open(self.data_root / dataset / 'train.json') as f:
                data = json.load(f)
            if dataset == 'Radiopaedia':
                train_data.extend(
                    [
                        {
                            'dataset_dir': self.data_root / dataset,
                            'image': random.choice(item['image']),
                            **random.choice(item['qa_list']),
                        }
                        for item in data

                    ]
                )
            else:
                train_data.extend(
                    [
                        {
                            'dataset_dir': self.data_root / dataset,
                            'image': item['image'],
                            'question': item['question'],
                            'answer': item['answer'],
                        }
                        for item in data
                    ]
                )
        return train_data

    def val_data(self) -> Sequence:
        val_data = []
        for dataset in self.datasets:
            with open(self.data_root / dataset / 'validate.json') as f:
                data = json.load(f)
            val_data.extend(
                [
                    {
                        'dataset_dir': self.data_root / dataset,
                        'image': item['image'],
                        'question': item['question'],
                        'answer': item['answer'],
                    }
                    for item in data
                ]
            )

        return val_data

    def train_transform(self) -> Callable:
        conf = self.trans_conf
        return VLTransform(
            conf.vit_patch_size,
            self.tokenizer,
        )

    # def get_train_collate_fn(self):
    #     return mmmm_collate_fn

class CapDataModule(ExpDataModuleBase):
    def __init__(
        self,
        trans,
        tokenizer: MMMMTokenizer,
        data_root: Path = PROCESSED_VL_DATA_ROOT,
        datasets: list[str] = ['PMC-OA'],
        prompts: list[str] = CAPTION_PROMPTS,
        *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.data_root = data_root
        self.datasets = datasets
        self.trans_conf = trans
        self.prompts = prompts

    def train_data(self) -> Sequence:
        train_data = []
        for dataset in self.datasets:
            with open(self.data_root / dataset / 'train.json') as f:
                data = json.load(f)
            train_data.extend(
                [
                    {
                        'dataset_dir': self.data_root / dataset,
                        'image': item['image'],
                        'question': random.choice(self.prompts),
                        'answer': item['caption'],
                    }
                    for item in data
                ]
            )
        return train_data

    def val_data(self) -> Sequence:
        val_data = []
        for dataset in self.datasets:
            with open(self.data_root / dataset / 'validate.json') as f:
                data = json.load(f)
            val_data.extend(
                [
                    {
                        'dataset_dir': self.data_root / dataset,
                        'image': item['image'],
                        'question': random.choice(self.prompts),
                        'answer': item['caption'],
                    }
                    for item in data
                ]
            )

        return val_data

    def train_transform(self) -> Callable:
        conf = self.trans_conf
        return VLTransform(
            conf.vit_patch_size,
            self.tokenizer,
        )

    # def get_train_collate_fn(self):
    #     return mmmm_collate_fn

class RepDataModule(ExpDataModuleBase):
    def __init__(
        self,
        trans,
        tokenizer: MMMMTokenizer,
        data_root: Path = PROCESSED_VL_DATA_ROOT,
        datasets: list[str] = ['Radiopaedia', 'OpenI'],
        prompts: list[str] = REPORT_PROMPTS,
        *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.data_root = data_root
        self.datasets = datasets
        self.trans_conf = trans
        self.prompts = prompts

    def train_data(self) -> Sequence:
        train_data = []
        for dataset in self.datasets:
            with open(self.data_root / dataset / 'train.json') as f:
                data = json.load(f)
            train_data.extend(
                [
                    {
                        'dataset_dir': self.data_root / dataset,
                        'image': random.choice(item['image']),
                        'question': random.choice(self.prompts),
                        'answer': item['caption'],
                    }
                    for item in data

                ]
            )
        return train_data

    def val_data(self) -> Sequence:
        val_data = []
        for dataset in self.datasets:
            with open(self.data_root / dataset / 'validate.json') as f:
                data = json.load(f)
            val_data.extend(
                [
                    {
                        'dataset_dir': self.data_root / dataset,
                        'image': random.choice(item['image']),
                        'question': random.choice(self.prompts),
                        'answer': item['caption'],
                    }
                    for item in data
                ]
            )

        return val_data

    def train_transform(self) -> Callable:
        conf = self.trans_conf
        return VLTransform(
            conf.vit_patch_size,
            self.tokenizer,
        )

    # def get_train_collate_fn(self):
    #     return mmmm_collate_fn
