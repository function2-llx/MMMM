from __future__ import annotations as _

from dataclasses import dataclass
from typing import TypedDict

import numpy as np
import orjson
import torch

from luolib.utils.misc import ensure_rgb
from monai import transforms as mt
from monai.utils import InterpolateMode, convert_to_tensor

import mmmm.data.dataset._dataset as _dataset
from mmmm.tokenizer import MMMMTokenizer
from ..defs import ConvTurn, PROCESSED_VL_DATA_ROOT, Split
from ..target_tax import get_target_tax
from ..utils import prepare_vlm_inputs
from .local.template import gen_general_conv
from .misc import gen_modality_conv, get_max_resize, intensity_norm, toss, load_image_byte, get_patch_size_z

CAPTION_PROMPTS = [
    'Briefly describe this {}.',
    'Please provide a brief description of this {}.',
    'Can you provide a brief description of this {}?',
    'Caption this {}.',
    'Please provide a caption for this {}.',
    'Can you provide a caption for this {}?',
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
    'Please write a radiology report for this {}.',
    'Please generate a radiology report for this {}.',
    'Please provide a report for this {}.',
    'Please generate a detailed radiology report for this {}, including a description of the findings and your impression.',
    'Evaluate the {} and generate a detailed report.',
    'Review the {} and provide a thorough clinical report.',
    'Report on this {}.',
    'Detail findings and impression from this {}.',
    'Analyze the {} with findings and impression.',
    'Generate a detailed radiology report for the given {}.',
    "Create a detailed report for this {}.",
    "Give a detailed radiology report for this {}.",
]

FINDINGS_PROMPTS = [
    'What are the findings presented in this {}?',
    'Can you provide the findings for this {}?',
    'Please report on this {} with findings.',
    'Describe this {} with findings.',
    'Please write a report consists of findings for this {}.',
    'Please provide a findings section of the report for this {}.',
    'Can you provide a summary consists of findings of this {}?',
    'Please write findings for this {}.',
    'Can you provide a description consists of findings of this {}?',
    'Please report on this {} with finding.',
    'Analyze this {} and provide a detailed findings section.',
    'Examine the {} and construct a findings section in the report.',
    'Based on your analysis, what would be the findings for this {}?',
    'What are the findings presented in this {}?',
]

PLANE_PROMPTS = [
    'In what plane is this {} oriented?',
    'In what plane is this {} taken?',
    'What imaging plane is depicted here?',
    'In what plane is this {}?',
    'What plane is this?',
    'What is the plane of this {}?',
    'What plane is the {} acquired in?',
    'What plane is this {} in?',
    'What is the scanning plane of this {}?',
    'Which plane is the {} shown in?',
]

REFERRINGS = ['image', 'medical image', 'radiograph', 'scan', 'radiology image', 'radiology scan', 'medical scan']

_REPORT_DATASETS = {'MIMIC-CXR', 'CT-RATE', 'OpenI'}
_CAPTION_DATASETS = {'ROCOv2'}
def get_vl_data_list(name: str, split: Split) -> list:
    dataset_dir = PROCESSED_VL_DATA_ROOT / name
    if name in _REPORT_DATASETS or name in _CAPTION_DATASETS:
        # load cleaned data for report dataset
        split_filename = f'{split}-processed.json'
    else:
        split_filename = f'{split}.json'
    data = orjson.loads((dataset_dir / split_filename).read_bytes())
    for item in data:
        item['dataset'] = name
    return data

@dataclass
class VLTransConf:
    max_tokens: int
    max_tokens_z: int
    log2_patch_size_z_std: float = 0.25
    ac_ratio: float = 0.2
    modality_prob: float = 0.2
    plane_prob: float = 0.2
    report_ratio: float = 0.8

class VLDataPoint(TypedDict):
    image: list[str]
    modality: list[str]
    findings: str
    impression: str
    anomaly_pos: list[str]
    anomaly_neg: list[str]
    vqa: list


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
        self.target_tax = get_target_tax()

    def __call__(self, data: dict):
        conf = self.conf
        trans_conf = conf.vl_trans
        dataset: str = data['dataset']
        # 1. sample image
        image_candidates = np.arange(len(data['image']))
        allow_report = True
        if dataset == 'MIMIC-CXR':
            # only use frontal view of MIMIC-CXR for report generation
            frontal_mask = np.array([plane in {'PA', 'AP'} for plane in data['plane']])
            if frontal_mask.all() or (frontal_mask.any() and toss(self.R, 0.9)):
                image_candidates = image_candidates[frontal_mask]
            else:
                image_candidates = image_candidates[~frontal_mask]
                allow_report = False
        image_idx = self.R.choice(image_candidates).item()
        image_path = data['image'][image_idx]
        if modalities := data.get('modality'):
            modality = modalities[image_idx]
        else:
            modality = None
        if planes := data.get('plane'):
            plane = planes[image_idx]
        else:
            plane = None
        # 2. image transform
        image = load_image_byte(image_path)
        patch_size_z, pool_size_z, stride_z, tokens_z = get_patch_size_z(
            conf.base_vit_patch_size_z,
            conf.base_pool_size_z,
            size_z := image.shape[1],
            trans_conf.max_tokens_z,
            trans_conf.log2_patch_size_z_std,
            self.R,
        )
        patch_size = (patch_size_z, conf.vit_patch_size_xy, conf.vit_patch_size_xy)
        stride = (stride_z, conf.stride_xy, conf.stride_xy)
        resize_shape = (
            min(size_z, tokens_z * stride_z),  # do not resize z if unnecessary
            *get_max_resize(
                image.shape[2:],
                conf.stride_xy,
                trans_conf.max_tokens // tokens_z,
            ),
        )
        if resize_shape != image.shape[1:]:
            resize = mt.Resize(resize_shape, mode=InterpolateMode.TRILINEAR, anti_aliasing=True)
            image = resize(image)
        image = mt.DivisiblePad(stride)(image)
        image = convert_to_tensor(image)
        image, _ = ensure_rgb(image, contiguous=True)
        image = intensity_norm(image)

        # 3. generate conversation
        referring: str = self.R.choice(REFERRINGS)
        conversation = []
        caption: str | None = data.get('processed_caption')
        report: str | None = data.get('processed_report') if allow_report else None
        vqa: list[dict] | None = data.get('vqa')
        _force = not caption and not report and not vqa
        if _force:
            assert modality or plane
        if modality and (_force or toss(self.R, trans_conf.modality_prob)):
            conversation.extend(gen_modality_conv(modality, self.R))
        if plane and (_force or toss(self.R, trans_conf.plane_prob)):
            _template: str = self.R.choice(PLANE_PROMPTS)
            conversation.append(
                ConvTurn(_template.format(referring), plane)
            )
        self.R.shuffle(conversation)
        if caption:
            conversation.append(
                ConvTurn(self.R.choice(CAPTION_PROMPTS).format(referring), caption),
            )
        elif report and (not vqa or toss(self.R, trans_conf.report_ratio)):
            if (
                (anomaly_pos := data.get('anomaly_pos')) is not None and
                (anomaly_neg := data.get('anomaly_neg')) is not None and
                (len(anomaly_pos) > 0 or len(anomaly_neg) > 0) and
                toss(self.R, trans_conf.ac_ratio)
            ):
                ac_conv, _ = gen_general_conv(
                    anomaly_pos,
                    anomaly_neg,
                    False,
                    False,
                    self.tokenizer,
                    self.target_tax,
                    self.R,
                )
                conversation.extend(ac_conv)
            else:
                conversation.append(
                    ConvTurn(self.R.choice(REPORT_PROMPTS).format(referring), report),
                )
        elif vqa:
            conv_vqa = [ConvTurn(qa['question'], qa['answer']) for qa in vqa]
            self.R.shuffle(conv_vqa)
            conversation.extend(conv_vqa)

        vlm_inputs, conversation_text = prepare_vlm_inputs(
            conversation,
            self.tokenizer,
            (np.array(image.shape[1:]) // stride).prod().item(),
            inference=self.inference,
            grounding=False,
            max_seq_len=conf.max_seq_len,
            bop_weight=1.,
        )
        data = {
            'src': (dataset, str(image_path)),
            'image': image,
            'grounding_image': torch.zeros(3, *patch_size),
            'patch_size': patch_size,
            'pool_size': (pool_size_z, conf.pool_size_xy, conf.pool_size_xy),
            'vlm_inputs': vlm_inputs,
            'masks': None,
            'boxes': None,
            'index_offsets': None,
            'instance_mask': False,
        }
        return data
