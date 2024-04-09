from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

from mashumaro import pass_through
from mashumaro.config import BaseConfig
from mashumaro.mixins.orjson import DataClassORJSONMixin
import nibabel as nib
import numpy as np
import numpy.typing as npt
import orjson
import pandas as pd
import torch

from luolib.types import PathLike, tuple3_t
from luolib.utils import load_pt_zst

DATA_ROOT = Path('data')
ORIGIN_DATA_ROOT = DATA_ROOT / 'origin'
PROCESSED_DATA_ROOT = DATA_ROOT / 'processed'
ORIGIN_SEG_DATA_ROOT = ORIGIN_DATA_ROOT / 'image'
ORIGIN_VL_DATA_ROOT = ORIGIN_DATA_ROOT / 'vision-language'
PROCESSED_SEG_DATA_ROOT = PROCESSED_DATA_ROOT / 'image'
PROCESSED_VL_DATA_ROOT = PROCESSED_DATA_ROOT / 'vision-language'
PROCESSED_VG_DATA_ROOT = PROCESSED_DATA_ROOT / 'visual-grounding'

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

def _numpy_field(dtype: np.dtype):
    return field(metadata={'serialize': pass_through, 'deserialize': partial(np.array, dtype=dtype)})

@dataclass
class Sparse(DataClassORJSONMixin):
    """
    Attributes:
        modalities: all images of different modalities must be co-registered
        mean: mean intensity for each modality
        normalized: whether the images are normalized during pre-processing
        anatomy: information for generating general conversation related to anatomy targets
        anomaly: information for generating general conversation related to anomaly targets
    """
    spacing: npt.NDArray[np.float64] = _numpy_field(np.float64)
    shape: npt.NDArray[np.int16] = _numpy_field(np.int16)
    modalities: list[str]
    mean: npt.NDArray[np.float32] = _numpy_field(np.float32)
    std: npt.NDArray[np.float32] = _numpy_field(np.float32)
    normalized: bool

    @dataclass
    class Anatomy:
        """
        Attributes:
            pos: anatomical structures that are assured to be observable in the image
            neg: anatomical structures that are assured to be unobservable in the image
        """
        pos: list[str]
        neg: list[str]
    anatomy: Anatomy

    @dataclass
    class Anomaly:
        """
        Attributes:
            pos: anomalies that are assured to be observable in the image
            neg: anomalies that are assured to be unobservable in the image
            complete: indicating that `pos` covers all anomalies in the image
        """
        pos: list[str]
        neg: list[str]
        complete: bool
    anomaly: Anomaly

    @dataclass
    class BBox:
        center: npt.NDArray[np.float64] = _numpy_field(np.float64)
        size: npt.NDArray[np.float64] = _numpy_field(np.float64)

    @dataclass
    class Annotation:
        """
        Attributes:
            mask: list of (name, mask size), where the order corresponds to the channel dimension of the mask
                file, and names may repeat for multiple anomalies with the same name
            bbox: list of (target name, 3D bounding box coordinates), coordinates range: [0, shape - 1]
        """
        mask: list[tuple[str, int]]
        bbox: list[tuple[str, Sparse.BBox]]
    annotation: Annotation

    extra: Any = None

    class Config(BaseConfig):
        orjson_options = orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_INDENT_2

def encode_patch_size(patch_size: tuple3_t[int]):
    return ','.join(map(str, patch_size))

def convert_to_slicer(data_dir: PathLike, output_dir: PathLike | None = None, multiclass: bool = True):
    """convert the processed data by MMMM to the format readable by Slicer"""
    data_dir = Path(data_dir)
    output_dir = data_dir / 'slicer' if output_dir is None else Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    img = torch.load(data_dir / 'images.pt')
    sparse: Sparse = pd.read_pickle(data_dir / 'sparse.pkl')
    for i, modality in enumerate(sparse.modalities):
        nib.save(
            nib.Nifti1Image(img[i].float().numpy(), np.diag([*sparse.spacing, 1])),
            output_dir / f'{modality}.nii.gz',
        )
    masks: torch.BoolTensor = load_pt_zst(data_dir / 'masks.pt.zst')
    if multiclass:
        seg = torch.zeros(masks.shape[1:], dtype=torch.int16)
        for c in range(masks.shape[0]):
            seg[masks[c]] = c + 1
    else:
        seg = masks
    nib.save(
        nib.Nifti1Image(seg.numpy(), np.diag([*sparse.spacing, 1])),
        output_dir / 'seg.nii.gz',
    )
