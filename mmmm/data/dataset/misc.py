import numpy as np
import torch

from luolib.types import tuple3_t

from mmmm.data.defs import ConvTurn

PROMPTS = [
    'What is the modality of this image?',
    'What type of imaging modality is used to acquire the given image?',
]

RESPONSES = [
    'The modality of this image is {}.',
]

def gen_modality_conv(modality: str, R: np.random.RandomState) -> list[ConvTurn]:
    return [
        ConvTurn(
            R.choice(PROMPTS),
            R.choice(RESPONSES).format(modality)
        ),
    ]

def toss(R: np.random.RandomState, prob: float):
    return R.uniform() < prob

def intensity_norm(
    image: torch.Tensor,
    mean: tuple3_t[float] = (0.48145466, 0.4578275, 0.40821073),
    std: tuple3_t[float] = (0.26862954, 0.26130258, 0.27577711),
):
    """default mean and std is adopted from CogVLM (, which is from CLIP)"""
    mean = image.new_tensor(mean)
    std = image.new_tensor(std)
    return (image - mean.view(-1, 1, 1, 1)) / std.view(-1, 1, 1, 1)
