import numpy as np

from mmmm.data.defs import ConvTurn

PROMPTS = [
    'What is the modality of this image?',
    'What type of imaging modality is used to acquire the given image?',
]

RESPONSES = [
    'The modality of this image is {}.',
]

def gen_modality_conversation(modality: str, R: np.random.RandomState) -> list[ConvTurn]:
    return [
        ConvTurn(
            R.choice(PROMPTS),
            R.choice(RESPONSES).format(modality)
        )
    ]

def toss(R: np.random.RandomState, prob: float):
    return R.uniform() < prob
