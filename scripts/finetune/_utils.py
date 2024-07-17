from typing import Mapping

import torch
from torch.nn.utils.rnn import pad_sequence

from luolib.types import tuple3_t

def intensity_norm_(
    image: torch.Tensor,
    mean: tuple3_t[float] = (0.48145466, 0.4578275, 0.40821073),
    std: tuple3_t[float] = (0.26862954, 0.26130258, 0.27577711),
):
    """default mean and std is adopted from CogVLM (, which is from CLIP)"""
    mean = image.new_tensor(mean)
    std = image.new_tensor(std)
    x = image.view(image.shape[0], -1)
    x.sub_(mean[:, None]).div_(std[:, None])

def _pad_inputs(inputs: list[Mapping], pad_token_id: int):
    pad_value = {
        'input_ids': pad_token_id,
        'labels': CE_IGNORE_INDEX,
    }
    return {
        key: pad_sequence(
            [x[key] for x in inputs],
            batch_first=True,
            padding_value=pad_value.get(key, 0),
        )
        for key in inputs[0].keys()
    }


CE_IGNORE_INDEX = -100
