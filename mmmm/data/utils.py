import torch

from mmmm.tokenizer import MMMMTokenizer
from mmmm.models.cogvlm import LANGUAGE_TOKEN_TYPE, VISION_TOKEN_TYPE
from .defs import CE_IGNORE_INDEX, ConvTurn

def get_text_position_ids(text_ids: torch.Tensor, tokenizer: MMMMTokenizer, start: int):
    ret = torch.empty_like(text_ids)
    ret[0] = start
    for i in range(1, ret.shape[0]):
        # when calling this function, *onp should already be converted to *op
        if text_ids[i - 1] == tokenizer.bop_token_id or text_ids[i] == tokenizer.eop_token_id:
            ret[i] = ret[i - 1]
        else:
            ret[i] = ret[i - 1] + 1
    return ret

def convert_np_input_ids(
    token_ids: torch.Tensor, bonp_mask: torch.BoolTensor, eonp_mask: torch.BoolTensor, tokenizer: MMMMTokenizer,
):
    ret = token_ids.clone()
    ret[bonp_mask] = tokenizer.bop_token_id
    ret[eonp_mask] = tokenizer.eop_token_id
    return ret

def prepare_vlm_inputs(
    conversation: list[ConvTurn],
    tokenizer: MMMMTokenizer,
    num_image_tokens: int,
    *,
    inference: bool,
    grounding: bool,
    max_seq_len: int | None = None,
):
    """
    Args:
        num_image_tokens: the number of tokens corresponding to image patches (does not include special tokens)
    """
    # TODO: refactor this function to support various VLM formats
    # user_start = 'Question:'
    # sys_start = 'Answer:'
    user_start = tokenizer.usr_token
    sys_start = tokenizer.sys_token
    # just for viewing, don't tokenize it directly
    text = '\n'.join(
        f'{user_start} {query}\n{sys_start} {answer}'
        for query, answer in conversation
    )
    dtype = torch.long
    text_ids = []
    # the last response is empty iff inference
    assert inference == (conversation[-1].response == '')
    lm_targets = None if inference else []
    for i, (query, answer) in enumerate(conversation):
        prompt = f'{user_start} {query} {sys_start}'
        prompt_ids = torch.tensor(tokenizer.encode(prompt, add_special_tokens=False))

        if answer == '':
            assert i == len(conversation) - 1 and inference
            text_ids.append(prompt_ids)
        else:
            answer_ids = torch.tensor(tokenizer.encode(answer, add_special_tokens=False))
            bonp_mask = answer_ids == tokenizer.bonp_token_id
            eonp_mask = answer_ids == tokenizer.eonp_token_id
            text_ids.append(
                torch.cat([
                    prompt_ids,
                    convert_np_input_ids(answer_ids, bonp_mask, eonp_mask, tokenizer),
                ]),
            )
            if not inference:
                lm_targets.append(
                    torch.cat([
                        torch.full((prompt_ids.shape[0] - 1, ), CE_IGNORE_INDEX),
                        answer_ids.masked_fill(bonp_mask | eonp_mask, CE_IGNORE_INDEX),
                        torch.tensor([tokenizer.eos_token_id]),
                    ]),
                )
    text_ids = torch.cat(text_ids)
    if not inference:
        lm_targets = torch.cat(lm_targets)
    num_image_tokens += 2  # to include boi and eoi
    input_ids = torch.cat([
        torch.tensor([tokenizer.bos_token_id]),
        torch.full((num_image_tokens, ), 0),
        torch.tensor([tokenizer.grd_token_id if grounding else tokenizer.ngrd_token_id]),
        text_ids,
    ])
    image_features_mask = torch.zeros(input_ids.shape[0], dtype=torch.bool)
    image_features_mask[1:1 + num_image_tokens] = True
    token_type_ids = torch.cat([
        torch.tensor([LANGUAGE_TOKEN_TYPE]),
        torch.full((num_image_tokens, ), VISION_TOKEN_TYPE),
        # all new tokens will be processed by VE
        # torch.where(text_ids < tokenizer.base_vocab_size, LANGUAGE_TOKEN_TYPE, VISION_TOKEN_TYPE),
        torch.full((1 + text_ids.shape[0], ), LANGUAGE_TOKEN_TYPE),
    ])

    position_ids = torch.cat([
        torch.tensor([0, 1]),  # bos, boi
        torch.full((num_image_tokens - 2,), 2),  # image patches
        torch.tensor([3, 4]),  # eoi, grounding
        get_text_position_ids(text_ids, tokenizer, start=5)
    ])
    attention_mask = torch.ones(input_ids.shape, dtype=dtype)
    if not inference:
        lm_targets = torch.cat([
            # bos, (boi, *image, eoi), grounding
            torch.full((1 + num_image_tokens + 1, ), CE_IGNORE_INDEX),
            lm_targets,
        ])
    inputs = {
        'input_ids': input_ids,
        'image_features_mask': image_features_mask,
        'token_type_ids': token_type_ids,
        'position_ids': position_ids,
        'attention_mask': attention_mask,
    }
    if not inference:
        inputs['lm_targets'] = lm_targets
    if max_seq_len is not None:
        for k, v in inputs.items():
            inputs[k] = v[:max_seq_len]
    return inputs, text
