import torch

from mmmm.data import CE_IGNORE_INDEX
from mmmm.models import MMMMTokenizer
from mmmm.models.cogvlm import LANGUAGE_TOKEN_TYPE, VISION_TOKEN_TYPE

def prepare_vlm_inputs(
    conversation: list[tuple[str, str]],
    tokenizer: MMMMTokenizer,
    num_image_tokens: int,
    inference: bool = False,
):
    """
    Args:
        num_image_tokens: the number of tokens corresponding to image patches (does not include special tokens)
    """
    # TODO: refactor this function to support various VLM formats
    # template: CogVLM `chat_old_history_to_prompt`
    # just for viewing, don't tokenize it directly
    user_start = 'Question:'
    sys_start = 'Answer:'
    text = '\n'.join(
        f'{user_start} {query} {sys_start} {answer}'
        for query, answer in conversation
    )
    dtype = torch.long
    text_ids = []
    # the last response is empty iff inference
    assert inference ^ (conversation[-1][1] == '')
    if not inference:
        lm_targets = []
    for i, (query, answer) in enumerate(conversation):
        prompt = f'{user_start} {query} {sys_start}'
        prompt_ids = torch.tensor(tokenizer.encode(prompt, add_special_tokens=False))

        if answer == '':
            assert i == len(conversation) - 1 and inference
            text_ids.append(prompt_ids)
        else:
            answer_ids = torch.tensor(tokenizer.encode(answer, add_special_tokens=False))
            text_ids.append(torch.cat([prompt_ids, answer_ids]))
            if not inference:
                lm_targets.append(
                    torch.cat([
                        torch.full((prompt_ids.shape[0] - 1, ), CE_IGNORE_INDEX),
                        answer_ids,
                        torch.tensor([tokenizer.eos_token_id]),
                    ]),
                )
    text_ids = torch.cat(text_ids)
    if not inference:
        lm_targets = torch.cat(lm_targets)

    # text_ids = torch.tensor(tokenizer.encode(text, add_special_tokens=False))
    # TODO: dynamically adjust patch size according to image spacing
    # num_vision_tokens = np.prod([s // ps for s, ps in zip(image_size, vit_patch_size)]).item() + 2  # including boi and eoi
    num_image_tokens += 2  # including boi and eoi
    input_ids = torch.cat([
        torch.tensor([tokenizer.bos_token_id]),
        torch.full((num_image_tokens,), 0),
        text_ids,
    ])
    image_features_mask = torch.zeros(input_ids.shape[0], dtype=torch.bool)
    image_features_mask[1:1 + num_image_tokens] = True
    token_type_ids = torch.cat([
        torch.tensor([LANGUAGE_TOKEN_TYPE]),
        torch.full((num_image_tokens,), VISION_TOKEN_TYPE),
        # all new tokens will be processed by VE
        # torch.where(text_ids < tokenizer.base_vocab_size, LANGUAGE_TOKEN_TYPE, VISION_TOKEN_TYPE),
        torch.full((text_ids.shape[0], ), LANGUAGE_TOKEN_TYPE),
    ])
    position_ids = torch.cat([
        torch.tensor([0, 1]),  # bos and boi
        torch.full((num_image_tokens - 2,), 2),
        torch.tensor([3]),  # eoi
        torch.arange(4, 4 + text_ids.shape[0]),
    ])
    attention_mask = torch.ones(input_ids.shape, dtype=dtype)
    if not inference:
        lm_targets = torch.cat([torch.full((1 + num_image_tokens,), CE_IGNORE_INDEX), lm_targets])
    inputs = {
        'input_ids': input_ids,
        'image_features_mask': image_features_mask,
        'token_type_ids': token_type_ids,
        'position_ids': position_ids,
        'attention_mask': attention_mask,
    }
    if not inference:
        inputs['lm_targets'] = lm_targets
    return inputs, text
