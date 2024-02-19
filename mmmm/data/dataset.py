import torch

from model.cogvlm import conversation as conversation_lib




def custom_collate_fn(batch, tokenizer=None, inference=False):
    # Initializing lists and counters
    global_enc_image_list, grounding_enc_image_list, conversation_list, masks_list  = [], [], [], []
    token_type_ids_list, label_list, inferences, resize_list, questions_list = [], [], [], [], []

    # Iterating through the batch
    for (global_enc_image, grounding_enc_image, conversations, masks, token_type_ids, label, resize, questions) in batch:
        length = len(conversations)
        global_enc_image_list.append(global_enc_image.expand(length, -1, -1, -1, -1))
        grounding_enc_image_list.append(grounding_enc_image.expand(length, -1, -1, -1, -1))
        conversation_list.extend(conversations)
        token_type_ids_list.append(token_type_ids)
        masks_list.append([] if masks is None else masks.float())
        label_list.append(label)
        resize_list.append(resize)
        questions_list.append(questions)
        inferences.append(inference)

    input_ids = torch.nn.utils.rnn.pad_sequence(conversation_list, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    # Preparing targets and handling conversation types
    conv = conversation_lib.default_conversation.copy()
    targets = input_ids.clone()
    sep = conv.sep + conv.roles[1] + ": "
    sep2 = conv.sep2

    for conversation, target in zip(conversation_list, targets):
        _process_conversation(conversation, target, tokenizer, sep, sep2)

    # Adjusting for inferences
    if not inferences[0]:
        truncate_len = tokenizer.model_max_length
        if input_ids.shape[1] > truncate_len:
            input_ids, targets, attention_masks = map(
                lambda x: x[:, :truncate_len], [input_ids, targets, attention_masks]
                )

    return {
        "global_enc_images": torch.stack(global_enc_image_list, dim=0),
        "grounding_enc_images": torch.stack(grounding_enc_image_list, dim=0),
        "input_ids": input_ids,
        "labels": targets,
        "attention_masks": attention_masks,
        "masks_list": masks_list,
        "label_list": label_list,
        "resize_list": resize_list,
        "questions_list": questions_list,
        "inference": inferences[0],
    }


def _process_conversation(conversation, target, tokenizer, sep, sep2):
    total_len = target.ne(tokenizer.pad_token_id).sum().item()
    rounds = conversation.split(sep2)
    cur_len = 1
    target[:cur_len] = IGNORE_INDEX

    for rou in rounds:
        if not rou:
            break

        parts = rou.split(sep)
        assert len(parts) == 2, (len(parts), rou)
        parts[0] += sep

        round_len = len(tokenizer(rou).input_ids)
        instruction_len = len(tokenizer(parts[0]).input_ids) - 2

        target[cur_len: cur_len + instruction_len] = IGNORE_INDEX
        cur_len += round_len

    target[cur_len:] = IGNORE_INDEX
    if cur_len < tokenizer.model_max_length:
        assert cur_len == total_len