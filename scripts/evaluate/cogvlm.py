import torch
from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer


def cogvlm_setup(checkpoint: str, tokenizer: str):
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    tokenizer = LlamaTokenizer.from_pretrained(tokenizer)

    model = model.to('cuda')
    model = model.eval()

    return model, tokenizer

def cogvlm_collate_fn(batch: list[dict]):
    assert len(batch) == 1

    return {
        'image': Image.open(batch[0]['image']).convert('RGB'),
        'question': batch[0]['question'],
        'answer': batch[0]['answer'],
    }
