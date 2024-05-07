from PIL import Image
import sys
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration


def setup_llavanext(checkpoint: str, tokenizer: str):
    model = LlavaNextForConditionalGeneration.from_pretrained(
        checkpoint, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True
    )
    model = model.to('cuda')
    model.eval()

    processor = LlavaNextProcessor.from_pretrained(tokenizer)

    return model, processor

def llavanext_collate_fn(batch: list[dict]):
    assert len(batch) == 1

    return {
        'image': Image.open(batch[0]['image']).convert('RGB'),
        'question': batch[0]['question'],
        'answer': batch[0]['answer'],
    }