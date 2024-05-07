from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
from PIL import Image


def setup_instructblip(checkpoint: str, tokenizer: str):
    model = InstructBlipForConditionalGeneration.from_pretrained(
        checkpoint,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    model = model.to('cuda')
    model.eval()

    processor = InstructBlipProcessor.from_pretrained(tokenizer)

    return model, processor

def instructblip_collate_fn(batch: list[dict]):
    assert len(batch) == 1

    return {
        'image': Image.open(batch[0]['image']).convert('RGB'),
        'question': batch[0]['question'],
        'answer': batch[0]['answer'],
    }
