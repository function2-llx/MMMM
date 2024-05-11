from PIL import Image
import torch
from tqdm import tqdm

from luolib.utils import load_pt_zst


def setup_llavanext(checkpoint: str, tokenizer: str):
    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

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

    if batch[0]['image'].endswith('.pt.zst'):
        image = load_pt_zst(batch[0]['image'])
    else:
        image = Image.open(batch[0]['image']).convert('RGB')
    return {
        'image': image,
        'question': batch[0]['question'],
        'answer': batch[0]['answer'],
    }


def llavanext_vl_evaluate(model, processor, dataloader):
    results = []

    for sample in tqdm(dataloader):
        prompt = 'A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human\'s questions. USER: <image>\n' + sample['question'] + ' ASSISTANT:'

        inputs = processor(prompt, sample['image'], return_tensors='pt').to('cuda')

        with torch.inference_mode():
            prediction = processor.decode(
                model.generate(
                    **inputs,
                    max_new_tokens=256,
                )[0],
                skip_special_tokens=True,
            ).split('ASSISTANT:')[1].strip()

        results.append(
            {
                'question': sample['question'],
                'answer': sample['answer'],
                'prediction': prediction,
            },
        )

        print(sample['question'])
        print(sample['answer'])
        print(prediction)

    return results