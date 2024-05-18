from monai import transforms as mt
from PIL import Image
import torch
import torchvision.transforms as transforms
from tqdm import tqdm

from luolib.utils.zstd import load_pt_zst
from scripts.evaluate.utils import dump_results


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
        transform = transforms.ToPILImage()
        image = load_pt_zst(batch[0]['image'])
        image = transform(image.squeeze(1))
    else:
        image = Image.open(batch[0]['image']).convert('RGB')
    return {
        'image': image,
        'question': batch[0]['question'],
        'answer': batch[0]['answer'],
    }


class LlavaNextTransform(mt.RandomizableTransform):
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, data: dict):
        if data['image'].endswith('.pt.zst'):
            transform = transforms.ToPILImage()
            image = load_pt_zst(data['image'])
            image = transform(image.squeeze(1))
        else:
            image = Image.open(data['image']).convert('RGB')

        prompt = 'A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human\'s questions. USER: <image>\n' + data['question'] + ' ASSISTANT:'

        inputs = self.processor(prompt, image, return_tensors='pt')

        return {
            'inputs': inputs,
            'question': data['question'],
            'answer': data['answer'],
        }


def llavanext_vl_evaluate(model, processor, dataloader, output):
    results = []

    for i, sample in enumerate(tqdm(dataloader)):
        
        with torch.inference_mode():
            prediction = processor.decode(
                model.generate(
                    **sample['inputs'],
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

        if i % 1000 == 0:
            dump_results(results, output)

        print(sample['question'])
        print(sample['answer'])
        print(prediction)

    dump_results(results, output)

    return results