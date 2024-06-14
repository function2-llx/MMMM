from monai import transforms as mt
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

from luolib.utils import load_pt_zst
from scripts.evaluate.utils import dump_results


def setup_instructblip(checkpoint: str, tokenizer: str):
    from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

    model = InstructBlipForConditionalGeneration.from_pretrained(
        checkpoint,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    model = model.to('cuda')
    model.eval()

    processor = InstructBlipProcessor.from_pretrained(tokenizer)

    return model, processor


class InstructBlipTransform(mt.RandomizableTransform):
    def __init__(self, processor, setting):
        self.processor = processor
        self.setting = setting

    def __call__(self, data: dict):
        if data['image'].endswith('.pt.zst'):
            transform = transforms.ToPILImage()
            image = load_pt_zst(data['image'])
            image = transform(image.squeeze(1))
        else:
            image = Image.open(data['image']).convert('RGB')

        inputs = self.processor(
            images=image,
            text='Question: ' + data['question'] + ' Answer: ' if 'finetuned' in self.setting else data['question'],
            return_tensors='pt',
        )

        return {
            'inputs': inputs,
            'question': data['question'],
            'answer': data['answer'],
        }

        

def instructblip_vl_evaluate(model, processor, dataloader, output):
    results = []

    for i, sample in enumerate(tqdm(dataloader)):
        with torch.inference_mode():
            prediction = processor.decode(
                model.generate(
                    **sample['inputs'].to('cuda'),
                    max_new_tokens=256,
                    do_sample=False,
                )[0],
                skip_special_tokens=True,
            )

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