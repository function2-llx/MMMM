import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

from luolib.utils import load_pt_zst


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


def instructblip_collate_fn(batch: list[dict]):
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


def instructblip_vl_evaluate(model, processor, dataloader):
    results = []

    for sample in tqdm(dataloader):
        inputs = processor(images=sample['image'], text=sample['question'], return_tensors='pt').to('cuda')

        with torch.inference_mode():
            prediction = processor.decode(
                model.generate(
                    **inputs,
                    max_new_tokens=256,
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

        print(sample['question'])
        print(sample['answer'])
        print(prediction)

    return results