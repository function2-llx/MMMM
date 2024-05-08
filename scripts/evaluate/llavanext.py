from PIL import Image
import torch
from tqdm import tqdm


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

    return {
        'image': Image.open(batch[0]['image']).convert('RGB'),
        'question': batch[0]['question'],
        'answer': batch[0]['answer'],
    }


def llavanext_vl_evaluate(model, processor, dataloader, metrics):
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
                **metrics.compute(prediction, sample['answer']),
            },
        )

        print(sample['question'])
        print(sample['answer'])
        print(prediction)

    return results