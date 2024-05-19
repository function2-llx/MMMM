import torch
from monai import transforms as mt
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
from transformers import AutoModelForCausalLM, LlamaTokenizer

from luolib.utils.zstd import load_pt_zst
from scripts.evaluate.utils import dump_results


def setup_cogvlm(checkpoint: str, tokenizer: str):
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


class CogVLMTransform(mt.RandomizableTransform):
    def __init__(self, build_conversation_input_ids, tokenizer):
        self.build_conversation_input_ids = build_conversation_input_ids
        self.tokenizer = tokenizer

    def __call__(self, data: dict):
        if data['image'].endswith('.pt.zst'):
            transform = transforms.ToPILImage()
            image = load_pt_zst(data['image'])
            image = transform(image.squeeze(1))
        else:
            image = Image.open(data['image']).convert('RGB')

        inputs = self.build_conversation_input_ids(
            self.tokenizer, query=data['question'], images=[image]
        )

        return {
            'image': image,
            'question': data['question'],
            'answer': data['answer'],
            'input_ids': inputs['input_ids'],
            'token_type_ids': inputs['token_type_ids'],
            'images': inputs['images'],
        }


def cogvlm_vl_evaluate(model, tokenizer, dataloader, output):
    results = []

    for i, sample in enumerate(tqdm(dataloader)):
        with torch.inference_mode():
            prediction = (
                tokenizer.decode(
                    model.generate(
                        input_ids=sample['input_ids'].unsqueeze(0).to('cuda'),
                        token_type_ids=sample['token_type_ids'].unsqueeze(0).to('cuda'),
                        attention_mask=sample['attention_mask'].unsqueeze(0).to('cuda'),
                        images=[[sample['images'][0].to('cuda').to(torch.bfloat16)]],
                        max_new_tokens=256,
                    )[0],
                    skip_special_tokens=True,
                )
                .split('Answer: ')[1]
                .strip()
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