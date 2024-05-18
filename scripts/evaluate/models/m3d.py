from einops import repeat, reduce
from monai import transforms as mt
from PIL import Image
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torchvision.transforms as transforms

from luolib.utils.zstd import load_pt_zst
from scripts.evaluate.utils import dump_results


def setup_m3d(checkpoint: str, tokenizer: str):
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer,
        model_max_length=512,
        padding_side='right',
        use_fast=False,
        trust_remote_code=True,
    )

    model = model.to('cuda')
    model = model.eval()

    return model, tokenizer


class M3DTransform(mt.RandomizableTransform):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, data: dict):
        if data['image'].endswith('.pt'):
            image = torch.load(data['image']).float()
            image = (image - image.min()) / (image.max() - image.min())
        elif data['image'].endswith('.pt.zst'):
            image = load_pt_zst(data['image']).float()
            image = (image - image.min()) / (image.max() - image.min())
        else:
            transform = transforms.ToTensor()
            image = Image.open(data['image']).convert('RGB')
            image = transform(image)
            image = repeat(image, 'c h w -> c 1 h w')

        image = reduce(image, 'c d h w -> d h w', 'mean')
        image = repeat(image, 'd h w -> 1 1 d h w')

        image = torch.nn.functional.interpolate(image, size=(32, 256, 256))

        language = self.tokenizer(
            '<im_patch>' * 256 + ' ' + data['question'],
            return_tensors='pt',
        )['input_ids']

        return {
            'vision': image,
            'language': language,
            'question': data['question'],
            'answer': data['answer'],
        }


def m3d_vl_evaluate(model, tokenizer, dataloader, output):
    results = []

    for i, sample in enumerate(tqdm(dataloader)):
        
        with torch.inference_mode():
            prediction = tokenizer.decode(
                model.generate(
                    sample['vision'].to(device='cuda', dtype=torch.bfloat16),
                    sample['language'].to('cuda'),
                    max_new_tokens=256
                )[0],
                skip_special_tokens=True,
            ).strip()

        results.append(
            {
                'question': sample['question'],
                'answer': sample['answer'],
                'prediction': prediction,
            },
        )

        if i % 1000 == 0:
            dump_results(results, output)
            results = []

        print(sample['question'])
        print(sample['answer'])
        print(prediction)

    dump_results(results, output)