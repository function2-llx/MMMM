from einops import repeat, reduce
from PIL import Image
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torchvision.transforms as transforms

from luolib.utils import load_pt_zst


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


def m3d_collate_fn(batch: list[dict]):
    assert len(batch) == 1

    if batch[0]['image'].endswith('.pt'):
        image = torch.load(batch[0]['image']).float()
        image = (image - image.min()) / (image.max() - image.min())
        image = reduce(image, 'c d h w -> d h w', 'mean')
        image = repeat(image, 'd h w -> 1 1 d h w')
    else:
        if batch[0]['image'].endswith('.pt.zst'):
            transform = transforms.ToPILImage()
            image = load_pt_zst(batch[0]['image'])
            image = transform(image.squeeze(1))
        else:
            image = Image.open(batch[0]['image']).convert('RGB')
        transform = transforms.ToTensor()
        image = transform(image)
        image = reduce(image, 'c h w -> h w', 'mean')
        image = repeat(image, 'h w -> 1 1 1 h w')

    image = torch.nn.functional.interpolate(image, size=(32, 256, 256))

    return {
        'image': image,
        'question': batch[0]['question'],
        'answer': batch[0]['answer'],
    }


def m3d_vl_evaluate(model, tokenizer, dataloader):
    results = []

    for sample in tqdm(dataloader):
        language = tokenizer(
            '<im_patch>' * 256 + ' ' + sample['question'],
            return_tensors='pt',
        )['input_ids'].to('cuda')
        vision = sample['image'].to(device='cuda', dtype=torch.bfloat16)
        
        with torch.inference_mode():
            prediction = tokenizer.decode(
                model.generate(vision, language, max_new_tokens=256)[0],
                skip_special_tokens=True,
            ).strip()

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