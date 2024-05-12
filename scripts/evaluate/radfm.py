from einops import rearrange, repeat
from PIL import Image
import sys
import torch
from torchvision import transforms
from tqdm import tqdm
from transformers import LlamaTokenizer

from luolib.utils import load_pt_zst

def setup_radfm(checkpoint: str, tokenizer: str):
    sys.path.append('third-party/RadFM/Quick_demo/Model')
    from RadFM.multimodality_model import MultiLLaMAForCausalLM

    model = MultiLLaMAForCausalLM(
        lang_model_path=tokenizer,
    )
    checkpoint = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint)
    model = model.to('cuda')
    model.eval()

    tokenizer = LlamaTokenizer.from_pretrained(tokenizer)
    special_tokens = {
        'additional_special_tokens': [f'<image{i}>' for i in range(32)] + ['<image>', '</image>']
    }
    tokenizer.add_special_tokens(special_tokens)
    tokenizer.pad_token_id = 0
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2

    return model, tokenizer


def radfm_collate_fn(batch: list[dict]):
    assert len(batch) == 1

    if batch[0]['image'].endswith('.pt'):
        image = rearrange(torch.load(batch[0]['image']).float(), 'c d h w -> c h w d')
        image = (image - image.min()) / (image.max() - image.min())
    else:
        if batch[0]['image'].endswith('.pt.zst'):
            image = load_pt_zst(batch[0]['image'])
            image = image.float() / 255.0
        else:
            transform = transforms.Compose([transforms.ToTensor()])
            image = Image.open(batch[0]['image']).convert('RGB')
            image = transform(image)

    target_d, max_d = 4, 4
    if len(image.shape) == 4:
        max_d = max(image.shape[3], max_d)
    for temp_d in range(4, 65, 4):
        if abs(temp_d - max_d) < abs(target_d - max_d):
            target_d = temp_d
    if len(image.shape) == 3:
        image = torch.nn.functional.interpolate(
            repeat(image, 'c h w -> 1 c h w 1'), size=(512, 512, target_d)
        ).unsqueeze(0)
    else:
        if image.shape[0] == 1:
            image = torch.nn.functional.interpolate(
                repeat(image, '1 h w d -> 1 3 h w d'), size=(512, 512, target_d)
            ).unsqueeze(0)
        else:
            image = torch.nn.functional.interpolate(
                repeat(image, 'c h w d -> 1 c h w d'), size=(512, 512, target_d)
            ).unsqueeze(0)

    return {
        'image': image,
        'question': batch[0]['question'],
        'answer': batch[0]['answer'],
    }


def radfm_vl_evaluate(model, tokenizer, dataloader):
    results = []

    for sample in tqdm(dataloader):
        question = '<image>' + ''.join([f'<image{i}>' for i in range(32)]) + '</image>' + sample['question']
        language = tokenizer(
            question,
            return_tensors='pt',
        )['input_ids'].to('cuda')
        vision = sample['image'].to('cuda')

        with torch.inference_mode():
            prediction = tokenizer.decode(
                model.generate(language, vision)[0], skip_special_tokens=True
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