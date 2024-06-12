from PIL import Image
from einops import rearrange, repeat
from monai import transforms as mt
from peft import PeftModel
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torchvision.transforms as transforms
from torchvision.transforms.v2 import functional as tvtf
from tqdm import tqdm
import sys
sys.path.append('scripts/finetune/')

from luolib.types import tuple3_t
from luolib.utils import load_pt_zst
from scripts.evaluate.utils import dump_results


def setup_llavamed(checkpoint: str, adapter: str, tokenizer: str):
    from llava.mm_utils import get_model_name_from_path
    from llava.model.builder import load_pretrained_model

    model_name = get_model_name_from_path(checkpoint)
    tokenizer, model, image_processor, context_len = load_pretrained_model(checkpoint, None, model_name)
    
    if adapter:
        pos_embed = model.model.vision_tower.vision_tower.vision_model.embeddings.position_embedding.weight
        cls_pos_embed, pos_embed = pos_embed[0:1], pos_embed[1:]
        pos_embed = rearrange(pos_embed, '(h w) c -> 1 c h w', h=24, w=24)
        pos_embed = nnf.interpolate(pos_embed, (16, 16), mode='area')
        pos_embed = torch.cat([cls_pos_embed, rearrange(pos_embed, '1 c h w ->(h w) c')])
        model.model.vision_tower.vision_tower.vision_model.embeddings.position_embedding = nn.Embedding(
            *pos_embed.shape[:2], _weight=pos_embed,
        )
        model.model.vision_tower.vision_tower.vision_model.embeddings.position_ids = torch.arange(257).expand((1, -1))
        model = PeftModel.from_pretrained(model, adapter)

    model.to('cuda')
    model.eval()

    return model, tokenizer, image_processor, context_len


class LlavaMedTransform(mt.RandomizableTransform):
    def __init__(self, config, tokenizer, processor, image_token_len, task, setting):
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.image_token_len = image_token_len
        self.task = task
        self.setting = setting

    def __call__(self, data: dict):
        from llava.mm_utils import process_images
        from llava.constants import DEFAULT_IMAGE_TOKEN

        if data['image'].endswith('.pt.zst'):
            transform = transforms.ToPILImage()
            image = load_pt_zst(data['image'])
            image = repeat(image, '1 1 h w -> 3 1 h w')
            image = transform(image.squeeze(1))
        else:
            image = Image.open(data['image']).convert('RGB')

        if self.setting == 'finetuned':
            def intensity_norm_(
                image: torch.Tensor,
                mean: tuple3_t[float] = (0.48145466, 0.4578275, 0.40821073),
                std: tuple3_t[float] = (0.26862954, 0.26130258, 0.27577711),
            ):
                """default mean and std is adopted from CogVLM (, which is from CLIP)"""
                mean = image.new_tensor(mean)
                std = image.new_tensor(std)
                x = image.view(image.shape[0], -1)
                x.sub_(mean[:, None]).div_(std[:, None])
            image = tvtf.to_image(image)
            image = tvtf.to_dtype(image, torch.float32, scale=True)
            image = tvtf.resize(image, (224, 224))
            intensity_norm_(image)
            vision = image.unsqueeze(0).half()
        else:
            vision = process_images([image], self.processor, self.config)[0]
            vision = repeat(vision, 'c h w -> 1 c h w').half()

        if self.task == 'vqa':
            prompt = f'{DEFAULT_IMAGE_TOKEN}\nQuestion: {data["question"]} Answer:'
        elif self.task == 'report':
            prompt = f'{DEFAULT_IMAGE_TOKEN}\nPlease write a radiology report for me:'
        language = torch.cat([
            torch.tensor([self.tokenizer.bos_token_id]).unsqueeze(1),
            torch.tensor(self.tokenizer.encode(prompt, add_special_tokens=False)).unsqueeze(1),
        ])
        language = rearrange(language, 'n l -> l n')

        return {
            'vision': vision,
            'language': language,
            'question': data['question'],
            'answer': data['answer'],
        }


def llavamed_vl_evaluate(model, tokenizer, processor, image_token_len, dataloader, output):
    results = []

    for i, sample in enumerate(tqdm(dataloader)):

        with torch.inference_mode():
            prediction = tokenizer.decode(
                model.generate(
                    sample['language'].to('cuda'),
                    images=sample['vision'].to('cuda'),
                    max_new_tokens=256,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id
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