from PIL import Image
from einops import repeat
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig, CLIPImageProcessor, StoppingCriteria
import sys

from luolib.utils.zstd import load_pt_zst


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
        else:
            outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len:], skip_special_tokens=True)[0]
            for keyword in self.keywords:
                if keyword in outputs:
                    return True
        return False


def setup_llavamed(checkpoint: str, tokenizer: str):
    sys.path.append('third-party/LLaVA-Med')
    from llava import LlavaLlamaForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    patch_dict = {
        'use_mm_proj': True,
        'mm_vision_tower': 'openai/clip-vit-large-patch14',
        'mm_hidden_size': 1024,
    }

    cfg = AutoConfig.from_pretrained(checkpoint)
    if not hasattr(cfg, 'mm_vision_tower'):
        print(
            f'`mm_vision_tower` not found in `{checkpoint}`, applying patch and save to disk.'
        )
        for k, v in patch_dict.items():
            setattr(cfg, k, v)
        cfg.save_pretrained(checkpoint)

    model = LlavaLlamaForCausalLM.from_pretrained(
        checkpoint, torch_dtype=torch.float16, use_cache=True
    )
    model = model.to('cuda')
    model.eval()

    image_processor = CLIPImageProcessor.from_pretrained(
        model.config.mm_vision_tower, torch_dtype=torch.float16
    )
    vision_tower = model.model.vision_tower[0]
    vision_tower.to(device='cuda', dtype=torch.float16)

    mm_use_im_start_end = getattr(model.config, 'mm_use_im_start_end', False)
    tokenizer.add_tokens(['<im_patch>'], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens(['<im_start>', '<im_end>'], special_tokens=True)

    vision_config = vision_tower.config
    vision_config.im_patch_token = tokenizer.convert_tokens_to_ids(['<im_patch>'])[0]
    vision_config.use_im_start_end = mm_use_im_start_end
    if mm_use_im_start_end:
        vision_config.im_start_token, vision_config.im_end_token = (
            tokenizer.convert_tokens_to_ids(['<im_start>', '<im_end>'])
        )
    image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2

    return model, tokenizer, image_processor, image_token_len


def llavamed_collate_fn(task, dataset, setting, model, tokenizer, processor, image_token_len, batch: list[dict]):
    sys.path.append('third-party/LLaVA-Med')
    from llava.conversation import conv_templates
    
    assert len(batch) == 1

    if batch[0]['image'].endswith('.pt.zst'):
        transform = transforms.ToPILImage()
        image = load_pt_zst(batch[0]['image'])
        image = transform(image.squeeze(1))
    else:
        image = Image.open(batch[0]['image'])

    question = batch[0]['question']
    if getattr(model.config, 'mm_use_im_start_end', False):
        question = (
            question
            + '\n'
            + '<im_start>'
            + '<im_patch>' * image_token_len
            + '<im_end>'
        )
    else:
        question = question + '\n' + '<im_patch>' * image_token_len
    conv = conv_templates['simple'].copy()
    conv.append_message(conv.roles[0], question)
    prompt = conv.get_prompt()
    language = torch.as_tensor(tokenizer([prompt]).input_ids).to('cuda')

    vision = processor.preprocess(image, return_tensors='pt')[
        'pixel_values'
    ][0]
    vision = repeat(vision, 'c h w -> 1 c h w').half().to('cuda')
    return {
        'image': image,
        'question': batch[0]['question'],
        'answer': batch[0]['answer'],
        'language': language,
        'vision': vision
    }


def llavamed_vl_evaluate(task, dataset, setting, model, tokenizer, processor, image_token_len, dataloader):
    sys.path.append('third-party/LLaVA-Med')
    from llava.conversation import conv_templates

    conv = conv_templates['simple'].copy()
    
    results = []

    for sample in tqdm(dataloader):
        stopping_criteria = KeywordsStoppingCriteria(['###'], tokenizer, sample['language'])
        with torch.inference_mode():
            prediction = tokenizer.decode(
                model.generate(
                    sample['language'],
                    images=sample['vision'],
                    stopping_criteria=[stopping_criteria],
                    max_new_tokens=256,
                )[0, sample['language'].shape[1] :],
                skip_special_tokens=True,
            )

        try:
            index = prediction.index(conv.sep)
        except ValueError:
            prediction += conv.sep
            index = prediction.index(conv.sep)

        prediction = prediction[:index].split('Assistant: ')[1].strip()

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