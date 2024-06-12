from einops import repeat
from lightning.fabric.utilities import move_data_to_device
from monai import transforms as mt
from peft import PeftModel
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.transforms.v2 import functional as tvtf
from tqdm import tqdm

from luolib.types import tuple3_t
from luolib.utils.zstd import load_pt_zst
from scripts.evaluate.utils import dump_results
from scripts.finetune._vqa.llavanext import MyLlavaNextForConditionalGeneration


def build_conversation_input_ids(tokenizer, prompt, image, return_tensors='pt'):
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
    image_size = torch.tensor((224, 224))
    intensity_norm_(image)

    prompt_ids = torch.tensor(tokenizer.encode(prompt, add_special_tokens=False))
    input_ids = torch.cat([
        torch.tensor([tokenizer.bos_token_id]).unsqueeze(1),
        prompt_ids.unsqueeze(1),
    ])

    input_ids = input_ids.reshape(input_ids.shape[-1], input_ids.shape[0])
    inputs = {
        'input_ids': input_ids,
        'pixel_values': repeat(image.unsqueeze(0), 'n ... -> n l2 ...', l2=2),
        'image_sizes': image_size.unsqueeze(0),
        'attention_mask': torch.ones_like(input_ids),
    }
    return inputs


def setup_llavanext(checkpoint: str, adapter: str, tokenizer: str):
    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

    if adapter:
        model = MyLlavaNextForConditionalGeneration.from_pretrained(
            checkpoint,
            image_grid_pinpoints=[[224, 224]],
            vision_config={
                "hidden_size": 1024,
                "image_size": 224,
                "intermediate_size": 4096,
                "model_type": "clip_vision_model",
                "num_attention_heads": 16,
                "num_hidden_layers": 24,
                "patch_size": 14,
                "projection_dim": 768,
                "vocab_size": 32000
            },
        )
        model = PeftModel.from_pretrained(model, adapter)
    else:
        model = LlavaNextForConditionalGeneration.from_pretrained(
            checkpoint, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True
        )

    model = model.to('cuda')
    model.eval()

    processor = LlavaNextProcessor.from_pretrained(tokenizer)

    return model, processor

class LlavaNextTransform(mt.RandomizableTransform):
    def __init__(self, processor, task):
        self.processor = processor
        self.task = task

    def __call__(self, data: dict):
        if data['image'].endswith('.pt.zst'):
            transform = transforms.ToPILImage()
            image = load_pt_zst(data['image'])
            image = repeat(image, '1 1 h w -> 3 1 h w')
            image = transform(image.squeeze(1))
        else:
            image = Image.open(data['image']).convert('RGB')

        if self.task == 'vqa':
            prompt = '<image>\nQuestion: ' + data['question'] + ' Answer:'
        elif self.task == 'report':
            prompt = f'<image>\nPlease write a radiology report for me:'

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
            inputs = move_data_to_device(sample['inputs'], 'cuda')
            prediction = processor.decode(
                model.generate(
                    **inputs,
                    max_new_tokens=256,
                )[0],
                skip_special_tokens=True,
            )
            try:
                prediction = prediction.split('Answer:')[1].strip()
            except:
                prediction = prediction.split('for me:')[1].strip()

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