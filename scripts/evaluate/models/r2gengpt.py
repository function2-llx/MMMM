import argparse
import sys
from einops import repeat
from monai import transforms as mt
from PIL import Image
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
from transformers import AutoImageProcessor

from luolib.utils.zstd import load_pt_zst
from scripts.evaluate.utils import dump_results


def setup_r2gengpt(adapter):
    sys.path.append('third-party/R2GenGPT/')
    from models.R2GenGPT import R2GenGPT

    model = R2GenGPT(
        argparse.Namespace(
            **{
                'vision_model': 'microsoft/swin-base-patch4-window7-224',
                'llama_model': '../Llama-2-7b-chat-hf',
                'freeze_vm': False,
                'llm_use_lora': True,
                'llm_r': 16,
                'llm_alpha': 16,
                'vis_use_lora': False,
                'vis_r': 16,
                'vis_alpha': 16,
                'lora_dropout': 0.1,
                'low_resource': False,
                'delta_file': adapter,
                'global_only': False,
                'end_sym': '</s>',
                'beam_size': 3,
                'do_sample': False,
                'no_repear_ngram_size': 2,
                'num_beam_groups': 1,
                'max_length': 100,
                'min_new_tokens': 80,
                'max_new_tokens': 120,
                'repetition_penalty': 2.0,
                'length_penalty': 2.0,
                'temperature': 0
            }
        )
    ).to(device='cuda', dtype=torch.bfloat16)

    processor = AutoImageProcessor.from_pretrained('microsoft/swin-base-patch4-window7-224')

    return model, processor


class R2GenGPTTransform(mt.RandomizableTransform):
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, data: dict):
        if data['image'].endswith('.pt.zst'):
            transform = transforms.ToPILImage()
            image = load_pt_zst(data['image'])
            image = repeat(image, '1 1 h w -> 3 1 h w')
            image = transform(image.squeeze(1))
        else:
            image = Image.open(data['image']).convert('RGB')
        image = self.processor(image, return_tensors='pt').pixel_values.unsqueeze(0).to(dtype=torch.bfloat16)

        return {
            'image': image,
            'input_text': data['answer'],
            'id': None,
        }
    

def r2gengpt_vl_evaluate(model, processor, dataloader, output):
    results = []

    for i, sample in enumerate(tqdm(dataloader)):
        sample['image'] = sample['image'].to('cuda')
        prediction, _ = model.test_step(sample, None)

        try:
            prediction = 'findings :' + ''.join(reversed(prediction[0].split('findings :')))
        except:
            pass

        results.append(
            {
                'answer': sample['input_text'],
                'prediction': prediction,
            },
        )

        if i % 1000 == 0:
            dump_results(results, output)

        print(sample['input_text'])
        print(prediction)

    dump_results(results, output)