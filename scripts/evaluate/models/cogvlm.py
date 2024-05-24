from einops import rearrange
import torch
from monai import transforms as mt
from peft import PeftModel
from PIL import Image
from torch import nn
from torch.nn import functional as nnf
from torchvision.transforms.v2 import functional as tvtf
import torchvision.transforms as transforms
from tqdm import tqdm
from transformers import AutoModelForCausalLM, LlamaTokenizer

from luolib.types import tuple3_t
from luolib.utils.zstd import load_pt_zst
from scripts.evaluate.utils import dump_results


def setup_cogvlm(checkpoint: str, tokenizer: str, setting: str):
    model = AutoModelForCausalLM.from_pretrained(
        'THUDM/cogvlm-chat-hf',
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    if setting == 'finetuned':
        pos_embed = model.model.vision.patch_embedding.position_embedding.weight
        cls_pos_embed, pos_embed = pos_embed[0:1], pos_embed[1:]
        pos_embed = rearrange(pos_embed, '(h w) c -> 1 c h w', h=35, w=35)

        pos_embed = nnf.interpolate(pos_embed, (16, 16), mode='area')
        pos_embed = torch.cat([cls_pos_embed, rearrange(pos_embed, '1 c h w ->(h w) c')])
        model.model.vision.patch_embedding.position_embedding = nn.Embedding(
            *pos_embed.shape[:2], _weight=pos_embed,
        )
        model.config.vision_config['image_size'] = 224
    if checkpoint:
        model = PeftModel.from_pretrained(model, checkpoint)
    tokenizer = LlamaTokenizer.from_pretrained(tokenizer if tokenizer else 'lmsys/vicuna-7b-v1.5')

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
            'inputs': inputs,
            'question': data['question'],
            'answer': data['answer'],
        }


def cogvlm_vl_evaluate(model, tokenizer, dataloader, output):
    results = []

    for i, sample in enumerate(tqdm(dataloader)):
        with torch.inference_mode():
            prediction = (
                tokenizer.decode(
                    model.generate(
                        input_ids=sample['inputs']['input_ids'].unsqueeze(0).to('cuda'),
                        token_type_ids=sample['inputs']['token_type_ids'].unsqueeze(0).to('cuda'),
                        attention_mask=sample['inputs']['attention_mask'].unsqueeze(0).to('cuda'),
                        images=[[sample['inputs']['images'][0].to('cuda').to(torch.bfloat16)]],
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