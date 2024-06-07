from luolib.lightning import LightningModule
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
import torch.nn as nn
import torch
from torchvision.transforms.v2 import functional as tvtf
from PIL import Image
from scripts.finetune._vqa._base import VQATransform, VQADataModule
from typing import Callable
from scripts.finetune._utils import CE_IGNORE_INDEX, intensity_norm_
from torchvision.io import read_image, ImageReadMode
import einops

class MyLlavaNextForConditionalGeneration(LlavaNextForConditionalGeneration):
    def _load_from_state_dict(self, state_dict: dict, prefix: str, *args, **kwargs):
        key = 'vision_tower.vision_model.embeddings.position_embedding.weight'
        if (pos_embed := state_dict.get(key)) is not None:
            cls_pos_embed, pos_embed = pos_embed[0:1], pos_embed[1:]
            pos_embed = einops.rearrange(pos_embed, '(h w) c -> 1 c h w', h=24, w=24)
            import torch.nn.functional as nnf
            pos_embed = nnf.interpolate(pos_embed, (16, 16), mode='area')
            pos_embed = torch.cat([cls_pos_embed, einops.rearrange(pos_embed, '1 c h w ->(h w) c')])
            state_dict[key] = pos_embed
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class FinetuneLlavaNEXT(LightningModule):
    def __init__(self, *, model_path: str):
        super().__init__()
        self.llavaN_model = MyLlavaNextForConditionalGeneration.from_pretrained(
            model_path,
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

        self.target_modules, self.modules_to_save = self.get_lora_modules_default(self.llavaN_model)
        self.llavaN_model.gradient_checkpointing_enable({"use_reentrant": False})
        self.train()

        # pos_embed = self.llavaN_model.vision_tower.vision_model.embeddings.position_embedding.weight
        # cls_pos_embed, pos_embed = pos_embed[0:1], pos_embed[1:]
        # pos_embed = einops.rearrange(pos_embed, '(h w) c -> 1 c h w', h=24, w=24)
        # import torch.nn.functional as nnf
        # pos_embed = nnf.interpolate(pos_embed, (16, 16), mode='area')
        # pos_embed = torch.cat([cls_pos_embed, einops.rearrange(pos_embed, '1 c h w ->(h w) c')])
        # self.llavaN_model.vision_tower.vision_model.embeddings.position_embedding = nn.Embedding(
        #     *pos_embed.shape[:2], _weight=pos_embed,
        # )
        # self.llavaN_model.vision_tower.vision_model.embeddings.position_ids = torch.arange(257).expand((1, -1))
        1

    def get_lora_modules_default(self, module: nn.Module, prefix: str = '', recursive: bool = True):
        target_modules, modules_to_save = [], []

        def dfs(m: nn.Module, prefix: str):
            if isinstance(m, nn.Linear):
                target_modules.append(prefix)  # Linear layers as LoRA targets
            elif isinstance(m, nn.Embedding):
                modules_to_save.append(prefix)  # Embedding layers to save

            for name, child in m.named_children():
                dfs(child, f"{prefix}.{name}" if prefix else name)

        dfs(module, prefix)
        return target_modules, modules_to_save

    def training_step(self, batch, *args, **kwargs):
        outputs = self.llavaN_model.forward(
            input_ids=batch['vlm_inputs']['input_ids'],
            pixel_values=einops.repeat(batch['image'], 'n ... -> n l2 ...', l2=2),
            image_sizes=batch['image_size'],
            attention_mask=batch['vlm_inputs']['attention_mask'],
            labels=batch['vlm_inputs']['labels'],
        )
        loss = outputs.loss
        self.log('train/loss', loss)
        return loss

class LLAVANVQATransform(VQATransform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-13b-hf")

        # self.processor.image_processor.crop_size = {"height": 224, "width": 224}
        # self.processor.image_processor.image_grid_pinpoints = [[224, 448], [448, 224], [448, 448], [672, 224], [224, 672]]
        # self.processor.image_processor.size = {"shortest_edge": 224}

    def __call__(self, data):
        image = read_image(data['image'], ImageReadMode.RGB)
        image = tvtf.to_dtype(image, torch.float32, scale=True)
        image = tvtf.resize(image, self.resize)
        image_size = torch.tensor(image.size)
        intensity_norm_(image)

        pairs = [(qa['question'], qa['answer']) for qa in data['vqa']]
        self.R.shuffle(pairs)
        tokenizer = self.tokenizer
        text_ids = []
        labels = []
        for i, (query, answer) in enumerate(pairs):
            prompt = f'Question: {query} Answer:'
            if i == 0:
                prompt = '<image>\n' + prompt
            prompt_ids = torch.tensor(tokenizer.encode(prompt, add_special_tokens=False))
            answer_ids = torch.tensor(tokenizer.encode(answer, add_special_tokens=False))
            text_ids.extend([prompt_ids, answer_ids])
            if i > 0:
                labels.extend([
                    torch.tensor([tokenizer.eos_token_id]),
                    torch.full((prompt_ids.shape[0] - 1,), CE_IGNORE_INDEX),
                    answer_ids,
                ])
            else:
                labels.extend(
                    [
                        torch.full((prompt_ids.shape[0],), CE_IGNORE_INDEX),
                        answer_ids,
                    ],
                )
        input_ids = torch.cat([
            torch.tensor([tokenizer.bos_token_id]),
            *text_ids,
            torch.tensor([tokenizer.eos_token_id]),
        ])
        labels = torch.cat([
            # the first item will be shifted
            torch.tensor([tokenizer.bos_token_id]),
            *labels,
            torch.tensor([tokenizer.eos_token_id]),
        ])
        return {
            'image': image,
            'image_size': image_size,
            'vlm_inputs': {
                'input_ids': input_ids,
                'labels': labels,
                'attention_mask': torch.ones_like(input_ids),
            }
        }


class LLAVANVQADataModule(VQADataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_transform(self) -> Callable:
        return LLAVANVQATransform(self.tokenizer, resize=self.resize)