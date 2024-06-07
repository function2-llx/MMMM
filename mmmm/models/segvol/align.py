from functools import cache

import cytoolz
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, CLIPTextModel, CLIPTokenizerFast, PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPooling

from luolib.lightning import LightningModule
from mmmm.data.defs import Batch
from mmmm.models import InstanceSam
from mmmm.models.segvol.modeling.sam import InstanceSamLoss, InstanceSamOutput

class TextEncoder(nn.Module):
    tokenizer: CLIPTokenizerFast
    clip_text_model: CLIPTextModel

    def forward(self, text: str | list[str], device: torch.device):
        inputs = self.tokenizer(text, padding=True, return_tensors='pt').to(device)
        clip_outputs: BaseModelOutputWithPooling = self.clip_text_model(**inputs, return_dict=True)
        text_embedding = clip_outputs.pooler_output
        assert text_embedding.shape[0] == 1
        text_embedding = self.dim_align(text_embedding[0])
        return text_embedding

def _add_prefix(log_dict: dict[str, ...], prefix: str) -> dict[str, ...]:
    if prefix != '' and not prefix.endswith('/'):
        prefix += '/'
    return {f'{prefix}{k}': v for k, v in log_dict.items()}

@torch.no_grad()
def _dice(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return 2 * (x * y).sum() / (x.sum() + y.sum())

class AlignSam(PreTrainedModel, LightningModule):
    text_encoder: TextEncoder
    supports_gradient_checkpointing = True

    def __init__(
        self,
        *,
        sam: InstanceSam,
        seg_vol_path: str,
        freeze_clip: bool = True,
        loss: InstanceSamLoss | None = None,
        num_classes: int | None = None,
        **kwargs,
    ):
        super().__init__(PretrainedConfig(), **kwargs)
        self.sam = sam
        sam.prompt_encoder.point_embeddings.requires_grad_(False)
        sam.prompt_encoder.not_a_point_embed.requires_grad_(False)
        sam.prompt_encoder.mask_downscaling.requires_grad_(False)
        self.loss = loss

        tokenizer = AutoTokenizer.from_pretrained(seg_vol_path)
        seg_vol = AutoModel.from_pretrained(seg_vol_path, trust_remote_code=True, test_mode=True)
        text_encoder = seg_vol.model.text_encoder
        text_encoder.tokenizer = tokenizer
        text_encoder.__class__ = TextEncoder
        self.text_encoder = text_encoder
        self.freeze_clip = freeze_clip
        if freeze_clip:
            self.text_encoder.requires_grad_(False)
            self.text_encoder.forward = cache(self.text_encoder.forward)
        else:
            self.text_encoder.requires_grad_(True)
            self.text_encoder.train()

        if num_classes is None:
            self.class_embeddings = None
        else:
            self.class_embeddings = nn.Embedding(num_classes, 768)
            self._class_cnt = 0

    def on_fit_start(self):
        super().on_fit_start()
        self.gradient_checkpointing_enable({'use_reentrant': False})

    @cache
    def _get_class_idx(self, class_name: str):
        ret = self._class_cnt
        self._class_cnt += 1
        return ret

    def get_class_embeddings(self, class_lists: list[list[str]]):
        if self.class_embeddings is None:
            class_lists = [
                [f'An image of the {class_name}.' for class_name in class_list]
                for class_list in class_lists
            ]
            if self.freeze_clip:
                return [
                    torch.stack([
                        self.text_encoder(class_name, device=self.device)
                        for class_name in class_list
                    ])
                    for class_list in class_lists
                ]
            else:
                classes = list(cytoolz.concat(class_lists))
                class_embeddings = self.text_encoder(classes)
                return class_embeddings.split(list(map(len, class_lists)))
        else:
            return [
                self.class_embeddings[[self._get_class_idx(class_name) for class_name in class_list]]
                for class_list in class_lists
            ]

    def forward(self, batch: Batch):
        class_lists: list[list[str]] = batch['grounding_classes']  # type: ignore
        class_embeddings = [
            torch.stack([
                self.text_encoder(class_name, device=self.device)
                for class_name in class_list
            ])
            for class_list in class_lists
        ]
        output: InstanceSamOutput = self.sam(batch['grounding_image'], batch['patch_size'], class_embeddings)
        return output

    def training_step(self, batch: Batch, *args, **kwargs):
        output: InstanceSamOutput = self(batch)
        loss, log_dict = self.loss(
            output.masks_logits,
            output.masks_logits_low_res,
            output.boxes,
            output.disc_logit,
            batch['masks'],
            batch['boxes'],
            batch['semantic_masks'],
            batch['semantic_boxes'],
            batch['index_offsets'],
            batch['semantic'],
            batch['num_uncertain'],
        )
        self.log_dict(_add_prefix(log_dict, 'train'))
        dice_pos = {}
        grounding_classes = batch['grounding_classes']  # type: ignore
        for batch_idx, targets in enumerate(grounding_classes):
            masks_preds = output.masks_logits[batch_idx].float().sigmoid() > 0.5
            if sem_masks := batch['semantic_masks'][batch_idx]:
                for i, target in enumerate(targets):
                    if (label := sem_masks[i]).any():
                        dice_pos.setdefault(target, []).append(
                            _dice(masks_preds[i, 0], label) * 100,
                        )
        dice_pos = {
            k: torch.stack(v).mean()
            for k, v in dice_pos.items()
        }
        self.log_dict(_add_prefix(dice_pos, 'train/dice-pos'))
        return loss
