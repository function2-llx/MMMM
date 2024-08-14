from functools import cache

import cytoolz
import torch
from torch import nn
from transformers import CLIPTokenizerFast, CLIPTextModel, PreTrainedModel, PretrainedConfig, AutoTokenizer, AutoModel
from transformers.modeling_outputs import BaseModelOutputWithPooling

from luolib.lightning import LightningModule

from mmmm.data.defs import Batch
from mmmm.models import InstanceSam
from mmmm.models.loss import DiceFocalLoss
from mmmm.models.segvol.modeling.sam import InstanceSamLoss, InstanceSamOutput, Sam

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
        sam: Sam,
        seg_vol_path: str,
        num_classes: int | None = None,
        loss: DiceFocalLoss | None = None,
        freeze_clip: bool = True,
        **kwargs,
    ):
        super().__init__(PretrainedConfig(), **kwargs)
        self.sam = sam
        # not used
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

    def _apply_clip_template(self, class_name: str):
        return f'An image of the {class_name}.'

    def get_class_embeddings(self, class_lists: list[list[str]]):
        if self.class_embeddings is None:
            class_lists = [
                [self._apply_clip_template(class_name) for class_name in class_list]
                for class_list in class_lists
            ]
            if self.freeze_clip:
                return [
                    torch.stack([self.text_encoder(class_name, device=self.device) for class_name in class_list])
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

    def forward(self, batch):
        class_lists = batch['classes']
        class_embeddings = self.get_class_embeddings(class_lists)
        masks_logits: list[torch.Tensor] = self.sam(batch['image'], batch['patch_size'], class_embeddings)
        log_dict = {}
        num_masks = 0
        # NOTE: it is assumed that:
        #   1. the batch sizes are the same across distributed ranks;
        #   2. no gradient accumulation, thanks
        if masks := batch.get('masks'):
            for masks_logits_, masks_ in zip(masks_logits, masks):
                num_masks += masks_.shape[0]
                log_dict_ = self.loss(masks_logits_[:, None], masks_[:, None], reduce_batch=False, return_dict=True)
                log_dict.setdefault('loss', []).append(log_dict_.pop('total'))
                for k, v in log_dict_.items():
                    log_dict.setdefault(k, []).append(v)
            all_num_masks = self.trainer.strategy.reduce(torch.tensor(num_masks, device=self.device), reduce_op='sum')
            log_dict_reduced = {
                k: torch.cat(v).sum() * self.trainer.world_size / all_num_masks
                for k, v in log_dict.items()
            }
            return log_dict_reduced, masks_logits
        else:
            return masks_logits

    def training_step(self, batch, *args, **kwargs):
        log_dict, masks_logits = self(batch)
        self.log_dict(_add_prefix(log_dict, 'train'), sync_dist=True)
        with torch.no_grad():
            masks: list[torch.BoolTensor] = batch['masks']
            class_lists: list[list[str]] = batch['classes']  # type: ignore
            dice_pos = {}
            for batch_idx, targets in enumerate(class_lists):
                masks_preds = masks_logits[batch_idx].float().sigmoid() > 0.5
                for i, target in enumerate(targets):
                    if (label := masks[batch_idx][i]).any():
                        dice_pos.setdefault(target, []).append(_dice(masks_preds[i], label) * 100)
            dice_pos_reduced = {
                k: torch.stack(v).mean()
                for k, v in dice_pos.items()
            }
        self.log_dict(_add_prefix(dice_pos_reduced, 'train/dice-pos'))
        return log_dict['loss']

class AlignInstanceSam(PreTrainedModel, LightningModule):
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
        # not used
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

    def apply_template(self, class_name: str):
        return f'An image of the {class_name}.'

    def get_class_embeddings(self, class_lists: list[list[str]]):
        if self.class_embeddings is None:
            class_lists = [
                [self.apply_template(class_name) for class_name in class_list]
                for class_list in class_lists
            ]
            if self.freeze_clip:
                return [
                    torch.stack([self.text_encoder(class_name, device=self.device) for class_name in class_list])
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
        class_lists: list[list[str]] = batch['classes']  # type: ignore
        class_embeddings = self.get_class_embeddings(class_lists)
        output: InstanceSamOutput = self.sam(batch['image'], batch['patch_size'], class_embeddings)
        return output

    def training_step(self, batch: Batch, *args, **kwargs):
        output: InstanceSamOutput = self(batch)
        loss, log_dict = self.loss.forward(
            output.masks_logits,
            output.masks_logits_low_res,
            output.boxes,
            output.disc_logit,
            batch['masks'],
            batch['boxes'],
            batch['index_offsets'],
        )
        self.log_dict(_add_prefix(log_dict, 'train'))
        dice_pos = {}
        classes = batch['classes']  # type: ignore
        for batch_idx, targets in enumerate(classes):
            if (sem_masks := batch['masks'][batch_idx]) is not None:
                masks_preds = output.masks_logits[batch_idx].float().sigmoid() > 0.5
                for i, target in enumerate(targets):
                    if (label := sem_masks[i]).any():
                        dice_pos.setdefault(target, []).append(
                            _dice(masks_preds[i, 0], label) * 100,
                        )
        dice_pos_reduced = {
            k: torch.stack(v).mean()
            for k, v in dice_pos.items()
        }
        self.log_dict(_add_prefix(dice_pos_reduced, 'train/dice-pos'))
        # class_indexes = [classes[0].index(name) for name in ['left kidney', 'right kidney']]
        # print(batch['src'][0])
        # IndexTrackerBinary(
        #     batch['image'][0][0].float(),
        #     torch.cat([
        #         batch['semantic_masks'][0][class_indexes, 0],
        #         output.masks_logits[0].float().sigmoid()[class_indexes, 0] > 0.5,
        #     ])
        # )
        return loss
