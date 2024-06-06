from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Literal

import einops
import einops.layers.torch as elt
from jsonargparse import class_from_function
from peft import PeftModel
from scipy.optimize import linear_sum_assignment
import torch
from torch import nn
from torch.nn import functional as nnf
from transformers.modeling_outputs import CausalLMOutputWithPast

from luolib.lightning import LightningModule
from luolib.losses import bce_neg, bce_pos, zero_loss
from luolib.types import PathLike, param3_t, tuple2_t, tuple3_t
from luolib.utils.misc import pairwise_forward
from mmmm.data.defs import Batch
from mmmm.tokenizer import MMMMTokenizer
from mmmm.utils import apply_prefix, get_lora_modules_default, get_lora_modules_finetune_all
from monai.data import box_pair_giou, convert_box_mode
from monai.data.box_utils import CenterSizeMode
from .cogvlm import CogVLMConfig, CogVLMForCausalLM
from .loss import DiceFocalLoss
from .segvol import InstanceSam

__all__ = [
    'MMMMForCausalLM',
    'build',
]

@dataclass
class VisionArgs:
    pos_embed_shape: tuple3_t[int]
    pt_pos_embed_shape: tuple2_t[int] | None = None
    patch_size: param3_t[int] = 16

@dataclass(kw_only=True)
class MMMMOutputWithPast:
    """modified from CausalLMOutputWithPast
    inheriting from ModelOutput will cause some trouble (e.g., "should not have more than one required field")
    """
    lm_logits: torch.Tensor
    lm_loss: torch.Tensor | None = None
    past_key_values: tuple[tuple[torch.Tensor, ...], ...] | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None
    attentions: tuple[torch.Tensor, ...] | None = None

    masks_logits: list[torch.Tensor]
    mask_loss: torch.Tensor | None

@dataclass
class VisualGroundingOutput:
    """(batch size, num targets, num queries, ...)"""
    masks_logits: list[torch.Tensor] = field(default_factory=list)
    masks_logits_ds: list[torch.Tensor] = field(default_factory=list)
    boxes: list[torch.FloatTensor] = field(default_factory=list)
    disc_logit: list[torch.FloatTensor] = field(default_factory=list)

@dataclass
class SlidingWindow:
    patch_size: tuple3_t[int]
    batch_size: int
    overlap: float = 0.5

MATCH_NEGATIVE = -1
MATCH_UNCERTAIN = -2

EPS = 1e-8

def _optional_index(x: torch.Tensor | None, index: ...):
    return None if x is None else x[index]

def _dice_metric(x: torch.BoolTensor, y: torch.BoolTensor):
    reduce = elt.Reduce('c ... -> c', 'sum')
    nominator = 2 * reduce(x & y)
    denominator = reduce(x) + reduce(y)
    return nominator / (denominator + EPS)

class MMMMForCausalLM(CogVLMForCausalLM, LightningModule):
    tokenizer: MMMMTokenizer
    sam_model: InstanceSam
    mask_loss: DiceFocalLoss | None

    @classmethod
    def build(
        cls,
        pretrained_model_name_or_path: str | os.PathLike | None,
        *args,
        lm_loss_weight: float = 1.,
        box_l1_loss_weight: float = 1.,
        box_giou_loss_weight: float = 0.5,
        disc_loss_weight: float = 1.,
        neg_mask_loss: bool = True,
        vision_override: VisionArgs,
        tokenizer: MMMMTokenizer,
        sam: 'SamArgs',
        torch_dtype: str | torch.dtype = 'auto',
        mask_loss: DiceFocalLoss | None = None,
        lora_lang: bool = True,
        disable_vg: bool = False,
        freeze_vg: bool = False,
    ):
        """make jsonargparse happy
        This works thanks to that AST does not support this (according to the debug information)
        TODO: refactor the construction of PreTrainedModel
        NOTE: mask loss handle the weights itself internally
        Args:
            neg_mask_loss: whether to compute loss on negative instance masks
            lora_lang: whether to fine-tune language weights
            vlm_only: do not perform vg during training
        """
        self: MMMMForCausalLM = super().from_pretrained(
            pretrained_model_name_or_path,
            vision_override=vision_override,
            torch_dtype=torch_dtype,
        )
        self.resize_token_embeddings(len(tokenizer))
        self.sam_model = build_sam_vit_3d(sam)
        self.tokenizer = tokenizer
        self.lm_loss_weight = lm_loss_weight
        self.box_l1_loss_weight = box_l1_loss_weight
        self.box_giou_loss_weight = box_giou_loss_weight
        self.disc_loss_weight = disc_loss_weight
        self.mask_loss = mask_loss
        self.neg_mask_loss = neg_mask_loss
        self.vg_proj = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.config.hidden_size, self.sam_model.prompt_dim),
        )
        # following DetrMLPPredictionHead
        self.box_head = nn.Sequential(
            nn.Linear(self.sam_model.mask_embed_dim, self.sam_model.mask_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.sam_model.mask_embed_dim, self.sam_model.mask_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.sam_model.mask_embed_dim, 6),
        )
        self.disc_head = nn.Sequential(
            nn.Linear(self.sam_model.mask_embed_dim, self.sam_model.mask_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.sam_model.mask_embed_dim, 1),
        )
        self.model.config.lora_lang = lora_lang
        self.check_grad = False
        self._setup_freeze(freeze_vg)
        self.disable_vg = disable_vg
        return self

    def on_load_checkpoint(self, checkpoint: dict):
        # we handle the state dict by cls.from_pretrained (in constructor) and peft_model.load_adapter (in CLI)
        checkpoint['state_dict'] = {}
        self.strict_loading = False

    def _init_weights(self, module):
        """Let's happily do nothing (necessary to make SAM pre-trained weights survive)"""

    def __init__(self, vlm_config: CogVLMConfig, *, vision_override: VisionArgs, **kwargs):
        # adapt vision config
        vision_config: dict = vlm_config.vision_config
        vision_config.update(vars(vision_override))
        super().__init__(vlm_config, **kwargs)

    def load_default_adapter(self, ckpt_dir: Path):
        self.peft_model.load_adapter(str(ckpt_dir / 'adapter'), 'default')

    def _setup_freeze(self, freeze_vg: bool):
        sam = self.sam_model
        if freeze_vg:
            self.vg_proj.requires_grad_(False)
            sam.requires_grad_(False)
            self.disc_head.requires_grad_(False)
            self.box_head.requires_grad_(False)
        else:
            # freeze unused parameters to make DDP work
            sam.prompt_encoder.point_embeddings.requires_grad_(False)
            sam.prompt_encoder.not_a_point_embed.requires_grad_(False)
            sam.prompt_encoder.mask_downscaling.requires_grad_(False)
            # sam.mask_decoder.iou_prediction_head.requires_grad_(False)
            if not sam.mask_decoder.text_sim:
                sam.mask_decoder.txt_align_upscaled_embedding.requires_grad_(False)

    def get_lora_modules(self, prefix: str):
        # apply LoRA on VLM, fully finetune others
        target_modules, modules_to_save = get_lora_modules_default(self.model, apply_prefix(prefix, 'model'))
        for name, child in self.named_children():
            if name == 'model':
                continue
            c_modules_to_save = get_lora_modules_finetune_all(child, apply_prefix(prefix, name))
            modules_to_save.extend(c_modules_to_save)
        return target_modules, modules_to_save

    def _get_vg_hidden_states(self, token_ids: torch.LongTensor, hidden_states: torch.Tensor) -> list[torch.Tensor]:
        eop_mask: torch.BoolTensor = token_ids == self.tokenizer.eop_token_id  # type: ignore
        vg_hidden_states = [
            hidden_states[i, eop_mask[i]]
            for i in range(hidden_states.shape[0])
        ]
        return vg_hidden_states

    def _predict_masks(
        self, vg_hidden_states: torch.Tensor, image_embeddings: torch.Tensor, patch_size_z: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sam = self.sam_model
        text_embedding = self.vg_proj(vg_hidden_states)
        sparse_embeddings, dense_embeddings = sam.prompt_encoder(image_embeddings.shape[2:], text_embedding=text_embedding)
        sparse_embeddings = sparse_embeddings.to(text_embedding.dtype)
        masks_logits_ds, masks_embeds = sam.mask_decoder(
            image_embeddings=image_embeddings,
            text_embedding=text_embedding,  # make SegVol happy
            image_pe=sam.prompt_encoder.get_dense_pe(image_embeddings.shape[2:]),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            patch_size_z=patch_size_z,
        )
        return masks_logits_ds, masks_embeds

    def visual_grounding(
        self,
        token_ids: torch.LongTensor,
        hidden_states: torch.Tensor,
        image: list[torch.Tensor],
        patch_size: list[tuple3_t[int]],
    ) -> VisualGroundingOutput:
        """
        Args:
            token_ids: generated token ids
            hidden_states: hidden states that generate tokens
        Returns: for each sample in the batch:
            - predicted masks logits, the first one is semantic
            - predicted bounding boxes
            - discrimination logits for instances
            each one has a size of (num_vg, num_queries, ...)
        """
        batch_size = token_ids.shape[0]
        vg_hidden_states = self._get_vg_hidden_states(token_ids, hidden_states)
        image_embeddings: list[torch.Tensor] = self.sam_model.image_encoder(image, patch_size)
        ret = VisualGroundingOutput()
        for i in range(batch_size):
            masks_logits_ds, masks_embeds = self._predict_masks(
                vg_hidden_states[i], image_embeddings[i], patch_size[i][0],
            )
            masks_logits = nnf.interpolate(masks_logits_ds, image[i].shape[1:], mode='trilinear')
            # calling sigmoid here to restrict range (CenterSizeMode), following DETR
            boxes = self.box_head(masks_embeds).float().sigmoid()
            disc_logit = einops.rearrange(self.disc_head(masks_embeds[:, 1:]).float(), 'nt nq 1 -> nt nq')
            ret.masks_logits.append(masks_logits)
            ret.masks_logits_ds.append(masks_logits_ds)
            ret.boxes.append(boxes)
            ret.disc_logit.append(disc_logit)
        return ret

    def on_fit_start(self) -> None:
        super().on_fit_start()
        # TODO: replace with checkpoint wrapper
        # https://github.com/pytorch/pytorch/blob/main/torch/distributed/algorithms/_checkpoint/checkpoint_wrapper.py
        self.gradient_checkpointing_enable({'use_reentrant': False})
        # NOTE: there may be some code setting lora.Linear.base_layer.eval(),
        #  however, let's keep it "training" to make DeepSpeed work, since it is just a linear layer
        #  and is not affected by the mode

    def training_step(self, batch: Batch, *args, **kwargs):
        vlm_inputs = batch['vlm_inputs']
        input_ids: torch.LongTensor = vlm_inputs['input_ids']  # type: ignore
        vlm_output: CausalLMOutputWithPast = self(
            **vlm_inputs,
            image=batch['image'],
            patch_size=batch['patch_size'],
            pool_size=batch['pool_size'],
            return_dict=True,
            output_hidden_states=True,
        )
        if self.disable_vg:
            lm_loss = vlm_output.loss
            self.log('train/loss', lm_loss, sync_dist=True)
            return self.lm_loss_weight * lm_loss
        vg_output = self.visual_grounding(
            # shift as suggested by GLaMM: https://github.com/mbzuai-oryx/groundingLMM/issues/16
            input_ids[:, 1:],
            vlm_output.hidden_states[-1][:, :-1],
            batch['grounding_image'],
            batch['patch_size'],
        )
        vg_loss, vg_log_dict = self._compute_vg_loss_batch(
            vg_output.masks_logits, vg_output.masks_logits_ds,
            vg_output.boxes, vg_output.disc_logit,
            batch['masks'], batch['boxes'],
            batch['semantic_masks'], batch['semantic_boxes'],
            batch['index_offsets'], batch['semantic'], batch['num_uncertain'],
            'train',
        )
        # weight for VG is controlled internally
        loss = vlm_output.loss * self.lm_loss_weight + vg_loss
        # make some custom log
        lm_labels = vlm_inputs['labels']
        token_log_dict = {
            f'train/token-lm/{name}_loss': nnf.cross_entropy(vlm_output.logits[token_mask], lm_labels[token_mask])
            for name, token_mask in {
                'bop': lm_labels == self.tokenizer.bop_token_id,
                'eop': lm_labels == self.tokenizer.eop_token_id,
            }.items()
            if token_mask.any()
        }
        self.log_dict(
            {
                'train/loss': loss,
                'train/lm_loss': vlm_output.loss,
                'train/vg_loss': vg_loss,
                **vg_log_dict,
                **token_log_dict,
            },
            # the logging keys can be inconsistent, setting sync_dist=True can make DDP hang
            # sync_dist=True,
        )
        return loss

    @torch.no_grad()
    def _match_instances(
        self,
        masks_logits: torch.Tensor,
        boxes_reg: torch.Tensor,
        disc_logit: torch.Tensor,
        masks_label: torch.BoolTensor | None,
        boxes_label: torch.Tensor,
        num_uncertain: int,
        offset: int,
    ) -> torch.LongTensor | int:
        num_queries = masks_logits.shape[0]
        num_pos = boxes_label.shape[0]
        num_uncertain = min(max(num_queries - num_pos, 0), num_uncertain)
        num_neg = max(num_queries - num_pos - num_uncertain, 0)
        if num_queries == num_neg:
            return MATCH_NEGATIVE
        disc_cost_pos = self.disc_loss(disc_logit, True, reduce_batch=False)
        disc_cost_neg = self.disc_loss(disc_logit, False, reduce_batch=False)
        # label order: pos, neg, uncertain,
        disc_cost = torch.cat(
            [
                einops.repeat(disc_cost_pos, 'n -> n m', m=num_pos),
                einops.repeat(disc_cost_neg, 'n -> n m', m=num_neg),
                disc_logit.new_zeros(num_queries, num_uncertain),
            ],
            dim=1,
        )
        if masks_label is None:
            box_cost = torch.cat(
                [
                    pairwise_forward(self.box_loss, boxes_reg, boxes_label, reduce_batch=False),
                    disc_cost.new_zeros(num_queries, num_neg + num_uncertain)
                ],
                dim=1,
            )
            mask_cost = torch.zeros_like(disc_cost)
        else:
            # not using box for matching when mask is available
            box_cost = torch.zeros_like(disc_cost)
            mask_cost_pos = pairwise_forward(self.mask_loss, masks_logits, masks_label, reduce_batch=False)
            if self.neg_mask_loss:
                mask_cost_neg = self.mask_loss(masks_logits, reduce_batch=False)
                mask_cost = torch.cat(
                    [
                        mask_cost_pos,
                        einops.repeat(mask_cost_neg, 'n -> n m', m=num_neg),
                        mask_cost_pos.new_zeros(num_queries, num_uncertain),
                    ],
                    dim=1,
                )
            else:
                mask_cost = torch.cat(
                    [
                        mask_cost_pos,
                        mask_cost_pos.new_zeros(num_queries, num_neg + num_uncertain)
                    ],
                    dim=1,
                )
        cost = mask_cost + box_cost + disc_cost
        row, col = linear_sum_assignment(cost.float().cpu().numpy())
        match = torch.empty(num_queries, dtype=torch.int64, device=self.device)
        match[row] = torch.as_tensor(col, device=self.device)
        match[match >= num_pos] = MATCH_NEGATIVE
        match[match >= num_pos + num_neg] = MATCH_UNCERTAIN
        match[match >= 0] += offset
        return match

    def box_loss(self, input: torch.Tensor, target: torch.Tensor, reduce_batch: bool = True, return_dict: bool = False):
        """input and target are in CenterSizeMode"""
        if reduce_batch:
            l1 = nnf.l1_loss(input, target)
        else:
            l1 = nnf.l1_loss(input, target, reduction='none').mean(dim=-1)
        giou = box_pair_giou(
            convert_box_mode(input, src_mode=CenterSizeMode),
            convert_box_mode(target, src_mode=CenterSizeMode),
        )
        if reduce_batch:
            giou = giou.mean()
        giou = 1 - giou
        total = self.box_l1_loss_weight * l1 + self.box_giou_loss_weight * giou
        if return_dict:
            return {
                'l1': l1,
                'giou': giou,
                'total': total,
            }
        else:
            return total

    def disc_loss(self, input: torch.Tensor, pos: bool, reduce_batch: bool = True, return_dict: bool = False):
        disc_loss = (bce_pos if pos else bce_neg)(input)
        if reduce_batch:
            disc_loss = disc_loss.mean()
        total = self.disc_loss_weight * disc_loss
        if return_dict:
            return {
                'ce': disc_loss,
                'total': total,
            }
        else:
            return total

    def _compute_vg_loss_batch(
        self,
        masks_logits: list[torch.Tensor],
        masks_logits_ds: list[torch.Tensor],
        boxes_reg: list[torch.Tensor],
        disc_logit: list[torch.Tensor],
        masks_label: list[torch.BoolTensor | None],
        boxes_label: list[torch.Tensor],
        sem_masks: list[torch.BoolTensor | None],
        sem_boxes: list[torch.Tensor],
        index_offsets: list[torch.LongTensor],
        semantic: list[torch.BoolTensor],
        num_uncertain: list[torch.LongTensor],
        prefix: str,
    ):
        batch_size = len(masks_logits)
        vg_loss_list = []
        vg_log_dict = {}
        for i in range(batch_size):
            _vg_loss, _vg_log_dict = self._compute_vg_loss(
                masks_logits[i], masks_logits_ds[i],
                boxes_reg[i], disc_logit[i],
                masks_label[i], boxes_label[i],
                sem_masks[i], sem_boxes[i],
                index_offsets[i], semantic[i], num_uncertain[i],
            )
            vg_loss_list.append(_vg_loss)
            for k, v in _vg_log_dict.items():
                vg_log_dict.setdefault(k, []).append(v)
        vg_loss = torch.stack(vg_loss_list).mean()
        with torch.no_grad():
            vg_log_dict = {
                f'{prefix}/vg/{k}': torch.stack(v).mean()
                for k, v in vg_log_dict.items()
            }
        return vg_loss, vg_log_dict

    def _compute_vg_loss(
        self,
        masks_logits: torch.Tensor,
        masks_logits_ds: torch.Tensor,
        boxes_reg: torch.Tensor,
        disc_logit: torch.Tensor,
        masks_label: torch.BoolTensor | None,
        boxes_label: torch.Tensor,
        sem_masks: torch.BoolTensor | None,
        sem_boxes: torch.Tensor | None,
        index_offsets: torch.LongTensor,
        semantic: torch.BoolTensor,
        num_uncertain: torch.LongTensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        num_targets = masks_logits.shape[0]
        # tokens for vg might be truncated by max_seq_len
        assert num_targets <= index_offsets.shape[0]
        loss = zero_loss(masks_logits_ds, boxes_reg, disc_logit)
        log_dict = {}
        if num_targets > 0:
            # convert to float since it is used for loss calculation
            masks_logits = masks_logits.float()
            index_offsets = index_offsets[:num_targets]
            semantic = semantic[:num_targets]
            num_uncertain = num_uncertain[:num_targets]

            def _accumulate(loss_log_dict: dict[str, torch.Tensor], task: Literal['semantic', 'instance'], prefix: str):
                ret = loss_log_dict.pop('total')
                for k, v in loss_log_dict.items():
                    log_dict[f'{task}-{prefix}-{k}'] = v
                return ret

            # 1. semantic part
            if sem_masks is not None and (_valid_mask := num_uncertain >= 0).any():
                sem_masks = sem_masks[:num_targets]
                sem_boxes = sem_boxes[:num_targets]
                loss += _accumulate(
                    self.mask_loss(masks_logits[_valid_mask, 0:1], sem_masks[_valid_mask], return_dict=True),
                    'semantic', 'mask',
                )
                loss += _accumulate(
                    self.box_loss(boxes_reg[:, 0][_valid_mask], sem_boxes, return_dict=True),
                    'semantic', 'box',
                )

            # 2. instance part
            # drop the semantic part
            masks_logits_ds = masks_logits_ds[:, 1:]
            masks_logits = masks_logits[:, 1:]
            boxes_reg = boxes_reg[:, 1:]
            # downsample the mask for matching to save computation
            if masks_label is None:
                masks_label_ds = None
            else:
                masks_label_ds = nnf.interpolate(
                    masks_label[None].byte(), masks_logits_ds.shape[2:], mode='nearest-exact',
                )[0].bool()

            num_uncertain_list = num_uncertain.tolist()
            offset_list = index_offsets[:, 0].tolist()
            match = torch.full((num_targets, masks_logits.shape[1]), MATCH_UNCERTAIN, dtype=torch.long, device=self.device)
            for i in range(num_targets):
                if semantic[i] or (_num_uncertain := num_uncertain_list[i]) == -1:
                    # we know that the target presents on the image, but no localized information available, skip
                    continue
                label_slice = slice(*index_offsets[i])
                match[i] = self._match_instances(
                    masks_logits_ds[i, :, None],
                    boxes_reg[i],
                    disc_logit[i],
                    None if masks_label_ds is None else masks_label_ds[label_slice, None],
                    boxes_label[label_slice],
                    _num_uncertain,
                    offset_list[i],
                )
            match_pos_mask: torch.BoolTensor = match >= 0  # type: ignore
            match_pos = match[match_pos_mask]
            match_neg_mask: torch.BoolTensor = match == MATCH_NEGATIVE  # type: ignore
            if match_pos.shape[0] > 0:
                loss += _accumulate(
                    self.disc_loss(disc_logit[match_pos_mask], True, return_dict=True),
                    'instance', 'disc-pos',
                )
                loss += _accumulate(
                    self.box_loss(
                        boxes_reg[match_pos_mask], boxes_label[match_pos], return_dict=True,
                    ),
                    'instance', 'box',
                )
                if masks_label is not None:
                    loss += _accumulate(
                        self.mask_loss(
                            masks_logits[match_pos_mask][:, None], masks_label[match_pos, None], return_dict=True,
                        ),
                        'instance', 'mask-pos',
                    )
            if match_neg_mask.any():
                loss += _accumulate(
                    self.disc_loss(disc_logit[match_neg_mask], False, return_dict=True),
                    'instance', 'disc-neg',
                )
                if self.neg_mask_loss:
                    loss += _accumulate(
                        self.mask_loss(masks_logits[match_neg_mask][:, None], return_dict=True),
                        'instance', 'mask-neg',
                    )
        return loss, log_dict

    # def sw_predictor(self, patch: torch.Tensor):
    #     output: MMMMOutputWithPast = self(
    #         global_enc_image=patch,
    #         grounding_enc_image=patch,
    #         **self._val_vlm_inputs,
    #         use_cache=False,
    #     )
    #     return output.masks_logits[0][None]
    #
    # def on_validation_epoch_start(self) -> None:
    #     self.dice_metric = DiceMetric(reduction=MetricReduction.MEAN_BATCH)
    #
    # def validation_step(self, batch: dict, *args, **kwargs):
    #     # the interface of sliding_window_inference only accepts image as input
    #     self._val_vlm_inputs = batch['vlm_inputs']
    #     conf = self.val_sw
    #     logits = sliding_window_inference(
    #         batch['image'],
    #         conf.patch_size,
    #         conf.batch_size,
    #         self.sw_predictor,
    #         conf.overlap,
    #         BlendMode.GAUSSIAN,
    #     )
    #     pred = logits.sigmoid() > 0.5
    #     dice = self.dice_metric(pred, batch['masks'])
    #     for i, name in enumerate(batch['mask_classes']):
    #         self.log(f'val/dice/{name}', dice[i])

    # noinspection PyMethodOverriding
    def prepare_inputs_for_generation(
        self,
        input_ids,
        *,
        token_type_ids,
        position_ids,
        image=None,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        patch_size,
        pool_size,
        **kwargs,
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]
            token_type_ids = token_type_ids[:, -1:]
            position_ids = position_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "image": image,
                "token_type_ids": token_type_ids,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "attention_mask": attention_mask,
                'patch_size': patch_size,
                'pool_size': pool_size,
                "use_cache": kwargs.get("use_cache"),
            },
        )
        return model_inputs


    def _inference_path(self, input_ids, token_type_ids, global_enc_images, attention_masks):
        # Process and return inference output
        output_hidden_states = []
        for i in range(input_ids.shape[0]):
            output_i = super().forward(
                input_ids=input_ids[i:i + 1],
                token_type_ids=token_type_ids[i:i + 1],
                image=global_enc_images[i],
                attention_mask=attention_masks[i:i + 1],
                output_hidden_states=True
            )
            output_hidden_states.append(output_i.hidden_states)
            # torch.cuda.empty_cache()

        output_hidden_states = torch.cat(output_hidden_states, dim=0)
        output_hidden_states = [output_hidden_states]
        return output_hidden_states

    def evaluate(self, input_ids, token_type_ids, global_enc_images, grounding_enc_images, resize_list, orig_sizes, max_tokens_new=32):
        with torch.no_grad():
            generation_outputs = self.generate(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                images=[[image] for image in global_enc_images],
                max_new_tokens=max_tokens_new,
                num_beams=1,
                output_hidden_states=True,
                return_dict_in_generate=True
            )

            output_hidden_states = generation_outputs.hidden_states
            generated_output_ids = generation_outputs.sequences

            seg_token_mask = generated_output_ids == self.seg_token_idx

            # Process hidden states
            _, predicted_embeddings = self._process_hidden_states(
                output_hidden_states, seg_token_mask, None, infer=True
            )
            image_embeddings = self.get_sam_model_embs(grounding_enc_images)
            # Generate and post-process masks
            pred_masks = self._generate_and_postprocess_masks(
                predicted_embeddings, image_embeddings, resize_list, orig_sizes, infer=True
            )
        return generated_output_ids, pred_masks

build = class_from_function(MMMMForCausalLM.build, MMMMForCausalLM)

def from_pretrained(conf_path: PathLike, adapter_dir: PathLike) -> tuple[MMMMForCausalLM, MMMMTokenizer]:
    from jsonargparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_subclass_arguments(MMMMForCausalLM, 'model')
    args = parser.parse_args(['--model', str(conf_path)])
    args = parser.instantiate_classes(args)
    model: MMMMForCausalLM = args.model
    peft_model = PeftModel.from_pretrained(model, adapter_dir)
    model.set_peft_model(peft_model)
    tokenizer = model.tokenizer
    return model, tokenizer
