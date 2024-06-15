import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import einops
import torch
from jsonargparse import class_from_function
from peft import PeftModel
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as nnf
from transformers.modeling_outputs import CausalLMOutputWithPast

from luolib.lightning import LightningModule
from luolib.losses import bce_neg, bce_pos, zero_loss
from luolib.types import PathLike, param3_t, tuple2_t, tuple3_t
from luolib.utils.misc import pairwise_forward
from monai.data import box_pair_giou, convert_box_mode
from monai.data.box_utils import CenterSizeMode

from mmmm.data.defs import Batch
from mmmm.tokenizer import MMMMTokenizer
from mmmm.utils import apply_prefix, get_lora_modules_default, get_lora_modules_finetune_all
from .cogvlm import CogVLMConfig, CogVLMForCausalLM
from .loss import DiceFocalLoss
from .segvol import InstanceSam
from .segvol.modeling.sam import Sam, InstanceSamLoss

__all__ = [
    'MMMMForCausalLM',
    'build',
]


@dataclass
class VisionArgs:
    pos_embed_shape: tuple3_t[int]
    pt_pos_embed_shape: tuple2_t[int] | None = None
    patch_size: param3_t[int] = 16

@dataclass
class VisualGroundingOutput:
    """(batch size, num targets, ...)"""
    masks_logits: list[torch.Tensor] = field(default_factory=list)
    masks_logits_ds: list[torch.Tensor] = field(default_factory=list)
    boxes: list[torch.FloatTensor] = field(default_factory=list)
    disc_logit: list[torch.FloatTensor] = field(default_factory=list)

MATCH_NEGATIVE = -1
MATCH_UNCERTAIN = -2

EPS = 1e-8

def _add_prefix(log_dict: dict[str, ...], prefix: str) -> dict[str, ...]:
    if prefix != '' and not prefix.endswith('/'):
        prefix += '/'
    return {f'{prefix}{k}': v for k, v in log_dict.items()}

class MMMMForCausalLM(CogVLMForCausalLM, LightningModule):
    tokenizer: MMMMTokenizer
    sam: Sam
    mask_loss: DiceFocalLoss | None
    isam: InstanceSam
    isam_loss: InstanceSamLoss

    @classmethod
    def build(
        cls,
        pretrained_model_name_or_path: str | os.PathLike | None,
        *args,
        vision_override: VisionArgs,
        tokenizer: MMMMTokenizer,
        torch_dtype: str | torch.dtype = 'auto',
        freeze_vision: bool = False,
        lm_loss_weight: float = 1.,
        sam: Sam | None = None,
        mask_loss: DiceFocalLoss | None = None,
        isam: InstanceSam | None = None,
        isam_loss: InstanceSamLoss | None = None,
    ):
        """make jsonargparse happy
        This works thanks to that AST does not support this (according to the debug information)
        TODO: refactor the construction of PreTrainedModel
        NOTE: mask loss handle the weights itself internally
        """
        self: MMMMForCausalLM = super().from_pretrained(
            pretrained_model_name_or_path,
            vision_override=vision_override,
            torch_dtype=torch_dtype,
        )
        self.resize_token_embeddings(len(tokenizer))
        self.tokenizer = tokenizer
        self.lm_loss_weight = lm_loss_weight
        self.sam = sam
        self.mask_loss = mask_loss
        self.isam = isam
        self.isam_loss = isam_loss
        if sam is not None:
            assert isam is not None
            sam.requires_grad_(False)
            sam.eval()
            isam.requires_grad_(False)
            isam.eval()
            assert sam.prompt_dim == isam.prompt_encoder
            self.vg_proj = nn.Sequential(
                nn.Linear(self.config.hidden_size, self.config.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.config.hidden_size, sam.prompt_dim),
            )
        if freeze_vision:
            self.model.vision.requires_grad_(False)
        self.model.config.lora_lang = not freeze_vision
        self.check_grad = False
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

    def get_lora_modules(self, prefix: str):
        # apply LoRA on VLM, fully finetune others
        target_modules, modules_to_save = get_lora_modules_default(self.model, apply_prefix(prefix, 'model'))
        for name, child in self.named_children():
            if name == 'model':
                continue
            c_modules_to_save = get_lora_modules_finetune_all(child, apply_prefix(prefix, name))
            modules_to_save.extend(c_modules_to_save)
        return target_modules, modules_to_save

    def _get_vg_prompts(self, token_ids: torch.LongTensor, hidden_states: torch.Tensor) -> list[torch.Tensor]:
        eop_mask: torch.BoolTensor = token_ids == self.tokenizer.eop_token_id  # type: ignore
        vg_hidden_states: torch.Tensor = self.vg_proj(hidden_states[eop_mask])
        return vg_hidden_states.split(eop_mask.sum(dim=-1).tolist())

    def visual_grounding(
        self,
        token_ids: torch.LongTensor,
        hidden_states: torch.Tensor,
        image: list[torch.Tensor],
        patch_size: list[tuple3_t[int]],
        instance_mask: list[bool] | None = None,
    ) -> VisualGroundingOutput:
        """
        Args:
            token_ids: generated token ids
            hidden_states: hidden states that generate tokens
        # Returns: for each sample in the batch:
        #     - predicted masks logits, the first one is semantic
        #     - predicted bounding boxes
        #     - discrimination logits for instances
        #     each one has a size of (num_vg, num_queries, ...)
        """
        vg_prompts = self._get_vg_prompts(token_ids, hidden_states)
        # image_embeddings: list[torch.Tensor] = self.sam.image_encoder(image, patch_size)
        masks_logits = self.sam(image, patch_size, vg_prompts)
        return masks_logits

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
        if self.sam is None:
            # pure VLM
            lm_loss = vlm_output.loss
            self.log('train/loss', lm_loss, sync_dist=True)
            return self.lm_loss_weight * lm_loss
        masks_logits = self.visual_grounding(
            # shift as suggested by GLaMM: https://github.com/mbzuai-oryx/groundingLMM/issues/16
            input_ids[:, 1:],
            vlm_output.hidden_states[-1][:, :-1],
            batch['grounding_image'],
            batch['patch_size'],
        )
        vg_loss, vg_log_dict = self._compute_vg_loss_batch(masks_logits, batch['masks'])
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
                **_add_prefix(vg_log_dict, 'train/vg'),
                **token_log_dict,
            },
            # the logging keys can be inconsistent, setting sync_dist=True can make DDP hang
            # sync_dist=True,
        )
        return loss

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
