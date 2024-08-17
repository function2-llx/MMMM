from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any

import cytoolz
from jsonargparse import class_from_function
from lightning.pytorch.plugins import HalfPrecision, Precision
from peft import PeftModel
import torch
from torch import nn
from torch.nn import Module, functional as nnf
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import ModelOutput

from luolib.lightning import LightningModule
from luolib.lightning.peft import PeftMixin
from luolib.losses import zero_loss
from luolib.types import PathLike, param3_t, tuple2_t, tuple3_t
from mmmm.tokenizer import MMMMTokenizer
from mmmm.utils import apply_prefix, get_lora_modules_default, get_lora_modules_finetune_all
from mmmm.data.defs import mmmm_debug
from .cogvlm import CogVLMConfig, CogVLMForCausalLM
from .loss import DiceFocalLoss
from .segvol import InstanceSam
from .segvol.modeling.sam import InstanceSamLoss, InstanceSamOutput, Sam

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

def _add_prefix(log_dict: dict[str, ...], prefix: str) -> dict[str, ...]:
    if prefix != '' and not prefix.endswith('/'):
        prefix += '/'
    return {f'{prefix}{k}': v for k, v in log_dict.items()}

class MMMMForCausalLM(CogVLMForCausalLM, PeftMixin, LightningModule):
    tokenizer: MMMMTokenizer
    sam: Sam
    mask_loss: DiceFocalLoss | None
    # PEFT is so clever, they match suffix and does not fully support regex of "^"
    # so I have to change "isam" to "isam_model" to avoid clashing with "sam"; don't name "sam" with "sam_model"!
    isam_model: InstanceSam
    isam_loss: InstanceSamLoss
    vg_proj: nn.Module
    _supports_param_buffer_assignment: bool = False  # otherwise, custom parameter class will be wiped out

    def _freeze_sam_unused(self):
        sam = self.sam
        isam = self.isam_model
        for module in (
            sam.prompt_encoder.point_embeddings,
            sam.prompt_encoder.not_a_point_embed,
            sam.prompt_encoder.mask_downscaling,
            isam.prompt_encoder.point_embeddings,
            isam.prompt_encoder.not_a_point_embed,
            isam.prompt_encoder.mask_downscaling,
            isam.mask_decoder.output_upscaling,
            isam.mask_decoder.output_hypernetworks_mlps,
            isam.mask_decoder.txt_align_upscaled_embedding,
        ):
            module.requires_grad_(False)

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
        freeze_sam: bool = True,
        mask_loss: DiceFocalLoss | None = None,
        isam: InstanceSam | None = None,
        freeze_isam: bool = True,
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
        self.isam_model = isam
        self.isam_loss = isam_loss
        if sam is not None:
            assert isam is not None
            if freeze_sam:
                sam.requires_grad_(False)
                sam.eval()
            if freeze_isam:
                isam.requires_grad_(False)
                isam.eval()
            self._freeze_sam_unused()
            assert sam.prompt_dim == isam.prompt_dim
            self.vg_proj = nn.Sequential(
                nn.Linear(self.config.hidden_size, self.config.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.config.hidden_size, sam.prompt_dim),
            )
            isam_loss.mask_loss = mask_loss
        if freeze_vision:
            self.model.vision.requires_grad_(False)
        self.model.config.lora_lang = not freeze_vision
        return self

    def get_fp32_children(self) -> list[str]:
        return ['sam', 'isam_model', 'vg_proj']

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

    def _get_vg_prompts(
        self, token_ids: torch.LongTensor, hidden_states: torch.Tensor, prompt_mask: list[torch.BoolTensor | None],
    ) -> list[torch.Tensor]:
        eop_mask: torch.BoolTensor = token_ids == self.tokenizer.eop_token_id  # type: ignore
        vg_hidden_states: torch.Tensor = self.vg_proj(hidden_states[eop_mask])
        vg_prompts = vg_hidden_states.split(eop_mask.sum(dim=-1).tolist())
        ret = []
        for vg_prompts_, prompt_mask_ in zip(vg_prompts, prompt_mask):
            if prompt_mask_ is not None:
                vg_prompts_ = vg_prompts_[prompt_mask_]
            ret.append(vg_prompts_)
        return ret

    def visual_grounding(
        self,
        token_ids: torch.LongTensor,
        hidden_states: torch.Tensor,
        image: list[torch.Tensor],
        patch_size: list[tuple3_t[int]],
        prompt_mask: list[torch.BoolTensor | None],
        instance_mask: list[bool] | None,
    ):
        """
        Args:
            token_ids: generated token ids
            hidden_states: hidden states that generate tokens (usually the hidden states of the last layer)
            prompt_mask: only generate results for specific prompts
        Returns: for each sample in the batch:
            - predicted semantic masks logits
            - predicted bounding boxes
            - discrimination logits for instances
            each one has a size of (num_vg, num_queries, ...)
        """
        if instance_mask is None:
            raise NotImplementedError
        batch_size = len(image)
        vg_prompts = self._get_vg_prompts(token_ids, hidden_states, prompt_mask)
        if all(instance_mask):
            masks_logits = [None] * batch_size
        else:
            args_list = (args for i, args in enumerate(zip(image, patch_size, vg_prompts)) if not instance_mask[i])
            masks_logits = self.sam(*zip(*args_list))
            for i, instance_mask_ in enumerate(instance_mask):
                if instance_mask_:
                    masks_logits.insert(i, None)
        if any(instance_mask):
            args_list = (args for i, args in enumerate(zip(image, patch_size, vg_prompts)) if instance_mask[i])
            output: InstanceSamOutput = self.isam_model(*zip(*args_list))
            boxes, disc_logit = output.boxes, output.disc_logit
            for i, instance_mask_ in enumerate(instance_mask):
                if not instance_mask_:
                    boxes.insert(i, None)
                    disc_logit.insert(i, None)
        else:
            boxes = [None] * batch_size
            disc_logit = [None] * batch_size
        return masks_logits, boxes, disc_logit

    def _compute_vg_loss(
        self,
        masks_logits: list[torch.Tensor | None],
        boxes_reg: list[torch.Tensor | None],
        disc_logit: list[torch.Tensor | None],
        masks_label: list[torch.BoolTensor | None],
        boxes_label: list[torch.Tensor | None],
        index_offsets: list[torch.LongTensor | None],
    ):
        batch_size = len(masks_logits)
        loss_list = []
        log_dict = {}
        for i in range(batch_size):
            if boxes_label[i] is not None:
                if masks_label[i] is not None:
                    # instance segmentation is not supported yet
                    raise NotImplementedError
                loss_, log_dict_ = self.isam_loss.compute_loss(
                    boxes_reg[i].new_empty((*boxes_reg[i].shape[:2], 0, 0, 0)),  # dummy mask logits
                    boxes_reg[i].new_empty((*boxes_reg[i].shape[:2], 0, 0, 0)),
                    boxes_reg[i],
                    disc_logit[i],
                    None,
                    boxes_label[i],
                    index_offsets[i],
                )
                loss_ += zero_loss(masks_logits[i])
            elif masks_label[i] is not None and masks_label[i].shape[0] > 0:
                log_dict_ = self.mask_loss(masks_logits[i][:, None], masks_label[i][:, None], return_dict=True)
                loss_ = log_dict_.pop('total')
                loss_ += zero_loss(disc_logit[i] ,boxes_reg[i])
            else:
                loss_ = zero_loss(masks_logits[i], disc_logit[i], boxes_reg[i])
                log_dict_ = {}
            loss_list.append(loss_)
            for k, v in log_dict_.items():
                log_dict.setdefault(k, []).append(v)
        loss = torch.stack(loss_list).mean()
        if self.trainer.is_parallel or mmmm_debug():
            if all(m is None for m in masks_logits):
                loss += zero_loss(
                    self.sam(
                        [torch.zeros(3, 2, 32, 32, device=self.device)],
                        [(1, 16, 16)],
                        [torch.zeros(1, self.sam.prompt_dim, device=self.device)],
                    )[0],
                )
            if all(b is None for b in boxes_reg):
                output: InstanceSamOutput = self.isam_model(
                    [torch.zeros(3, 2, 32, 32, device=self.device)],
                    [(1, 16, 16)],
                    [torch.zeros(1, self.sam.prompt_dim, device=self.device)],
                )
                loss += zero_loss(output.disc_logit[0], output.boxes[0])

        with torch.no_grad():
            log_dict = {
                k: torch.stack(v).mean()
                for k, v in log_dict.items()
            }
        return loss, log_dict

    def on_fit_start(self) -> None:
        super().on_fit_start()
        # TODO: replace with checkpoint wrapper
        # https://github.com/pytorch/pytorch/blob/main/torch/distributed/algorithms/_checkpoint/checkpoint_wrapper.py
        self.gradient_checkpointing_enable({'use_reentrant': False})
        # NOTE: there may be some code setting lora.Linear.base_layer.eval(),
        #  however, let's keep it "training" to make DeepSpeed work, since it is just a linear layer
        #  and is not affected by the mode

    def training_step(self, batch: dict, *args, **kwargs):
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
        masks_logits, boxes, disc_logit = self.visual_grounding(
            # shift as suggested by GLaMM: https://github.com/mbzuai-oryx/groundingLMM/issues/16
            input_ids[:, 1:],
            vlm_output.hidden_states[-1][:, :-1].float(),
            batch['grounding_image'],
            batch['patch_size'],
            batch['vg_label_mask'],
            batch['instance_mask'],
        )
        vg_loss, vg_log_dict = self._compute_vg_loss(
            masks_logits,
            boxes,
            disc_logit,
            batch['masks'],
            batch['boxes'],
            batch['index_offsets'],
        )
        # NOTE: weight for VG is controlled internally
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

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: dict[str, ...],
        *args,
        **kwargs,
    ):
        position_ids = model_kwargs.pop('position_ids')
        model_kwargs = super()._update_model_kwargs_for_generation(outputs, model_kwargs, *args, **kwargs)
        position_ids = torch.cat([position_ids, position_ids[:, -1:] + 1], dim=1)  # `position_ids` will be corrected based on `input_ids` later in `prepare_inputs_for_generation`
        model_kwargs['position_ids'] = position_ids
        return model_kwargs

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
        # we only have access to `input_ids` here, not in `_update_model_kwargs_for_generation`, humor hf
        if past_key_values:
            keep_position = (input_ids[:, -2] == self.tokenizer.bop_token_id) | (input_ids[:, -1] == self.tokenizer.eop_token_id)
            position_ids[:, -1] -= keep_position.long()  # modify `position_ids` in-place, hope it keeps working
            input_ids = input_ids[:, -1:]
            token_type_ids = token_type_ids[:, -1:]
            position_ids = position_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update({
            "image": image,
            "token_type_ids": token_type_ids,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            'patch_size': patch_size,
            'pool_size': pool_size,
            "use_cache": kwargs.get("use_cache"),
        })
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

def from_pretrained(conf_path: PathLike, adapter_dir: PathLike, trainable: bool = False) -> tuple[MMMMForCausalLM, MMMMTokenizer]:
    from jsonargparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_subclass_arguments(MMMMForCausalLM, 'model')
    args = parser.parse_args(['--model', str(conf_path)])
    args = parser.instantiate_classes(args)
    model: MMMMForCausalLM = args.model
    peft_model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=trainable)
    model.set_peft_model(peft_model)
    tokenizer = model.tokenizer
    return model, tokenizer

class MyPrecision(Precision):
    def __init__(self):
        self._bf16 = HalfPrecision('bf16-true')

    def convert_input(self, data: dict) -> Any:
        fp16_mixed_keys = ['grounding_image', 'boxes']
        ret = {
            **self._bf16.convert_input(cytoolz.dissoc(data, *fp16_mixed_keys)),
            **{
                key: value for key in fp16_mixed_keys
                if (value := data.get(key)) is not None
            },
        }
        return ret

    def convert_module(self, module: Module) -> MMMMForCausalLM:
        assert isinstance(module, MMMMForCausalLM)
        # NOTE: module._dtype is not set since module.to is not called
        for param in module.parameters(recurse=False):
            param.to(dtype=self._bf16._desired_input_dtype)
        fp32_children = set(module.get_fp32_children())
        for name, child in module.named_children():
            if name not in fp32_children:
                self._bf16.convert_module(child)
        return module
