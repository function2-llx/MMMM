from dataclasses import dataclass
import os

import einops
from jsonargparse import class_from_function
import torch
from torch.nn import functional as nnf
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from luolib.lightning import LightningModule
from luolib.types import tuple2_t, tuple3_t
from monai.losses import DiceFocalLoss as DiceFocalLossBase

from .cogvlm import CogVLMConfig, CogVLMForCausalLM
from .segvol import SamArgs, build_sam_vit_3d
from .tokenizer import MMMMTokenizer

__all__ = [
    'MMMMForCausalLM',
    'from_pretrained',
]

from mmmm.utils import apply_prefix, get_lora_modules_default, get_lora_modules_finetune_all

@dataclass
class VisionConf:
    pos_embed_shape: tuple3_t[int]
    pt_pos_embed_shape: tuple2_t[int] | None = None
    patch_size: int = 16

class DiceFocalLoss(DiceFocalLossBase):
    """reduce the results to channel only"""

    def __init__(self, **kwargs):
        super().__init__(reduction='none', sigmoid=True, **kwargs)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dice_loss = einops.reduce(self.dice(input, target), 'n c ... -> c', 'mean')
        focal_loss = einops.reduce(self.focal(input, target), 'n c ... -> c', 'mean')
        total_loss: torch.Tensor = self.lambda_dice * dice_loss + self.lambda_focal * focal_loss
        return total_loss

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

class MMMMForCausalLM(CogVLMForCausalLM, LightningModule):
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike | None,
        *args,
        lm_loss_weight: float = 1.,
        mask_loss_weight: float = 1.,
        vision_override: VisionConf,
        tokenizer: MMMMTokenizer,
        sam: SamArgs,
        torch_dtype: str | torch.dtype = 'auto',
        **kwargs,
    ):
        """make jsonargparse happy
        This works thanks to that AST does not support this (according to the debug information)
        TODO: refactor the construction of PreTrainedModel
        """
        return super().from_pretrained(
            pretrained_model_name_or_path, *args,
            lm_loss_weight=lm_loss_weight,
            mask_loss_weight=mask_loss_weight,
            vision_override=vision_override,
            tokenizer=tokenizer,
            sam_args=sam,
            torch_dtype=torch_dtype,
            **kwargs,
        )

    def __init__(
        self,
        vlm_config: CogVLMConfig,
        *,
        lm_loss_weight: float,
        mask_loss_weight: float,
        vision_override: VisionConf,
        tokenizer: MMMMTokenizer,
        sam_args: SamArgs,
        **kwargs,
    ):
        # adapt vision config
        vision_config: dict = vlm_config.vision_config
        vision_config.update(vars(vision_override))
        super().__init__(vlm_config, **kwargs)
        self.sam_model = build_sam_vit_3d(sam_args)
        self.seg_proj = nn.Sequential(
            nn.Linear(vlm_config.hidden_size, vlm_config.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(vlm_config.hidden_size, self.sam_model.prompt_encoder.embed_dim),
        )
        self.mask_loss = DiceFocalLoss()
        self.lm_loss_weight = lm_loss_weight
        self.mask_loss_weight = mask_loss_weight
        self.seg_token_id = tokenizer.seg_token_id

        self._setup_sam_requires_grad()

        self.post_init()

    def _setup_sam_requires_grad(self):
        # make DDP work
        # if this dissatisfies you, go and construct it
        sam = self.sam_model
        sam.prompt_encoder.point_embeddings.requires_grad_(False)
        sam.prompt_encoder.not_a_point_embed.requires_grad_(False)
        sam.prompt_encoder.mask_downscaling.requires_grad_(False)
        sam.mask_decoder.iou_prediction_head.requires_grad_(False)

    def get_lora_modules(self, prefix: str):
        # apply LoRA on VLM, fully finetune others
        target_modules, modules_to_save = get_lora_modules_default(self.model, apply_prefix(prefix, 'model'))
        for name, child in self.named_children():
            if name == 'model':
                continue
            c_target_modules, c_modules_to_save = get_lora_modules_finetune_all(child, apply_prefix(prefix, name))
            target_modules.extend(c_target_modules)
            modules_to_save.extend(c_modules_to_save)
        return target_modules, modules_to_save

    def forward(
        self,
        *,
        global_enc_image: torch.FloatTensor,
        grounding_enc_image: torch.FloatTensor,
        masks: list[torch.BoolTensor] | None = None,
        input_ids: torch.LongTensor,
        lm_targets: torch.LongTensor | None = None,
        **kwargs,
    ) -> MMMMOutputWithPast:
        """
        TODO:
         - make it compatible with HF interface, e.g., adapt return_dict
         - support cache for segmentation output
        """
        # VLM part
        vlm_output: CausalLMOutputWithPast = CogVLMForCausalLM.forward(
            self,
            input_ids=input_ids,
            image=global_enc_image,
            labels=lm_targets,
            return_dict=True,
            output_hidden_states=True,
            **kwargs,
        )
        # SAM part
        masks_logits = self._generate_and_postprocess_masks(
            grounding_enc_image,
            vlm_output.hidden_states[-1],
            input_ids == self.seg_token_id,
        )
        mask_loss = None if masks is None else self._compute_mask_loss(masks_logits, masks)
        return MMMMOutputWithPast(
            lm_loss=vlm_output.loss,
            lm_logits=vlm_output.logits,
            past_key_values=vlm_output.past_key_values,
            hidden_states=vlm_output.hidden_states,
            attentions=vlm_output.attentions,
            masks_logits=masks_logits,
            mask_loss=mask_loss,
        )

    def training_step(self, batch: dict, *args, **kwargs):
        image = batch['image']
        output: MMMMOutputWithPast = self(
            global_enc_image=image,
            grounding_enc_image=image,
            masks=batch['masks'],
            **batch['vlm_inputs'],
            use_cache=False,
        )
        loss = output.lm_loss * self.lm_loss_weight + output.mask_loss * self.mask_loss_weight
        self.log_dict({
            'train/lm_loss': output.lm_loss,
            'train/mask_loss': output.mask_loss,
            'train/loss': loss,
        })
        return loss

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
            torch.cuda.empty_cache()

        output_hidden_states = torch.cat(output_hidden_states, dim=0)
        output_hidden_states = [output_hidden_states]
        return output_hidden_states

    def _generate_and_postprocess_masks(
        self,
        image: torch.Tensor,
        hidden_states: torch.Tensor,
        seg_token_mask: torch.BoolTensor,
    ) -> list[torch.Tensor]:
        sam = self.sam_model
        # TODO: check grad vs no_grad
        image_embeddings = sam.image_encoder(image)
        masks_logits_list = []
        for i in range(hidden_states.shape[0]):
            if seg_token_mask[i].any():
                text_embedding = self.seg_proj(hidden_states[i, seg_token_mask[i]])
                sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                    image_embeddings.shape[2:], text_embedding=text_embedding,
                )
                sparse_embeddings = sparse_embeddings.to(text_embedding.dtype)
                masks_logits, _ = sam.mask_decoder(
                    image_embeddings=image_embeddings[i:i + 1],
                    text_embedding=text_embedding,  # make SegVol happy
                    image_pe=sam.prompt_encoder.get_dense_pe(image_embeddings.shape[2:]),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
                masks_logits = nnf.interpolate(masks_logits, image.shape[2:], mode='trilinear')
                masks_logits_list.append(masks_logits[:, 0])
            else:
                masks_logits_list.append(image.new_empty(0, *image.shape[2:]))
        return masks_logits_list

    def _compute_mask_loss(self, masks_logits: list[torch.Tensor], masks_label: list[torch.BoolTensor]):
        assert (batch_size := len(masks_label)) == len(masks_logits)
        loss_list = []
        for i in range(batch_size):
            loss_list.append(self.mask_loss(masks_logits[i][None], masks_label[i][None]))
        loss = torch.cat(loss_list).mean()
        return loss

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

from_pretrained = class_from_function(MMMMForCausalLM.from_pretrained, MMMMForCausalLM)