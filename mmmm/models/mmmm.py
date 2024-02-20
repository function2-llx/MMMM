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
from monai.losses import DiceFocalLoss

from .cogvlm import CogVLMConfig, CogVLMForCausalLM
from .segvol import SamArgs, build_sam_vit_3d
from .tokenizer import MMMMTokenizer

@dataclass
class VisionConf:
    pos_embed_shape: tuple3_t[int]
    pt_pos_embed_shape: tuple2_t[int] | None = None
    patch_size: int = 16

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
        **kwargs,
    ):
        """make jsonargparse happy"""
        return super().from_pretrained(
            pretrained_model_name_or_path, *args,
            lm_loss_weight=lm_loss_weight,
            mask_loss_weight=mask_loss_weight,
            vision_override=vision_override,
            tokenizer=tokenizer,
            sam_args=sam,
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
        self.mask_loss = DiceFocalLoss(sigmoid=True, reduction='none')
        self.lm_loss_weight = lm_loss_weight
        self.mask_loss_weight = mask_loss_weight
        self.seg_token_id = tokenizer.mask_close_id

        self.post_init()

    def _encode_single_image(self, image):
        return self.model.sam_model.image_encoder(image.unsqueeze(0))
    
    def forward(self, **kwargs):
        return super().forward(**kwargs) if "past_key_values" in kwargs else self.model_forward(**kwargs)

    def training_step(self, batch: dict, *args, **kwargs):
        image = batch['image']
        vlm_inputs = batch['vlm_inputs']
        vlm_output, masks_logits = self.model_forward(
            global_enc_image=image,
            grounding_enc_image=image,
            **vlm_inputs,
        )
        lm_loss = vlm_output.loss
        mask_loss = self._compute_mask_loss(masks_logits, batch['masks'])
        loss = lm_loss * self.lm_loss_weight + mask_loss * self.mask_loss_weight
        self.log_dict({
            'train/lm_loss': lm_loss,
            'train/mask_loss': mask_loss,
            'train/loss': loss,
        })
        return loss

    def model_forward(
        self,
        global_enc_image: torch.FloatTensor,
        grounding_enc_image: torch.FloatTensor,
        input_ids: torch.LongTensor,
        token_type_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        lm_targets: torch.LongTensor,
        attention_mask: torch.LongTensor,
    ):
        # VLM part
        # CogVLMForCausalLM.__call__ will not work
        vlm_output: CausalLMOutputWithPast = CogVLMForCausalLM.forward(
            self,
            image=global_enc_image,
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=lm_targets,
            output_hidden_states=True,
            return_dict=True,
        )
        # SAM part
        masks_logits = self._generate_and_postprocess_masks(
            grounding_enc_image,
            vlm_output.hidden_states[-1],
            input_ids == self.seg_token_id,
        )
        return vlm_output, masks_logits

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
        return masks_logits_list

    def _compute_mask_loss(self, masks_logits: list[torch.Tensor], masks_label: list[torch.BoolTensor]):
        assert (batch_size := len(masks_label)) == len(masks_logits)
        loss_list = []
        for i in range(batch_size):
            loss_list.append(
                einops.reduce(
                    self.mask_loss(masks_logits[i][None], masks_label[i][None]),
                    'n c ... -> c', 'mean',
                )
            )
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
