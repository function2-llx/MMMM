from dataclasses import dataclass
import os
from pathlib import Path
from typing import Literal, Self

from jsonargparse import class_from_function
import torch
from torch import nn
from torch.nn import functional as nnf
from transformers.modeling_outputs import CausalLMOutputWithPast

from luolib.lightning import LightningModule
from luolib.types import param3_t, tuple2_t, tuple3_t

from mmmm.data.defs import Batch
from mmmm.tokenizer import MMMMTokenizer
from mmmm.utils import apply_prefix, get_lora_modules_default, get_lora_modules_finetune_all
from .cogvlm import CogVLMConfig, CogVLMForCausalLM
from .loss import DiceFocalLoss
from .segvol import Sam, SamArgs, build_sam_vit_3d

__all__ = [
    'MMMMForCausalLM',
    'build_finetune',
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
class SlidingWindow:
    patch_size: tuple3_t[int]
    batch_size: int
    overlap: float = 0.5

# class EmbeddingWrapper(nn.Module):
#     def __init__(self, base: nn.Embedding, new_size: int):
#         self.base = base
#         self.new = nn.Embedding(new_size)
#
#     def forward(self, input: torch.Tensor) -> torch.Tensor:
#         return F.embedding(
#             input, self.weight, self.padding_idx, self.max_norm,
#             self.norm_type, self.scale_grad_by_freq, self.sparse)

class MMMMForCausalLM(CogVLMForCausalLM, LightningModule):
    tokenizer: MMMMTokenizer
    sam_model: Sam
    mask_loss: DiceFocalLoss | None

    @classmethod
    def build_finetune(
        cls,
        pretrained_model_name_or_path: str | os.PathLike | None,
        *args,
        lm_loss_weight: float = 1.,
        mask_loss_weight: float = 1.,
        vision_override: VisionArgs,
        tokenizer: MMMMTokenizer,
        sam: SamArgs,
        torch_dtype: str | torch.dtype = 'auto',
        mask_loss: DiceFocalLoss | None = None,
        val_sw: SlidingWindow | None = None,
        seg_hidden_layer: Literal[0, -1] = -1,
        lora_lang: bool = True,
    ):
        """make jsonargparse happy
        This works thanks to that AST does not support this (according to the debug information)
        TODO: refactor the construction of PreTrainedModel
        Args:
            lora_lang: whether to fine-tune language weights
        """
        self: Self = super().from_pretrained(
            pretrained_model_name_or_path,
            vision_override=vision_override,
            torch_dtype=torch_dtype,
        )
        self.resize_token_embeddings(len(tokenizer))
        self.sam_model = build_sam_vit_3d(sam)
        self._setup_sam_requires_grad()
        self.tokenizer = tokenizer
        self.lm_loss_weight = lm_loss_weight
        self.mask_loss_weight = mask_loss_weight
        self.mask_loss = mask_loss
        self.seg_proj = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.config.hidden_size, self.sam_model.prompt_dim),
        )
        self.val_sw = val_sw
        self.seg_hidden_layer = seg_hidden_layer
        self.model.config.lora_lang = lora_lang
        # make the `from_pretrained` interface consistent
        # since `resize_token_embeddings` will create new modules without preserving original attributes
        # self.sam_model.requires_grad_(False)
        self.model.tokenizer = tokenizer
        self.eval()
        return self

    def _init_weights(self, module):
        """Let's happily do nothing (necessary to make SAM pre-trained weights survive)"""

    def __init__(self, vlm_config: CogVLMConfig, *, vision_override: VisionArgs, **kwargs):
        # adapt vision config
        vision_config: dict = vlm_config.vision_config
        vision_config.update(vars(vision_override))
        super().__init__(vlm_config, **kwargs)

    def load_default_adapter(self, ckpt_dir: Path):
        self.peft_model.load_adapter(str(ckpt_dir / 'adapter'), 'default')

    def on_fit_start(self) -> None:
        super().on_fit_start()
        # model.train() will not be called anymore in fit loop: https://github.com/Lightning-AI/pytorch-lightning/pull/18951
        # and from_pretrained call .eval() by default https://github.com/huggingface/transformers/blob/v4.38.2/src/transformers/modeling_utils.py#L3523-L3524
        # also see: https://huggingface.co/docs/transformers/v4.38.2/en/main_classes/model#transformers.PreTrainedModel (search "model.eval()")
        self.train()
        # self.model will be adapted by LoRA; FIXME: but what about LoRA?
        self.model.eval()
        self.gradient_checkpointing_enable({'use_reentrant': False})

    def _setup_sam_requires_grad(self):
        # make DDP work
        # if this dissatisfies you, go and construct it
        sam = self.sam_model
        sam.prompt_encoder.point_embeddings.requires_grad_(False)
        sam.prompt_encoder.not_a_point_embed.requires_grad_(False)
        sam.prompt_encoder.mask_downscaling.requires_grad_(False)
        sam.mask_decoder.iou_prediction_head.requires_grad_(False)
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

    def forward_with_sam(
        self,
        *,
        global_enc_image: list[torch.Tensor],
        grounding_enc_image: list[torch.Tensor | None],
        patch_size: list[tuple3_t[int]],
        pool_size: list[tuple3_t[int]],
        mask: list[torch.BoolTensor] | None = None,
        bbox: list[torch.Tensor] | None = None,
        input_ids: torch.LongTensor,
        lm_targets: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple[MMMMOutputWithPast, dict]:
        """
        TODO:
         - make it compatible with HF interface, e.g., adapt return_dict
         - support cache for segmentation output
        """
        # VLM part
        vlm_output: CausalLMOutputWithPast = CogVLMForCausalLM.forward(
            self,
            input_ids,
            image=global_enc_image,
            patch_size=patch_size,
            pool_size=pool_size,
            labels=lm_targets,
            return_dict=True,
            output_hidden_states=True,
            **kwargs,
        )
        # SAM part
        match self.seg_hidden_layer:
            case 0:
                # for debugging
                seg_layer_hidden_states = vlm_output.hidden_states[0]
                seg_token_mask = self.tokenizer.create_seg_token_mask(input_ids)
            case -1:
                # shift as suggested by GLaMM: https://github.com/mbzuai-oryx/groundingLMM/issues/16
                seg_layer_hidden_states = vlm_output.hidden_states[-1][:, :-1]
                seg_token_mask = self.tokenizer.create_seg_token_mask(input_ids[:, 1:])
            case _:
                raise ValueError
        seg_hidden_states = [
            seg_layer_hidden_states[i, seg_token_mask[i]]
            for i in range(seg_layer_hidden_states.shape[0])
        ]
        # for i in range(seg_layer_hidden_states.shape[0]):
        #     if seg_token_mask[i].any():
        #         seg_hidden_states.append(seg_layer_hidden_states[i, seg_token_mask[i]])
        #     else:
        #         # create a dummy to make DDP works
        #         seg_hidden_states.append(seg_layer_hidden_states[i, 0:1])
        masks_logits = self._generate_and_postprocess_masks(grounding_enc_image, patch_size, seg_hidden_states)
        if mask is None:
            mask_loss = None
            sam_log_dict = {}
        else:
            mask_loss, sam_log_dict = self._compute_mask_loss(masks_logits, mask)
        output = MMMMOutputWithPast(
            lm_loss=vlm_output.loss,
            lm_logits=vlm_output.logits,
            past_key_values=vlm_output.past_key_values,
            hidden_states=vlm_output.hidden_states,
            attentions=vlm_output.attentions,
            masks_logits=masks_logits,
            mask_loss=mask_loss,
        )
        return output, sam_log_dict

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
        image_features_mask: torch.BoolTensor | None = None,
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
            model_inputs = {"input_ids": input_ids, 'image_features_mask': image_features_mask}

        model_inputs.update(
            {
                "image": image,
                "token_type_ids": token_type_ids,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "attention_mask": attention_mask,
                "use_cache": kwargs.get("use_cache"),
            }
        )
        return model_inputs

    def training_step(self, batch: Batch, batch_idx: int, *args, **kwargs):
        vlm_inputs = batch['vlm_inputs']
        output, sam_log_dict = self.forward_with_sam(
            global_enc_image=batch['image'],
            grounding_enc_image=batch['grounding_image'],
            patch_size=batch['patch_size'],
            pool_size=batch['pool_size'],
            mask=batch['mask'],
            **vlm_inputs,
            use_cache=False,
        )
        loss = output.lm_loss * self.lm_loss_weight + output.mask_loss * self.mask_loss_weight
        # make some custom log
        lm_targets = vlm_inputs['lm_targets']
        lm_loss_dict = {
            f'train/token-lm/{name}_loss': nnf.cross_entropy(output.lm_logits[token_mask], lm_targets[token_mask])
            for name, token_mask in {
                'seg': self.tokenizer.create_seg_token_mask(lm_targets),
                'bop': lm_targets == self.tokenizer.bop_token_id,
                'eop': lm_targets == self.tokenizer.eop_token_id,
            }.items()
            if token_mask.any()
        }
        self.log_dict(
            {
                'train/lm_loss': output.lm_loss,
                **lm_loss_dict,
                'train/mask_loss': output.mask_loss,
                'train/loss': loss,
                **{f'train/mask-{k}_loss': v for k, v in sam_log_dict.items()},
            },
            # the logging keys are inconsistent, setting sync_dist=True will make DDP hang
            # sync_dist=True,
        )
        return loss

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

    def _generate_and_postprocess_masks(
        self,
        image_list: list[torch.Tensor],
        patch_size_list: list[tuple3_t[int]],
        seg_hidden_states: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        sam = self.sam_model
        # TODO: check grad vs no_grad
        image_embedding_list: list[torch.Tensor] = sam.image_encoder(image_list, patch_size_list)
        masks_logits_list = []
        for i, (image, image_embedding, patch_size) in enumerate(zip(image_list, image_embedding_list, patch_size_list)):
            text_embedding = self.seg_proj(seg_hidden_states[i])
            sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                image_embedding.shape[2:], text_embedding=text_embedding,
            )
            sparse_embeddings = sparse_embeddings.to(text_embedding.dtype)
            masks_logits, _ = sam.mask_decoder(
                image_embeddings=image_embedding_list[i],
                text_embedding=text_embedding,  # make SegVol happy
                image_pe=sam.prompt_encoder.get_dense_pe(image_embedding.shape[2:]),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                patch_size_z=patch_size[0],
            )
            masks_logits = nnf.interpolate(masks_logits, image.shape[1:], mode='trilinear')
            masks_logits_list.append(masks_logits[:, 0])
        return masks_logits_list

    def _compute_mask_loss(self, masks_logits: list[torch.Tensor], masks_label: list[torch.BoolTensor]):
        assert (batch_size := len(masks_label)) == len(masks_logits)
        mask_loss_list: dict[str, list[torch.Tensor]] = {}
        for i in range(batch_size):
            sample_mask_loss: dict = self.mask_loss(masks_logits[i][None], masks_label[i][None])
            for k, v in sample_mask_loss.items():
                mask_loss_list.setdefault(k, []).append(v)
        # loss for optimization
        loss = torch.stack(mask_loss_list['total']).nan_to_num().mean()
        log_dict = {
            k: v for k, _list in mask_loss_list.items()
            if not torch.isnan(v := torch.stack(_list).nanmean())
        }
        return loss, log_dict

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

build_finetune = class_from_function(MMMMForCausalLM.build_finetune, MMMMForCausalLM)
