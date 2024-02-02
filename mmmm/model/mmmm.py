from monai.losses import DiceFocalLoss
import torch
import torch.nn as nn
from typing import List
import torch.nn.functional as F

from model.segment_anything_volumetric.build_sam import build_sam_vit_3d
from model.cogvlm.modeling_cogvlm import CogVLMForCausalLM, CogVLMModel


class MMMMBaseModel:
    def __init__(self, config, **kwargs):
        super(MMMMBaseModel, self).__init__(config)
        self.config = config
        self.sam_pretrained = kwargs.get("sam_pretrained", None)

        self._initialize_sam_model()
        self._initialize_text_projection_layer()


    def _initialize_sam_model(self):
        self.sam_model = build_sam_vit_3d(self.sam_pretrained)

    def _initialize_text_projection_layer(self):
        in_dim, out_dim = self.config.hidden_size, self.config.out_dim
        text_projection_layer = [nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True), nn.Linear(in_dim, out_dim),
            nn.Dropout(0.0), ]
        self.text_projection_layer = nn.ModuleList([nn.Sequential(*text_projection_layer)])
        self.text_projection_layer.train()


class MMMMModel(MMMMBaseModel, CogVLMModel):
    def __init__(self, config, **kwargs):
        super(MMMMModel, self).__init__(config, **kwargs)


class MMMMForCausalLM(CogVLMForCausalLM):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.config = config

        self._setup_model_configurations(config, kwargs)
        self._setup_model_metrics()
        self.post_init()

    def _setup_model_configurations(self, kwargs):
        self.ce_loss_weight = kwargs.get("ce_loss_weight", None)
        self.loss = DiceFocalLoss(
            sigmoid=True,
            reduction="none",
            lambda_dice=kwargs.get("dice_loss_weight", None),
            lambda_focal=kwargs.get("focal_loss_weight", None),
        )
        self.seg_token_idx = kwargs.get("seg_token_idx", None)

    def get_sam_model_embs(self, pixel_values):
        with torch.no_grad():
            return torch.cat([self._encode_single_image(img) for img in pixel_values], dim=0)

    def _encode_single_image(self, image):
        torch.cuda.empty_cache()
        return self.model.sam_model.image_encoder(image.unsqueeze(0))
    
    def forward(self, **kwargs):
        return super().forward(**kwargs) if "past_key_values" in kwargs else self.model_forward(**kwargs)

    def model_forward(
        self,
        global_enc_images: torch.FloatTensor,
        grounding_enc_images: torch.FloatTensor,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        offset: torch.LongTensor,
        masks_list: List[torch.FloatTensor],
        label_list: List[torch.Tensor],
        resize_list: List[tuple],
        inference: bool = False
    ):
        # Extract grounding encoder image embeddings
        image_embeddings = self.get_sam_model_embs(grounding_enc_images)
        assert image_embeddings.shape[0] == len(offset) - 1

        # Create segmentation token mask
        seg_token_mask = self._create_seg_token_mask(input_ids)

        # Handle inference or training paths
        if inference:
            output_hidden_states = self._inference_path(input_ids, global_enc_images, attention_masks)
        else:
            output, output_hidden_states = self._training_path(
                global_enc_images, input_ids, labels, attention_masks, offset
            )

        # Process hidden states
        hidden_states, pred_embeddings = self._process_hidden_states(output_hidden_states, seg_token_mask, offset)

        # Generate and post-process masks
        pred_masks = self._generate_and_postprocess_masks(
            pred_embeddings, image_embeddings, resize_list, label_list
        )

        if inference:
            return {"pred_masks": pred_masks, "gt_masks": masks_list, }

        # Calculate losses
        return self._compute_losses(pred_masks, masks_list, output)
    
    def _create_seg_token_mask(self, input_ids):
        mask = input_ids[:, 1:] == self.seg_token_idx
        return torch.cat(
            [torch.zeros((mask.shape[0], 575)).bool().cuda(), mask, torch.zeros((mask.shape[0], 1)).bool().cuda()],
            dim=1
        )

    def _inference_path(self, input_ids, global_enc_images, attention_masks):
        length = input_ids.shape[0]
        global_enc_images_extended = global_enc_images.expand(length, -1, -1, -1).contiguous()

        # Process and return inference output
        output_hidden_states = []
        for i in range(input_ids.shape[0]):
            output_i = super().forward(
                images=global_enc_images_extended[i:i + 1],
                attention_mask=attention_masks[i:i + 1],
                input_ids=input_ids[i:i + 1],
                output_hidden_states=True
            )
            output_hidden_states.append(output_i.hidden_states)
            torch.cuda.empty_cache()

        output_hidden_states = torch.cat(output_hidden_states, dim=0)
        output_hidden_states = [output_hidden_states]
        return output_hidden_states

    def _training_path(self, global_enc_images, input_ids, labels, attention_masks, offset):
        global_enc_images = self._prepare_global_enc_image(global_enc_images, offset)

        output = super().forward(
            images=global_enc_images,
            attention_mask=attention_masks,
            input_ids=input_ids,
            labels=labels,
            output_hidden_states=True
        )
        output_hidden_states = output.hidden_states
        return output, output_hidden_states

    def _prepare_global_enc_image(self, global_enc_image, offset):
        global_enc_image_list = []
        for i in range(len(offset) - 1):
            start_i, end_i = offset[i], offset[i + 1]
            global_enc_image_i = global_enc_image[i].unsqueeze(0).expand(end_i - start_i, -1, -1, -1).contiguous()
            global_enc_image_list.append(global_enc_image_i)
        return torch.cat(global_enc_image_list, dim=0)

    def _process_hidden_states(self, output_hidden_states, seg_token_mask, offset, infer=False):
        hidden_states = [self.model.text_hidden_fcs[0](output_hidden_states[-1])]
        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
        pred_embeddings = last_hidden_state[seg_token_mask]
        seg_token_counts = seg_token_mask.int().sum(-1)

        seg_token_offset = seg_token_counts.cumsum(-1)
        seg_token_offset = torch.cat([torch.zeros(1).long().cuda(), seg_token_offset], dim=0)
        if not infer:
            seg_token_offset = seg_token_offset[offset]

        pred_embeddings_list = []
        for i in range(len(seg_token_offset) - 1):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            pred_embeddings_list.append(pred_embeddings[start_i:end_i])
        return hidden_states, pred_embeddings_list

    def _generate_and_postprocess_masks(self, pred_embeddings, image_embeddings, resize_list, label_list, infer=False):
        pred_masks = []
        for i, pred_embedding in enumerate(pred_embeddings):
            sparse_embeddings, dense_embeddings = self.model.sam_model.prompt_encoder(
                points=None, boxes=None, masks=None, text_embeds=pred_embedding.unsqueeze(1)
            )
            sparse_embeddings = sparse_embeddings.to(pred_embedding.dtype)
            low_res_masks, _ = self.model.sam_model.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),
                image_pe=self.model.sam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings, dense_prompt_embeddings=dense_embeddings,
                multimask_output=False
            )
            orig_size = label_list[i].shape if not infer else label_list[i]
            # During inference, we have original size list in place of label list
            pred_mask = self.model.sam_model.postprocess_masks(
                low_res_masks, input_size=resize_list[i], original_size=orig_size, )
            pred_masks.append(pred_mask[:, 0])
        return pred_masks

    def _compute_losses(self, pred_masks, masks_list, output):
        # Initialize loss components
        ce_loss = output.loss * self.ce_loss_weight
        mask_loss = torch.tensor(0.0, device=ce_loss.device)
        num_masks = 0

        # Iterate over batch and compute mask-related losses
        for batch_idx, pred_mask in enumerate(pred_masks):
            if pred_mask.numel() > 0:  # Ensure pred_mask is not empty
                gt_mask = masks_list[batch_idx]
                # Resize gt_mask to match pred_mask if needed
                if gt_mask.shape[0] != pred_mask.shape[0]:
                    gt_mask = gt_mask[:pred_mask.shape[0]]

                assert gt_mask.shape[0] == pred_mask.shape[0], f"Shape mismatch: gt_mask {gt_mask.shape}, pred_mask {pred_mask.shape}"

                mask_loss += self.loss(pred_mask.unsqueeze(0), gt_mask.unsqueeze(0))
                num_masks += gt_mask.shape[0]

        mask_loss /= (num_masks + 1e-8)

        # Aggregate all loss components
        total_loss = ce_loss + mask_loss
        return {
            "loss": total_loss,
            "ce_loss": ce_loss,
            "mask_loss": mask_loss
        }

    def evaluate(self, global_enc_images, grounding_enc_images, input_ids, resize_list, orig_sizes, max_tokens_new=32):
        with torch.no_grad():
            generation_outputs = self.generate(
                images=global_enc_images, input_ids=input_ids, max_new_tokens=max_tokens_new,
                num_beams=1, output_hidden_states=True, return_dict_in_generate=True, )

            output_hidden_states = generation_outputs.hidden_states
            generated_output_ids = generation_outputs.sequences

            seg_token_mask = generated_output_ids[:, 1:] == self.seg_token_idx
            # Adjusting for IMAGE_TOKEN_INDEX (assuming single image at start)
            seg_token_mask = torch.cat(
                [torch.zeros((seg_token_mask.shape[0], 575), dtype=torch.bool).cuda(), seg_token_mask],
                dim=1
            )
            # Process hidden states
            hidden_states, predicted_embeddings = self._process_hidden_states(
                output_hidden_states, seg_token_mask, None, infer=True
            )
            image_embeddings = self.get_sam_model_embs(grounding_enc_images)
            # Generate and post-process masks
            pred_masks = self._generate_and_postprocess_masks(
                predicted_embeddings, image_embeddings, resize_list, orig_sizes, infer=True
            )
        return generated_output_ids, pred_masks