from jsonargparse import class_from_function
import torch

from luolib.models.param import NoWeightDecayParameter
from mmmm.models import MMMMForCausalLM

class MMMMDebug(MMMMForCausalLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model.requires_grad_(False)
        self.lm_head.requires_grad_(False)
        self.cls_embed = NoWeightDecayParameter(torch.randn(16, self.sam_model.prompt_encoder.embed_dim))

    def training_step(self, batch: dict, *args, **kwargs):
        image = batch['image']
        vlm_inputs = batch['vlm_inputs']
        input_ids = vlm_inputs['input_ids']
        masks = batch['masks']
        seg_token_mask = self.tokenizer.create_seg_token_mask(input_ids)
        seg_hidden_states = [
            self.cls_embed[input_ids[i, seg_token_mask[i]] - self.tokenizer.seg_token_id_start]
            for i in range(seg_token_mask.shape[0])
        ]
        masks_logits = self._generate_and_postprocess_masks(image, seg_hidden_states)
        mask_loss = None if masks is None else self._compute_mask_loss(masks_logits, masks)
        loss = mask_loss['total']
        self.log_dict(
            {
                'train/mask_loss': mask_loss['total'],
                'train/loss': loss,
                **{f'train/{k}_loss': v for k, v in mask_loss.items() if k != 'total'},
            }
        )
        return loss

from_debug = class_from_function(MMMMDebug.from_pretrained, MMMMDebug, name='mmmm_debug_t')

class MMMMDebugSAM(MMMMForCausalLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model.requires_grad_(False)
        self.lm_head.requires_grad_(False)
        self.cls_embed = NoWeightDecayParameter(torch.randn(15, self.sam_model.prompt_encoder.embed_dim))

    def _setup_sam_requires_grad(self):
        pass

    def training_step(self, batch: dict, *args, **kwargs):
        image = batch['img']
        masks = batch['seg']
        seg_hidden_states = [self.cls_embed for _ in range(image.shape[0])]
        masks_logits = self._generate_and_postprocess_masks(image, seg_hidden_states)
        mask_loss = self._compute_mask_loss(masks_logits, masks)
        loss = mask_loss['total']
        self.log_dict(
            {
                'train/mask_loss': mask_loss['total'],
                'train/loss': loss,
                **{f'train/{k}_loss': v for k, v in mask_loss.items() if k != 'total'},
            }
        )
        return loss

from_debug_sam = class_from_function(MMMMDebugSAM.from_pretrained, MMMMDebugSAM, name='mmmm_debug_sam_t')
