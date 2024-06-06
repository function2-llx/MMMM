# modified from SegVol (from SAM)

from dataclasses import dataclass
from typing import Literal

import einops
import torch
from torch import nn
from torch.nn import functional as nnf

from luolib.losses import bce_neg, bce_pos, zero_loss
from luolib.types import tuple3_t
from monai.data import box_pair_giou, convert_box_mode
from monai.data.box_utils import CenterSizeMode
from monai.losses.focal_loss import sigmoid_focal_loss

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder
from ...loss import DiceFocalLoss

@dataclass
class InstanceSamOutput:
    """(batch size, num targets, num queries, ...)"""
    masks_logits: list[torch.Tensor]
    masks_logits_low_res: list[torch.Tensor]
    boxes: torch.FloatTensor
    disc_logit: torch.FloatTensor

MATCH_NEGATIVE = -1
MATCH_UNCERTAIN = -2

class InstanceSam(nn.Module):
    mask_threshold: float = 0.0

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        disc_focal_gamma: float = 2,
        disc_focal_alpha: float | None = None,
        mask_loss: DiceFocalLoss | None = None,
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder

        self.box_head = nn.Sequential(
            nn.Linear(self.mask_embed_dim, self.mask_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.mask_embed_dim, self.mask_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.mask_embed_dim, 6),
        )
        self.disc_head = nn.Sequential(
            nn.Linear(self.mask_embed_dim, self.mask_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.mask_embed_dim, self.prompt_dim),
        )
        self.focal_gamma = disc_focal_gamma
        self.focal_alpha = disc_focal_alpha

    @property
    def prompt_dim(self):
        return self.prompt_encoder.embed_dim

    @property
    def mask_embed_dim(self):
        return self.mask_decoder.transformer_dim

    @property
    def num_mask_tokens(self):
        return self.mask_decoder.num_mask_tokens

    def _predict_masks(
        self, text_embedding: torch.Tensor, image_embeddings: torch.Tensor, patch_size_z: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sparse_embeddings, dense_embeddings = self.prompt_encoder(image_embeddings.shape[2:], text_embedding=text_embedding)
        sparse_embeddings = sparse_embeddings.to(text_embedding.dtype)
        masks_logits_low_res, masks_embeds = self.mask_decoder(
            image_embeddings=image_embeddings,
            text_embedding=text_embedding,  # make SegVol happy
            image_pe=self.prompt_encoder.get_dense_pe(image_embeddings.shape[2:]),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            patch_size_z=patch_size_z,
        )
        return masks_logits_low_res, masks_embeds

    def forward(self, image: list[torch.Tensor], patch_size: list[tuple3_t[int]], text_embedding: torch.Tensor):
        batch_size = len(image)
        device = text_embedding.device
        image_embeddings: list[torch.Tensor] = self.image_encoder(image, patch_size)
        masks_logits_low_res = []
        masks_embeds = torch.empty(batch_size, self.num_mask_tokens, self.mask_embed_dim, dtype=torch.float, device=device)
        for i in range(batch_size):
            _masks_logits_low_res, masks_embeds[i] = self._predict_masks(
                text_embedding[i], image_embeddings[i], patch_size[i][0],
            )
            masks_logits_low_res.append(_masks_logits_low_res)
        masks_logits = [
            nnf.interpolate(m, image[i].shape[1:], mode='trilinear')
            for i, m in enumerate(masks_logits_low_res)
        ]
        # calling sigmoid here to restrict range (CenterSizeMode), following DETR
        boxes = self.box_head(masks_embeds).float().sigmoid()
        disc_logit = einops.einsum(self.disc_proj(masks_embeds), text_embedding, 'n nq c, n c -> n nq').float()
        output = InstanceSamOutput(masks_logits, masks_logits_low_res, boxes, disc_logit)
        return output

    def disc_loss(self, input: torch.Tensor, pos: bool, reduce_batch: bool = True, return_dict: bool = False):
        # disc_loss = (bce_pos if pos else bce_neg)(input)
        label = (torch.ones_like if pos else torch.zeros_like)(input)
        disc_loss = sigmoid_focal_loss(input, label, self.focal_gamma, self.focal_alpha)
        if reduce_batch:
            disc_loss = disc_loss.mean()
        total = self.disc_loss_weight * disc_loss
        if return_dict:
            return {
                f'focal-{self.focal_gamma:.1f}': disc_loss,
                'total': total,
            }
        else:
            return total

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

    def _compute_loss(
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

    def compute_loss_batch(
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
        loss_list = []
        log_dict = {}
        for i in range(batch_size):
            _loss, _vg_log_dict = self._compute_loss(
                masks_logits[i], masks_logits_ds[i],
                boxes_reg[i], disc_logit[i],
                masks_label[i], boxes_label[i],
                sem_masks[i], sem_boxes[i],
                index_offsets[i], semantic[i], num_uncertain[i],
            )
            loss_list.append(_loss)
            for k, v in _vg_log_dict.items():
                log_dict.setdefault(k, []).append(v)
        loss = torch.stack(loss_list).mean()
        with torch.no_grad():
            log_dict = {
                f'{prefix}/vg/{k}': torch.stack(v).mean()
                for k, v in log_dict.items()
            }
        return loss, log_dict
