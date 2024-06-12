# modified from SegVol (from SAM)

from dataclasses import dataclass
from typing import Literal

import einops
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as nnf

from luolib.losses import zero_loss, sigmoid_focal_loss
from luolib.types import tuple3_t
from luolib.utils import pairwise_forward
from monai.data import box_pair_giou, convert_box_mode
from monai.data.box_utils import CenterSizeMode

from mmmm.models.loss import DiceFocalLoss
from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder

@dataclass
class InstanceSamOutput:
    """(batch size, num targets (varied), num queries, ...)"""
    masks_logits: list[torch.Tensor]
    masks_logits_low_res: list[torch.Tensor]
    boxes: list[torch.FloatTensor]
    disc_logit: list[torch.FloatTensor]

MATCH_NEGATIVE = -1
MATCH_UNCERTAIN = -2

class InstanceSamLoss(nn.Module):
    def __init__(
        self,
        *,
        mask_loss: DiceFocalLoss | None = None,
        use_neg_mask: bool,
        box_l1_weight: float,
        box_giou_weight: float,
        disc_weight: float,
        disc_focal_gamma: float,
        disc_focal_alpha: float | None = None,
        sem_only: bool = False,
    ):
        super().__init__()
        self.mask_loss = mask_loss
        self.use_neg_mask = use_neg_mask
        self.box_l1_weight = box_l1_weight
        self.box_giou_weight = box_giou_weight
        self.disc_weight = disc_weight
        self.disc_focal_gamma = disc_focal_gamma
        self.disc_focal_alpha = disc_focal_alpha
        self.sem_only = sem_only

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
        total = self.box_l1_weight * l1 + self.box_giou_weight * giou
        if return_dict:
            return {
                'l1': l1,
                'giou': giou,
                'total': total,
            }
        else:
            return total

    def disc_loss(
        self,
        input: torch.Tensor,
        label: torch.Tensor | bool,
        reduce_batch: bool = True,
        return_dict: bool = False,
        alpha: bool = True,
    ):
        # disc_loss = (bce_pos if pos else bce_neg)(input)
        if isinstance(label, bool):
            label = (torch.ones_like if label else torch.zeros_like)(input)

        disc_loss = sigmoid_focal_loss(
            input, label,
            self.disc_focal_gamma,
            self.disc_focal_alpha if alpha else None,
        )
        if reduce_batch:
            disc_loss = disc_loss.mean()
        total = self.disc_weight * disc_loss
        if return_dict:
            return {
                f'focal-{self.disc_focal_gamma:.1f}': disc_loss,
                'total': total,
            }
        else:
            return total

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
            if self.use_neg_mask:
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
        device = masks_logits.device
        match = torch.empty(num_queries, dtype=torch.int64, device=device)
        match[row] = torch.as_tensor(col, device=device)
        match[match >= num_pos] = MATCH_NEGATIVE
        match[match >= num_pos + num_neg] = MATCH_UNCERTAIN
        match[match >= 0] += offset
        return match

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
            # TODO: fix _valid_mask checking
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
            if not self.sem_only:
                # 2. instance part
                # drop the semantic part
                masks_logits_ds = masks_logits_ds[:, 1:]
                masks_logits = masks_logits[:, 1:]
                boxes_reg = boxes_reg[:, 1:]
                disc_logit = disc_logit.float()
                # downsample the mask for matching to save computation
                if masks_label is None:
                    masks_label_ds = None
                else:
                    masks_label_ds = nnf.interpolate(
                        masks_label[None].byte(), masks_logits_ds.shape[2:], mode='nearest-exact',
                    )[0].bool()

                num_uncertain_list = num_uncertain.tolist()
                offset_list = index_offsets[:, 0].tolist()
                match = torch.full((num_targets, masks_logits.shape[1]), MATCH_UNCERTAIN, dtype=torch.long, device=masks_logits.device)
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
                match_certain_mask = match != MATCH_UNCERTAIN
                loss += _accumulate(
                    self.disc_loss(
                        disc_logit[match_certain_mask],
                        match_pos_mask[match_certain_mask],
                        return_dict=True,
                    ),
                    'instance', 'disc',
                )
                if match_pos.shape[0] > 0:
                    with torch.no_grad():
                        _accumulate(
                            self.disc_loss(
                                disc_logit[match_pos_mask], True, return_dict=True, alpha=False,
                            ),
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
                    with torch.no_grad():
                        _accumulate(
                            self.disc_loss(
                                disc_logit[match_neg_mask], False, return_dict=True, alpha=False,
                            ),
                            'instance', 'disc-neg',
                        )
                    with torch.set_grad_enabled(self.use_neg_mask):
                        neg_mask_loss = _accumulate(
                            self.mask_loss(masks_logits[match_neg_mask][:, None], return_dict=True),
                            'instance', 'mask-neg',
                        )
                        if self.use_neg_mask:
                            loss += neg_mask_loss
        return loss, log_dict

    def forward(
        self,
        masks_logits: list[torch.Tensor],
        masks_logits_low_res: list[torch.Tensor],
        boxes_reg: list[torch.Tensor],
        disc_logit: list[torch.Tensor],
        masks_label: list[torch.BoolTensor | None],
        boxes_label: list[torch.Tensor],
        sem_masks: list[torch.BoolTensor | None],
        sem_boxes: list[torch.Tensor],
        index_offsets: list[torch.LongTensor],
        semantic: list[torch.BoolTensor],
        num_uncertain: list[torch.LongTensor],
    ):
        batch_size = len(masks_logits)
        loss_list = []
        log_dict = {}
        for i in range(batch_size):
            _loss, _log_dict = self._compute_loss(
                masks_logits[i], masks_logits_low_res[i],
                boxes_reg[i], disc_logit[i],
                masks_label[i], boxes_label[i],
                sem_masks[i], sem_boxes[i],
                index_offsets[i], semantic[i], num_uncertain[i],
            )
            loss_list.append(_loss)
            for k, v in _log_dict.items():
                log_dict.setdefault(k, []).append(v)
        loss = torch.stack(loss_list).mean()
        with torch.no_grad():
            log_dict = {
                k: torch.stack(v).mean()
                for k, v in log_dict.items()
            }
        return loss, log_dict

class InstanceSam(nn.Module):
    mask_threshold: float = 0.0

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
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
            nn.Linear(self.mask_embed_dim, 1),
        )

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
        masks_logits_low_res, masks_embeds = self.mask_decoder.forward(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(image_embeddings.shape[2:]),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            text_embedding=text_embedding,
            patch_size_z=patch_size_z,
        )
        return masks_logits_low_res, masks_embeds

    def forward(self, image: list[torch.Tensor], patch_size: list[tuple3_t[int]], text_embedding: list[torch.Tensor]):
        batch_size = len(image)
        image_embeddings: list[torch.Tensor] = self.image_encoder(image, patch_size)
        masks_logits_low_res = []
        boxes = []
        disc_logit = []
        for i in range(batch_size):
            _masks_logits_low_res, masks_embeds = self._predict_masks(
                text_embedding[i], image_embeddings[i], patch_size[i][0],
            )
            # calling sigmoid here to restrict range (CenterSizeMode), following DETR
            _boxes = self.box_head(masks_embeds).float().sigmoid()
            _disc_logit = einops.rearrange(self.disc_head(masks_embeds[:, 1:]), 'nt ni 1 -> nt ni')
            masks_logits_low_res.append(_masks_logits_low_res)
            boxes.append(_boxes)
            disc_logit.append(_disc_logit)

        masks_logits = [
            nnf.interpolate(m, image[i].shape[1:], mode='trilinear')
            for i, m in enumerate(masks_logits_low_res)
        ]
        output = InstanceSamOutput(masks_logits, masks_logits_low_res, boxes, disc_logit)
        return output
