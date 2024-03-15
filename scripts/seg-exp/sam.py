import einops
import torch
from torch.nn import functional as nnf
from transformers import PreTrainedModel, PretrainedConfig

from luolib.lightning.cli import LightningCLI
from luolib.models.param import NoWeightDecayParameter
from mmmm.models.segvol import SamArgs, build_sam_vit_3d

from base import DataModuleDebug, SemanticSegModel

class SAMForSemanticSeg(PreTrainedModel, SemanticSegModel):
    supports_gradient_checkpointing: bool = True
    datamodule: DataModuleDebug

    def __init__(
        self,
        *,
        num_fg_classes: int = 15,
        sam: SamArgs,
        freeze_sam: bool = False,
        hidden_size: int = 768,
        empty_cache: bool = False,
        lambda_focal: float = 1.,
        **kwargs,
    ):
        super().__init__(PretrainedConfig(), lambda_focal=lambda_focal, **kwargs)  # make HF happy
        self.sam = build_sam_vit_3d(sam)
        if freeze_sam:
            self.requires_grad_(False)
        self.cls_embeds = NoWeightDecayParameter(torch.randn(num_fg_classes, hidden_size))
        self.empty_cache = empty_cache

    def on_fit_start(self) -> None:
        super().on_fit_start()
        self.gradient_checkpointing_enable({'use_reentrant': False})

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        sam = self.sam
        image_embeddings = sam.image_encoder(image)
        sparse_embeddings, dense_embeddings = sam.prompt_encoder(
            image_embeddings.shape[2:], text_embedding=self.cls_embeds,
        )
        masks_logits_list = []
        for i in range(image.shape[0]):
            masks_logits, _ = sam.mask_decoder(
                image_embeddings=image_embeddings[i:i + 1],
                text_embedding=self.cls_embeds,  # make SegVol happy
                image_pe=sam.prompt_encoder.get_dense_pe(image_embeddings.shape[2:]),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            masks_logits_list.append(masks_logits)
        masks_logits = einops.rearrange(masks_logits_list, 'n c 1 ... -> n c ...')
        masks_logits = nnf.interpolate(masks_logits, image.shape[2:], mode='trilinear')
        return masks_logits

    def forward_wow(self, image: torch.Tensor) -> list[torch.Tensor]:
        sam = self.sam
        image_embeddings = sam.image_encoder(image)
        masks_logits_list = []
        for i in range(image.shape[0]):
            text_embedding = self.cls_embeds
            sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                image_embeddings.shape[2:], text_embedding=text_embedding,
            )
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

    def _compute_mask_loss(self, masks_logits: list[torch.Tensor], masks_label: list[torch.BoolTensor]) -> dict[str, torch.Tensor]:
        assert (batch_size := len(masks_label)) == len(masks_logits)
        mask_loss_list: dict[str, list[torch.Tensor]] = {}
        for i in range(batch_size):
            sample_mask_loss: dict = self.loss(masks_logits[i][None], masks_label[i][None])
            sample_mask_loss.pop('dice-pos-batch')
            dice_pos_loss = sample_mask_loss.pop('dice-pos')
            mask_loss_list.setdefault('dice-pos', []).append(dice_pos_loss[dice_pos_loss.isfinite()])
            for k, v in sample_mask_loss.items():
                mask_loss_list.setdefault(k, []).append(v)

        mask_loss = {k: torch.cat(v).mean() for k, v in mask_loss_list.items()}
        return mask_loss

    def training_step(self, batch: dict, *args: ..., **kwargs: ...):
        image = batch['img']
        masks_logits_list: list[torch.Tensor] = self.forward_wow(image)
        mask_loss = self._compute_mask_loss(masks_logits_list, batch['seg'])
        loss = mask_loss['total']
        self.log_dict(
            {
                'train/mask_loss': mask_loss['total'],
                'train/loss': loss,
                **{f'train/{k}_loss': v for k, v in mask_loss.items() if k != 'total'},
            }
        )
        return loss

    def on_validation_epoch_end(self) -> None:
        if self.empty_cache:
            torch.cuda.empty_cache()
        super().on_validation_epoch_end()

class CLI(LightningCLI):
    pass

def main():
    CLI(
        model_class=SAMForSemanticSeg,
        subclass_mode_data=False,
        datamodule_class=DataModuleDebug,
        subclass_mode_model=False,
    )

if __name__ == '__main__':
    main()
