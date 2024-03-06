import einops
import torch
from torch.nn import functional as nnf
from transformers import PreTrainedModel, PretrainedConfig

from luolib.lightning import LightningModule
from luolib.lightning.cli import LightningCLI
from luolib.models.param import NoWeightDecayParameter

from mmmm.models.mmmm import DiceFocalLoss
from mmmm.models.segvol import SamArgs, build_sam_vit_3d

from datamodule import DataModule

class SAMForSemanticSeg(PreTrainedModel, LightningModule):
    supports_gradient_checkpointing: bool = True

    def __init__(
        self,
        *,
        num_fg_classes: int = 15,
        sam: SamArgs,
        hidden_size: int = 768,
        **kwargs,
    ):
        super().__init__(PretrainedConfig(), **kwargs)  # make HF happy
        self.sam = build_sam_vit_3d(sam)
        self.cls_embeds = NoWeightDecayParameter(torch.randn(num_fg_classes, hidden_size))
        self.loss = DiceFocalLoss()

    def on_fit_start(self) -> None:
        super().on_fit_start()
        self.gradient_checkpointing_enable({'use_reentrant': False})

    def training_step(self, batch: dict, *args: ..., **kwargs: ...):
        image = batch['img']
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
        mask_loss = {
            k: v.mean()
            for k, v in self.loss(masks_logits, batch['seg']).items()
        }
        loss = mask_loss['total']
        self.log_dict({
            'train/loss': loss,
            **{f'train/{k}_loss': v for k, v in mask_loss.items() if k != 'total'},
        })
        return loss

class CLI(LightningCLI):
    pass

def main():
    CLI(
        model_class=SAMForSemanticSeg,
        subclass_mode_data=False,
        datamodule_class=DataModule,
        subclass_mode_model=False,
    )

if __name__ == '__main__':
    main()
