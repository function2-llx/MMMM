import torch
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig

from luolib.lightning.cli import LightningCLI
from luolib.models import PlainConvUNetDecoder, UNetBackbone
from mmmm.models.mmmm import DiceFocalLoss

from base import DataModule, SemanticSegModel

class UNetForSemanticSeg(PreTrainedModel, SemanticSegModel):
    supports_gradient_checkpointing: bool = True

    def __init__(
        self,
        *,
        num_fg_classes: int = 15,
        backbone: UNetBackbone,
        decoder: PlainConvUNetDecoder,
        **kwargs,
    ):
        super().__init__(PretrainedConfig(), **kwargs)  # make HF happy
        self.backbone = backbone
        self.decoder = decoder
        self.seg_head = nn.Conv3d(self.decoder.layer_channels[0], num_fg_classes, 1)
        self.loss = DiceFocalLoss()

    def on_fit_start(self) -> None:
        super().on_fit_start()
        self.gradient_checkpointing_enable({'use_reentrant': False})

    def training_step(self, batch: dict, *args: ..., **kwargs: ...):
        image = batch['img']
        feature_maps: list[torch.Tensor] = self.backbone(image)
        feature_maps = self.decoder(feature_maps)
        masks_logits = self.seg_head(feature_maps[0])
        mask_loss = self.loss(masks_logits, batch['seg'])
        dice_loss = mask_loss['dice']
        self.log_dict({
            f'train/dice/{self.class_names[i]}': (1 - dice_loss[i]) * 100
            for i in range(dice_loss.shape[0])
        })
        mask_loss_reduced = {
            k: v.mean()
            for k, v in self.loss(masks_logits, batch['seg']).items()
        }
        loss = mask_loss_reduced['total']
        self.log_dict({
            'train/loss': loss,
            **{f'train/{k}_loss': v for k, v in mask_loss_reduced.items() if k != 'total'},
        })
        return loss

class CLI(LightningCLI):
    pass

def main():
    CLI(
        model_class=UNetForSemanticSeg,
        subclass_mode_data=False,
        datamodule_class=DataModule,
        subclass_mode_model=False,
    )

if __name__ == '__main__':
    main()
