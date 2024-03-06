import einops
import torch
from torch import nn
from torch.nn import functional as nnf
from transformers import PreTrainedModel, PretrainedConfig

from luolib.lightning import LightningModule
from luolib.lightning.cli import LightningCLI
from luolib.models import PlainConvUNetDecoder, UNetBackbone
from luolib.models.param import NoWeightDecayParameter

from mmmm.models.mmmm import DiceFocalLoss
from mmmm.models.segvol import SamArgs, build_sam_vit_3d

from datamodule import DataModule

class UNetForSemanticSeg(PreTrainedModel, LightningModule):
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
        model_class=UNetForSemanticSeg,
        subclass_mode_data=False,
        datamodule_class=DataModule,
        subclass_mode_model=False,
    )

if __name__ == '__main__':
    main()
