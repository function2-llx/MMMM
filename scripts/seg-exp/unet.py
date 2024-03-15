import torch
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig

from luolib.lightning.cli import LightningCLI
from luolib.models import PlainConvUNetDecoder, UNetBackbone

from base import DataModuleDebug, SemanticSegModel

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

    def on_fit_start(self) -> None:
        super().on_fit_start()
        self.gradient_checkpointing_enable({'use_reentrant': False})

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feature_maps: list[torch.Tensor] = self.backbone(image)
        feature_maps = self.decoder(feature_maps)
        masks_logits = self.seg_head(feature_maps[0])
        return masks_logits

class CLI(LightningCLI):
    pass

def main():
    CLI(
        model_class=UNetForSemanticSeg,
        subclass_mode_data=False,
        datamodule_class=DataModuleDebug,
        subclass_mode_model=False,
    )

if __name__ == '__main__':
    main()
