from lightning.pytorch.cli import LightningArgumentParser
from transformers import AutoModel, AutoTokenizer

from luolib.lightning import LightningModule
from luolib.lightning.cli import LightningCLI

from mmmm.data import MMMMDataModule
from mmmm.models.loss import DiceFocalLoss
from mmmm.models.segvol import InstanceSam

class AlignSam(LightningModule):
    def __init__(self, *, sam: InstanceSam, seg_vol_path: str, **kwargs):
        super().__init__(**kwargs)
        self.sam = sam
        tokenizer = AutoTokenizer.from_pretrained(seg_vol_path)
        seg_vol = AutoModel.from_pretrained(seg_vol_path, trust_remote_code=True, test_mode=True)
        self.text_encoder = seg_vol.model.text_encoder
        self.text_encoder.tokenizer = tokenizer

class CLI(LightningCLI):
    model: AlignSam
    datamodule: MMMMDataModule

    def add_arguments_to_parser(self, parser: LightningArgumentParser):
        super().add_arguments_to_parser(parser)
        parser.add_class_arguments(DiceFocalLoss, 'mask_loss')
        parser.link_arguments('mask_loss', f'{self.model_prefix}.mask_loss', apply_on='instantiate')

def main():
    CLI(AlignSam, )

if __name__ == '__main__':
    main()
