from lightning.pytorch.cli import LightningArgumentParser

from luolib.lightning.cli import LightningCLI
from luolib.lightning.trainer import Trainer

from mmmm.data import MMMMDataModule
from mmmm.models import MMMMTokenizer
from mmmm.models.loss import DiceFocalLoss
from mmmm_debug.data import DataModuleDebug
from mmmm_debug.model import MMMMDebugSAM

class CLI(LightningCLI):
    model: MMMMDebugSAM
    datamodule: MMMMDataModule

    def add_arguments_to_parser(self, parser: LightningArgumentParser):
        super().add_arguments_to_parser(parser)
        parser.add_subclass_arguments(MMMMTokenizer, 'tokenizer')
        parser.link_arguments('tokenizer', f'{self.data_prefix}.tokenizer', apply_on='instantiate')
        parser.link_arguments('tokenizer', f'{self.model_prefix}.tokenizer', apply_on='instantiate')
        # dataclass as class: https://github.com/omni-us/jsonargparse/issues/287
        # parser.add_class_arguments(LoraConfig, 'lora')
        parser.add_class_arguments(DiceFocalLoss, 'mask_loss')
        parser.link_arguments('mask_loss', f'{self.model_prefix}.mask_loss', apply_on='instantiate')

def main():
    CLI(
        model_class=MMMMDebugSAM,
        datamodule_class=DataModuleDebug,
        trainer_class=Trainer,
    )

if __name__ == '__main__':
    main()
