from dataclasses import dataclass

from lightning.pytorch.cli import LightningArgumentParser
from peft import LoraConfig, get_peft_model

from luolib.lightning.cli import LightningCLI, OptimDict as OptimDictBase
from luolib.lightning.trainer import PeftTrainer
from luolib.lightning.utils import OptimConf

from mmmm.data import MMMMDataModule
from mmmm.models import MMMMForCausalLM, MMMMTokenizer
from mmmm.models.loss import DiceFocalLoss
from mmmm.utils import get_lora_modules_default

@dataclass
class OptimDict(OptimDictBase):
    sam: OptimConf
    seg_proj: OptimConf
    default: OptimConf

class CLI(LightningCLI):
    model: MMMMForCausalLM
    datamodule: MMMMDataModule

    def add_arguments_to_parser(self, parser: LightningArgumentParser):
        super().add_arguments_to_parser(parser)
        parser.add_subclass_arguments(MMMMTokenizer, 'tokenizer')
        parser.link_arguments('tokenizer', f'{self.data_prefix}.tokenizer', apply_on='instantiate')
        parser.link_arguments('tokenizer', f'{self.model_prefix}.tokenizer', apply_on='instantiate')
        # dataclass as class: https://github.com/omni-us/jsonargparse/issues/287
        parser.add_class_arguments(LoraConfig, 'lora')
        parser.add_class_arguments(DiceFocalLoss, 'mask_loss')
        parser.link_arguments('mask_loss', f'{self.model_prefix}.mask_loss', apply_on='instantiate')

    def instantiate_classes(self) -> None:
        super().instantiate_classes()
        model = self.model
        config = self.active_config_init
        lora_config: LoraConfig = config.lora
        lora_config.target_modules, lora_config.modules_to_save = get_lora_modules_default(model)
        model.set_peft_model(get_peft_model(model, lora_config))
        # MetaTensor is required for lazy transform orz
        # set_track_meta(False)

def main():
    CLI(
        model_class=MMMMForCausalLM,
        datamodule_class=MMMMDataModule,
        trainer_class=PeftTrainer,
        optim_dict_class=OptimDict,
    )

if __name__ == '__main__':
    main()
