from lightning.pytorch.cli import LightningArgumentParser
from transformers import AutoTokenizer

from luolib.datamodule import ExpDataModuleBase
from luolib.lightning import LightningModule
from luolib.lightning.cli import LightningCLI
from luolib.lightning.trainer import Trainer


class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser):
        super().add_arguments_to_parser(parser)
        parser.add_argument('--tokenizer', type=str)

    def instantiate_classes(self) -> None:
        super().instantiate_classes()
        tokenizer = AutoTokenizer.from_pretrained(self.active_config_init.tokenizer)
        self.datamodule.tokenizer = tokenizer
    #     model = self.model
    #     config = self.active_config_init
    #     lora_config: LoraConfig = config.lora
    #     lora_config.target_modules, lora_config.modules_to_save = get_lora_modules_default(model)
    #     peft_model = get_peft_model(model, lora_config)
    #     model.set_peft_model(peft_model)
    #     if (lora_adapter_path := config.lora_adapter_path) is not None:
    #         peft_model.load_adapter(str(lora_adapter_path), 'default', is_trainable=self.subcommand == 'fit')
    #         print(f'load adapter from {lora_adapter_path}')

def main():
    CLI(
        model_class=LightningModule,
        datamodule_class=ExpDataModuleBase,
        trainer_class=Trainer,
    )

if __name__ == '__main__':
    main()
