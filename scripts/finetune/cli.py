from lightning.pytorch.cli import LightningArgumentParser
from transformers import AutoTokenizer

from _vqa.cogvlm import FinetuneCogVLM
from _vqa.llavanext import FinetuneLlavaNEXT
from _vqa.llavamed import FinetuneLlavaMed
from luolib.datamodule import ExpDataModuleBase
from luolib.lightning import LightningModule
from luolib.lightning.cli import LightningCLI
from luolib.lightning.trainer import Trainer, PeftTrainer

from peft import LoraConfig, get_peft_model

class MyPeftTrainer(PeftTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, save_embedding_layers=True)

class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser):
        super().add_arguments_to_parser(parser)
        parser.add_argument('--tokenizer', type=str)
        parser.add_argument('--lora', type=LoraConfig | None, default=None)
        # parser.add_class_arguments(LoraConfig, 'lora')

    def instantiate_classes(self) -> None:
        if self.active_config.lora is None:
            self.trainer_class = Trainer
        else:
            self.trainer_class = MyPeftTrainer
        super().instantiate_classes()
        tokenizer = AutoTokenizer.from_pretrained(self.active_config_init.tokenizer)
        self.datamodule.tokenizer = tokenizer
        if self.active_config_init.lora:
            model = self.model
            config = self.active_config_init
            lora_config: LoraConfig = config.lora
            lora_config.target_modules = model.target_modules
            lora_config.modules_to_save = model.modules_to_save
            if isinstance(model, FinetuneCogVLM):
                peft_model = get_peft_model(model.cogvlm_model, lora_config)
                model.set_peft_model(peft_model, prefix='cogvlm_model')
            elif isinstance(model, FinetuneLlavaNEXT):
                peft_model = get_peft_model(model.llavaN_model, lora_config)
                model.set_peft_model(peft_model, prefix='llavaN_model')
            elif isinstance(model, FinetuneLlavaMed):
                peft_model = get_peft_model(model.llavaM_model, lora_config)
                model.set_peft_model(peft_model, prefix='llavaM_model')
            else:
                raise NotImplementedError
            # if (lora_adapter_path := config.lora_adapter_path) is not None:
            #     peft_model.load_adapter(str(lora_adapter_path), 'default', is_trainable=self.subcommand == 'fit')
            #     print(f'load adapter from {lora_adapter_path}')

def main():
    CLI(
        model_class=LightningModule,
        datamodule_class=ExpDataModuleBase,
        # trainer_class=Trainer,
    )

if __name__ == '__main__':
    main()
