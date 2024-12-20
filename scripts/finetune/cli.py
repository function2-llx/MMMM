from lightning.pytorch.cli import LightningArgumentParser
from transformers import AutoTokenizer

from luolib.datamodule import ExpDataModuleBase
from luolib.lightning import LightningModule
from luolib.lightning.cli import LightningCLI
from luolib.lightning.trainer import Trainer, PeftTrainer

from peft import LoraConfig, get_peft_model

class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser):
        super().add_arguments_to_parser(parser)
        parser.add_argument('--tokenizer', type=str)
        parser.add_argument('--lora', type=LoraConfig | None, default=None)
        parser.add_argument('--peft', action='store_true')
        # parser.add_class_arguments(LoraConfig, 'lora')

    def instantiate_classes(self) -> None:
        if self.active_config.lora is not None or self.active_config.peft:
            self.trainer_class = PeftTrainer
        else:
            self.trainer_class = Trainer
        super().instantiate_classes()
        tokenizer = AutoTokenizer.from_pretrained(self.active_config_init.tokenizer)
        # if isinstance(self.model, FinetuneM3D):
        #     tokenizer = AutoTokenizer.from_pretrained(self.active_config_init.tokenizer, model_max_length=1024)
        self.datamodule.tokenizer = tokenizer
        if self.active_config_init.lora:
            model = self.model
            config = self.active_config_init
            lora_config: LoraConfig = config.lora
            lora_config.target_modules = model.target_modules
            lora_config.modules_to_save = model.modules_to_save
            if model.__class__.__name__ == 'FinetuneCogVLM':
                peft_model = get_peft_model(model.cogvlm_model, lora_config)
                model.set_peft_model(peft_model, prefix='cogvlm_model')
            # elif isinstance(model, FinetuneLlavaNEXT):
            #     peft_model = get_peft_model(model.llavaN_model, lora_config)
            #     model.set_peft_model(peft_model, prefix='llavaN_model')
            # elif isinstance(model, FinetuneLlavaMed):
            #     peft_model = get_peft_model(model.llavaM_model, lora_config)
            #     model.set_peft_model(peft_model, prefix='llavaM_model')
            elif model.__class__.__name__ == 'FinetuneM3D':
                peft_model = get_peft_model(model.m3d_model, lora_config)
                model.set_peft_model(peft_model, prefix='m3d_model')
            elif model.__class__.__name__ == 'FinetuneRadFM':
                peft_model = get_peft_model(model.radfm_model, lora_config)
                model.set_peft_model(peft_model, prefix='radfm_model')
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
