from lightning.pytorch.cli import LightningArgumentParser
from peft import LoraConfig, get_peft_model

from luolib.lightning.cli import LightningCLI as CLIBase

from mmmm.models import MMMMForCausalLM, MMMMTokenizer
from mmmm.utils import get_lora_modules_default

class CLI(CLIBase):
    model: MMMMForCausalLM

    def add_arguments_to_parser(self, parser: LightningArgumentParser):
        super().add_arguments_to_parser(parser)
        parser.add_subclass_arguments(MMMMTokenizer, 'tokenizer')
        parser.link_arguments('tokenizer', 'data.init_args.tokenizer', apply_on='instantiate')
        parser.link_arguments('tokenizer', 'model.init_args.tokenizer', apply_on='instantiate')
        # https://github.com/omni-us/jsonargparse/issues/287
        parser.add_class_arguments(LoraConfig, 'lora')

    def instantiate_classes(self) -> None:
        super().instantiate_classes()
        lora_config: LoraConfig = self._get(self.config_init, 'lora')
        lora_config.target_modules, lora_config.modules_to_save = get_lora_modules_default(self.model)
        self.model.peft_model = get_peft_model(self.model, lora_config)

def main():
    CLI()

if __name__ == '__main__':
    main()
