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
        model = self.model
        config = self.config_init[self.subcommand]
        tokenizer: MMMMTokenizer = config.tokenizer
        model.resize_token_embeddings(len(tokenizer))
        model.gradient_checkpointing_enable({'use_reentrant': False})
        lora_config: LoraConfig = config.lora
        lora_config.target_modules, lora_config.modules_to_save = get_lora_modules_default(model)
        model.peft_model = get_peft_model(model, lora_config)

def main():
    CLI()

if __name__ == '__main__':
    main()
