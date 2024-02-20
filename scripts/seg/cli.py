from lightning.pytorch.cli import LightningArgumentParser

from luolib.lightning.cli import LightningCLI as CLIBase
from mmmm.models import MMMMTokenizer

from transformers import PretrainedConfig

class CLI(CLIBase):
    def add_arguments_to_parser(self, parser: LightningArgumentParser):
        super().add_arguments_to_parser(parser)
        parser.add_subclass_arguments(MMMMTokenizer, 'tokenizer')
        parser.link_arguments('tokenizer', 'data.init_args.tokenizer', apply_on='instantiate')

def main():
    CLI()

if __name__ == '__main__':
    main()
