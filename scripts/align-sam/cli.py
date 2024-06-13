from pathlib import Path

import torch
from lightning.pytorch.cli import LightningArgumentParser

from luolib.lightning.cli import LightningCLI

from _data import DataModule
from _model import AlignSam

class CLI(LightningCLI):
    model: AlignSam
    datamodule: DataModule

    def add_arguments_to_parser(self, parser: LightningArgumentParser):
        super().add_arguments_to_parser(parser)
        parser.add_argument('--state_dict_path', type=Path | None, default=None)

    def instantiate_classes(self) -> None:
        super().instantiate_classes()
        ckpt = torch.load(self.active_config_init.state_dict_path)
        self.model.load_state_dict(ckpt['state_dict'])

def main():
    CLI(
        model_class=AlignSam,
        subclass_mode_model=False,
        datamodule_class=DataModule,
        subclass_mode_data=False,
    )

if __name__ == '__main__':
    main()
