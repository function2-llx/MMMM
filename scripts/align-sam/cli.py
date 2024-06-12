from luolib.lightning.cli import LightningCLI

from _data import DataModule
from _model import AlignSam

class CLI(LightningCLI):
    model: AlignSam
    datamodule: DataModule

def main():
    CLI(
        model_class=AlignSam,
        subclass_mode_model=False,
        datamodule_class=DataModule,
        subclass_mode_data=False,
    )

if __name__ == '__main__':
    main()
