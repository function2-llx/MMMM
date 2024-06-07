from luolib.lightning.cli import LightningCLI

from mmmm.data import MMMMDataModule
from mmmm.models.segvol.align import AlignSam

class CLI(LightningCLI):
    model: AlignSam
    datamodule: MMMMDataModule

def main():
    CLI(
        model_class=AlignSam,
        subclass_mode_model=False,
        datamodule_class=MMMMDataModule,
        subclass_mode_data=False,
    )

if __name__ == '__main__':
    main()
