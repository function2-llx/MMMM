from dataclasses import dataclass

import torch

from luolib.datamodule import ExpDataModuleBase
from luolib.lightning.cli import LightningCLI
from luolib.types import tuple2_t

from mmmm.data.dataset import DatasetSpec
from mmmm.data.dataset.local import get_local_data_list
from mmmm.models.segvol.align import AlignSam


class CLI(LightningCLI):
    model: AlignSam
    datamodule: MMMMDataModule

class DataModule(ExpDataModuleBase):
    pass

def main():
    CLI(
        model_class=AlignSam,
        subclass_mode_model=False,
        datamodule_class=MMMMDataModule,
        subclass_mode_data=False,
    )

if __name__ == '__main__':
    main()
