from collections.abc import Callable
from functools import partial
from pathlib import Path

from lightning.pytorch.cli import LightningArgumentParser
from lightning.pytorch.plugins import FSDPPrecision
from lightning.pytorch.strategies import FSDPStrategy
from peft import LoraConfig, get_peft_model
from peft.tuners import lora
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.nn import Module

from luolib.lightning.cli import LightningCLI
from luolib.lightning.trainer import PeftTrainer

from mmmm.data import MMMMDataModule
from mmmm.models import MMMMForCausalLM
from mmmm.models.loss import DiceFocalLoss
from mmmm.tokenizer import MMMMTokenizer
from mmmm.utils import get_lora_modules_default

def wrap_lora_linear(model: nn.Module, fsdp_fn: Callable[[nn.Module], FullyShardedDataParallel]) -> FullyShardedDataParallel:
    vis = set()
    def dfs_wrap(module: nn.Module):
        if module in vis:
            return
        vis.add(module)
        if isinstance(module, lora.Linear):
            # only wrap the base layer for LoRA Linear
            module.base_layer = fsdp_fn(module.base_layer)
            return
        for child in module.children():
            dfs_wrap(child)
    dfs_wrap(model)
    model = fsdp_fn(model)
    return model

class MyFSDPStrategy(FSDPStrategy):
    def _setup_model(self, model: Module):
        model = wrap_lora_linear(
            model,
            partial(
                FullyShardedDataParallel,
                cpu_offload=self.cpu_offload,
                mixed_precision=self.mixed_precision_config,
                sharding_strategy=self.sharding_strategy,
                device_id=self.root_device.index,
                **self.kwargs,
            ),
        )
        return super()._setup_model(model)

def _FSDP_convert_module(self: FSDPPrecision, module: nn.Module) -> nn.Module:
    # fix https://github.com/Lightning-AI/pytorch-lightning/issues/19721, humor
    return module.to(self._desired_input_dtype)

FSDPPrecision.convert_module = _FSDP_convert_module

class CLI(LightningCLI):
    model: MMMMForCausalLM
    datamodule: MMMMDataModule
    trainer: PeftTrainer

    def add_arguments_to_parser(self, parser: LightningArgumentParser):
        super().add_arguments_to_parser(parser)
        parser.add_subclass_arguments(MMMMTokenizer, 'tokenizer')
        parser.link_arguments('tokenizer', f'{self.data_prefix}.tokenizer', apply_on='instantiate')
        parser.link_arguments('tokenizer', f'{self.model_prefix}.tokenizer', apply_on='instantiate')
        # dataclass as class: https://github.com/omni-us/jsonargparse/issues/287
        parser.add_class_arguments(LoraConfig, 'lora')
        parser.add_class_arguments(DiceFocalLoss, 'mask_loss')
        parser.link_arguments('mask_loss', f'{self.model_prefix}.mask_loss', apply_on='instantiate')
        parser.add_argument('--lora_adapter_path', type=Path | None, default=None)

    def instantiate_classes(self) -> None:
        super().instantiate_classes()
        model = self.model
        config = self.active_config_init
        lora_config: LoraConfig = config.lora
        lora_config.target_modules, lora_config.modules_to_save = get_lora_modules_default(model)
        peft_model = get_peft_model(model, lora_config)
        model.set_peft_model(peft_model)
        if (lora_adapter_path := config.lora_adapter_path) is not None:
            peft_model.load_adapter(str(lora_adapter_path), 'default', is_trainable=self.subcommand == 'fit')
            print(f'load adapter from {lora_adapter_path}')

def main():
    CLI(
        model_class=MMMMForCausalLM,
        datamodule_class=MMMMDataModule,
        trainer_class=PeftTrainer,
    )

if __name__ == '__main__':
    main()
