from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any, ContextManager

import cytoolz
from jsonargparse import lazy_instance
from lightning import Trainer
from lightning.pytorch.cli import LightningArgumentParser
from lightning.pytorch.plugins import FSDPPrecision, HalfPrecision, Precision, MixedPrecision
from lightning.pytorch.strategies import FSDPStrategy
from peft import LoraConfig, get_peft_model
from peft.tuners import lora
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.nn import Module

from luolib.lightning.cli import LightningCLI
from luolib.lightning.trainer import PeftTrainer

from mmmm.data import MMMMDataModule
from mmmm.models import MMMMForCausalLM, Sam, InstanceSam
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

class MyPrecision(Precision):
    def __init__(self):
        self._bf16 = HalfPrecision('bf16-true')

    def convert_input(self, data: dict) -> Any:
        fp16_mixed_keys = ['grounding_image', 'boxes']
        ret = {
            **self._bf16.convert_input(cytoolz.dissoc(data, *fp16_mixed_keys)),
            **{key: data[key] for key in fp16_mixed_keys}
        }
        return ret

    def convert_module(self, module: Module) -> MMMMForCausalLM:
        assert isinstance(module, MMMMForCausalLM)
        # NOTE: module._dtype is not set since module.to is not called
        for param in module.parameters(recurse=False):
            param.to(dtype=self._bf16._desired_input_dtype)
        fp32_children = set(module.get_fp32_children())
        for name, child in module.named_children():
            if name not in fp32_children:
                self._bf16.convert_module(child)
        return module

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
        parser.add_argument('--lora_adapter_path', type=Path | None, default=None)

    def instantiate_classes(self) -> None:
        super().instantiate_classes()
        model = self.model
        config = self.active_config_init
        lora_config: LoraConfig = config.lora
        lora_config.target_modules, lora_config.modules_to_save = get_lora_modules_default(model)
        if len(lora_config.target_modules) > 0:
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
        trainer_defaults={
            'precision': None,
            'plugins': lazy_instance(MyPrecision),
        }
    )

if __name__ == '__main__':
    main()
