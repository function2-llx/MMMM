from functools import partial

import cytoolz
from torch import nn

from luolib.types import tuple2_t

def apply_prefix(prefix: str, path: str):
    return f'{prefix}{path}' if prefix.endswith('.') or not prefix else f'{prefix}.{path}'

def _check_leaf_module(module: nn.Module):
    requires_grad = [p.requires_grad for p in module.parameters()]
    if len(requires_grad) > 0 and any(requires_grad):
        assert all(requires_grad), "What's going on?"
        return True
    else:
        return False

def get_lora_modules_default(module: nn.Module, prefix: str = '', recursive: bool = True) -> tuple2_t[list[str]]:
    target_modules, modules_to_save = [], []

    # noinspection PyShadowingNames
    def dfs(m: nn.Module, prefix: str):
        if recursive and hasattr(m, 'get_lora_modules'):
            # if the module has defined the `get_lora_modules` method, use it
            m_target_modules, m_modules_to_save = m.get_lora_modules(prefix='')
            filter_func = cytoolz.compose(_check_leaf_module, m.get_submodule)
            map_func = partial(apply_prefix, prefix)
            m_target_modules = [*map(map_func, filter(filter_func, m_target_modules))]
            m_modules_to_save = [*map(map_func, filter(filter_func, m_modules_to_save))]
            target_modules.extend(m_target_modules)
            modules_to_save.extend(m_modules_to_save)
        elif isinstance(m, (nn.Linear, nn.Embedding)):
            target_modules.append(prefix)
        elif len(named_children := list(m.named_children())) == 0:
            if _check_leaf_module(m):
                modules_to_save.append(prefix)
        else:
            for name, child in named_children:
                dfs(child, apply_prefix(prefix, name))

    dfs(module, prefix)
    return target_modules, modules_to_save

def get_lora_modules_finetune_all(module: nn.Module, prefix: str) -> list[str]:
    modules_to_save = []

    def dfs(m: nn.Module, p: str):
        if len(named_children := list(m.named_children())) == 0:
            if _check_leaf_module(m):
                modules_to_save.append(p)
        else:
            for name, child in named_children:
                dfs(child, apply_prefix(p, name))

    dfs(module, prefix)
    return modules_to_save

class ParameterWrapper(nn.Module):
    """peft does not support parameter in `modules_to_save`
    use "weight" as the name for the wrapped parameter because peft happens to support this
    see https://github.com/huggingface/peft/issues/1492
    """
    def __init__(self, weight: nn.Parameter):
        super().__init__()
        self.weight = weight

    @classmethod
    def wrap(cls, module: nn.Module, state_dict: dict, prefix: str):
        for name, child in module.named_children():
            if isinstance(child, ParameterWrapper):
                name = apply_prefix(prefix, name)
                if (weight := state_dict.pop(name, None)) is not None:
                    state_dict[f'{name}.weight'] = weight

    def extra_repr(self) -> str:
        return f'shape={self.weight.shape}'
