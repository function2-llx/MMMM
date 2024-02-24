from torch import nn

def apply_prefix(prefix: str, name: str):
    return f'{prefix}{name}' if prefix.endswith('.') or not prefix else f'{prefix}.{name}'

def get_lora_modules_default(module: nn.Module, prefix: str):
    target_modules, modules_to_save = [], []

    # noinspection PyShadowingNames
    def dfs(m: nn.Module, prefix: str):
        if hasattr(m, 'get_lora_modules'):
            # if the module has defined the `get_lora_modules` method, use it
            m_target_modules, m_modules_to_save = m.get_lora_modules(prefix=prefix)
            target_modules.extend(m_target_modules)
            modules_to_save.extend(m_modules_to_save)
        elif isinstance(m, (nn.Linear, nn.Embedding)):
            target_modules.append(prefix)
        elif len(named_children := list(m.named_children())) == 0:
            # is leaf module
            modules_to_save.append(prefix)
        else:
            for name, child in named_children:
                dfs(child, apply_prefix(prefix, name))

    dfs(module, prefix)
    return target_modules, modules_to_save

def get_lora_modules_finetune_all(module: nn.Module, prefix: str):
    target_modules, modules_to_save = [], []

    # noinspection PyShadowingNames
    def dfs(m: nn.Module, prefix: str):
        if len(named_children := list(m.named_children())) == 0:
            # is leaf module
            modules_to_save.append(prefix)
        else:
            for name, child in named_children:
                dfs(child, apply_prefix(prefix, name))

    dfs(module, prefix)
    return target_modules, modules_to_save

class ParameterWrapper(nn.Module):
    """peft does not support parameter in `modules_to_save`
    use "weight" as the name for the wrapped parameter because peft happens to support this
    """
    def __init__(self, weight: nn.Parameter):
        super().__init__()
        self.weight = weight

    @classmethod
    def wrap(cls, module: nn.Module, state_dict: dict, prefix: str):
        for name, child in module.named_children():
            if isinstance(child, ParameterWrapper):
                name = apply_prefix(prefix, name)
                state_dict[f'{name}.weight'] = state_dict.pop(name)
