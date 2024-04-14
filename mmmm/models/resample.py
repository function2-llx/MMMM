import einops
import torch
from torch import nn
from torch.nn import functional as nnf

from luolib.models import spadop
from luolib.types import param3_t, tuple3_t
from monai.utils import StrEnum

class Inflation(StrEnum):
    MEAN = 'mean'
    CENTER = 'center'

class Downsample(nn.Conv3d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: param3_t[int],
        bias: bool = True,
        inflation: Inflation = 'mean',
        interpolate_2d: bool = False,
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, kernel_size, bias=bias,
        )
        self.inflation = inflation
        self.interpolate_2d = interpolate_2d

    # noinspection DuplicatedCode
    def _load_from_state_dict(self, state_dict: dict[str, torch.Tensor], prefix: str, *args, **kwargs):
        # copy from luolib.spadop
        weight_key = f'{prefix}weight'
        if (weight := state_dict.get(weight_key)) is not None and weight.ndim + 1 == self.weight.ndim:
            if weight.shape[2:] != self.kernel_size[1:] and self.interpolate_2d:
                weight = spadop.resample(weight, self.kernel_size[1:], scale=True)
            d = self.kernel_size[0]
            match self.inflation:
                case 'mean':
                    weight = einops.repeat(weight / d, 'co ci ... -> co ci d ...', d=d)
                case 'center':
                    new_weight = weight.new_zeros(*weight.shape[:2], d, *weight.shape[2:])
                    if d & 1:
                        new_weight[:, :, d >> 1] = weight
                    else:
                        new_weight[:, :, [d - 1 >> 1, d >> 1]] = weight[:, :, None] / 2
                    weight = new_weight
                case _:
                    raise ValueError
            state_dict[weight_key] = weight
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, x: torch.Tensor, kernel_size: tuple3_t[int]):
        if self.kernel_size[0] == kernel_size[0]:
            weight = self.weight
        else:
            if self.kernel_size[0] % kernel_size[0] != 0:
                raise NotImplementedError
            weight = einops.reduce(self.weight, '... (d dr) h w -> ... d h w', 'sum', d=kernel_size[0])
        return nnf.conv3d(x, weight, self.bias, kernel_size)

class Upsample(nn.ConvTranspose3d):
    """
    This class should be able to load most pre-trained 3D checkpoints, since the shape for upsampling is usually 2
    TODO: load from 2D checkpoint
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        *,
        cnt: int,
    ):
        """
        Args:
            cnt: the number of upsampling layers before this layer
        """
        super().__init__(
            in_channels, out_channels, 2, 2, bias=bias,
        )
        self.patch_size_th = 1 << cnt + 1

    def forward(self, x: torch.Tensor, patch_size_z: int):
        kernel_size = list(self.kernel_size)
        if patch_size_z < self.patch_size_th:
            weight = einops.reduce(self.weight, '... d h w -> ... 1 h w', 'sum')
            kernel_size[0] = 1
        else:
            weight = self.weight
        return nnf.conv_transpose3d(x, weight, self.bias, kernel_size)
