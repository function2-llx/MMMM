import einops
import torch
from torch import nn

from luolib.losses import bce_with_binary_label, sigmoid_focal_loss

__all__ = [
    'DiceFocalLoss',
]

_EPS = 1e-8

class DiceFocalLoss(nn.Module):
    """
    fix smooth issue of dice
    """
    def __init__(
        self,
        dice_weight: float,
        focal_weight: float,
        focal_gamma: float,
        focal_alpha: float | None = None,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        assert focal_gamma >= 0
        self.focal_weight = focal_weight

    def dice(self, input: torch.Tensor, target: torch.Tensor | None):
        """
        copy from monai.losses.DiceLoss.forward, but fix the smooth issue, fix: https://github.com/MIC-DKFZ/nnUNet/issues/812
        """
        if target is None:
            return input.new_ones(input.shape[:2])
        input = torch.sigmoid(input)
        intersection = einops.reduce(target * input, 'n c ... -> n c', 'sum')
        ground_o = einops.reduce(target, 'n c ... -> n c ', 'sum')
        pred_o = einops.reduce(input, 'n c ... -> n c', 'sum')
        denominator = ground_o + pred_o
        # NOTE: no smooth item for nominator, or it will become unfortunate
        f: torch.Tensor = 1.0 - 2.0 * intersection / torch.clip(denominator, min=_EPS)
        return f

    def focal(self, input: torch.Tensor, target: torch.Tensor | None):
        # let's be happy
        if self.focal_gamma < _EPS:
            return bce_with_binary_label(input, target)
        else:
            if target is None:
                target = torch.zeros_like(input)
            return sigmoid_focal_loss(input, target, self.focal_gamma, self.focal_alpha)

    def forward(
        self,
        input: torch.Tensor,
        target: torch.BoolTensor | None = None,
        *,
        reduce_batch: bool = True,
        return_dict: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        assert input.ndim == 5
        if target is not None:
            assert input.shape == target.shape
        dice_loss = self.dice(input, target)
        focal_loss = self.focal(input, target)
        if reduce_batch:
            dice_loss = dice_loss.mean()
            focal_loss = focal_loss.mean()
        else:
            dice_loss = einops.reduce(dice_loss, 'n ... -> n', 'mean')
            focal_loss = einops.reduce(focal_loss, 'n ... -> n', 'mean')
        total_loss: torch.Tensor = self.dice_weight * dice_loss + self.focal_weight * focal_loss
        if return_dict:
            focal_key = 'ce' if self.focal_gamma < _EPS else f'focal-{self.focal_gamma:.1f}'
            return {
                'dice': dice_loss,
                focal_key: focal_loss,
                'total': total_loss,
            }
        else:
            return total_loss
