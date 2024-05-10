import einops
import torch
from torch import nn

from monai.losses.focal_loss import sigmoid_focal_loss

__all__ = [
    'DiceFocalLoss',
]

class DiceFocalLoss(nn.Module):
    """fix smooth issue of dice"""
    def __init__(
        self,
        dice_weight: float = 1.,
        focal_weight: float = 1.,
        smooth_dr: float = 1e-8,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.smooth_dr = smooth_dr

    def dice(self, input: torch.Tensor, target: torch.Tensor):
        """
        copy from monai.losses.DiceLoss.forward, but fix the smooth issue, fix: https://github.com/MIC-DKFZ/nnUNet/issues/812
        """
        input = torch.sigmoid(input)
        intersection = einops.reduce(target * input, 'n c ... -> n c', 'sum')
        ground_o = einops.reduce(target, 'n c ... -> n c ', 'sum')
        pred_o = einops.reduce(input, 'n c ... -> n c', 'sum')
        denominator = ground_o + pred_o
        # NOTE: no smooth item for nominator, or it will become unfortunate
        f: torch.Tensor = 1.0 - 2.0 * intersection / torch.clip(denominator + self.smooth_dr, min=1e-8)
        return f

    def focal(self, input: torch.Tensor, target: torch.Tensor):
        # let's be happy
        return sigmoid_focal_loss(input, target)

    def forward(
        self, input: torch.Tensor, target: torch.Tensor, *, reduce_batch: bool = True, return_dict: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
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
            return {
                'dice': dice_loss,
                'focal': focal_loss,
                'total': total_loss,
            }
        else:
            return total_loss
