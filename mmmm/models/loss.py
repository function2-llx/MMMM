from functools import partial
import warnings

from einops import einops
import torch

from monai.losses import DiceFocalLoss as _MONAIDiceFocalLoss

__all__ = [
    'DiceFocalLoss',
]

from monai.networks import one_hot
from monai.utils import LossReduction

# fix: https://github.com/MIC-DKFZ/nnUNet/issues/812

def patched_dice_forward(self, input: torch.Tensor, target: torch.Tensor):
    if self.sigmoid:
        input = torch.sigmoid(input)

    n_pred_ch = input.shape[1]
    if self.softmax:
        if n_pred_ch == 1:
            warnings.warn("single channel prediction, `softmax=True` ignored.")
        else:
            input = torch.softmax(input, 1)

    if self.other_act is not None:
        input = self.other_act(input)

    if self.to_onehot_y:
        if n_pred_ch == 1:
            warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
        else:
            target = one_hot(target, num_classes=n_pred_ch)

    if not self.include_background:
        if n_pred_ch == 1:
            warnings.warn("single channel prediction, `include_background=False` ignored.")
        else:
            # if skipping background, removing first channel
            target = target[:, 1:]
            input = input[:, 1:]

    if target.shape != input.shape:
        raise AssertionError(f"ground truth has different shape ({target.shape}) from input ({input.shape})")

    # reducing only spatial dimensions (not batch nor channels)
    reduce_axis: list[int] = torch.arange(2, len(input.shape)).tolist()
    if self.batch:
        # reducing spatial dimensions and batch
        reduce_axis = [0] + reduce_axis

    intersection = torch.sum(target * input, dim=reduce_axis)

    if self.squared_pred:
        ground_o = torch.sum(target ** 2, dim=reduce_axis)
        pred_o = torch.sum(input ** 2, dim=reduce_axis)
    else:
        ground_o = torch.sum(target, dim=reduce_axis)
        pred_o = torch.sum(input, dim=reduce_axis)

    denominator = ground_o + pred_o

    if self.jaccard:
        denominator = 2.0 * (denominator - intersection)

    # NOTE: the only change is here
    f: torch.Tensor = 1.0 - (2.0 * intersection + self.smooth_nr) / torch.clip(denominator + self.smooth_dr, min=1e-8)

    num_of_classes = target.shape[1]
    if self.class_weight is not None and num_of_classes != 1:
        # make sure the lengths of weights are equal to the number of classes
        if self.class_weight.ndim == 0:
            self.class_weight = torch.as_tensor([self.class_weight] * num_of_classes)
        else:
            if self.class_weight.shape[0] != num_of_classes:
                raise ValueError(
                    """the length of the `weight` sequence should be the same as the number of classes.
                    If `include_background=False`, the weight should not include
                    the background category class 0."""
                )
        if self.class_weight.min() < 0:
            raise ValueError("the value/values of the `weight` should be no less than 0.")
        # apply class_weight to loss
        f = f * self.class_weight.to(f)

    if self.reduction == LossReduction.MEAN.value:
        f = torch.mean(f)  # the batch and channel average
    elif self.reduction == LossReduction.SUM.value:
        f = torch.sum(f)  # sum over the batch and channel dims
    elif self.reduction == LossReduction.NONE.value:
        # If we are not computing voxelwise loss components at least
        # make sure a none reduction maintains a broadcastable shape
        broadcast_shape = list(f.shape[0:2]) + [1] * (len(input.shape) - 2)
        f = f.view(broadcast_shape)
    else:
        raise ValueError(
            f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].'
        )

    return f

class DiceFocalLoss(_MONAIDiceFocalLoss):
    """reduce the results to channel only; also fix smooth issue of dice"""

    def __init__(self, **kwargs):
        super().__init__(reduction='none', smooth_nr=0, smooth_dr=0, sigmoid=True, **kwargs)
        self.dice.forward = partial(patched_dice_forward, self.dice)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        input = input.float()
        dice_loss = self.dice(input, target)
        # this won't take long, right?
        pos_class_mask = einops.reduce(target, 'n c ... -> n c', 'any')
        # FIXME: not pos_class_mask.any() ??
        dice_pos_batch_loss = dice_loss[pos_class_mask].mean()
        dice_pos_loss = dice_loss.new_empty(dice_loss.shape[1])
        for c in range(input.shape[1]):
            dice_pos_loss[c] = dice_loss[pos_class_mask[:, c], c].mean()
        dice_loss = einops.reduce(dice_loss, 'n c ... -> c', 'mean')
        focal_loss = einops.reduce(self.focal(input, target), 'n c ... -> c', 'mean')
        total_loss: torch.Tensor = self.lambda_dice * dice_loss + self.lambda_focal * focal_loss
        return {
            'dice': dice_loss,
            'dice-pos': dice_pos_loss,
            'dice-pos-batch': dice_pos_batch_loss,
            'focal': focal_loss,
            'total': total_loss,
        }
