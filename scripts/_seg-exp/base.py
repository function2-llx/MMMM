import einops

from luolib.lightning import LightningModule
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.utils import BlendMode, MetricReduction

from mmmm.models.loss import DiceFocalLoss
from mmmm_debug.data import DataModuleDebug

class SemanticSegModel(LightningModule):
    datamodule: DataModuleDebug

    def __init__(self, *, lambda_focal: float = 1., **kwargs):
        super().__init__(**kwargs)
        self.dice_metric = DiceMetric()
        self.loss = DiceFocalLoss(lambda_focal=lambda_focal)

    def training_step(self, batch: dict, *args: ..., **kwargs: ...):
        image = batch['img']
        masks_logits = self(image)
        mask_loss = self.loss(masks_logits, batch['seg'])
        dice_pos = mask_loss.pop('dice-pos')
        for c in range(dice_pos.shape[0]):
            if dice_pos[c].isfinite():
                self.log(f'train/dice-pos/{self.datamodule.class_names[c]}', (1 - dice_pos[c]) * 100)
        mask_loss_reduced = {k: v.mean() for k, v in mask_loss.items()}
        loss = mask_loss_reduced.pop('total')
        self.log_dict({
            'train/loss': loss,
            **{f'train/{k}_loss': v for k, v in mask_loss_reduced.items()},
        })
        return loss

    def on_validation_epoch_start(self) -> None:
        self.dice_metric.reset()
        self.recall = []

    def validation_step(self, batch: dict, *args: ..., **kwargs: ...):
        image = batch['img']
        logits = sliding_window_inference(
            image,
            self.datamodule.patch_size,
            sw_batch_size=8,
            predictor=self,
            overlap=0.5,
            mode=BlendMode.GAUSSIAN,
            progress=False,
        )
        pred = logits.sigmoid() > 0.5
        label = batch['seg']
        self.dice_metric(pred, label)
        recall = einops.reduce(pred & label, 'n c ... -> c', 'sum') / einops.reduce(label, 'n c ... -> c', 'sum')
        self.recall.append(recall)

    def on_validation_epoch_end(self) -> None:
        dice = self.dice_metric.aggregate(MetricReduction.MEAN_BATCH) * 100
        recall = einops.rearrange(self.recall, 'n c -> c n')
        for i in range(dice.shape[0]):
            self.log(f'val/dice/{self.datamodule.class_names[i]}', dice[i])
            r = recall[i][recall[i].isfinite()].mean() * 100
            self.log(f'val/recall/{self.datamodule.class_names[i]}', r)
        self.log(f'val/dice/avg', dice.mean())
