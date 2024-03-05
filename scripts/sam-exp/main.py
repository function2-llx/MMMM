from pathlib import Path
from typing import Sequence

import einops
import torch
from torch.nn import functional as nnf
from transformers import PreTrainedModel, PretrainedConfig

from luolib import transforms as lt
from luolib.datamodule import ExpDataModuleBase
from luolib.lightning import LightningModule
from luolib.lightning.cli import LightningCLI
from luolib.models.param import NoWeightDecayParameter
from luolib.types import tuple3_t
from luolib.utils.misc import ensure_rgb
from monai import transforms as mt
from monai.networks import one_hot

from mmmm.models.mmmm import DiceFocalLoss
from mmmm.models.segvol import SamArgs, build_sam_vit_3d

class SAMForSemanticSeg(PreTrainedModel, LightningModule):
    supports_gradient_checkpointing: bool = True

    def __init__(
        self,
        *,
        num_fg_classes: int = 15,
        sam: SamArgs,
        hidden_size: int = 768,
        **kwargs,
    ):
        super().__init__(PretrainedConfig(), **kwargs)  # make HF happy
        self.sam = build_sam_vit_3d(sam)
        self.num_fg_classes = num_fg_classes
        self.cls_embeds = NoWeightDecayParameter(torch.randn(num_fg_classes, hidden_size))
        self.loss = DiceFocalLoss()

    def on_fit_start(self) -> None:
        super().on_fit_start()
        self.gradient_checkpointing_enable({'use_reentrant': False})

    def training_step(self, batch: dict, *args: ..., **kwargs: ...):
        image = batch['img']
        sam = self.sam
        image_embeddings = sam.image_encoder(image)
        sparse_embeddings, dense_embeddings = sam.prompt_encoder(
            image_embeddings.shape[2:], text_embedding=self.cls_embeds,
        )
        masks_logits_list = []
        for i in range(image.shape[0]):
            masks_logits, _ = sam.mask_decoder(
                image_embeddings=image_embeddings[i:i + 1],
                text_embedding=self.cls_embeds,  # make SegVol happy
                image_pe=sam.prompt_encoder.get_dense_pe(image_embeddings.shape[2:]),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            masks_logits_list.append(masks_logits)
        masks_logits = einops.rearrange(masks_logits_list, 'n c 1 ... -> n c ...')
        masks_logits = nnf.interpolate(masks_logits, image.shape[2:], mode='trilinear')
        mask_loss = {
            k: v.mean()
            for k, v in self.loss(masks_logits, batch['seg']).items()
        }
        loss = mask_loss['total']
        self.log_dict({
            'train/loss': loss,
            **{f'train/{k}_loss': v for k, v in mask_loss.items() if k != 'total'},
        })
        return loss

class InputTransformD(mt.Transform):
    def __init__(self, num_fg_classes: int):
        self.num_fg_classes = num_fg_classes

    def __call__(self, data: dict):
        img, _ = ensure_rgb(data['img'].as_tensor())
        seg = one_hot(data['seg'].as_tensor(), self.num_fg_classes + 1, dim=0, dtype=torch.bool)
        seg = seg[1:]  # remove bg
        return {
            'img': img,
            'seg': seg,
        }

class DataModule(ExpDataModuleBase):
    def __init__(
        self,
        *,
        patch_size: tuple3_t[int],
        num_fg_classes: int = 15,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.data_dir = Path('nnUNet-data/preprocessed/Dataset219_AMOS2022_postChallenge_task2/nnUNetPlans_3d_fullres')
        self.patch_size = patch_size
        self.num_fg_classes = num_fg_classes

    def train_transform(self):
        return mt.Compose(
            [
                lt.nnUNetLoaderD('case', self.data_dir, unravel_class_locations=True),
                mt.SpatialPadD(['img', 'seg'], self.patch_size),
                lt.OneOf(
                    [
                        mt.RandSpatialCropD(['img', 'seg'], self.patch_size, random_center=True, random_size=False),
                        mt.RandCropByLabelClassesD(
                            ['img', 'seg'],
                            'seg',
                            spatial_size=self.patch_size,
                            # who knows why they pop? https://github.com/Project-MONAI/MONAI/blob/1.3.0/monai/transforms/croppad/dictionary.py#L1153
                            indices_key='class_locations',
                            warn=False,
                        ),
                    ],
                    weights=(2, 1),
                ),
                InputTransformD(self.num_fg_classes),
            ],
        )

    def train_data(self) -> Sequence:
        return [
            {'case': case_file.stem}
            for case_file in self.data_dir.glob('*.npz')
        ]
        # ret = []
        # for label_path in self.data_dir.glob('*_seg.npy'):
        #     case = label_path.stem[:-4]
        #     ret.append({'case': case})
        # return ret

class CLI(LightningCLI):
    pass

def main():
    CLI(
        model_class=SAMForSemanticSeg,
        subclass_mode_data=False,
        datamodule_class=DataModule,
        subclass_mode_model=False,
    )

if __name__ == '__main__':
    main()
