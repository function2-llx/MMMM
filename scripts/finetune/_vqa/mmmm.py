from pathlib import Path
from typing import Callable

import einops
import math
import numpy as np
import torch
import torchvision.transforms.v2.functional as tvtf
from torchvision.io import read_image

from luolib.lightning import LightningModule
from luolib.lightning.peft import PeftMixin
from luolib.utils import load_pt_zst
from luolib.utils.misc import ensure_rgb
from monai.utils import InterpolateMode, convert_to_tensor
import monai.transforms as mt

from mmmm.data.defs import ConvTurn
from mmmm.data.utils import prepare_vlm_inputs
from mmmm.data.dataset.misc import get_max_resize, intensity_norm
from mmmm.models.mmmm import from_pretrained, MMMMTokenizer
from _vqa._base import VQADataModule, VQATransform

LANGUAGE_TOKEN_TYPE = 0
VISION_TOKEN_TYPE = 1

class FinetuneMMMM(PeftMixin, LightningModule):
    def __init__(self, *, adapter_path: Path):
        super().__init__()

        model, tokenizer = from_pretrained('conf/model.yaml', adapter_path, trainable=True)
        self.mmmm_model = model
        self.set_peft_model(self.mmmm_model.peft_model, 'mmmm_model.')
        self.mmmm_model.gradient_checkpointing_enable({'use_reentrant': False})
        self.train()

    def training_step(self, batch, *args, **kwargs):
        image: list[torch.Tensor] = batch['image']
        batch_size = len(image)
        outputs = self.mmmm_model(
            **batch['vlm_inputs'],
            image=image,
            patch_size=[(1, 16, 16)] * batch_size,
            pool_size=[(1, 2, 2)] * batch_size,
        )
        loss = outputs.loss
        self.log('train/loss', loss)
        return loss

class MMMMVQATransform(VQATransform):
    base_pool_size_z = 2
    pool_size_xy = 2
    max_tokens_z = 4
    base_patch_size_z = 16
    patch_size_xy = 16

    def __init__(self, *args, max_vision_tokens: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_vision_tokens = max_vision_tokens

    def image_transform(self, image_path: str):
        if image_path.endswith('.pt'):
            image = torch.load(image_path).float()
        elif image_path.endswith('.pt.zst'):
            image = load_pt_zst(Path(image_path)).float()
        else:
            image = read_image(image_path)
            image = einops.rearrange(image, 'c h w -> c 1 h w')
        image = tvtf.to_dtype(image, torch.float, scale=True)
        if (size_z := image.shape[1]) <= self.max_tokens_z:
            patch_size_z = pool_size_z = stride_z = 1
            tokens_z = size_z
        else:
            pool_size_z = self.base_pool_size_z
            log2_patch_size_z = np.log2(size_z / (pool_size_z * self.max_tokens_z)),
            log2_patch_size_z = np.clip(
                np.rint(log2_patch_size_z), 0, self.base_patch_size_z.bit_length() - 1,
            )
            patch_size_z = 1 << int(log2_patch_size_z)
            stride_z = patch_size_z * pool_size_z
            tokens_z = min(math.ceil(size_z / stride_z), self.max_tokens_z)
        patch_size = (patch_size_z, self.patch_size_xy, self.patch_size_xy)
        stride_xy = self.patch_size_xy * self.pool_size_xy
        stride = (stride_z, stride_xy, stride_xy)
        resize_shape = (
            min(size_z, tokens_z * stride_z),  # do not resize z if unnecessary
            *get_max_resize(
                image.shape[2:],
                stride_xy,
                self.max_vision_tokens // tokens_z,
            ),
        )
        if resize_shape != image.shape[1:]:
            resize = mt.Resize(resize_shape, mode=InterpolateMode.TRILINEAR, anti_aliasing=True)
            image = resize(image)
        image = mt.DivisiblePad(stride)(image)
        image = convert_to_tensor(image)
        image, _ = ensure_rgb(image, contiguous=True)
        image = intensity_norm(image)
        pool_size = (pool_size_z, self.pool_size_xy, self.pool_size_xy)
        num_vision_tokens = (np.array(image.shape[1:]) // stride).prod().item()
        return image, patch_size, pool_size, num_vision_tokens

    def __call__(self, data):
        image, patch_size, pool_size, num_vision_tokens = self.image_transform(data['image'])
        pairs = [(qa['question'], qa['answer']) for qa in data['vqa']]
        self.R.shuffle(pairs)
        vlm_inputs, _ = prepare_vlm_inputs(
            [ConvTurn(*pair) for pair in pairs],
            self.tokenizer,
            self.max_vision_tokens,
            inference=False,
            grounding=False,
            max_seq_len=self.max_seq_len,
        )

        return {
            'image': image,
            'vlm_inputs': vlm_inputs,
        }


class MMMMVQADataModule(VQADataModule):
    def __init__(self, *args, max_vision_tokens: int, resize=(0, 0), **kwargs):
        assert resize == (0, 0)
        super().__init__(*args, **kwargs, resize=resize)
        self.max_vision_tokens = max_vision_tokens

    def train_transform(self) -> Callable:
        self.tokenizer = MMMMTokenizer.build('lmsys/vicuna-7b-v1.5')
        return MMMMVQATransform(self.tokenizer, resize=self.resize, max_seq_len=None, max_vision_tokens=self.max_vision_tokens)

    def _collate_fn(self, batch: list[dict]):
        vlm_inputs_list: list[dict] = []
        for x in batch:
            vlm_inputs_list.append(x.pop('vlm_inputs'))
        from _utils import _pad_inputs
        ret = {
            'image': [x['image'] for x in batch],
            'vlm_inputs': _pad_inputs(vlm_inputs_list, self.tokenizer.pad_token_id),
        }
        return ret
