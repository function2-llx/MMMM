from pathlib import Path

from einops import einops
from jsonargparse import ArgumentParser
from lightning.fabric.utilities import move_data_to_device
from lightning.pytorch.plugins import HalfPrecision
import math

from luolib.utils.misc import ensure_rgb
import monai.transforms as mt
from monai.utils import InterpolateMode, convert_to_tensor
import numpy as np
import orjson
import torch
from torchvision.io import read_image
import torchvision.transforms.v2.functional as tvtf
from transformers import GenerationConfig
from transformers.generation import GenerateDecoderOnlyOutput

from luolib.types import PathLike
from mmmm.data.datamodule import _collate_fn
from mmmm.data.dataset.misc import get_max_resize, intensity_norm
from mmmm.data.defs import ConvTurn, PROCESSED_VL_DATA_ROOT
from mmmm.data.utils import prepare_vlm_inputs

base_pool_size_z = 2
pool_size_xy = 2
max_tokens_z = 4
base_patch_size_z = 16
patch_size_xy = 16
max_vision_tokens = 100

def image_transform(image_path: PathLike):
    image = read_image(str(image_path))
    image = einops.rearrange(image, 'c h w -> c 1 h w')
    image = tvtf.to_dtype(image, torch.float, scale=True)
    if (size_z := image.shape[1]) <= max_tokens_z:
        patch_size_z = pool_size_z = stride_z = 1
        tokens_z = size_z
    else:
        pool_size_z = base_pool_size_z
        log2_patch_size_z = np.log2(size_z / (pool_size_z * max_tokens_z)),
        log2_patch_size_z = np.clip(
            np.rint(log2_patch_size_z), 0, base_patch_size_z.bit_length() - 1,
        )
        patch_size_z = 1 << int(log2_patch_size_z)
        stride_z = patch_size_z * pool_size_z
        tokens_z = min(math.ceil(size_z / stride_z), max_tokens_z)
    patch_size = (patch_size_z, patch_size_xy, patch_size_xy)
    stride_xy = patch_size_xy * pool_size_xy
    stride = (stride_z, stride_xy, stride_xy)
    resize_shape = (
        min(size_z, tokens_z * stride_z),  # do not resize z if unnecessary
        *get_max_resize(
            image.shape[2:],
            stride_xy,
            max_vision_tokens // tokens_z,
        ),
    )
    if resize_shape != image.shape[1:]:
        resize = mt.Resize(resize_shape, mode=InterpolateMode.TRILINEAR, anti_aliasing=True)
        image = resize(image)
    image = mt.DivisiblePad(stride)(image)
    image = convert_to_tensor(image)
    image, _ = ensure_rgb(image, contiguous=True)
    image = intensity_norm(image)
    pool_size = (pool_size_z, pool_size_xy, pool_size_xy)
    num_vision_tokens = (np.array(image.shape[1:]) // stride).prod().item()
    return image, patch_size, pool_size, num_vision_tokens

def main():
    parser = ArgumentParser()
    parser.add_argument('adapter_dir', type=Path)
    args = parser.parse_args()
    from mmmm.models.mmmm import from_pretrained
    model, tokenizer = from_pretrained('conf/model.yaml', args.adapter_dir)
    model = model.cuda()
    data: list = orjson.loads(Path(PROCESSED_VL_DATA_ROOT / 'VQA-RAD/test.json').read_bytes())
    precision = HalfPrecision('bf16-true')
    for sample in data:
        conv = []
        image, patch_size, pool_size, num_vision_tokens = image_transform(sample['image'][0])
        for i in range(len(sample['vqa'])):
            prompt = sample['vqa'][i]['question']
            conv.append(ConvTurn(prompt, ''))
            vlm_inputs, _ = prepare_vlm_inputs(
                conv,
                tokenizer,
                num_vision_tokens,
                inference=True,
                grounding=False,
            )
            input_len = len(vlm_inputs['input_ids'])

            batch = _collate_fn([
                {
                    'image': image,
                    'vlm_inputs': vlm_inputs,
                    'patch_size': patch_size,
                    'pool_size': pool_size,
                }
            ])
            batch = precision.convert_input(batch)
            batch = move_data_to_device(batch, 'cuda')
            gen_config = GenerationConfig(max_new_tokens=512, do_sample=False)
            output: GenerateDecoderOnlyOutput = model.generate(
                generation_config=gen_config,
                return_dict_in_generate=True,
                output_hidden_states=True,
                **batch['vlm_inputs'],
                image=batch['image'],
                patch_size=batch['patch_size'],
                pool_size=batch['pool_size'],
            )
            token_ids = output.sequences[0].tolist()
            token_ids = token_ids[input_len:]
            if token_ids[-1] == tokenizer.eos_token_id:
                token_ids = token_ids[:-1]
            response = tokenizer.decode(token_ids, clean_up_tokenization_spaces=False)
            print(prompt)
            print(response)
            conv[-1] = ConvTurn(prompt, response)

if __name__ == '__main__':
    main()
