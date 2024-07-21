from einops import repeat
from lightning.fabric.utilities import move_data_to_device
from lightning.pytorch.plugins import HalfPrecision
import math

from tqdm import tqdm
import monai.transforms as mt
from monai.utils import InterpolateMode, convert_to_tensor
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from transformers import GenerationConfig

from luolib.types import PathLike
from luolib.utils.misc import ensure_rgb
from luolib.utils.zstd import load_pt_zst
from mmmm.data.datamodule import _collate_fn
from mmmm.data.dataset.misc import get_max_resize, intensity_norm
from mmmm.data.defs import ConvTurn
from mmmm.data.utils import prepare_vlm_inputs
from scripts.evaluate.utils import dump_results


base_pool_size_z = 2
pool_size_xy = 2
max_tokens_z = 4
base_patch_size_z = 16
patch_size_xy = 16
max_vision_tokens = 256


def setup_mmmm(adapter: str):
    from mmmm.models.mmmm import from_pretrained

    model, tokenizer = from_pretrained('conf/model.yaml', adapter)
    model = model.to('cuda')
    model.eval()

    return model, tokenizer


def image_transform(image_path: PathLike):
    if image_path.endswith('.pt'):
        image = torch.load(image_path).float()
        image = (image - image.min()) / (image.max() - image.min())
    elif image_path.endswith('.pt.zst'):
        image = load_pt_zst(image_path).float()
        image = (image - image.min()) / (image.max() - image.min())
    else:
        transform = transforms.ToTensor()
        image = Image.open(image_path).convert('RGB')
        image = transform(image)
        image = repeat(image, 'c h w -> c 1 h w')
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


class MMMMTransform(mt.RandomizableTransform):
    def __init__(self, tokenizer, task):
        self.tokenizer = tokenizer
        self.precision = HalfPrecision('bf16-true')
        self.task = task

    def __call__(self, data: dict):
        image, patch_size, pool_size, num_vision_tokens = image_transform(data['image'])
        if self.task == 'report':
            data['question'] = 'Can you provide a radiology report for this medical image?'
        vlm_inputs, _ = prepare_vlm_inputs(
            [ConvTurn(data['question'], '')],
            self.tokenizer,
            num_vision_tokens,
            inference=True,
            grounding=False,
        )
        input_len = len(vlm_inputs['input_ids'])
        batch = _collate_fn([
            {
                'image': image,
                'patch_size': patch_size,
                'pool_size': pool_size,
                'vlm_inputs': vlm_inputs,
            },
        ])
        batch = self.precision.convert_input(batch)

        return {
            'batch': batch,
            'input_len': input_len,
            'question': data['question'],
            'answer': data['answer'],
        }


def mmmm_vl_evaluate(model, tokenizer, dataloader, output):
    gen_config = GenerationConfig(max_new_tokens=1024, do_sample=False, eos_token_id=tokenizer.eos_token_id)

    results = []

    for i, sample in enumerate(tqdm(dataloader)):
        with torch.inference_mode():
            
            batch = move_data_to_device(sample['batch'], 'cuda')
            outputs = model.generate(
                generation_config=gen_config,
                return_dict_in_generate=True,
                output_hidden_states=True,
                **batch['vlm_inputs'],
                image=batch['image'],
                patch_size=batch['patch_size'],
                pool_size=batch['pool_size'],
            )
            token_ids = outputs.sequences[0].tolist()
            token_ids = token_ids[sample['input_len']:]
            if token_ids[-1] == tokenizer.eos_token_id:
                token_ids = token_ids[:-1]
            prediction = tokenizer.decode(token_ids, clean_up_tokenization_spaces=False)

        results.append(
            {
                'question': sample['question'],
                'answer': sample['answer'],
                'prediction': prediction,
            },
        )

        if i % 1000 == 0:
            dump_results(results, output)

        print(sample['question'])
        print(sample['answer'])
        print(prediction)

    dump_results(results, output)