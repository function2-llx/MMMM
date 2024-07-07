from pathlib import Path

import cytoolz
import einops
import inflect
import math
import numpy as np
import orjson
import torch
import torchvision.transforms.v2.functional as tvtf
from jsonargparse import ActionConfigFile, ArgumentParser
from lightning.fabric.utilities import move_data_to_device
from lightning.pytorch.plugins import HalfPrecision
from tqdm import tqdm

from luolib.types import PathLike
from luolib.utils import load_pt_zst
from luolib.utils.misc import ensure_rgb
from mmmm.data.dataset.misc import get_max_resize
from mmmm.misc import load_image
from monai.inferers import sliding_window_inference
from monai.utils import BlendMode

from mmmm.data import get_target_tax
from mmmm.data.defs import PROCESSED_VG_DATA_ROOT, Split
from mmmm.data.sparse import Sparse
from mmmm.data.target_tax import TargetClass
from _data import _collate_fn
from _model import AlignSam, AlignInstanceSam

engine = inflect.engine()
_stop_words = {'the'}
supported_phrases = [
    'aortic enlargement', 'atelectasis', 'calcification', 'cardiomegaly', 'clavicle fracture', 'pulmonary consolidation', 'pulmonary edema', 'pulmonary emphysema', 'pulmonary artery enlargement', 'interstitial lung disease', 'pulmonary infiltrate', 'pulmonary cavity', 'pulmonary cyst', 'pulmonary opacification', 'mediastinal shift', 'lung nodule', 'pleural effusion', 'pleural thickening', 'pneumothorax', 'pulmonary fibrosis', 'rib fracture'
]

def singularize(word: str) -> str:
    result = engine.singular_noun(word)
    return result if result else word

def normalize(text: str):
    words = text.split()
    words = map(str.lower, words)
    words = map(singularize, words)
    words = filter(lambda w: w not in _stop_words, words)
    text = ' '.join(words)
    return text

def parse_phrases(text: str):
    start = '<p>'
    end = '</p>'
    ret = []
    while (i := text.find(start)) >= 0:
        text = text[i + len(start):]
        j = text.find(end)
        assert j >= 0
        phrase = text[:j]
        ret.append(phrase)
        text = text[j + len(end):]
    return ret

target_tax = get_target_tax()
phrase_mapping: dict[str, TargetClass]
def build_phrase_mapping():
    global phrase_mapping
    phrase_mapping = {}
    for target in target_tax.values():
        for phrase in target.synonyms:
            phrase_mapping[normalize(phrase)] = target

@torch.no_grad()
def _dice(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return 2 * (x * y).sum() / (x.sum() + y.sum())

def image_transform(
    image_path: PathLike,
    max_vision_tokens,
    max_tokens_z: int = 4,
    base_patch_size_z: int = 16,
    base_pool_size_z: int = 2,
    patch_size_xy: int = 16,
    pool_size_xy: int = 2,
    norm: bool = True,
):
    image = load_image(image_path)
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
        image_resized = resize(image)
    else:
        image_resized = image
    image_resized = mt.DivisiblePad(stride)(image_resized)
    image_resized = convert_to_tensor(image_resized)
    image_resized, _ = ensure_rgb(image_resized, contiguous=True)
    if norm:
        image_resized = intensity_norm(image_resized)
    pool_size = (pool_size_z, pool_size_xy, pool_size_xy)
    num_vision_tokens = (np.array(image_resized.shape[1:]) // stride).prod().item()
    return image_resized, image, patch_size, pool_size, num_vision_tokens

def main():
    parser = ArgumentParser()
    parser.add_argument('-c', action=ActionConfigFile)
    parser.add_class_arguments(AlignInstanceSam, 'model')
    parser.add_argument('--ckpt_path', type=Path | None)
    # parser.add_argument('--max_vision_tokens', type=int, required=True)
    args = parser.parse_args()
    args = parser.instantiate_classes(args)
    model: AlignInstanceSam = args.model
    if args.ckpt_path is None:
        print('no checkpoint provided')
    else:
        model.load_state_dict(torch.load(args.ckpt_path)['state_dict'])
    model.to(dtype=torch.bfloat16,  device='cuda')
    build_phrase_mapping()
    precision = HalfPrecision('bf16-true')
    data = orjson.loads((PROCESSED_VG_DATA_ROOT / 'MIMIC-CXR/train.json').read_bytes())
    # data = get_local_data_list('BTCV-Abdomen', Split.TRAIN)
    for item_idx, item in enumerate(tqdm(data)):
        for image_path, plane in zip(item['image'], item['plane']):
            if plane in {'AP', 'PA'}:
                pass
            else:
                pass
        annotated_report: str = item['annotation']
        # findings_ann: str = item['findings-ann']
        # findings_remove_ann = findings_ann.replace('<p>', '').replace('</p>', '')
        # if findings_ann != findings_remove_ann:
        #     pass
        phrases = parse_phrases(annotated_report)
        supported_phrases = []
        unsupported_phrases = []
        supported_indexes = []
        # image =
        for i, phrase in enumerate(phrases):
            if (target := phrase_mapping.get(normalize(phrase))) is None:
                unsupported_phrases.append(phrase)
            else:
                item = {
                    'image':
                }
                model.forward(

                )
                supported_indexes.append(i)
                supported_phrases.append((phrase, target.name))

        # print(supported_phrases)
        # print(unsupported_phrases)
        # targets = list({target for _, target in supported_phrases})
        # if 'adrenal gland' in targets:
        #     targets = ['adrenal gland']
        #     targets.extend(['left adrenal gland', 'right adrenal gland', 'kidney', 'left kidney', 'right kidney'])
        # for image_path in item['image']:
        case_dir = item['dataset_dir'] / 'data' / item['key']
        image_path = case_dir / 'images.pt.zst'
        sparse = Sparse.from_json((case_dir / 'sparse.json').read_bytes())
        targets = [
            target.name
            for target in cytoolz.concat(sparse.targets.values())
        ]
        image = load_pt_zst(image_path)
        image = tvtf.to_dtype(image, torch.float, scale=True)
        roi_size = (32, 256, 256)
        vit_patch_size = (4, 16, 16)
        masks = load_pt_zst(case_dir / 'masks.pt.zst').cuda()
        # masks = nnf.interpolate(masks[None].float(), trans_image.shape[1:], mode='trilinear')[0] > 0.5
        targets_dict = {
            target.name: target
            for y in sparse.targets.values()
            for target in y
        }
        batch = _collate_fn(
            [
                {
                    'patch_size': vit_patch_size,
                    'classes': targets,
                },
            ]
        )
        batch = precision.convert_input(batch)
        image, _ = ensure_rgb(image, contiguous=True)
        image_plot = image.clone()
        # image = intensity_norm(image)
        image = precision.convert_input(image).cuda()
        batch = move_data_to_device(batch, 'cuda')

        def _patch_infer(patch: torch.Tensor):
            batch['image'] = [patch[0]]
            patch_logits = model(batch)[0][None]
            return patch_logits

        with torch.inference_mode():
            for rotate in [0, 1]:
                masks_logits = sliding_window_inference(
                    image[None].rot90(dims=(-1, -2), k=rotate),
                    roi_size,
                    8,
                    _patch_infer,
                    0.8,
                    BlendMode.GAUSSIAN,
                    progress=True,
                )[0]
                sem_masks_pred = masks_logits.sigmoid() > 0.5
                for i, target in enumerate(targets):
                    label = einops.reduce(masks[slice(*targets_dict[target].index_offset)], 'c ... -> ...', 'any')
                    label = label.rot90(dims=(-1, -2), k=rotate)
                    dice = _dice(sem_masks_pred[i], label) * 100
                    print(target, f'rotate={rotate}', f'{dice.item():.2f}')
                    # IndexTrackerBinary(
                    #     image_plot,
                    #     torch.stack([label, sem_masks_pred[i]]),
                    #     # sem_masks_pred[i, 0:1],
                    #     # torch.stack([label[i], sem_masks_pred[i, 0]]),
                    #     choose_max=True,
                    #     title=f'{image_path}\n{target}\ndice={dice:.1f}',
                    #     # title=f'{image_path}\n{target}',
                    # )

if __name__ == '__main__':
    main()
