from pathlib import Path

import einops
import inflect
from jsonargparse import ArgumentParser
from lightning.fabric.utilities import move_data_to_device
from lightning.pytorch.plugins import HalfPrecision
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import numpy as np
import orjson
from peft import PeftModel
import torch
from tqdm import tqdm
from transformers.modeling_outputs import CausalLMOutputWithPast

from monai.config import NdarrayOrTensor
from monai.utils import convert_to_tensor

from mmmm.data import get_target_tax
from mmmm.data.datamodule import _collate_fn
from mmmm.data.dataset.local.template import gen_general_conv
from mmmm.data.defs import PROCESSED_VG_DATA_ROOT
from mmmm.data.target_tax import TargetClass
from mmmm.data.utils import prepare_vlm_inputs
from mmmm.misc import image_transform
from mmmm.models.mmmm import MMMMForCausalLM, from_pretrained

class IndexTracker:
    def __init__(
        self,
        img: NdarrayOrTensor,
        seg: NdarrayOrTensor | None = None,
        boxes: torch.Tensor | None = None,
        block: bool = True,
        title: str = "",
        zyx: bool = True,
        choose_max: bool = False,
        vmin: float | None = None,
        vmax: float | None = None,
        labels: list[str] | None = None,
    ):
        img_t: torch.Tensor = convert_to_tensor(img, device='cpu')
        if seg is not None:
            seg: torch.Tensor = convert_to_tensor(seg, device='cpu')
            if seg.ndim == 3:
                labels = seg.unique()
                masks = labels[:, None, None, None] == seg[None]
            else:
                masks = seg.bool()
        else:
            masks = torch.empty(0, *img_t.shape)
        if boxes is None:
            boxes = torch.empty(0, 6, dtype=torch.int64)
        elif not zyx:
            raise NotImplementedError
        if not zyx:
            img_t = einops.rearrange(img_t, '... h w d -> ... d h w')
            masks = einops.rearrange(masks, 'c h w d -> c d h w')
        if boxes.is_floating_point():
            boxes = boxes * einops.repeat(torch.tensor(img_t.shape[-3:]), 'd -> (l2 d)', l2=2)
            boxes = boxes.round().long()
        # make matplotlib happy
        img_t = img_t.mT
        img_t = einops.rearrange(img_t, '... d w h -> d w h ...')
        masks = masks.mT
        img_cmap = 'gray' if img_t.ndim == 3 else None

        fig, ax = plt.subplots()
        fig: plt.Figure
        ax: plt.Axes
        ax.set_title('use scroll wheel to navigate images')
        self.ax = ax
        self.img = img_t.numpy()
        self.masks = masks.numpy()
        self.boxes = boxes.numpy()
        self.slices = img_t.shape[0]
        self.ind = self.slices // 2
        if choose_max:
            self.ind = einops.reduce(masks, 'c d h w -> d', 'sum').argmax().item()
        self.ax_img = ax.imshow(self.img[self.ind], img_cmap, vmin=vmin, vmax=vmax)
        colormap: ListedColormap = matplotlib.colormaps['tab20']
        self.ax_masks = [
            ax.imshow(
                self.masks[c, self.ind],
                ListedColormap(['none', colormap.colors[c]]),
                # vmax=num_classes,
                alpha=0.5,
            )
            for c in range(masks.shape[0])
        ]
        # from matplotlib.colorbar import Colorbar
        # cbar: matplotlib.colorbar.Colorbar = fig.colorbar(self.ax_seg)
        # cbar.set_ticks(np.arange(num_classes), labels=labels)

        self.update()
        fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        plt.title(title)
        plt.show(block=block)

    def on_scroll(self, event):
        # print("%s %s" % (event.button, event.step))
        last = self.ind
        if event.button == 'up':
            self.ind = min(self.slices - 1, self.ind + 1)
        else:
            self.ind = max(0, self.ind - 1)
        if last != self.ind:
            self.update()

    def update(self):
        self.ax.set_ylabel('slice %s' % self.ind)
        for patch in list(self.ax.patches):
            patch.remove()
        for box in self.boxes:
            if self.ind in range(box[0], box[3]):
                self.ax.add_patch(
                    Rectangle(
                        (box[1], box[2]), (box[4] - box[1]), (box[5] - box[2]),
                        linewidth=1, edgecolor='r', facecolor='none',
                    ),
                )
        self.ax_img.set_data(self.img[self.ind])
        for c, ax_mask in enumerate(self.ax_masks):
            ax_mask.set_data(self.masks[c, self.ind])
        self.ax.figure.canvas.draw()


engine = inflect.engine()
_stop_words = {'the'}

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

def main():
    parser = ArgumentParser()
    parser.add_argument('--adapter_path', type=Path, required=True)
    parser.add_subclass_arguments(MMMMForCausalLM, 'model')
    parser.add_argument('--max_vision_tokens', type=int, required=True)
    args = parser.parse_args()
    args = parser.instantiate_classes(args)
    model: MMMMForCausalLM = args.model
    PeftModel.from_pretrained(model, args.adapter_path)
    tokenizer = model.tokenizer
    # model, tokenizer = from_pretrained(model_conf_path, args.adapter_path)
    model.to(dtype=torch.bfloat16,  device='cuda')
    R = np.random.RandomState(42)
    build_phrase_mapping()
    data = orjson.loads((PROCESSED_VG_DATA_ROOT / 'CT-RATE/train.json').read_bytes())
    precision = HalfPrecision('bf16-true')
    for item_idx, item in enumerate(tqdm(data)):
        findings_ann: str = item['findings-ann']
        findings_remove_ann = findings_ann.replace('<p>', '').replace('</p>', '')
        if findings_ann != findings_remove_ann:
            pass
        phrases = parse_phrases(findings_ann)
        supported_phrases = []
        unsupported_phrases = []
        for phrase in phrases:
            if (target := phrase_mapping.get(normalize(phrase))) is None:
                unsupported_phrases.append(phrase)
            else:
                supported_phrases.append((phrase, target.name))
        print(supported_phrases)
        print(unsupported_phrases)
        targets = list({target for _, target in supported_phrases})
        if 'adrenal gland' in targets:
            targets = ['adrenal gland']
            targets.extend(['left adrenal gland', 'right adrenal gland', 'kidney', 'left kidney', 'right kidney'])
        conv, grounding_classes = gen_general_conv(
            targets,
            [],
            True,
            False,
            tokenizer,
            target_tax,
            R,
        )
        for image_path in item['image']:
            image, patch_size, pool_size, num_vision_tokens = image_transform(
                image_path, max_vision_tokens=args.max_vision_tokens,
            )
            vlm_inputs, _ = prepare_vlm_inputs(
                conv,
                tokenizer,
                num_vision_tokens,
                inference=False,
                grounding=False,
                bop_weight=1.,  # dummy
            )
            vlm_inputs.pop('labels')
            batch = _collate_fn(
                [
                    {
                        'image': image,
                        'patch_size': patch_size,
                        'pool_size': pool_size,
                        'vlm_inputs': vlm_inputs,
                    },
                ]
            )
            batch = precision.convert_input(batch)
            batch = move_data_to_device(batch, 'cuda')
            vlm_inputs = batch['vlm_inputs']
            image_plot = (image - image.min()) / (image.max() - image.min())
            with torch.inference_mode():
                vlm_output: CausalLMOutputWithPast = model(
                    **vlm_inputs,
                    image=batch['image'],
                    patch_size=batch['patch_size'],
                    pool_size=batch['pool_size'],
                    return_dict=True,
                    output_hidden_states=True,
                )
                vg_output = model.visual_grounding(
                    # shift as suggested by GLaMM: https://github.com/mbzuai-oryx/groundingLMM/issues/16
                    vlm_inputs['input_ids'][:, 1:],
                    vlm_output.hidden_states[-1][:, :-1],
                    batch['image'],
                    batch['patch_size'],
                )
                masks_pred = vg_output.masks_logits[0].sigmoid() > 0.5

                for i, target in enumerate(grounding_classes):
                    print(target)
                    IndexTracker(image_plot, masks_pred[i, 0:1], choose_max=True, title=f'{image_path}\n{target}')

if __name__ == '__main__':
    main()
