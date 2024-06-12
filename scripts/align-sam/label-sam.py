from pathlib import Path

import cytoolz
import einops
import inflect
import matplotlib
import torch
from jsonargparse import ActionConfigFile, ArgumentParser
from lightning.fabric.utilities import move_data_to_device
from lightning.pytorch.plugins import HalfPrecision
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import torchvision.transforms.v2.functional as tvtf
from tqdm import tqdm

from luolib.utils import load_pt_zst
from luolib.utils.misc import ensure_rgb
from monai.config import NdarrayOrTensor
from monai.inferers import sliding_window_inference
from monai.utils import BlendMode, convert_to_tensor

from mmmm.data import get_target_tax
from mmmm.data.defs import Split
from mmmm.data.sparse import Sparse
from mmmm.data.target_tax import TargetClass
from mmmm.models.segvol.modeling.sam import InstanceSamOutput

from _data import _collate_fn
from _model import AlignSam

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

@torch.no_grad()
def _dice(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return 2 * (x * y).sum() / (x.sum() + y.sum())

def main():
    parser = ArgumentParser()
    parser.add_argument('-c', action=ActionConfigFile)
    parser.add_class_arguments(AlignSam, 'model')
    parser.add_argument('--ckpt_path', type=Path | None)
    # parser.add_argument('--max_vision_tokens', type=int, required=True)
    args = parser.parse_args()
    args = parser.instantiate_classes(args)
    model: AlignSam = args.model
    if args.ckpt_path is None:
        print('no checkpoint provided')
    else:
        model.load_state_dict(torch.load(args.ckpt_path)['state_dict'])
    model.to(dtype=torch.bfloat16,  device='cuda')
    # R = np.random.RandomState(42)
    build_phrase_mapping()
    precision = HalfPrecision('bf16-true')
    # data = orjson.loads((PROCESSED_VG_DATA_ROOT / 'CT-RATE/train.json').read_bytes())
    from mmmm.data.dataset.local import get_local_data_list
    data = get_local_data_list('BTCV-Abdomen', Split.TRAIN)
    for item_idx, item in enumerate(tqdm(data)):
        # findings_ann: str = item['findings-ann']
        # findings_remove_ann = findings_ann.replace('<p>', '').replace('</p>', '')
        # if findings_ann != findings_remove_ann:
        #     pass
        # phrases = parse_phrases(findings_ann)
        # supported_phrases = []
        # unsupported_phrases = []
        # for phrase in phrases:
        #     if (target := phrase_mapping.get(normalize(phrase))) is None:
        #         unsupported_phrases.append(phrase)
        #     else:
        #         supported_phrases.append((phrase, target.name))
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
        roi_size = (48, 192, 192)
        vit_patch_size = (8, 16, 16)
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
            output: InstanceSamOutput = model(batch)
            return output.masks_logits[0][None, :, 0]

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
                    # IndexTracker(
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
