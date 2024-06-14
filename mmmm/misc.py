from pathlib import Path

import einops
import math
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import numpy as np
import torch
from torchvision.io import read_image
import torchvision.transforms.v2.functional as tvtf

from luolib.types import PathLike
from luolib.utils import load_pt_zst
from luolib.utils.misc import ensure_rgb
from monai.config import NdarrayOrTensor
import monai.transforms as mt
from monai.utils import InterpolateMode, convert_to_tensor

from mmmm.data.dataset.misc import get_max_resize, intensity_norm

def load_image(image_path: PathLike):
    image_path_str = str(image_path)
    if image_path_str.endswith('.pt.zst'):
        image = load_pt_zst(Path(image_path))
    else:
        image = read_image(image_path_str)
        image = einops.rearrange(image, 'c h w -> c 1 h w')
    image = tvtf.to_dtype(image, torch.float, scale=True)
    return image

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

class IndexTrackerBinary:
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
        # rotate to make matplotlib happy
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
