from pathlib import Path

from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import pandas as pd
import torch

from luolib.utils import process_map
from monai.data import set_track_meta
import monai.transforms as mt

split = 'test'
data_dir = Path('data/origin/vision-language/VinDr-CXR')
df = pd.read_csv(data_dir / f'annotations/annotations_{split}.csv')
loader = mt.LoadImage(reader='itkreader')

targets: dict[str, list] = {}
for t in df.itertuples():
    if t.class_name != 'No finding':
        targets.setdefault(t.image_id, []).append(t)

def plot(image_id: str):
    x: torch.Tensor = loader(data_dir / f'{split}/{image_id}.dicom')[..., 0]
    x = (x - x.min()) / (x.max() - x.min())
    x = x.T
    dpi = 100
    fig, ax = plt.subplots(figsize=(x.shape[0] / dpi, x.shape[1] / dpi))
    ax: plt.Axes
    ax.imshow(x, cmap='gray')
    ax.set_axis_off()
    for t in targets[image_id]:
        if t.class_name == 'No findings':
            continue
        ax.add_patch(
            Rectangle(
                (t.x_min, t.y_min), t.x_max - t.x_min, t.y_max - t.y_min,
                linewidth=1, edgecolor='r', facecolor='none',
            ),
        )
        ax.text(
            (t.x_min + t.x_max) / 2, t.y_min - 15, t.class_name,
            color='white', fontsize=8, backgroundcolor='black', horizontalalignment='center',
        )
    save_dir = Path(f'VinDr-CXR-plot/{image_id}')
    save_dir.mkdir(exist_ok=True, parents=True)
    fig.savefig(save_dir / 'plot.pdf', dpi=dpi, bbox_inches='tight', pad_inches=0)
    fig.savefig(save_dir / 'plot.png', dpi=dpi, bbox_inches='tight', pad_inches=0)

def main():
    set_track_meta(False)
    process_map(
        plot, list(targets.keys()),
        max_workers=0, chunksize=8, ncols=80,
    )

if __name__ == '__main__':
    main()
