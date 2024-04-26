from pathlib import Path

from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import pandas as pd
import torch

from monai.data import set_track_meta
import monai.transforms as mt

split = 'test'
data_dir = Path('data/origin/vision-language/VinDr-CXR')
df = pd.read_csv(data_dir / f'annotations/annotations_{split}.csv')
loader = mt.LoadImage(reader='itkreader')

def plot(image_id: str):
    x: torch.Tensor = loader(data_dir / f'{split}/{image_id}.dicom')[..., 0]
    x = (x - x.min()) / (x.max() - x.min())
    x = x.T
    case = df[df['image_id'] == image_id]
    dpi = 100
    fig, ax = plt.subplots(figsize=(x.shape[0] / dpi, x.shape[1] / dpi))
    ax: plt.Axes
    ax.imshow(x, cmap='gray')
    ax.set_axis_off()
    for t in case.itertuples():
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
    plt.show()
    fig.savefig('vindr-plot.pdf', dpi=dpi, bbox_inches='tight', pad_inches=0)
    fig.savefig('vindr-plot.png', dpi=dpi, bbox_inches='tight', pad_inches=0)

def main():
    set_track_meta(False)
    plot('01ded16689539deb30d0981fafd18465')

if __name__ == '__main__':
    main()
