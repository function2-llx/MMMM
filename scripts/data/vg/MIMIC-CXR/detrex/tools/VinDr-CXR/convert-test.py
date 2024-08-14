from collections.abc import Hashable
from dataclasses import dataclass
from pathlib import Path

import einops
import monai.transforms as mt
import orjson
import pandas as pd
import torch
import torchvision.transforms.v2.functional as tvtf
from monai.data.box_utils import clip_boxes_to_image
from tqdm import tqdm

from _utils import src_dir, save_dir, local_labels, _save_image
from detectron2.structures import BoxMode
from luolib.transforms.box_ops import round_boxes
from luolib.utils import get_cuda_device, process_map

@dataclass(kw_only=True)
class VinDrCXRDataPoint:
    image_path: Path
    objects: list[Hashable]

split = 'test'
# save_dir = save_dir.with_name(f'{save_dir.name}-sub')

loader = mt.LoadImage(reader='itkreader', ensure_channel_first=True)
objects_df = pd.read_csv(src_dir / f'annotations/annotations_{split}.csv')
labels_df = pd.read_csv(src_dir / f'annotations/image_labels_{split}.csv', index_col='image_id')

def process_item(data_point: VinDrCXRDataPoint):
    key = data_point.image_path.stem
    image = einops.rearrange(loader(data_point.image_path).as_tensor(), '1 w h 1 -> 1 h w')
    h, w = image.shape[1:]
    image = image.to(device=get_cuda_device())
    image = tvtf.to_dtype(image / image.max(), dtype=torch.uint8, scale=True)
    image = tvtf.equalize(image)
    file_name = f'{split}/{key}.jpeg'
    _save_image(image, save_dir / file_name)
    objects: pd.DataFrame = objects_df.loc[data_point.objects]
    annotations = []
    if labels_df.at[key, 'No finding']:
        assert len(objects) == 1 and objects.iloc[0]['class_name'] == 'No finding'
    else:
        boxes = objects[['x_min', 'y_min', 'x_max', 'y_max']].to_numpy()
        boxes = round_boxes(boxes)
        boxes, keep = clip_boxes_to_image(boxes, (w, h))
        for i, class_name in enumerate(objects[keep]['class_name']):
            if class_name == 'No finding':
                raise ValueError
            if class_name == 'Other lesion':
                continue
            class_idx = local_labels.index(class_name)
            annotations.append({
                'bbox': boxes[i].tolist(),
                'bbox_mode': BoxMode.XYXY_ABS,
                'category_id': class_idx,
            })
    return {
        'file_name': file_name,
        'image_id': key,
        'height': image.shape[1],
        'width': image.shape[2],
        'annotations': annotations,
    }

def main():
    image_objects = {}
    for index, row in tqdm(objects_df.iterrows(), 'iterating objects', total=objects_df.shape[0], dynamic_ncols=True):
        image_objects.setdefault(row['image_id'], []).append(index)
    items = (
        VinDrCXRDataPoint(
            image_path=src_dir / f'{split}/{image_id}.dicom',
            objects=objects,
        )
        for image_id, objects in image_objects.items()
    )
    # import cytoolz
    # items = cytoolz.take(100, items)
    (save_dir / split).mkdir(exist_ok=True, parents=True)
    items = process_map(process_item, items, total=len(image_objects), max_workers=24, chunksize=10, dynamic_ncols=True)
    (save_dir / f'{split}.json').write_bytes(orjson.dumps(items, option=orjson.OPT_INDENT_2))

if __name__ == '__main__':
    main()
