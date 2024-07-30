from collections.abc import Hashable
from dataclasses import dataclass
from pathlib import Path

import cytoolz
from detectron2.structures import BoxMode
import einops
from monai.data import box_iou
from monai.data.box_utils import clip_boxes_to_image
import monai.transforms as mt
import numpy as np
import orjson
import pandas as pd
from scipy.sparse.csgraph import connected_components
import torch
import torchvision.transforms.v2.functional as tvtf
from tqdm import tqdm

from luolib.transforms.box_ops import round_boxes
from luolib.utils import get_cuda_device, process_map

from _utils import src_dir, save_dir, local_labels, _save_image

split = 'train'

@dataclass(kw_only=True)
class VinDrCXRDataPoint:
    image_path: Path
    labels: list[Hashable]
    objects: list[Hashable]

def _remove_duplicate(objects: pd.DataFrame, boxes: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    rad_ids = objects['rad_id'].to_numpy()
    class_names = objects['class_name'].to_numpy()
    identical = (
        (box_iou(boxes, boxes) >= 0.9) &
        (rad_ids[:, None] == rad_ids[None, :]) &
        (class_names[:, None] == class_names[None, :])
    )
    nc, labels = connected_components(identical, directed=False)
    cc_list: list[list[int]] = list(cytoolz.groupby(lambda i: labels[i], range(labels.shape[0])).values())
    boxes_ret = np.empty((nc, 4), dtype=np.float64)
    objects_ret = objects.iloc[[cc[0] for cc in cc_list]]
    for i, cc in enumerate(cc_list):
        boxes_ret[i] = boxes[cc].mean(axis=0)
    boxes_ret = round_boxes(boxes_ret)
    return objects_ret, boxes_ret

def _cluster(objects: pd.DataFrame, h: int, w: int):
    boxes = objects[['x_min', 'y_min', 'x_max', 'y_max']].to_numpy()
    boxes = round_boxes(boxes)
    boxes, keep = clip_boxes_to_image(boxes, (w, h))
    if boxes.shape[0] == 0:
        return boxes
    objects = objects[keep]
    objects, boxes = _remove_duplicate(objects, boxes)
    rad_ids = objects['rad_id'].to_numpy()
    iou: np.ndarray = box_iou(boxes.astype(np.float64), boxes.astype(np.float64))
    rad_mask = rad_ids[:, None] != rad_ids[None, :]
    # use a low threshold
    iou_th = 0.25
    nc, labels = connected_components((iou >= iou_th) & rad_mask, directed=False)
    mean_boxes = np.empty((nc, 4), dtype=np.float64)
    for i, cc in enumerate(cytoolz.groupby(lambda i: labels[i], range(labels.shape[0])).values()):
        mean_boxes[i] = boxes[cc].mean(axis=0)
    mean_boxes = round_boxes(mean_boxes)
    return mean_boxes

loader = mt.LoadImage(reader='itkreader', ensure_channel_first=True)
objects_df = pd.read_csv(src_dir / f'annotations/annotations_{split}.csv')
labels_df = pd.read_csv(src_dir / f'annotations/image_labels_{split}.csv')

def process_item(data_point: VinDrCXRDataPoint):
    key = data_point.image_path.stem
    image = einops.rearrange(loader(data_point.image_path).as_tensor(), '1 w h 1 -> 1 h w')
    image = image.to(device=get_cuda_device())
    image = tvtf.to_dtype(image / image.max(), dtype=torch.uint8, scale=True)
    image = tvtf.equalize(image)
    image_path = save_dir / f'{split}/{key}.jpeg'
    _save_image(image, image_path)
    labels: pd.DataFrame = labels_df.loc[data_point.labels]
    objects: pd.DataFrame = objects_df.loc[data_point.objects]
    annotations = []
    if labels['No finding'].all():
        assert (objects['class_name'] == 'No finding').all()
    else:
        for class_name, class_objects_indexes in objects.groupby('class_name').groups.items():
            if class_name in {'No finding', 'Other lesion'}:
                continue
            boxes_np = _cluster(objects.loc[class_objects_indexes], *image.shape[1:])
            boxes = boxes_np.tolist()
            class_idx = local_labels.index(class_name)
            for box in boxes:
                annotations.append({
                    'bbox': box,
                    'bbox_mode': BoxMode.XYXY_ABS,
                    'category_id': class_idx,
                })
    return {
        'file_name': str(image_path),
        'height': image.shape[1],
        'width': image.shape[2],
        'annotations': annotations,
    }

def main():
    image_objects = {}
    for index, row in tqdm(objects_df.iterrows(), 'iterating objects', total=objects_df.shape[0], dynamic_ncols=True):
        image_objects.setdefault(row['image_id'], []).append(index)
    image_labels = {}
    for index, row in tqdm(labels_df.iterrows(), 'iterating labels', total=labels_df.shape[0], dynamic_ncols=True):
        image_labels.setdefault(row['image_id'], []).append(index)
    items = (
        VinDrCXRDataPoint(
            image_path=src_dir / f'{split}/{image_id}.dicom',
            labels=labels,
            objects=image_objects[image_id],
        )
        for image_id, labels in image_labels.items()
    )
    (save_dir / split).mkdir(exist_ok=True, parents=True)
    items = process_map(process_item, items, total=len(image_labels), max_workers=24, chunksize=10, dynamic_ncols=True)
    (save_dir / f'{split}.json').write_bytes(orjson.dumps(items, option=orjson.OPT_INDENT_2))

if __name__ == '__main__':
    main()
