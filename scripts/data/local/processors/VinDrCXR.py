from dataclasses import dataclass
from pathlib import Path

import cytoolz
from monai.data import MetaTensor, box_iou
from monai.data.box_utils import clip_boxes_to_image
import monai.transforms as mt
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
import torch

from luolib.transforms.box_ops import round_boxes
from luolib.types import tuple3_t

from ._base import DataPoint, DefaultImageLoaderMixin, Processor

@dataclass(kw_only=True)
class VinDrCXRDataPoint(DataPoint):
    complete_anomaly = True
    labels: list[pd.DataFrame]
    objects: list[pd.DataFrame]

local_labels = {
    'Aortic enlargement': 'aortic enlargement',
    'Atelectasis': 'atelectasis',
    'Calcification': 'calcification',
    'Cardiomegaly': 'cardiomegaly',
    'Clavicle fracture': 'clavicle fracture',
    'Consolidation': 'pulmonary consolidation',
    'Edema': 'pulmonary edema',
    'Emphysema': 'pulmonary emphysema',
    'Enlarged PA': 'pulmonary artery enlargement',
    'ILD': 'interstitial lung disease',
    'Infiltration': 'pulmonary infiltrate',
    'Lung cavity': 'pulmonary cavity',
    'Lung cyst': 'pulmonary cyst',
    'Lung Opacity': 'pulmonary opacification',
    'Mediastinal shift': 'mediastinal shift',
    'Nodule/Mass': 'lung nodule',
    # 'Other lesion': 'other lesion',
    'Pleural effusion': 'pleural effusion',
    'Pleural thickening': 'pleural thickening',
    'Pneumothorax': 'pneumothorax',
    'Pulmonary fibrosis': 'pulmonary fibrosis',
    'Rib fracture': 'rib fracture',
}

global_labels = {
    'COPD': 'chronic obstructive pulmonary disease',
    'Lung tumor': 'lung tumor',
    'Pneumonia': 'pneumonia',
    'Tuberculosis': 'tuberculosis',
    'Other diseases': 'other diseases',
    # 'No finding': 'no finding',
}

def _remove_duplicate(objects: list[pd.Series], boxes: np.ndarray) -> tuple[list[pd.Series], np.ndarray]:
    rad_ids = np.array([x['rad_id'] for x in objects])
    identical = (boxes[:, None] == boxes[None, :]).all(-1) & (rad_ids[:, None] == rad_ids[None, :])
    nc, labels = connected_components(identical, directed=False)
    objects_ret = [None] * nc
    boxes_ret = np.empty((nc, 6), dtype=boxes.dtype)
    for i in range(boxes.shape[0]):
        if objects_ret[label := labels[i]] is not None:
            continue
        objects_ret[label] = objects[i]
        boxes_ret[label] = boxes[i]
    return objects_ret, boxes_ret

def _cluster(objects: list[pd.Series], image_size: tuple3_t[int]) -> torch.Tensor:
    boxes = np.array([
        (0, obj['x_min'], obj['y_min'], 1, obj['x_max'], obj['y_max'])
        for obj in objects
    ])
    boxes = round_boxes(boxes)
    boxes, keep = clip_boxes_to_image(boxes, image_size)
    if boxes.shape[0] == 0:
        return boxes
    objects = [obj for i, obj in enumerate(objects) if keep[i]]
    objects, boxes = _remove_duplicate(objects, boxes)
    rad_ids = np.array([x['rad_id'] for x in objects])
    iou: np.ndarray = box_iou(boxes.astype(np.float64), boxes.astype(np.float64))
    num_rads = len(np.unique(rad_ids))
    rad_mask = rad_ids[:, None] != rad_ids[None, :]
    # use a low threshold
    iou_th = 0.25
    step_size = 0.05
    while True:
        nc, labels = connected_components((iou >= iou_th) & rad_mask, directed=False)
        _, counts = np.unique(labels, return_counts=True)
        if (iou_th := iou_th + step_size) > 1 or counts.max() <= num_rads:
            break
    assert labels.min() == 0 and labels.max() == nc - 1
    mean_boxes = np.concatenate([boxes[labels == i].mean(axis=0, keepdims=True) for i in range(nc)])
    return torch.from_numpy(mean_boxes)

class VinDrCXRProcessor(DefaultImageLoaderMixin, Processor):
    name = 'VinDr-CXR'
    # pydicom reader seems to produce reversed intensities
    image_reader = 'itkreader'

    def image_loader(self, path: Path) -> MetaTensor:
        loader = mt.LoadImage(self.image_reader, image_only=True, dtype=self.image_dtype, ensure_channel_first=True)
        # FIXME: h-w orientation
        image = loader(path)
        assert image.ndim == 4 and image.shape[-1] == 1
        image = self._adapt_to_3d(image[..., 0])
        return image

    def load_annotations(
        self, data_point: VinDrCXRDataPoint, images: MetaTensor,
    ) -> tuple[list[str], set[str], MetaTensor | None, torch.Tensor | None]:
        labels = pd.concat(data_point.labels, axis='columns').T
        if labels['No finding'].all():
            targets = []
            neg_targets = set(local_labels.values())
            boxes = None
        else:
            neg_targets = []
            for local_label_key in local_labels:
                if not labels[local_label_key].any():
                    neg_targets.append(local_labels[local_label_key])
            neg_targets = set(neg_targets)
            targets = []
            boxes = []
            class_objects: dict = cytoolz.groupby(lambda x: x.class_name, data_point.objects)  # type: ignore
            class_objects.pop('No finding', None)
            for class_name_key, objects in class_objects.items():
                objects_boxes = _cluster(objects, images.shape[1:])
                targets.extend([local_labels[class_name_key]] * objects_boxes.shape[0])
                boxes.append(objects_boxes)
            boxes = torch.cat(boxes)
        return targets, neg_targets, None, boxes

    def get_data_points(self):
        # only process the training set here
        image_objects = {}
        objects_df = pd.read_csv(self.dataset_root / 'annotations/annotations_train.csv')
        objects_df = objects_df[objects_df['class_name'] != 'Other lesion']
        for _, row in objects_df.iterrows():
            image_objects.setdefault(row['image_id'], []).append(row)
        labels_df = pd.read_csv(self.dataset_root / 'annotations/image_labels_train.csv')
        labels_df.drop(columns='Other lesion', inplace=True)
        image_labels = {}
        for _, row in labels_df.iterrows():
            image_labels.setdefault(row['image_id'], []).append(row)
        ret = [
            VinDrCXRDataPoint(
                key=key,
                images={'X-ray': self.dataset_root / f'train/{key}.dicom'},
                labels=labels,
                objects=image_objects[key],
            )
            for key, labels in image_labels.items() if key in image_objects
        ]
        return ret, None
