from __future__ import annotations as _

import json
from dataclasses import dataclass
from idlelib.pyparse import trans
from operator import index
from pathlib import Path
from typing import TypedDict

import numpy as np
import orjson
import torch
import torchvision.transforms.v2.functional as tvtf
from monai import transforms as mt

import mmmm.data.dataset._dataset as _dataset
from luolib.transforms.box_ops import round_boxes
from luolib.utils import load_pt_zst
from luolib.utils.misc import ensure_rgb
from mmmm.tokenizer import MMMMTokenizer
from .misc import get_max_resize, intensity_norm, toss, load_image_byte_as_float, get_patch_size_z, \
    spatial_transform_image_labels, norm_boxes
from .vl import REFERRINGS, REPORT_PROMPTS
from ..defs import ConvTurn, Split, PROCESSED_VG_DATA_ROOT
from ..target_tax import get_target_tax
from ..utils import prepare_vlm_inputs

def get_grg_data_list(name: str, split: Split) -> list:
    dataset_dir = PROCESSED_VG_DATA_ROOT / name
    split_filename = f'{split}.json'
    data = orjson.loads((dataset_dir / split_filename).read_bytes())
    if name == 'MIMIC-CXR':
        # filter items with at least one frontal image
        data = [
            item for item in data
            if any(plane in ('PA', 'AP') for plane in item['plane'])
        ]
    for item in data:
        item['dataset'] = name
    return data

@dataclass
class GRGTransConf:
    max_tokens: int
    max_tokens_z: int
    log2_patch_size_z_std: float = 0.25
    ac_ratio: float = 0.2
    modality_prob: float = 0.2
    plane_prob: float = 0.2
    report_ratio: float = 0.8

class GRGDataPoint(TypedDict):
    image: list[str]
    modality: list[str]
    findings: str
    impression: str
    anomaly_pos: list[str]
    anomaly_neg: list[str]

def resolve_image_path(vl_path: str, dataset_name: str) -> tuple[Path, str, Path]:
    suffix = '.pt.zst'
    assert vl_path.endswith(suffix)
    vl_path = Path(vl_path)
    key = vl_path.name[:-len(suffix)]
    # sorry
    if dataset_name == 'MIMIC-CXR':
        base_dir = Path(PROCESSED_VG_DATA_ROOT, *Path(vl_path).parts[-6:-1])
        grg_path = base_dir / f'{key}.jpg'
    else:
        base_dir = Path(PROCESSED_VG_DATA_ROOT, *Path(vl_path).parts[-4:-1])
        grg_path = base_dir / f'{key}-t/{key}.pt.zst'
    return grg_path, key, base_dir

@dataclass
class GRGTransConf:
    max_tokens: int
    max_tokens_z: int
    log2_patch_size_z_std: float = 0.25
    ac_ratio: float = 0.2
    modality_prob: float = 0.2
    plane_prob: float = 0.2
    report_ratio: float = 0.8
    grounding_prob: float = 0.99
    max_num_vg: int = 12

class GRGTransform(mt.RandomizableTransform):
    def __init__(
        self,
        conf: _dataset.DatasetConf,
        tokenizer: MMMMTokenizer,
        inference: bool = False,
    ):
        super().__init__()
        self.conf = conf
        self.tokenizer = tokenizer
        self.inference = inference
        self.target_tax = get_target_tax()

    def _reduce_items_(self, mask: torch.BoolTensor, max_num: int):
        if (num := mask.sum().item()) <= max_num:
            return
        indexes, = torch.where(mask)
        neg = self.R.choice(indexes.numpy(), num - max_num, replace=False)
        mask[neg] = False

    def __call__(self, data: dict):
        conf = self.conf
        trans_conf = conf.grg_trans
        dataset: str = data['dataset']

        # 1. sample image & resolve image path
        image_candidates = np.arange(len(data['image']))
        if dataset == 'MIMIC-CXR':
            # only use frontal view of MIMIC-CXR for report generation
            frontal_mask = np.array([plane in {'PA', 'AP'} for plane in data['plane']])
            # see `get_grg_data_list`
            assert frontal_mask.any()
            image_candidates = image_candidates[frontal_mask]
        image_idx = self.R.choice(image_candidates).item()
        vl_image_path = data['image'][image_idx]
        image_path, key, base_dir = resolve_image_path(vl_image_path, dataset)

        # 2. load image, calculate patch size, resize
        image = load_image_byte_as_float(image_path)
        patch_size_z, pool_size_z, stride_z, tokens_z = get_patch_size_z(
            conf.base_vit_patch_size_z,
            conf.base_pool_size_z,
            size_z := image.shape[1],
            trans_conf.max_tokens_z,
            trans_conf.log2_patch_size_z_std,
            self.R,
        )
        patch_size = (patch_size_z, conf.vit_patch_size_xy, conf.vit_patch_size_xy)
        stride = (stride_z, conf.stride_xy, conf.stride_xy)
        resize_shape = (
            min(size_z, tokens_z * stride_z),  # do not resize z if unnecessary
            *get_max_resize(
                image.shape[2:],
                conf.stride_xy,
                trans_conf.max_tokens // tokens_z,
            ),
        )

        # 3. load localization labels
        tags: list[dict] = data['tags']
        grounding = toss(self.R, trans_conf.grounding_prob)
        # vg_label_mask: torch.BoolTensor
        if grounding:
            vg_label_mask = torch.zeros(len(tags), dtype=torch.bool)
        else:
            vg_label_mask = torch.zeros(0, dtype=torch.bool)
        if (box_path := (base_dir / f'{key}_box.json')).exists():
            masks = None
            if grounding:
                target_boxes: dict[str, ...] = orjson.loads(box_path.read_bytes())
                for name, boxes in target_boxes.items():
                    # mode: XY(Z)XY(Z)
                    boxes = torch.tensor(boxes, dtype=torch.float64)
                    if boxes.shape[1] == 4:
                        boxes_3d = torch.empty(boxes.shape[0], 6, dtype=torch.float64)
                        boxes_3d[:, 0] = 0
                        boxes_3d[:, 3] = 1
                        boxes_3d[:, [1, 2, 4, 5]] = boxes
                        boxes = boxes_3d
                    target_boxes[name] = boxes
                for i, tag in enumerate(tags):
                    if target_boxes.get(tag['target']) is not None:
                        vg_label_mask[i] = True
                self._reduce_items_(vg_label_mask, trans_conf.max_num_vg)
                boxes_list = []
                index_offset = 0
                index_offsets = []
                for i, tag in enumerate(tags):
                    if not vg_label_mask[i]:
                        continue
                    boxes_list.append(boxes)
                    index_offsets.append((index_offset, index_offset := index_offset + boxes.shape[0]))
                if len(boxes_list) == 0:
                    boxes = torch.empty(0, 6, dtype=torch.int64)
                    index_offsets = torch.empty(0, 2, dtype=torch.int64)
                else:
                    boxes = round_boxes(torch.cat(boxes_list, dim=0))
                    index_offsets = torch.tensor(index_offsets)
            else:
                boxes = None
                index_offsets = None
        else:
            boxes = None
            index_offsets = None
            if grounding:
                targets: list[str] = json.loads((base_dir / f'{key}_seg.json').read_bytes())
                ref_masks: torch.BoolTensor = load_pt_zst(image_path.with_name(f'{key}_seg.pt.zst'))
                target_to_idx = {
                    target: i
                    for i, target in enumerate(targets)
                }
                for i, tag in enumerate(tags):
                    if tag['target'] in target_to_idx:
                        vg_label_mask[i] = True
                self._reduce_items_(vg_label_mask, trans_conf.max_num_vg)
                masks_list = []
                for i, tag in enumerate(tags):
                    if not vg_label_mask[i]:
                        continue
                    masks_list.append(ref_masks[target_to_idx[tag['target']]])
                if len(masks_list) > 0:
                    masks = torch.stack(masks_list, dim=0)
                else:
                    masks = torch.empty(0, *image.shape[1:], dtype=torch.bool)
            else:
                masks = None

        # 4. prepare data
        image, masks, boxes = spatial_transform_image_labels(image, masks, boxes, resize_shape, stride, R=self.R)
        if boxes is not None:
            boxes = norm_boxes(boxes, image.shape[1:])
        image, _ = ensure_rgb(image, contiguous=True)
        # no normalization for grounding image, see `LocalTransform`
        grounding_image = image
        image = intensity_norm(image)
        # 5. generate conversation
        referring: str = self.R.choice(REFERRINGS)
        conversation = []
        report: str = data['ref_report']
        # inject bracket tokens
        tokenizer = self.tokenizer
        if grounding:
            last_end = 0
            report_pieces = []
            for tag in tags:
                start = tag['start']
                end = tag['end']
                if start > 1 and report[start - 1] == ' ':
                    # ask how does llama tokenizer handle "a b", see also in `MMMMTokenizer.wrap_name`
                    start -= 1
                report_pieces.extend([
                    report[last_end:start],
                    tokenizer.bop_token,
                    report[start:end],
                    tokenizer.eop_token,
                ])
                last_end = end
            report = ''.join((*report_pieces, report[last_end:]))
        conversation.append(
            ConvTurn(self.R.choice(REPORT_PROMPTS).format(referring), report),
        )
        vlm_inputs, conversation_text = prepare_vlm_inputs(
            conversation,
            self.tokenizer,
            (np.array(image.shape[1:]) // stride).prod().item(),
            inference=self.inference,
            grounding=grounding,
            max_seq_len=conf.max_seq_len,
            bop_weight=conf.bop_weight,
        )
        data = {
            'src': (dataset, str(image_path)),
            'image': image,
            'grounding_image': grounding_image,
            'patch_size': patch_size,
            'pool_size': (pool_size_z, conf.pool_size_xy, conf.pool_size_xy),
            'vlm_inputs': vlm_inputs,
            'masks': masks,
            'boxes': None if boxes is None else boxes.float(),
            'index_offsets': index_offsets,
            'instance_mask': boxes is not None,
            'vg_label_mask': vg_label_mask,
        }
        return data