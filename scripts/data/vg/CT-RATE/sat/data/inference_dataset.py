import math

import einops
import torch
from torch.utils.data import Dataset

import monai
from monai.data import MetaTensor

def split_3d(image_tensor, crop_size=(288, 288, 96)):
    image_tensor = image_tensor.as_tensor()
    # C H W D
    interval_h, interval_w, interval_d = crop_size[0] // 2, crop_size[1] // 2, crop_size[2] // 2
    split_idx = []
    split_patch = []

    c, h, w, d = image_tensor.shape
    h_crop = max(math.ceil(h / interval_h) - 1, 1)
    w_crop = max(math.ceil(w / interval_w) - 1, 1)
    d_crop = max(math.ceil(d / interval_d) - 1, 1)

    for i in range(h_crop):
        h_s = i * interval_h
        h_e = h_s + crop_size[0]
        if h_e > h:
            h_s = h - crop_size[0]
            h_e = h
            if h_s < 0:
                h_s = 0
        for j in range(w_crop):
            w_s = j * interval_w
            w_e = w_s + crop_size[1]
            if w_e > w:
                w_s = w - crop_size[1]
                w_e = w
                if w_s < 0:
                    w_s = 0
            for k in range(d_crop):
                d_s = k * interval_d
                d_e = d_s + crop_size[2]
                if d_e > d:
                    d_s = d - crop_size[2]
                    d_e = d
                    if d_s < 0:
                        d_s = 0
                split_idx.append([h_s, h_e, w_s, w_e, d_s, d_e])
                split_patch.append(image_tensor[:, h_s:h_e, w_s:w_e, d_s:d_e])

    return split_patch, split_idx

def contains(text, key):
    if isinstance(key, str):
        return key in text
    elif isinstance(key, list):
        for k in key:
            if k in text:
                return True
        return False

def quantile(x: torch.Tensor, q: float, dim: int | None = None, keepdim: bool = False) -> torch.Tensor:
    # workaround for https://github.com/pytorch/pytorch/issues/64947
    assert 0 <= q <= 1
    if dim is None:
        x = x.view(-1)
        k = round(x.numel() * q)
        dim = 0
    else:
        k = round(x.shape[dim] * q)
    if k == 0:
        k = 1
    return x.kthvalue(k, dim, keepdim).values

def Normalization(image: torch.Tensor, modality: str):
    image = image.contiguous()
    if modality == 'ct':
        lower_bound, upper_bound = -500, 1000
    else:
        lower_bound = quantile(image, 0.5 / 100)
        upper_bound = quantile(image, 99.5 / 100)
    image = image.clip(lower_bound, upper_bound)
    image = (image - image.mean()) / image.std()
    return image

def load_image(datum):
    orientation_code = datum['orientation_code'] if 'orientation_code' in datum else "RAS"

    monai_loader = monai.transforms.Compose(
        [
            monai.transforms.LoadImaged(keys=['image']),
            monai.transforms.EnsureChannelFirstD(keys=['image']),
            monai.transforms.Orientationd(axcodes=orientation_code, keys=['image']),  # zyx
            monai.transforms.Spacingd(keys=["image"], pixdim=(1, 1, 3), mode=("bilinear")),
            monai.transforms.CropForegroundd(keys=["image"], source_key="image"),
            monai.transforms.ToTensord(keys=["image"]),
        ]
    )
    dictionary = monai_loader({'image': datum['image']})
    img = dictionary['image']
    img = Normalization(img, datum['modality'].lower())

    return img, datum['label'], datum['modality'], datum['image']

class Inference_Dataset(Dataset):
    def __init__(self, lines: list[dict], max_queries=256, batch_size=2, patch_size=(288, 288, 96)):
        """
        max_queries: num of queries in a batch. can be very large.
        batch_size: num of image patch in a batch. be careful with this if you have limited gpu memory.
        evaluated_samples: to resume from an interrupted evaluation
        """
        self.lines = lines

        self.max_queries = max_queries
        self.batch_size = batch_size
        self.patch_size = patch_size

        print(f'** DATASET ** : load {len(self.lines)} samples')

    def __len__(self):
        return len(self.lines)

    def _split_labels(self, label_list):
        # split the labels into sub-lists
        if len(label_list) < self.max_queries:
            return [label_list], [[0, len(label_list)]]
        else:
            split_idx = []
            split_label = []
            query_num = len(label_list)
            n_crop = (query_num // self.max_queries + 1) if (query_num % self.max_queries != 0) else (
                    query_num // self.max_queries)
            for n in range(n_crop):
                n_s = n * self.max_queries
                n_f = min((n + 1) * self.max_queries, query_num)
                split_label.append(label_list[n_s:n_f])
                split_idx.append([n_s, n_f])
            return split_label, split_idx

    def _merge_modality(self, mod):
        if contains(mod, ['t1', 't2', 'mri', 'mr', 'flair', 'dwi']):
            return 'mri'
        if contains(mod, 'ct'):
            return 'ct'
        if contains(mod, 'pet'):
            return 'pet'
        else:
            return mod

    def __getitem__(self, idx):
        datum = self.lines[idx]
        img: MetaTensor
        img, labels, modality, image_path = load_image(datum)
        if img.shape[0] == 1:
            img = einops.repeat(img, '1 ... -> c ...', c=3)
        # split labels into batches
        split_labels, split_n1n2 = self._split_labels(labels)  # [xxx, ...] [[n1, n2], ...]
        modality = self._merge_modality(modality.lower())
        for i in range(len(split_labels)):
            split_labels[i] = [label.lower() for label in split_labels[i]]

        return {
            'image': img.as_tensor(),
            'meta': img.meta,
            'save_path': datum['save_path'],
            'split_queries': split_labels,
            'split_n1n2': split_n1n2,
            'labels': labels,
            'modality': modality,
        }

def collate_fn(data):
    return data[0]
