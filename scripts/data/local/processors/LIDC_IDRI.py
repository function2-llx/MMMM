from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pylidc as pl
from pylidc.utils import consensus
import torch

from luolib.utils import get_cuda_device
from monai.data import MetaTensor

from mmmm.data.sparse import Sparse

from ._base import DataPoint as _DataPoint, DefaultImageLoaderMixin, Processor

@dataclass(kw_only=True)
class LIDR_IDRIDataPoint(_DataPoint):
    scan: pl.Scan
    images: ... = None

class LIDC_IDRIProcessor(DefaultImageLoaderMixin, Processor):
    name = 'LIDC-IDRI'

    @property
    def dataset_root(self):
        return super().dataset_root / 'download'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scans: list[pl.Scan] = list(pl.query(pl.Scan))

    def load_images(self, data_point: LIDR_IDRIDataPoint) -> tuple[list[str], MetaTensor]:
        scan = data_point.scan
        volume = scan.to_volume(verbose=False)
        image = MetaTensor(torch.from_numpy(volume)[None], torch.diag(torch.tensor((*scan.spacings.tolist(), 1))))
        modality = 'contrast-enhanced CT' if scan.contrast_used else 'CT'
        return [modality], image

    def load_masks(self, data_point: LIDR_IDRIDataPoint, images: MetaTensor):
        scan = data_point.scan
        nodules = scan.cluster_annotations()
        masks = np.zeros((len(nodules), *images.shape[1:]), dtype=np.bool_)
        for i, nodule in enumerate(nodules):
            mask, mask_bbox = consensus(nodule, ret_masks=False)
            masks[i, *mask_bbox] = mask
        masks = MetaTensor(torch.as_tensor(masks, device=get_cuda_device()), images.affine)
        return ['nodule'] * masks.shape[0], masks

    def get_data_points(self) -> list[LIDR_IDRIDataPoint]:
        return [
            LIDR_IDRIDataPoint(
                key=f'{scan.patient_id}_{scan.series_instance_uid}',
                scan=scan,
                complete_anomaly=True,
            )
            for scan in self.scans
        ]
