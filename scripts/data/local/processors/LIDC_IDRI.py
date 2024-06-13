from dataclasses import dataclass

import cytoolz
import numpy as np
import pylidc as pl
from pylidc.utils import consensus
import torch

from mmmm.data.defs import ORIGIN_LOCAL_DATA_ROOT
from monai.data import MetaTensor
from ._base import DefaultImageLoaderMixin, Processor, SegDataPoint

@dataclass(kw_only=True)
class LIDR_IDRIDataPoint(SegDataPoint):
    series_instance_uid: str
    images: ... = None


class LIDC_IDRIProcessor(DefaultImageLoaderMixin, Processor):
    name = 'LIDC-IDRI'
    class_mapping = {
        3: 'left lung',
        4: 'right lung',
        5: 'trachea',
    }

    @property
    def dataset_root(self):
        return super().dataset_root / 'download'

    def load_images(self, data_point: LIDR_IDRIDataPoint) -> tuple[list[str], MetaTensor]:
        scan: pl.Scan = cytoolz.first(pl.query(pl.Scan).filter(pl.Scan.series_instance_uid == data_point.series_instance_uid))
        volume = scan.to_volume(verbose=False)
        image = MetaTensor(torch.from_numpy(volume)[None], torch.diag(torch.tensor((*scan.spacings, 1))))
        modality = 'contrast-enhanced CT' if scan.contrast_used else 'CT'
        return [modality], image

    def _load_nodule_masks(self, data_point: LIDR_IDRIDataPoint, images: MetaTensor):
        scan: pl.Scan = cytoolz.first(pl.query(pl.Scan).filter(pl.Scan.series_instance_uid == data_point.series_instance_uid))
        nodules = scan.cluster_annotations()
        if len(nodules) == 0:
            # TODO: calm down
            return ['lung nodule'], torch.zeros(1,  dtype=torch.bool, device=self.device)
        masks = np.zeros((len(nodules), *images.shape[1:]), dtype=np.bool_)
        for i, nodule in enumerate(nodules):
            assert len(nodule) <= 4
            mask, mask_bbox = consensus(nodule, ret_masks=False)
            masks[i, *mask_bbox] = mask
        masks = MetaTensor(torch.as_tensor(masks, device=self.device), images.affine)
        return masks

    def load_masks(self, data_point: LIDR_IDRIDataPoint, images: MetaTensor):
        lung_targets, lung_masks = self._load_multi_class_masks(
            ORIGIN_LOCAL_DATA_ROOT / f'LUNA16/seg-lungs-LUNA16/{data_point.series_instance_uid}.mhd',
            self.class_mapping,
        )
        nodule_masks = self._load_nodule_masks(data_point, images)
        targets = lung_targets + ['lung nodule'] * lung_masks.shape[0]
        masks = torch.cat([lung_masks, nodule_masks])
        return targets, masks

    def get_data_points(self):
        return [
            LIDR_IDRIDataPoint(
                key=f'{scan.patient_id}_{scan.series_instance_uid}',
                series_instance_uid=scan.series_instance_uid,
                complete_anomaly=True,
            )
            for scan in pl.query(pl.Scan)
        ], None
