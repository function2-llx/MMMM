from dataclasses import dataclass

import cytoolz
import numpy as np
import pylidc as pl
from pylidc.utils import consensus
import torch

from monai.data import MetaTensor
import monai.transforms as mt

from mmmm.data.defs import ORIGIN_LOCAL_DATA_ROOT
from ._base import DefaultImageLoaderMixin, Processor, SegDataPoint, DefaultMaskLoaderMixin

@dataclass(kw_only=True)
class LIDR_IDRIDataPoint(SegDataPoint):
    series_instance_uid: str
    images: ... = None


class LIDC_IDRIProcessor(DefaultImageLoaderMixin, DefaultMaskLoaderMixin, Processor):
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
        image_path = scan.get_path_to_dicom_files()
        image = self.image_loader(image_path)
        image = mt.Orientation('PLS')(image)
        volume = torch.from_numpy(scan.to_volume(verbose=False))
        # from mmmm.misc import IndexTrackerBinary
        # IndexTrackerBinary(mt.Orientation('ASR')(image)[0], zyx=False)
        # IndexTrackerBinary(volume, zyx=False)
        modality = 'contrast-enhanced CT' if scan.contrast_used else 'CT'
        assert torch.equal(image[0], volume), 'I hate you'
        return [modality], image.to(device=self.device)

    def _load_nodule_masks(self, data_point: LIDR_IDRIDataPoint, images: MetaTensor):
        scan: pl.Scan = cytoolz.first(pl.query(pl.Scan).filter(pl.Scan.series_instance_uid == data_point.series_instance_uid))
        nodules = scan.cluster_annotations()
        if len(nodules) == 0:
            # no nodule, then create a negative mask for it
            return torch.zeros(1, *images.shape[1:], dtype=torch.bool, device=self.device)
        masks = np.zeros((len(nodules), *images.shape[1:]), dtype=np.bool_)
        for i, nodule in enumerate(nodules):
            assert len(nodule) <= 4
            mask, mask_bbox = consensus(nodule, ret_masks=False)
            masks[i, *mask_bbox] = mask
        masks = MetaTensor(torch.as_tensor(masks, device=self.device), images.affine)
        return masks

    def load_masks(self, data_point: LIDR_IDRIDataPoint, images: MetaTensor):
        label_path = (ORIGIN_LOCAL_DATA_ROOT / f'LUNA16/seg-lungs-LUNA16/{data_point.series_instance_uid}.mhd')
        if label_path.exists():
            lung_targets, lung_masks = self._load_multi_class_masks(label_path, self.class_mapping)
        else:
            lung_targets, lung_masks = [], MetaTensor(
                torch.empty(0, *images.shape[1:], dtype=torch.bool, device=self.device), images.affine,
            )
        nodule_masks = self._load_nodule_masks(data_point, images)
        targets = lung_targets + ['lung nodule'] * nodule_masks.shape[0]
        orientation = mt.Orientation('RAS')
        lung_masks = orientation(lung_masks)
        nodule_masks = orientation(nodule_masks)
        self._check_affine(lung_masks.affine, nodule_masks.affine)
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
