import os
from dataclasses import dataclass

import cytoolz
import numpy as np
import pydicom as dicom
import pylidc as pl
from pylidc.utils import consensus
import torch

from monai.data import MetaTensor
import monai.transforms as mt

from mmmm.data.defs import ORIGIN_LOCAL_DATA_ROOT
from ._base import DefaultImageLoaderMixin, Processor, SegDataPoint, DefaultMaskLoaderMixin, SkipException

@dataclass(kw_only=True)
class LIDR_IDRIDataPoint(SegDataPoint):
    series_instance_uid: str
    images: ... = None

def _patch_Scan_load_all_dicom_images(self: pl.Scan, verbose=True):
    if verbose:
        print("Loading dicom files ... This may take a moment.")

    path = self.get_path_to_dicom_files()
    fnames = [fname for fname in os.listdir(path)
              if fname.endswith('.dcm') and not fname.startswith(".")]
    images = []
    for fname in fnames:
        image = dicom.dcmread(os.path.join(path, fname))

        seid = str(image.SeriesInstanceUID).strip()
        stid = str(image.StudyInstanceUID).strip()

        if seid == self.series_instance_uid and \
            stid == self.study_instance_uid:
            images.append(image)

    # ##############################################
    # Clean multiple z scans.
    #
    # Some scans contain multiple slices with the same `z` coordinate
    # from the `ImagePositionPatient` tag.
    # The arbitrary choice to take the slice with lesser
    # `InstanceNumber` tag is made.
    # This takes some work to accomplish...
    zs = [float(img.ImagePositionPatient[-1]) for img in images]
    inums = [float(img.InstanceNumber) for img in images]
    inds = list(range(len(zs)))
    while np.unique(zs).shape[0] != len(inds):
        for i in inds:
            for j in inds:
                if i != j and zs[i] == zs[j]:
                    k = i if inums[i] > inums[j] else j
                    inds.pop(inds.index(k))

    # Prune the duplicates found in the loops above.
    zs = [zs[i] for i in range(len(zs)) if i in inds]
    images = [images[i] for i in range(len(images)) if i in inds]

    # Sort everything by (now unique) ImagePositionPatient z coordinate.
    sort_inds = np.argsort(zs)
    images = [images[s] for s in sort_inds]
    # End multiple z clean.
    # ##############################################

    return images, sort_inds

def _patch_Scan_to_volume(self: pl.Scan, verbose: bool = True):
    """
    Return the scan as a 3D numpy array volume.
    """
    images, sort_inds = _patch_Scan_load_all_dicom_images(self, verbose=verbose)

    volume = np.stack(
        [
            x.pixel_array * x.RescaleSlope + x.RescaleIntercept
            for x in images
        ],
        axis=-1,
    ).astype(np.int16)
    return volume, sort_inds


class LIDC_IDRIProcessor(DefaultImageLoaderMixin, DefaultMaskLoaderMixin, Processor):
    name = 'LIDC-IDRI'
    clip_min = True
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
        volume, sort_inds = _patch_Scan_to_volume(scan, verbose=False)
        volume = torch.from_numpy(volume)
        if image.shape[1:] != volume.shape:
            raise SkipException('some slices are cleaned by pylidc')
        assert torch.equal(image[0], volume)
        modality = 'contrast-enhanced CT' if scan.contrast_used else 'CT'
        return [modality], image.to(device=self.device)

    def _load_nodule_masks(self, data_point: LIDR_IDRIDataPoint, images: MetaTensor):
        scan: pl.Scan = cytoolz.first(pl.query(pl.Scan).filter(pl.Scan.series_instance_uid == data_point.series_instance_uid))
        nodules = scan.cluster_annotations()
        if len(nodules) == 0:
            # no nodule, then create a negative mask for it
            return MetaTensor(torch.zeros(1, *images.shape[1:], dtype=torch.bool, device=self.device), affine=images.affine)
        masks = np.zeros((len(nodules), *images.shape[1:]), dtype=np.bool_)
        for i, nodule in enumerate(nodules):
            if len(nodule) > 4:
                raise SkipException('too many nodules')
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
        if lung_masks.shape[1:] != nodule_masks.shape[1:]:
            raise SkipException
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
