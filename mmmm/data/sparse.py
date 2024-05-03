from __future__ import annotations as _

from dataclasses import dataclass, field
from functools import partial
from typing import Any

from mashumaro import pass_through
from mashumaro.config import BaseConfig
from mashumaro.mixins.orjson import DataClassORJSONMixin
import numpy as np
from numpy import typing as npt
import orjson

from mmmm.data.target_tax import TargetCategory

def _numpy_field(dtype: np.dtype):
    return field(metadata={'serialize': pass_through, 'deserialize': partial(np.array, dtype=dtype)})

@dataclass(kw_only=True)
class Sparse(DataClassORJSONMixin):
    """
    Attributes:
        modalities: all images of different modalities must be co-registered
        mean: mean intensity for each modality
        complete_anomaly: indicating that `pos` covers all anomalies in the image
    """
    spacing: npt.NDArray[np.float64] = _numpy_field(np.float64)
    shape: npt.NDArray[np.int64] = _numpy_field(np.int64)
    modalities: list[str]
    mean: npt.NDArray[np.float32] = _numpy_field(np.float32)
    std: npt.NDArray[np.float32] = _numpy_field(np.float32)

    @dataclass(kw_only=True)
    class Annotation:
        """
        indistinguishable instances of the same class
        Attributes:
            name: class name
            num: number of instances (== len(boxes) == len(masks), if available)
            merged: whether different instances are merged (e.g., semantic), bbox makes less sense in this case
            position_offset: offsets of the corresponding class positions
            boxes: use MONAI's StandardMode (CornerCornerModeTypeA)
            masks: mask index of each instance corresponding to the mask file; if None, no mask for the instance available
        """
        name: str
        num: int
        merged: bool
        position_offset: tuple[int, int] | None

        @dataclass
        class MaskInfo:
            index: int
            size: int
        masks: list[MaskInfo] | None
        boxes: npt.NDArray[np.int64] = _numpy_field(np.int64)

    annotations: dict[TargetCategory, list[Annotation]]
    neg_targets: dict[TargetCategory, set[str]]
    complete_anomaly: bool
    extra: Any = None

    class Config(BaseConfig):
        orjson_options = orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_INDENT_2
