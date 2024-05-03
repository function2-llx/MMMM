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

from luolib.types import tuple2_t
from mmmm.data.target_tax import TargetCategory

def _numpy_field(dtype: np.dtype, *, optional: bool = False):
    to_array = partial(np.array, dtype=dtype)
    if optional:
        deserialize = lambda x: None if x is None else to_array(x)
    else:
        deserialize = to_array
    return field(
        metadata={
            'serialize': pass_through,
            'deserialize': deserialize,
        },
    )

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
            semantic: whether different instances are merged in the mask (i.e., semantic), bbox makes less sense in this case
            position_offset: offsets of the corresponding class positions
            index_offset: index offset among all instances of all targets
            boxes: use MONAI's StandardMode (CornerCornerModeTypeA)
        """
        name: str
        semantic: bool
        position_offset: tuple2_t[int]
        index_offset: tuple2_t[int] | None
        mask_sizes: npt.NDArray[np.int64] | None = _numpy_field(np.int64, optional=True)
        boxes: npt.NDArray[np.int64] = _numpy_field(np.int64)

    annotations: dict[TargetCategory, list[Annotation]]
    neg_targets: dict[TargetCategory, list[str]]
    complete_anomaly: bool
    extra: Any = None

    class Config(BaseConfig):
        orjson_options = orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_INDENT_2
