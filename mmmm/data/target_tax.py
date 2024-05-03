from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import pandas as pd

from monai.utils import StrEnum
from .defs import ORIGIN_DATA_ROOT

def _split_items(items_str: str, sep: str = '; ') -> list[str]:
    return [] if pd.isna(items_str) else items_str.split(sep)

class TargetCategory(StrEnum):
    ANATOMY = 'anatomy'
    ANOMALY = 'anomaly'

@dataclass
class TargetClass:
    name: str
    category: TargetCategory
    synonyms: list[str]
    parents: list[TargetClass]
    children: list[TargetClass]

    def _update(self, info: pd.Series, classes: dict[str, TargetClass]):
        self.synonyms = _split_items(info['synonyms'])
        self.parents = [
            classes[parent_name]
            for parent_name in _split_items(info['parents'])
        ]
        for parent in self.parents:
            parent.children.append(self)

@cache
def get_target_tax() -> dict[str, TargetClass]:
    tax_dict = pd.read_excel(ORIGIN_DATA_ROOT / 'target-tax.xlsx', ['anatomy', 'anomaly'])
    ret = {}
    for category, tax in tax_dict.items():
        tax.set_index('name', inplace=True)
        category = TargetCategory(category)
        ret.update({
            name: TargetClass(name, category, [], [], [])
            for name in tax.index
        })
        for name, row in tax.iterrows():
            ret[name]._update(row, ret)
    return ret

load_target_tax = get_target_tax
