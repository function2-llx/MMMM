from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .defs import ORIGIN_DATA_ROOT

def _split_items(items_str: str, sep: str = '; ') -> list[str]:
    return [] if pd.isna(items_str) else items_str.split(sep)

@dataclass
class SegClass:
    name: str
    synonyms: list[str]
    parents: list[SegClass]
    children: list[SegClass]

    def update(self, info: pd.Series, classes: dict[str, SegClass]):
        self.synonyms = _split_items(info['synonyms'])
        self.parents = [
            classes[parent_name]
            for parent_name in _split_items(info['parents'])
        ]
        for parent in self.parents:
            parent.children.append(self)

def load_seg_tax() -> dict[str, SegClass]:
    tax = pd.read_excel(ORIGIN_DATA_ROOT / 'seg-tax.xlsx')
    tax = tax[~tax['name'].isna()]
    tax.set_index('name', inplace=True)
    classes = {
        name: SegClass(name, [], [], [])
        for name in tax.index
    }
    for name, row in tax.iterrows():
        classes[name].update(row, classes)

    return classes
