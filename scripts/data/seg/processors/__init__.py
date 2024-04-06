# FIXME: these boilerplate import codes are stupid, and inefficient for importing too much :(

from .ACDC import ACDCProcessor
from .AMOS22 import AMOS22Processor
from .ATM22 import ATM22Processor
from .BTCV import BTCVAbdomenProcessor, BTCVCervixProcessor
from .BUSI import BUSIProcessor
from .BraTS2023 import (
    BraTS2023GLIProcessor, BraTS2023MENProcessor, BraTS2023METProcessor, BraTS2023PEDProcessor, BraTS2023SSAProcessor,
)
from .CHAOS import CHAOSProcessor
from .CTPelvic1K import CTPelvic1KProcessor
from .CTSpine1K import CTSpine1KProcessor
from .CT_ORG import CT_ORGProcessor
from .HaNSeg import HaNSegProcessor
from .ISLES22 import ISLES22Processor
from .TotalSegmentator import TotalSegmentatorProcessor
from .VerSe import VerSeProcessor
