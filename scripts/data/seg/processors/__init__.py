# FIXME: these boilerplate import codes are stupid, and inefficient for importing too much :(

from .ACDC import ACDCProcessor
from .AMOS22 import AMOS22DebugProcessor, AMOS22Processor
from .ATM22 import ATM22Processor
from .BraTS2023 import (
    BraTS2023GLIProcessor, BraTS2023MENProcessor, BraTS2023METProcessor, BraTS2023PEDProcessor, BraTS2023SSAProcessor,
)
from .BTCV import BTCVAbdomenProcessor, BTCVCervixProcessor
from .BUSI import BUSIProcessor
from .CHAOS import CHAOSProcessor
from .CT_ORG import CT_ORGProcessor
from .TotalSegmentator import TotalSegmentatorProcessor
