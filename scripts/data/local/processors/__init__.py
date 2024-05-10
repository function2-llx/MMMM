# FIXME: these boilerplate import codes are stupid, and inefficient for importing too much :(

from .ACDC import ACDCProcessor
from .AMOS22 import AMOS22Processor
from .ATLAS import ATLASProcessor
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
from .LIDC_IDRI import LIDC_IDRIProcessor
from .LNQ2023 import LNQ2023Processor
from .LiTS import LiTSProcessor
from .MRSpineSeg import MRSpineSegProcessor
from .MSD import (
    MSDColonProcessor, MSDHeartProcessor, MSDHepaticVesselProcessor, MSDHippocampusProcessor, MSDLiverProcessor,
    MSDLungProcessor, MSDPancreasProcessor, MSDProstateProcessor, MSDSpleenProcessor,
)
from .PARSE2022 import PARSE2022Processor
from .PENGWIN import (
    PENGWINT1Processor,
    # PENGWINT2Processor,
)
# from .PI_CAI import PI_CAIProcessor
from .Prostate158 import Prostate158Processor
from .RibFrac import RibFracProcessor
from .SEGA2023 import SEGA2022Processor
from .SegRap2023 import SegRap2023Processor
from .SegTHOR import SegTHORProcessor
from .TotalSegmentator import TotalSegmentatorProcessor
from .VerSe import VerSeProcessor
from .VinDrCXR import VinDrCXRProcessor
from .WORD import WORDProcessor
from .autoPET_III import AutoPETIIIProcessor
