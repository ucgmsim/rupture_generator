from dataclasses import dataclass

from .enums import Stype


@dataclass
class PointSourceParams:
    stype: Stype
    risetime: float
    risetimefac: float
    risetimedep: float
    inittime: float
