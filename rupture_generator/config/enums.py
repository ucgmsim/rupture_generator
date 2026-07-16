from enum import IntEnum, StrEnum


class Stype(StrEnum):
    esg2006 = "esg2006"
    urs = "urs"
    ucsb = "ucsb"
    ucsb2 = "ucsb2"
    ucsb_T = "ucsb-T"
    ucsb_varT1 = "ucsb-varT1"
    cos = "cos"
    seki = "seki"


class KModel(IntEnum):
    SOMERVILLE = 1
    MAI = 2
    FRANKEL = 3
    MAI_SOMERVILLE = 4
    SUZUKI = 5
    INPUT_CORNERS = -1
