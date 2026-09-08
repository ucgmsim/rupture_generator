import json
from dataclasses import dataclass
from typing import TextIO, Self
from rupture_generator.geometry.geometry import Geometry


import numpy as np

type TraceArray = np.ndarray[tuple[int, int], np.float64]


@dataclass(frozen=True)
class Segment:
    trace: TraceArray
    name: str
    dip_deg: float
    dip_direction_deg: float
    upper_depth_km: float
    lower_depth_km: float

    @classmethod
    def from_feature(cls, feature: dict) -> Self:
        # obvious implementation
        pass

    def to_geometry(self) -> Geometry:
        # obvious implementation 
        pass

        


    



def geometry_from_geojson(handle: TextIO) -> dict[str, Geometry]:
   def _hook():
       # ---> nice match statement breakout here
   return json.load(handle, object_hook=_hook) # obvious implementation comes for free
