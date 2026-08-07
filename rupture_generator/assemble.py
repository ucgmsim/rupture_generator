"""Turning a generated rupture into an SRF file.

The core produces physics: how much each subfault slips, in what direction, when it
starts, and the shape of the pulse. It knows nothing about where the fault is. This
puts the two together.

There is no geodesy here and no projection. The subfault coordinates come from
whoever discretised the fault — `rupture_generator.geometry` — because that is the
only place that knows how the mesh became a grid. genslip recomputes a plane centre
from a fault width and a dip with a tangent-plane approximation that is off by a
kilometre at subduction scale; the caller here already has the answer.
"""

import dataclasses

import numpy as np
import pandas as pd
import scipy as sp

from rupture_generator._core import GeneratedRupture
from rupture_generator.srf import SrfFile

FloatArray = np.ndarray[tuple[int], np.dtype[np.float32]]


@dataclasses.dataclass(frozen=True)
class SubfaultGeometry:
    """Where each subfault is, in the along-strike-fastest order the core uses.

    Every array holds one value per subfault. `area_cm2` is the subfault's area in
    square centimetres, which is what the SRF format stores and what the moment sum
    is expressed in.
    """

    longitude_deg: FloatArray
    latitude_deg: FloatArray
    depth_km: FloatArray
    strike_deg: FloatArray
    dip_deg: FloatArray
    area_cm2: FloatArray

    def __post_init__(self) -> None:
        """Check every array describes the same set of subfaults.

        Raises
        ------
        ValueError
            If the arrays are not all the same length, or any is empty.
        """
        lengths = {
            name: len(getattr(self, name))
            for name in ("longitude_deg", "latitude_deg", "depth_km", "strike_deg", "dip_deg", "area_cm2")
        }
        if len(set(lengths.values())) != 1:
            raise ValueError(f"subfault arrays disagree on length: {lengths}")
        if not any(lengths.values()):
            raise ValueError("a fault needs at least one subfault")


@dataclasses.dataclass(frozen=True)
class PlaneHeader:
    """One segment's entry in the SRF header.

    `hypocentre_strike_km` is measured from the segment's along-strike **centre** and
    `hypocentre_dip_km` from its top edge — genslip's convention, and the one
    `realisation_to_srf.py` already converts into.
    """

    centre_longitude_deg: float
    centre_latitude_deg: float
    strike_count: int
    dip_count: int
    length_km: float
    width_km: float
    strike_deg: float
    dip_deg: float
    top_depth_km: float
    hypocentre_strike_km: float
    hypocentre_dip_km: float


def to_srf_file(
    rupture: GeneratedRupture,
    geometry: SubfaultGeometry,
    header: PlaneHeader,
    shear_speed_km_s: FloatArray,
    density_g_cm3: FloatArray,
) -> SrfFile:
    """Assemble an SRF version 2.0 file from a generated rupture.

    Parameters
    ----------
    rupture : GeneratedRupture
        The generated model.
    geometry : SubfaultGeometry
        Where each subfault is.
    header : PlaneHeader
        The segment's header entry.
    shear_speed_km_s : FloatArray
        Shear-wave speed at each subfault. Version 2.0 carries it per point.
    density_g_cm3 : FloatArray
        Density at each subfault.

    Returns
    -------
    SrfFile
        A single-segment SRF, version 2.0.

    Raises
    ------
    ValueError
        If the geometry, the material properties and the rupture disagree about how
        many subfaults there are.
    """
    subfaults = len(geometry.longitude_deg)
    strike_count, dip_count = rupture.shape
    if strike_count * dip_count != subfaults:
        raise ValueError(
            f"the rupture covers {strike_count}x{dip_count} subfaults and the "
            f"geometry describes {subfaults}"
        )
    for name, values in (
        ("shear_speed_km_s", shear_speed_km_s),
        ("density_g_cm3", density_g_cm3),
    ):
        if len(values) != subfaults:
            raise ValueError(
                f"{name} has {len(values)} entries for {subfaults} subfaults"
            )

    headers = pd.DataFrame(
        [
            {
                "elon": header.centre_longitude_deg,
                "elat": header.centre_latitude_deg,
                "nstk": header.strike_count,
                "ndip": header.dip_count,
                "len": header.length_km,
                "wid": header.width_km,
                "stk": header.strike_deg,
                "dip": header.dip_deg,
                "dtop": header.top_depth_km,
                "shyp": header.hypocentre_strike_km,
                "dhyp": header.hypocentre_dip_km,
            }
        ]
    )
    headers["nstk"] = headers["nstk"].astype(int)
    headers["ndip"] = headers["ndip"].astype(int)

    points = pd.DataFrame(
        {
            "lon": geometry.longitude_deg,
            "lat": geometry.latitude_deg,
            "dep": geometry.depth_km,
            "stk": geometry.strike_deg,
            "dip": geometry.dip_deg,
            "area": geometry.area_cm2,
            "tinit": rupture.onset_s,
            "dt": np.full(subfaults, rupture.sample_interval_s, dtype=np.float32),
            "vs": np.asarray(shear_speed_km_s, dtype=np.float32),
            "den": np.asarray(density_g_cm3, dtype=np.float32),
            "rake": rupture.rake_deg,
            "slip": rupture.slip_cm,
            "rise": rupture.rise_time_s,
        }
    )

    # The pulses are already concatenated with offsets that index into them, which is
    # a CSR matrix in all but name: `slip_rate_offsets` is the row pointer and the
    # column index of each sample is its position within its own pulse.
    offsets = np.asarray(rupture.slip_rate_offsets, dtype=np.int64)
    lengths = np.diff(offsets)
    longest = int(lengths.max()) if len(lengths) else 0
    columns = np.concatenate(
        [np.arange(length, dtype=np.int64) for length in lengths]
    ) if longest else np.empty(0, dtype=np.int64)

    slip_rate = sp.sparse.csr_array(
        (np.asarray(rupture.slip_rate, dtype=np.float32), columns, offsets),
        shape=(subfaults, longest),
    )

    return SrfFile(
        version="2.0",
        header=headers,
        points=points,
        slipt1_array=slip_rate,
    )
