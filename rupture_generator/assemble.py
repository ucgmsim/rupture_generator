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
import scipy as sp

from rupture_generator._core import GeneratedRupture
from rupture_generator.srf import FloatArray, PlaneHeader, Points, SrfFile

CM_PER_KM = np.float32(1.0e5)
"""What an SRF's shear speed is in, over what a velocity model's is in."""


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
            for name in (
                "longitude_deg",
                "latitude_deg",
                "depth_km",
                "strike_deg",
                "dip_deg",
                "area_cm2",
            )
        }
        if len(set(lengths.values())) != 1:
            raise ValueError(f"subfault arrays disagree on length: {lengths}")
        if not any(lengths.values()):
            raise ValueError("a fault needs at least one subfault")


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
        Shear-wave speed at each subfault, in the kilometres per second a velocity
        model is written in. Version 2.0 carries it per point, in centimetres per
        second, and this converts.
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

    points = Points(
        longitude_deg=geometry.longitude_deg,
        latitude_deg=geometry.latitude_deg,
        depth_km=geometry.depth_km,
        strike_deg=geometry.strike_deg,
        dip_deg=geometry.dip_deg,
        area_cm2=geometry.area_cm2,
        onset_s=rupture.onset_s,
        sample_interval_s=np.full(
            subfaults, rupture.sample_interval_s, dtype=np.float32
        ),
        rake_deg=rupture.rake_deg,
        slip_cm=rupture.slip_cm,
        rise_time_s=rupture.rise_time_s,
        shear_speed_cm_s=np.asarray(shear_speed_km_s, dtype=np.float32) * CM_PER_KM,
        density_g_cm3=np.asarray(density_g_cm3, dtype=np.float32),
    )

    # The pulses are already concatenated with offsets that index into them, which is
    # a CSR matrix in all but name: `slip_rate_offsets` is the row pointer and the
    # column index of each sample is its position within its own pulse.
    offsets = np.asarray(rupture.slip_rate_offsets, dtype=np.int64)
    lengths = np.diff(offsets)
    longest = int(lengths.max()) if len(lengths) else 0
    columns = (
        np.concatenate([np.arange(length, dtype=np.int64) for length in lengths])
        if longest
        else np.empty(0, dtype=np.int64)
    )

    slip_rate = sp.sparse.csr_array(
        (np.asarray(rupture.slip_rate, dtype=np.float32), columns, offsets),
        shape=(subfaults, longest),
    )

    return SrfFile(
        version="2.0",
        planes=[header],
        points=points,
        slip_rate=slip_rate,
    )
