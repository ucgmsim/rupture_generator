"""Turning a generated rupture into an SRF file.

The core produces physics: how much each subfault slips, in what direction, when it
starts, and the shape of the pulse. It knows nothing about where the fault is. This
puts the two together.

There is no geodesy here and no projection. The subfault coordinates arrive in
`SubfaultGeometry`, from whoever discretised the fault, because that is the only
place that knows how the mesh became a grid. genslip instead recomputes a plane
centre from a fault width and a dip, with a tangent-plane approximation that is off
by 43 m on a crustal fault and 1.9 km at subduction scale — `test_corpus.py`'s
`TestTheGeometryDivergence` measures it. The caller here already has the answer, so
this asks for it rather than deriving it worse.

**The supplier now exists.** `rupture_generator.mesh.to_subfault_geometry` is it: a
fault is discretised in a projected Cartesian CRS by `genslip::mesh`, where every
derived quantity is an exact identity, and that module converts to WGS84 at a single
seam — adding the grid convergence angle to strike, because grid north is not true
north and in NZTM the difference reaches five degrees.

This docstring used to name two other things as the source, and both are gone.
`rupture_generator.geometry` was pre-port scaffold with `pass` for a body and no
importers. `genslip::geodesy::Wgs84Geodesic` was the ellipsoidal placer the
discretiser was expected to use, and when the discretiser got written it turned out
not to want geodesy at all — see `PRUNED.md`.

The contract here did not change through any of that, which is the point of having
had one: this module asks for arrays and is handed arrays.
"""

import dataclasses

import numpy as np
import scipy as sp

from rupture_generator._core import GeneratedRupture
from rupture_generator.srf import FloatArray, PlaneHeader, Points, SrfFile
from rupture_generator.units import CM_PER_KM, SRF_FLOAT


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
            subfaults, rupture.sample_interval_s, dtype=SRF_FLOAT
        ),
        rake_deg=rupture.rake_deg,
        slip_cm=rupture.slip_cm,
        rise_time_s=rupture.rise_time_s,
        shear_speed_cm_s=np.asarray(shear_speed_km_s, dtype=SRF_FLOAT)
        * SRF_FLOAT(CM_PER_KM),
        density_g_cm3=np.asarray(density_g_cm3, dtype=SRF_FLOAT),
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
        (np.asarray(rupture.slip_rate, dtype=SRF_FLOAT), columns, offsets),
        shape=(subfaults, longest),
    )

    return SrfFile(
        version="2.0",
        planes=[header],
        points=points,
        slip_rate=slip_rate,
    )
