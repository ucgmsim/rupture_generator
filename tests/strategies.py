"""Hypothesis strategies for the things the pipeline takes in.

One module per language, per `PLAN.md` section 6: *"A property holds for generated
grids, seeds, magnitudes, hypocentres, and trees -- shrinkable strategies, shared in
one module per language."* The Rust half is ``crates/kernels/tests/common``.

Ranges are physical, and stated as such: a fault is between a kilometre and a couple
of hundred long, dips somewhere in ``(0, 90]``, reaches tens of kilometres down. Where
a range is narrower than the config permits, the narrowing is a *conditioning* choice
and says so -- a dip of a tenth of a degree is legal and makes every horizontal
distance enormous, which tests arithmetic rather than geometry.

**Origins are generated at CRS scale as well as at zero.** An NZTM northing reaches
5,180 km, and the absolute-coordinate precision defect only appears there: a property
asserted at the origin and nowhere else would have missed it entirely.
"""

from __future__ import annotations

import hypothesis.strategies as st
import numpy as np

from rupture_generator.config.geometry import (
    Discretisation,
    FaultConfig,
    LonLat,
    PlaneConfig,
    PointConfig,
)
from rupture_generator.mesh import RuptureMesh
from rupture_generator.sampling import VonKarmanFilterParameters
from rupture_generator.stages import DepthRamp

# New Zealand, where EPSG:2193 is valid and where the convergence angle is worth
# something. The latitude band is what keeps a generated trace inside the projection's
# useful area rather than out in the distortion.
LONGITUDES = st.floats(min_value=167.0, max_value=178.0, allow_nan=False)
LATITUDES = st.floats(min_value=-46.5, max_value=-35.0, allow_nan=False)

DIPS = st.floats(min_value=5.0, max_value=90.0, allow_nan=False)
"""Dips from 5 degrees. The config permits ``(0, 90]``; below about 5 the horizontal
reach exceeds the fault's own length and the chart is a sliver, which conditions every
tolerance on arithmetic rather than on geometry. `test_mesh.py` covers the shallow end
separately where it matters."""

DEPTHS = st.floats(min_value=4.0, max_value=40.0, allow_nan=False)
TOP_DEPTHS = st.floats(min_value=0.0, max_value=5.0, allow_nan=False)
SUBFAULT_SIZES = st.floats(min_value=0.5, max_value=4.0, allow_nan=False)


@st.composite
def lonlats(draw: st.DrawFn) -> LonLat:
    """A point in the region."""
    return LonLat(
        longitude_deg=draw(LONGITUDES),
        latitude_deg=draw(LATITUDES),
    )


@st.composite
def discretisations(draw: st.DrawFn) -> Discretisation:
    """A cut, given either way -- the two forms round differently and both need
    exercising.

    Only sound for a **single-plane** fault. Explicit counts on several planes of
    different lengths give them different spacings by construction, which a fused
    surface refuses (and rightly: one grid needs one spacing), so
    :func:`straight_faults` asks for a size when it draws more than one plane.
    """
    if draw(st.booleans()):
        return Discretisation(subfault_size_km=draw(SUBFAULT_SIZES))
    return Discretisation(
        strike_count=draw(st.integers(min_value=1, max_value=20)),
        dip_count=draw(st.integers(min_value=1, max_value=12)),
    )


@st.composite
def straight_faults(draw: st.DrawFn, *, planes: int = 1) -> FaultConfig:
    """A fault whose planes all hang the same way: one surface, however it bends.

    Every plane shares a dip, a dip direction and a bottom depth, which is exactly
    the condition that makes their seam columns coincide. The trace itself is a walk
    of bounded turns, so a generated fault bends without doubling back.
    """
    dip_deg = draw(DIPS)
    bottom_depth_km = draw(DEPTHS)
    top_depth_km = draw(
        st.floats(min_value=0.0, max_value=max(0.0, bottom_depth_km - 3.0))
    )
    dip_direction = draw(st.sampled_from(["right", "left"]))

    if planes == 1:
        discretisation = draw(discretisations())
    else:
        # A size, not counts, and a size small enough against the shortest trace
        # segment below that rounding cannot push two planes' realised spacings more
        # than 10% apart: the realised size is within `s / 2L` of `s`, so at
        # s <= 1.5 km on segments of at least ~16 km that spread is under 5%.
        discretisation = Discretisation(
            subfault_size_km=draw(
                st.floats(min_value=0.5, max_value=1.5, allow_nan=False)
            )
        )

    # The trace walks in steps of a bounded bearing change. Turning by less than the
    # builder's ceiling is what keeps a generated fault buildable rather than
    # refused, and the walk is **never clamped**: clamping is what puts two trace
    # points on top of each other, which is a different refusal and not the one any
    # property here is about. The start is drawn well inside the region and the walk
    # is bounded by `planes * 0.3` degrees, so it cannot leave it.
    bearing = draw(st.floats(min_value=0.0, max_value=360.0, allow_nan=False))
    margin = 0.35 * planes
    longitude = draw(
        st.floats(min_value=168.0 + margin, max_value=177.0 - margin, allow_nan=False)
    )
    latitude = draw(
        st.floats(min_value=-45.5 + margin, max_value=-36.0 - margin, allow_nan=False)
    )
    origin = LonLat(longitude_deg=longitude, latitude_deg=latitude)

    # How sharply a *fused* fault may turn is not a constant: a bend rotates the
    # down-dip direction by half the deflection, which swings the surface's bottom
    # edge by the horizontal reach of the dip. That reach is `depth_span / tan(dip)`
    # -- 46 km on a 4 km fault dipping 5 degrees, against a ~27 km plane -- so a
    # shallow fault turning sharply is not one grid at all, and `validate_chart`
    # rightly refuses it. Capping the turn so the swing stays under a quarter of the
    # segment keeps generated faults inside what "one surface" can mean.
    reach_km = (bottom_depth_km - top_depth_km) / _tan_deg(dip_deg)
    shortest_segment_km = 16.0
    swing_ratio = min(1.0, 0.25 * shortest_segment_km / max(reach_km, 1.0e-9))
    max_turn_deg = min(60.0, 2.0 * _asin_deg(swing_ratio))

    ends = []
    for _ in range(planes):
        bearing += draw(
            st.floats(min_value=-max_turn_deg, max_value=max_turn_deg, allow_nan=False)
        )
        # At least ~16 km, which is what keeps a size-based cut giving every plane
        # the same spacing to within the fusion bound (see `discretisations`).
        length_deg = draw(st.floats(min_value=0.15, max_value=0.3, allow_nan=False))
        # A crude local-tangent step: the exact geometry does not matter, only that
        # the points are distinct and the turn is bounded.
        longitude += length_deg * _sin_deg(bearing) / 0.74
        latitude += length_deg * _cos_deg(bearing)
        ends.append(LonLat(longitude_deg=longitude, latitude_deg=latitude))

    return FaultConfig(
        name="generated",
        origin=origin,
        top_depth_km=top_depth_km,
        planes=[
            PlaneConfig(
                end=end,
                dip_deg=dip_deg,
                bottom_depth_km=bottom_depth_km,
                discretisation=discretisation,
                dip_direction=dip_direction,
            )
            for end in ends
        ],
    )


@st.composite
def point_sources(draw: st.DrawFn) -> PointConfig:
    """A point source deep enough that its top edge is under the ground."""
    size_km = draw(st.floats(min_value=0.2, max_value=5.0, allow_nan=False))
    dip_deg = draw(DIPS)
    return PointConfig(
        name="generated_point",
        centre=draw(lonlats()),
        # Half the cell's own down-dip extent, so the constructor's refusal is not
        # what this strategy is exercising.
        depth_km=draw(st.floats(min_value=size_km, max_value=30.0, allow_nan=False)),
        strike_deg=draw(st.floats(min_value=0.0, max_value=360.0, allow_nan=False)),
        dip_deg=dip_deg,
        size_km=size_km,
    )


@st.composite
def planar_charts(
    draw: st.DrawFn,
    *,
    min_cells: int = 4,
    max_cells: int = 24,
    at_crs_scale: bool | None = None,
) -> RuptureMesh:
    """A single planar chart, built from nodes rather than from a config.

    The field stages do not care how a chart was digitised -- only that it is a
    regular grid with a depth at every subfault -- so building one directly is both
    cheaper than routing through `build_fault` and able to reach shapes a trace
    cannot conveniently produce (a very long thin fault, a nearly square one).

    The plane strikes due north and dips east, which is a choice with no content:
    every stage here reads depths, spacings and cell counts, and none of them reads
    an azimuth.
    """
    cells_j = draw(st.integers(min_value=min_cells, max_value=max_cells))
    cells_i = draw(st.integers(min_value=min_cells, max_value=max_cells))
    strike_km = draw(st.floats(min_value=0.5, max_value=3.0, allow_nan=False))
    dip_km = draw(st.floats(min_value=0.5, max_value=3.0, allow_nan=False))
    dip_deg = draw(DIPS)
    top_depth_km = draw(st.floats(min_value=0.0, max_value=4.0, allow_nan=False))

    if at_crs_scale is None:
        at_crs_scale = draw(st.booleans())
    origin = (1500.0, 5180.0) if at_crs_scale else (0.0, 0.0)

    along = np.arange(cells_j + 1) * strike_km
    down = np.arange(cells_i + 1) * dip_km
    east = np.tile(down[:, None] * _cos_deg(dip_deg), (1, cells_j + 1))
    north = np.tile(along[None, :], (cells_i + 1, 1))
    depth = top_depth_km + np.tile(down[:, None] * _sin_deg(dip_deg), (1, cells_j + 1))

    return RuptureMesh.from_nodes(
        east,
        north,
        depth,
        origin_east_km=origin[0],
        origin_north_km=origin[1],
        surface="generated",
    )


@st.composite
def covariances(
    draw: st.DrawFn, mesh: RuptureMesh | None = None
) -> VonKarmanFilterParameters:
    """A field structure a real earthquake could have on a fault this size.

    Mai & Beroza (2002) figure 13: across all 44 finite-source models the correlation
    length is **0.25 to 0.6 of the source dimension** on each axis, with a median near
    0.4 and no dependence on magnitude. So a correlation length is not free of the
    fault it sits on, and drawing the two independently generates earthquakes that do
    not exist -- a 20 km down-dip correlation length on a 4 km-wide fault, whose field
    is constant over the whole rupture.

    That pairing is not merely unrealistic, it is unsamplable: a covariance whose
    structure outruns its grid has no circulant embedding, and `sampling._embed`
    refuses it. Tying the two here is what keeps the property tests inside the model
    the paper describes, rather than testing the refusal path by accident.

    Passing no chart gives the range for a fault of about 20 km, which is what a test
    that does not have a chart to hand wants.
    """
    strike_km, dip_km = mesh.spacing_km() if mesh is not None else (1.0, 1.0)
    cells_i, cells_j = mesh.cell_counts if mesh is not None else (20, 20)

    fraction = st.floats(min_value=0.25, max_value=0.6, allow_nan=False)
    return VonKarmanFilterParameters(
        correlation_length_strike_km=draw(fraction) * cells_j * strike_km,
        correlation_length_dip_km=draw(fraction) * cells_i * dip_km,
    )


@st.composite
def charts_with_covariances(
    draw: st.DrawFn, **chart_arguments: object
) -> tuple[RuptureMesh, VonKarmanFilterParameters]:
    """A chart and a field structure that belong to the same earthquake."""
    mesh = draw(planar_charts(**chart_arguments))  # ty: ignore[invalid-argument-type]
    return mesh, draw(covariances(mesh))


@st.composite
def depth_ramps(draw: st.DrawFn) -> DepthRamp:
    """A linear depth transition, always with a positive width."""
    return DepthRamp(
        centre_km=draw(st.floats(min_value=1.0, max_value=30.0, allow_nan=False)),
        half_width_km=draw(st.floats(min_value=0.25, max_value=5.0, allow_nan=False)),
    )


MAGNITUDES = st.floats(min_value=5.0, max_value=8.5, allow_nan=False)
SEEDS = st.integers(min_value=0, max_value=2**32 - 1)


def _sin_deg(degrees: float) -> float:
    import math

    return math.sin(math.radians(degrees))


def _cos_deg(degrees: float) -> float:
    import math

    return math.cos(math.radians(degrees))


def _tan_deg(degrees: float) -> float:
    import math

    return math.tan(math.radians(degrees))


def _asin_deg(ratio: float) -> float:
    import math

    return math.degrees(math.asin(min(1.0, max(-1.0, ratio))))
