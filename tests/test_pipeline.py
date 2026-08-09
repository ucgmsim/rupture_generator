"""End-to-end properties of a generated rupture.

One fixture per class of fault -- a single plane, a bent trace, a point source -- run
at a fixed seed, asserting **invariants of the output** rather than stored arrays. A
stored array is a second transcription of the code that produced it, and it goes red
for every deliberate change as loudly as for every accidental one; an invariant only
goes red when the rupture stops being one.

What is asserted here is what a consumer of the file would notice: that the moment is
the magnitude's, that the rupture starts where it was told and spreads outward, that
every subfault which slips has a pulse carrying exactly its slip, and that the file
says the same thing after a round trip through either container and through the SRF.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyproj
import pytest
import xarray as xr

from rupture_generator import assemble, moment
from rupture_generator.config import read_config, read_geometry
from rupture_generator.config.geometry import (
    GeometryConfig,
    PredeterminedPropagation,
)
from rupture_generator.config.rupture import RuptureConfig
from rupture_generator.formats.mesh import read_mesh, write_mesh
from rupture_generator.formats.rupture import (
    read_rupture,
    segments_in,
    to_datatree,
    write_rupture,
)
from rupture_generator.mesh import build_surface, fuse, validate_chart
from rupture_generator.pipeline import (
    Realisation,
    charts_for,
    generate,
    segments_of,
)
from rupture_generator.srf import read_srf, write_srf

EXAMPLES = Path(__file__).parent.parent / "examples"

Generated = tuple[Realisation, RuptureConfig, pyproj.CRS]
"""A run of the pipeline, and what it was run with."""

MIN_SLIP_M = 1.0e-4
"""The kernel's own no-pulse guard. Below it a subfault gets no pulse at all, which is
not the same as a pulse of zeros."""


def only(realisation: Realisation) -> xr.Dataset:
    """The one segment of a single-fault rupture."""
    (segment,) = realisation.segments.values()
    return segment


def _config() -> RuptureConfig:
    return read_config(EXAMPLES / "crustal.toml")


def _run(geometry_name: str) -> Generated:
    geometry = read_geometry(EXAMPLES / geometry_name)
    config = _config()
    segments = charts_for(geometry, None)
    return generate(config, segments, geometry.crs), config, geometry.crs


@pytest.fixture(scope="module")
def bent() -> Generated:
    """The shipped bent trace: two planes that share a seam, fused into one segment."""
    return _run("hope.geometry.toml")


# ============================================================================
# What the file says about the earthquake
# ============================================================================


def test_a_bent_fault_generates_as_one_segment(bent: Generated) -> None:
    """Planes that share a seam are one rupture, not two.

    The whole reason fusion exists: the trace bends, the strike varies along the
    grid, and the wavefront crosses the bend because in index space there is no bend
    to cross.
    """
    realisation, _, _ = bent
    assert len(realisation.segments) == 1
    assert realisation.tree == {realisation.root: None}
    assert realisation.jumps == {}

    segment = only(realisation)
    # Both config planes are present, and the seam column is counted once.
    planes = segment["plane"].to_numpy()
    assert set(np.unique(planes)) == {0, 1}
    assert segment.sizes["j_node"] == segment.sizes["j"] + 1


def test_the_moment_is_the_magnitudes(bent: Generated) -> None:
    """Recomputed from the file's own numbers, not from the pipeline's bookkeeping.

    The file carries area and slip; rigidity comes from the velocity model at each
    subfault's stored depth. If those three do not reproduce the target, then what
    was written is not what was scaled.
    """
    realisation, config, _ = bent
    segment = only(realisation)

    _, rigidity = moment.sample_velocity_model(
        segment["centre_depth_km"].to_numpy(),
        np.asarray(config.velocity_model.bottom_depth_km),
        np.asarray(config.velocity_model.shear_speed_km_s),
        np.asarray(config.velocity_model.density_g_cm3),
    )
    recomputed = float(
        np.sum(rigidity * segment["area_m2"].to_numpy() * segment["slip_m"].to_numpy())
    )
    expected = moment.seismic_moment_nm(config.source.magnitude)

    assert recomputed == pytest.approx(expected, rel=1.0e-9)
    assert realisation.moment_newton_m == pytest.approx(expected, rel=1.0e-12)


def test_the_rupture_starts_where_it_was_told(bent: Generated) -> None:
    """The hypocentre's onset is the delay, and nothing precedes it.

    `DEFECTS.md` 17's property, asserted on the output rather than on the stage: a
    consumer reading the file should find the earliest subfault at the cell the
    config named, and the config names it in arc lengths that the file records back.
    """
    realisation, config, _ = bent
    segment = only(realisation)
    onset = segment["onset_s"].to_numpy()

    assert float(onset[realisation.hypocentre]) == pytest.approx(
        config.timing.rupture_delay_s, abs=1.0e-9
    )
    assert float(onset.min()) == pytest.approx(config.timing.rupture_delay_s, abs=1e-9)
    assert segment.attrs["hypocentre_strike_km"] == pytest.approx(
        config.hypocentre.strike_km
    )


def test_the_front_spreads_outward_from_the_hypocentre(bent: Generated) -> None:
    """Onset grows with distance from where the rupture started.

    Not an exact statement -- the perturbation moves individual subfaults, by design,
    so that high-slip patches rupture early -- but a strong correlation is what
    separates a propagating front from a field of noise. A rupture whose onsets did
    not track distance would be one whose wavefront never ran.
    """
    realisation, _, _ = bent
    segment = only(realisation)

    centres = np.stack(
        [
            segment["node_east_km"].to_numpy()[:-1, :-1],
            segment["node_north_km"].to_numpy()[:-1, :-1],
            segment["centre_depth_km"].to_numpy(),
        ],
        axis=-1,
    )
    origin = centres[realisation.hypocentre]
    distance_km = np.linalg.norm(centres - origin, axis=-1).ravel()
    onset_s = segment["onset_s"].to_numpy().ravel()

    assert float(np.corrcoef(distance_km, onset_s)[0, 1]) > 0.9


def test_every_slipping_subfault_has_a_pulse_carrying_its_slip(bent: Generated) -> None:
    """`DEFECTS.md` 21, on the output: nothing that slips is silently empty.

    The kernel guarantees the integral exactly and refuses a rise time it cannot
    sample; this is the end-to-end statement of both. An empty row means one thing
    only -- a subfault below the slip guard -- and 0.63% of the moment once went
    missing because it could also mean a pulse that was thrown away.
    """
    realisation, _, _ = bent
    segment = only(realisation)

    slip_m = segment["slip_m"].to_numpy().ravel()
    offsets = segment["slip_rate_offset"].to_numpy()
    samples = segment["slip_rate"].to_numpy()
    interval_s = float(segment.attrs["sample_interval_s"])

    assert len(offsets) == slip_m.size + 1

    for subfault, slip in enumerate(slip_m):
        pulse = samples[offsets[subfault] : offsets[subfault + 1]]
        if slip <= MIN_SLIP_M:
            assert pulse.size == 0
            continue
        assert pulse.size > 0
        assert float(interval_s * pulse.sum()) == pytest.approx(slip, rel=1.0e-9)


def test_the_rupture_is_reproducible(bent: Generated) -> None:
    """The same seed gives the same earthquake, bit for bit."""
    realisation, _, _ = bent
    again, _, _ = _run("hope.geometry.toml")

    assert np.array_equal(
        only(realisation)["slip_m"].to_numpy(),
        only(again)["slip_m"].to_numpy(),
    )
    assert np.array_equal(
        only(realisation)["onset_s"].to_numpy(),
        only(again)["onset_s"].to_numpy(),
    )


# ============================================================================
# The other two classes of fault
# ============================================================================


def test_a_single_plane_fault_generates(tmp_path: Path) -> None:
    """The simplest case, and the one every other reduces to.

    Built from the bent example's first plane alone, so the fixture is the shipped
    geometry rather than a second one invented for the test.
    """
    geometry = read_geometry(EXAMPLES / "hope.geometry.toml")
    surface = geometry.surfaces[0]
    surface.planes = surface.planes[:1]

    charts = build_surface(surface, geometry.crs)
    segments = fuse(charts)
    assert len(segments) == 1
    validate_chart(segments[0])

    realisation = generate(_config(), segments, geometry.crs)
    segment = only(realisation)

    assert set(np.unique(segment["plane"].to_numpy())) == {0}
    assert (segment["slip_m"].to_numpy() >= 0.0).all()
    assert float(segment["onset_s"].to_numpy().min()) == pytest.approx(0.0, abs=1e-9)


def test_a_point_source_is_the_pipeline_with_constant_fields() -> None:
    """One cell, constant everything, and the same stages.

    `PLAN.md` section 5: a point source is not a separate path. Its slip is uniform,
    its rake is the configured base, its rise time is the one it was given rather
    than one derived from the moment -- and it still carries the magnitude's moment
    and still goes through pulse synthesis.
    """
    geometry = read_geometry(EXAMPLES / "hope.geometry.toml")
    config = read_config(EXAMPLES / "crustal.toml")

    from rupture_generator.config.geometry import LonLat, PointConfig
    from rupture_generator.config.rupture import PointSourceConfig

    point = PointConfig(
        name="point",
        centre=LonLat(longitude_deg=172.5, latitude_deg=-42.5),
        depth_km=8.0,
        strike_deg=45.0,
        dip_deg=70.0,
        size_km=2.0,
    )
    config.source = PointSourceConfig(
        magnitude=5.5,
        rise_time_s=0.8,
        average_dip_deg=70.0,
        average_rake_deg=175.0,
    )
    config.hypocentre.strike_km = 1.0
    config.hypocentre.dip_km = 1.0

    segments = fuse(build_surface(point, geometry.crs))
    realisation = generate(config, segments, geometry.crs)
    segment = only(realisation)

    assert (segment.sizes["i"], segment.sizes["j"]) == (1, 1)
    assert (segment.sizes["i_node"], segment.sizes["j_node"]) == (2, 2)
    assert segment.sizes["cell_edge"] == 2
    assert float(segment["rise_time_s"].to_numpy()[0, 0]) == pytest.approx(0.8)
    assert float(segment["rake_deg"].to_numpy()[0, 0]) == pytest.approx(
        config.field.base_rake_deg
    )
    assert float(segment["onset_s"].to_numpy()[0, 0]) == pytest.approx(0.0, abs=1e-9)

    # Still carries the magnitude's moment, and still has a pulse.
    _, rigidity = moment.sample_velocity_model(
        segment["centre_depth_km"].to_numpy(),
        np.asarray(config.velocity_model.bottom_depth_km),
        np.asarray(config.velocity_model.shear_speed_km_s),
        np.asarray(config.velocity_model.density_g_cm3),
    )
    recomputed = float(
        np.sum(rigidity * segment["area_m2"].to_numpy() * segment["slip_m"].to_numpy())
    )
    assert recomputed == pytest.approx(moment.seismic_moment_nm(5.5), rel=1e-9)
    assert segment.sizes["sample"] > 0


# ============================================================================
# The formats
# ============================================================================


@pytest.mark.parametrize("suffix", [".h5", ".zarr"])
def test_a_rupture_file_round_trips(
    bent: Generated, suffix: str, tmp_path: Path
) -> None:
    """Every variable, attribute and the CSR structure survive both containers.

    One round trip per format and no assertions about xarray or Zarr internals --
    except this package's own trap, that Zarr does not preserve group order, which
    `segments_in` sorts around.
    """
    realisation, _, crs = bent
    tree = to_datatree(
        {"hope/segment_0": only(realisation)},
        crs,
        attrs={"seed": 1234, "moment_newton_m": realisation.moment_newton_m},
    )
    path = tmp_path / f"rupture{suffix}"
    write_rupture(tree, path)

    with read_rupture(path) as back:
        found = segments_in(back)
        assert len(found) == 1
        restored = found[0][2]

        original = only(realisation)
        for name in original.data_vars:
            assert np.array_equal(
                restored[name].to_numpy(), original[name].to_numpy()
            ), name
        assert (
            restored.attrs["sample_interval_s"] == original.attrs["sample_interval_s"]
        )
        assert float(back.attrs["moment_newton_m"]) == pytest.approx(
            realisation.moment_newton_m
        )


def test_a_mesh_file_carries_a_bent_fault_into_the_pipeline(tmp_path: Path) -> None:
    """The subcommand boundary: `mesh` writes planes, `generate` fuses them back.

    The two steps are separate because a geometry is digitised once and reused, so
    what passes between them is a file -- and a bent fault has to survive that trip
    as the same surface, or the boundary is a place ruptures change.
    """
    geometry = read_geometry(EXAMPLES / "hope.geometry.toml")
    charts = build_surface(geometry.surfaces[0], geometry.crs)

    path = tmp_path / "mesh.h5"
    write_mesh({"hope": charts}, geometry.crs, path)
    restored, crs, _ = read_mesh(path)

    direct = generate(_config(), fuse(charts), geometry.crs)
    through_file = generate(_config(), fuse(restored["hope"]), crs)

    assert np.array_equal(
        only(direct)["slip_m"].to_numpy(),
        only(through_file)["slip_m"].to_numpy(),
    )


def test_an_srf_carries_the_same_rupture_in_its_own_units(
    bent: Generated, tmp_path: Path
) -> None:
    """The SRF is this rupture in centimetres, and the conversion is the only change.

    Read back through the format's own parser rather than compared in memory, so what
    is asserted is what a consumer downstream would actually get -- at the format's
    own resolution, which is float32: six significant figures, and the reason the
    tolerances here are 1e-5 rather than 1e-9.
    """
    realisation, config, _ = bent
    segment = only(realisation)

    depth_km = segment["centre_depth_km"].to_numpy()
    bottoms = np.asarray(config.velocity_model.bottom_depth_km)
    shear_speed, _ = moment.sample_velocity_model(
        depth_km,
        bottoms,
        np.asarray(config.velocity_model.shear_speed_km_s),
        np.asarray(config.velocity_model.density_g_cm3),
    )
    layer = np.minimum(
        np.searchsorted(bottoms, depth_km, side="left"), len(bottoms) - 1
    )
    density = np.asarray(config.velocity_model.density_g_cm3)[layer]

    path = tmp_path / "rupture.srf"
    write_srf(
        path,
        assemble.to_srf_file([segment], [shear_speed.ravel()], [density.ravel()]),
    )
    srf = read_srf(path)

    assert srf.version == "2.0"
    assert len(srf.planes) == 1
    assert srf.points.shear_speed_cm_s is not None

    assert np.allclose(
        srf.points.slip_cm, segment["slip_m"].to_numpy().ravel() * 100.0, rtol=1e-5
    )
    assert np.allclose(
        srf.points.area_cm2, segment["area_m2"].to_numpy().ravel() * 1.0e4, rtol=1e-5
    )
    assert np.allclose(
        srf.points.onset_s, segment["onset_s"].to_numpy().ravel(), atol=1e-4
    )


def test_the_srf_hypocentre_is_measured_from_the_plane_centre(bent: Generated) -> None:
    """``shyp`` is the one convention conversion, and it happens at the writer.

    The config and the mesh measure the hypocentre from the fault's ``j = 0`` end;
    the SRF measures it from the along-strike centre. Getting this wrong puts the
    hypocentre half a fault away, which on a 56 km fault is 28 km -- and the file
    still parses, still looks like a rupture, and still has its slip in the right
    place.
    """
    realisation, config, _ = bent
    segment = only(realisation)
    header = assemble.plane_header(
        segment,
        hypocentre_km=(config.hypocentre.strike_km, config.hypocentre.dip_km),
    )

    assert header.hypocentre_strike_km == pytest.approx(
        config.hypocentre.strike_km - header.length_km / 2.0
    )
    assert header.hypocentre_dip_km == pytest.approx(config.hypocentre.dip_km)


def test_the_moment_survives_the_trip_into_cgs(bent: Generated) -> None:
    """Summed from the SRF's own columns, in the units it stores them in.

    The conversion touches slip, area, shear speed and density, and a mistake in any
    one of them moves the moment by a power of ten. Asserting it after the conversion
    is what makes the four of them one claim instead of four.
    """
    realisation, config, _ = bent
    segment = only(realisation)

    depth_km = segment["centre_depth_km"].to_numpy()
    bottoms = np.asarray(config.velocity_model.bottom_depth_km)
    shear_speed, _ = moment.sample_velocity_model(
        depth_km,
        bottoms,
        np.asarray(config.velocity_model.shear_speed_km_s),
        np.asarray(config.velocity_model.density_g_cm3),
    )
    layer = np.minimum(
        np.searchsorted(bottoms, depth_km, side="left"), len(bottoms) - 1
    )
    density = np.asarray(config.velocity_model.density_g_cm3)[layer]

    srf = assemble.to_srf_file([segment], [shear_speed.ravel()], [density.ravel()])
    points = srf.points

    # In CGS: rigidity is rho * vs^2 in dyne per square centimetre.
    rigidity_dyne_cm2 = points.density_g_cm3 * points.shear_speed_cm_s**2
    moment_dyne_cm = float(
        np.sum(
            rigidity_dyne_cm2.astype(np.float64)
            * points.area_cm2.astype(np.float64)
            * points.slip_cm.astype(np.float64)
        )
    )
    # A dyne-centimetre is 1e-7 newton-metres.
    assert moment_dyne_cm * 1.0e-7 == pytest.approx(
        realisation.moment_newton_m, rel=1.0e-5
    )


# ============================================================================
# Several faults, one earthquake
# ============================================================================


def _two_fault_geometry() -> tuple[GeometryConfig, RuptureConfig]:
    """The shipped two-segment example, with a hypocentre that names its fault."""
    geometry = read_geometry(EXAMPLES / "kaikoura.geometry.toml")
    config = _config()
    config.hypocentre.fault = "kaikoura:0"
    return geometry, config


@pytest.fixture(scope="module")
def two_faults() -> Generated:
    """A rupture that crosses between segments."""
    geometry, config = _two_fault_geometry()
    segments = segments_of(geometry)
    return (
        generate(
            config, segments, geometry.crs, propagation_config=geometry.propagation
        ),
        config,
        geometry.crs,
    )


def test_a_multi_segment_rupture_has_a_causality_tree(two_faults: Generated) -> None:
    """Two segments, one root, and the other triggered by it."""
    realisation, _, _ = two_faults

    assert set(realisation.segments) == {"kaikoura:0", "kaikoura:1"}
    assert realisation.tree == {"kaikoura:0": None, "kaikoura:1": "kaikoura:0"}
    assert realisation.root == "kaikoura:0"
    assert set(realisation.jumps) == {"kaikoura:1"}


def test_the_front_reaches_a_child_after_it_leaves_its_parent(
    two_faults: Generated,
) -> None:
    """Causality across the jump, in both of its halves.

    The seed time is the parent's own arrival at the jump-off point plus a
    non-negative delay -- so the child cannot start before the parent got there --
    and nothing on the child precedes that seed. Together those are what make the
    tree a statement about time rather than only about which faults broke.
    """
    realisation, _, _ = two_faults
    jump = realisation.jumps["kaikoura:1"]

    parent = realisation.segments["kaikoura:0"]
    child = realisation.segments["kaikoura:1"]

    assert jump.departure_s == pytest.approx(
        float(parent["onset_s"].to_numpy()[jump.parent_cell]), abs=1e-9
    )
    assert jump.arrival_s >= jump.departure_s
    assert float(child["onset_s"].to_numpy().min()) >= jump.departure_s - 1e-9


def test_onsets_do_not_decrease_along_a_path_of_the_tree(
    two_faults: Generated,
) -> None:
    """A child fault never begins before its parent did.

    The whole-rupture form of causality: walking the tree from the root, each
    segment's earliest onset is at or after its parent's earliest. A rupture whose
    second fault began before the first would be two earthquakes filed as one.
    """
    realisation, _, _ = two_faults

    earliest = {
        name: float(segment["onset_s"].to_numpy().min())
        for name, segment in realisation.segments.items()
    }
    for name, parent in realisation.tree.items():
        if parent is not None:
            assert earliest[name] >= earliest[parent] - 1e-9

    # The root still starts at the delay, and is still the earliest thing anywhere.
    assert earliest[realisation.root] == pytest.approx(0.0, abs=1e-9)
    assert earliest[realisation.root] == min(earliest.values())


def test_the_moment_is_shared_between_the_faults(two_faults: Generated) -> None:
    """One factor over every segment: the parts sum to the whole.

    Each fault's own moment is whatever its pattern and the shared factor give it, so
    neither hits a target of its own -- which is the point. Scaling each alone would
    make the partition between faults an artefact of how the two fields happened to
    normalise rather than of the earthquake.
    """
    realisation, config, _ = two_faults

    moments = []
    for segment in realisation.segments.values():
        _, rigidity = moment.sample_velocity_model(
            segment["centre_depth_km"].to_numpy(),
            np.asarray(config.velocity_model.bottom_depth_km),
            np.asarray(config.velocity_model.shear_speed_km_s),
            np.asarray(config.velocity_model.density_g_cm3),
        )
        moments.append(
            float(
                np.sum(
                    rigidity
                    * segment["area_m2"].to_numpy()
                    * segment["slip_m"].to_numpy()
                )
            )
        )

    expected = moment.seismic_moment_nm(config.source.magnitude)
    assert sum(moments) == pytest.approx(expected, rel=1e-9)
    assert all(part > 0.0 for part in moments)
    assert not any(part == pytest.approx(expected, rel=1e-3) for part in moments)


def test_only_the_nucleating_segment_records_a_hypocentre(
    two_faults: Generated,
) -> None:
    """One earthquake, one hypocentre.

    Writing the arc lengths into every group claimed a hypocentre per fault, which a
    reader has no way to tell from a rupture that genuinely nucleated three times.
    """
    realisation, _, _ = two_faults

    holders = [
        name
        for name, segment in realisation.segments.items()
        if "hypocentre_strike_km" in segment.attrs
    ]
    assert holders == [realisation.root]


def test_a_stated_propagation_gives_the_tree_it_states() -> None:
    """Predetermined mode, end to end, and it does not consult the sampler."""
    geometry, config = _two_fault_geometry()
    geometry.propagation = PredeterminedPropagation(
        parents={"kaikoura:1": "kaikoura:0"}
    )
    segments = segments_of(geometry)

    realisation = generate(
        config, segments, geometry.crs, propagation_config=geometry.propagation
    )
    assert realisation.tree == {"kaikoura:0": None, "kaikoura:1": "kaikoura:0"}


def test_a_stated_root_that_is_not_the_hypocentres_fault_is_refused() -> None:
    """Two statements of one fact, refused when they disagree."""
    geometry, config = _two_fault_geometry()
    geometry.propagation = PredeterminedPropagation(
        parents={"kaikoura:1": "kaikoura:0"}
    )
    config.hypocentre.fault = "kaikoura:1"
    segments = segments_of(geometry)

    with pytest.raises(ValueError, match="rooted at"):
        generate(
            config, segments, geometry.crs, propagation_config=geometry.propagation
        )


def test_a_rupture_over_several_segments_needs_to_say_where_it_starts() -> None:
    """With one fault there is nothing to choose; with two there is."""
    geometry, config = _two_fault_geometry()
    config.hypocentre.fault = None
    segments = segments_of(geometry)

    with pytest.raises(ValueError, match="which one it is on"):
        generate(config, segments, geometry.crs)


def test_a_multi_segment_rupture_writes_one_srf_plane_per_segment(
    two_faults: Generated, tmp_path: Path
) -> None:
    """The gap the old writer refused: a multi-fault rupture as one SRF.

    genslip emits one PLANE record per segment and orders its points the same way, so
    this is the format's own shape rather than an extension of it.
    """
    realisation, config, _ = two_faults

    segments = list(realisation.segments.values())
    speeds, densities = [], []
    bottoms = np.asarray(config.velocity_model.bottom_depth_km)
    layer_densities = np.asarray(config.velocity_model.density_g_cm3)
    for segment in segments:
        depth_km = segment["centre_depth_km"].to_numpy()
        shear_speed, _ = moment.sample_velocity_model(
            depth_km,
            bottoms,
            np.asarray(config.velocity_model.shear_speed_km_s),
            layer_densities,
        )
        layer = np.minimum(
            np.searchsorted(bottoms, depth_km, side="left"), len(bottoms) - 1
        )
        speeds.append(shear_speed.ravel())
        densities.append(layer_densities[layer].ravel())

    path = tmp_path / "multi.srf"
    write_srf(path, assemble.to_srf_file(segments, speeds, densities))
    srf = read_srf(path)

    assert len(srf.planes) == 2
    assert len(srf.points.longitude_deg) == sum(
        segment.sizes["i"] * segment.sizes["j"] for segment in segments
    )
    total_slip_cm = np.concatenate(
        [segment["slip_m"].to_numpy().ravel() * 100.0 for segment in segments]
    )
    assert np.allclose(srf.points.slip_cm, total_slip_cm, rtol=1e-5)
