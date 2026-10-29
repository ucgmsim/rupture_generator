"""Bradley & Faulkner's moment apportionment, on fault systems built by hand."""

import math

import numpy as np
import pyproj
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from rupture_generator import Geometry, Realisation, RuptureGeneratorError
from rupture_generator import config as rupture_config
from rupture_generator.rupture.generator import moment_rng
from rupture_generator.rupture.medium import Layers, Medium, layered_medium
from rupture_generator.rupture.source import (
    MOMENT_SIGMA_LN,
    moment_components,
    split_moment,
)

from .conftest import EXAMPLE

CRS = pyproj.CRS("EPSG:2193")

MOMENT_NM = 1.0e20

MEDIUM = layered_medium(
    Layers(np.array([2.0, 6.0, 40.0])),
    np.array([2.0, 3.2, 3.7]),
    np.array([2.3, 2.6, 2.8]),
)
"""Rigidity that grows with depth, so that the shares depend on more than area."""


def plane(
    east_km: float,
    length_km: float,
    *,
    north_km: float = 0.0,
    top_km: float = 0.0,
    bottom_km: float = 15.0,
    dip_deg: float = 90.0,
) -> Geometry:
    """A plane striking east from ``(east_km, north_km)``, dipping north, 1 km cells.

    A negative dip dips south.
    """
    columns = round(length_km)
    rows = max(1, round(bottom_km - top_km))
    depth = np.linspace(top_km, bottom_km, rows + 1)[:, None]
    east = east_km + np.linspace(0.0, length_km, columns + 1)[None, :]
    north = north_km + (depth - top_km) / math.tan(math.radians(dip_deg))
    nodes = np.stack(np.broadcast_arrays(east, north, depth), axis=-1).astype(float)
    return Geometry(nodes, np.ones((rows, columns), bool), (columns,))


def system(**segments: Geometry) -> Realisation[Geometry]:
    return Realisation(segments, CRS)


def master_split_moment(
    moment_nm: float, realisation: Realisation[Geometry], medium: Medium
) -> dict[str, float]:
    """``split_moment`` as it was before components: one mean slip for the event."""
    weights = {
        name: float(
            np.sum(
                (medium.rigidity_pa(chart.centres) * chart.areas_km2)[chart.occupied]
            )
        )
        for name, chart in realisation.items()
    }
    total = sum(weights.values())
    return {name: moment_nm * weight / total for name, weight in weights.items()}


def mean_slip(
    shares: dict[str, float], realisation: Realisation[Geometry]
) -> dict[str, float]:
    """Each segment's moment over its ``sum(mu * A)``, which is its mean slip."""
    return {
        name: shares[name]
        / float(np.sum(MEDIUM.rigidity_pa(chart.centres) * chart.areas_km2))
        for name, chart in realisation.items()
    }


# ------------------------------------------------------------------- components


def test_segments_far_apart_are_separate_components():
    faults = system(a=plane(0.0, 20.0), b=plane(30.0, 20.0))
    assert moment_components(faults) == [("a",), ("b",)]


def test_close_segments_of_one_dip_are_one_component():
    faults = system(a=plane(0.0, 20.0), b=plane(24.0, 20.0, dip_deg=75.0))
    assert moment_components(faults) == [("a", "b")]


def test_close_segments_of_different_dip_are_separate_components():
    # Touching end to end, and 25 degrees apart.
    faults = system(a=plane(0.0, 20.0), b=plane(20.0, 20.0, dip_deg=65.0))
    assert moment_components(faults) == [("a",), ("b",)]


def test_components_join_through_a_chain():
    # a and c are 23 km apart, and each is 3 km from b.
    faults = system(c=plane(46.0, 20.0), a=plane(0.0, 20.0), b=plane(23.0, 20.0))
    assert moment_components(faults) == [("c", "a", "b")]


def test_separation_is_measured_below_the_connection_depth():
    # Equal dips in opposite directions: 0.5 km apart along their top edges, more than 6 km
    # apart below 5 km.
    a = plane(0.0, 20.0, dip_deg=60.0)
    b = plane(0.0, 20.0, north_km=-0.5, dip_deg=-60.0)
    faults = system(a=a, b=b)
    assert moment_components(faults) == [("a",), ("b",)]
    assert moment_components(faults, depth_km=0.0) == [("a", "b")]


def test_a_segment_above_the_connection_depth_is_measured_at_its_base():
    # b stops at 3 km. Below 2.97 km, 99% of that, the two lie 2 km apart.
    shallow = system(a=plane(0.0, 20.0), b=plane(22.0, 10.0, bottom_km=3.0))
    assert moment_components(shallow) == [("a", "b")]
    far = system(a=plane(0.0, 20.0), b=plane(26.0, 10.0, bottom_km=3.0))
    assert moment_components(far) == [("a",), ("b",)]


def test_a_cut_off_between_rows_of_nodes_is_sampled_where_it_falls():
    # a's far end slants back with depth, and its rows of nodes lie 4 km apart. Below
    # 5 km it comes closest to b at 5 km, 4.5 km away, between two rows; its nearest
    # node below 5 km is 7.5 km from b.
    depth = np.array([0.0, 4.0, 8.0, 12.0])[:, None]
    east = np.linspace(0.0, 1.0, 21)[None, :] * (20.0 - depth)
    nodes = np.stack(np.broadcast_arrays(east, 0.0, depth), axis=-1).astype(float)
    a = Geometry(nodes, np.ones((3, 20), bool), (20,))
    assert moment_components(system(a=a, b=plane(19.5, 10.0))) == [("a", "b")]


# -------------------------------------------------------------------- the shares


def test_one_component_shares_exactly_as_master_did():
    faults = system(
        a=plane(0.0, 20.0),
        b=plane(21.0, 12.0, dip_deg=80.0, bottom_km=12.0),
        c=plane(34.0, 30.0, top_km=1.0),
    )
    assert len(moment_components(faults)) == 1
    assert split_moment(MOMENT_NM, faults, MEDIUM) == master_split_moment(
        MOMENT_NM, faults, MEDIUM
    )


def test_one_segment_takes_the_whole_moment():
    faults = system(a=plane(0.0, 20.0))
    assert split_moment(MOMENT_NM, faults, MEDIUM) == {"a": MOMENT_NM}


def test_components_share_by_area_to_the_three_halves():
    faults = system(a=plane(0.0, 40.0), b=plane(60.0, 10.0))
    shares = split_moment(MOMENT_NM, faults, MEDIUM)
    # Equal widths in one medium: mu_c A_c^(3/2) goes as length^(3/2).
    assert shares["a"] / shares["b"] == pytest.approx(4.0**1.5, rel=1e-12)


lengths = st.integers(min_value=3, max_value=30).map(float)
gaps = st.sampled_from([0.0, 1.0, 3.0, 8.0, 20.0])
dips = st.sampled_from([90.0, 80.0, 65.0, 45.0])


@st.composite
def fault_systems(draw) -> Realisation[Geometry]:
    """A row of 2 to 5 segments along strike, some joined and some not."""
    count = draw(st.integers(min_value=2, max_value=5))
    segments, east = {}, 0.0
    for index in range(count):
        length = draw(lengths)
        segments[f"s{index}"] = plane(
            east, length, dip_deg=draw(dips), bottom_km=draw(st.sampled_from([8, 15]))
        )
        east += length + draw(gaps)
    return Realisation(segments, CRS)


@settings(max_examples=40, deadline=None)
@given(
    fault_systems(),
    st.sampled_from([0.0, 0.3, MOMENT_SIGMA_LN, 2.0]),
    st.integers(0, 2**32),
)
def test_the_shares_are_positive_and_sum_to_the_moment(faults, sigma_ln, seed):
    shares = split_moment(
        MOMENT_NM, faults, MEDIUM, sigma_ln=sigma_ln, rng=moment_rng(seed)
    )
    assert list(shares) == list(faults)
    assert all(share > 0.0 for share in shares.values())
    assert math.fsum(shares.values()) == pytest.approx(MOMENT_NM, rel=1e-12)


@settings(max_examples=40, deadline=None)
@given(fault_systems(), st.integers(0, 2**32))
def test_every_segment_of_a_component_has_one_mean_slip(faults, seed):
    for sigma_ln in (0.0, MOMENT_SIGMA_LN):
        slip = mean_slip(
            split_moment(
                MOMENT_NM, faults, MEDIUM, sigma_ln=sigma_ln, rng=moment_rng(seed)
            ),
            faults,
        )
        for names in moment_components(faults):
            first = slip[names[0]]
            assert [slip[name] for name in names] == pytest.approx(
                [first] * len(names), rel=1e-12
            )


def test_cutting_a_segment_in_two_leaves_every_mean_slip_as_it_was():
    whole = system(x=plane(0.0, 30.0), y=plane(31.0, 10.0), z=plane(70.0, 15.0))
    cut = system(
        x1=plane(0.0, 15.0),
        x2=plane(15.0, 15.0),
        y=plane(31.0, 10.0),
        z=plane(70.0, 15.0),
    )
    assert moment_components(whole) == [("x", "y"), ("z",)]
    assert moment_components(cut) == [("x1", "x2", "y"), ("z",)]
    for sigma_ln in (0.0, MOMENT_SIGMA_LN):
        # Both systems' components fall in the same order, so they take the same draws.
        before = mean_slip(
            split_moment(
                MOMENT_NM, whole, MEDIUM, sigma_ln=sigma_ln, rng=moment_rng(3)
            ),
            whole,
        )
        after = mean_slip(
            split_moment(MOMENT_NM, cut, MEDIUM, sigma_ln=sigma_ln, rng=moment_rng(3)),
            cut,
        )
        assert after["x1"] == pytest.approx(before["x"], rel=1e-12)
        assert after["x2"] == pytest.approx(before["x"], rel=1e-12)
        assert after["y"] == pytest.approx(before["y"], rel=1e-12)
        assert after["z"] == pytest.approx(before["z"], rel=1e-12)


# --------------------------------------------------------------- the departures

THREE_COMPONENTS = system(
    a=plane(0.0, 20.0),
    b=plane(21.0, 10.0),
    c=plane(50.0, 25.0),
    d=plane(100.0, 8.0),
    e=plane(109.0, 12.0, dip_deg=80.0),
)


def test_departures_scale_whole_components():
    reference = split_moment(MOMENT_NM, THREE_COMPONENTS, MEDIUM)
    drawn = split_moment(
        MOMENT_NM,
        THREE_COMPONENTS,
        MEDIUM,
        sigma_ln=MOMENT_SIGMA_LN,
        rng=moment_rng(11),
    )
    assert moment_components(THREE_COMPONENTS) == [("a", "b"), ("c",), ("d", "e")]
    assert drawn != pytest.approx(reference)
    assert drawn["a"] / drawn["b"] == pytest.approx(
        reference["a"] / reference["b"], rel=1e-12
    )
    assert drawn["d"] / drawn["e"] == pytest.approx(
        reference["d"] / reference["e"], rel=1e-12
    )
    assert math.fsum(drawn.values()) == pytest.approx(MOMENT_NM, rel=1e-12)


def test_departures_reproduce_from_the_seed():
    def draw(seed):
        return split_moment(
            MOMENT_NM, THREE_COMPONENTS, MEDIUM, sigma_ln=0.85, rng=moment_rng(seed)
        )

    assert draw(5) == draw(5)
    assert draw(5) != draw(6)


def test_no_departure_is_the_reference():
    assert split_moment(
        MOMENT_NM, THREE_COMPONENTS, MEDIUM, sigma_ln=0.0, rng=moment_rng(5)
    ) == split_moment(MOMENT_NM, THREE_COMPONENTS, MEDIUM)


def test_departures_follow_the_lognormal_about_the_reference():
    reference = split_moment(MOMENT_NM, THREE_COMPONENTS, MEDIUM)
    rng = moment_rng(0)
    residuals = []
    for _ in range(1500):
        drawn = split_moment(
            MOMENT_NM, THREE_COMPONENTS, MEDIUM, sigma_ln=MOMENT_SIGMA_LN, rng=rng
        )
        ratio = np.log([drawn[name] / reference[name] for name in ("a", "c", "d")])
        residuals.append(ratio - ratio.mean())
    # Centred within the rupture, as the calibration was: the spread of a residual of
    # three i.i.d. departures is sigma * sqrt(2/3).
    spread = float(np.std(np.asarray(residuals)))
    assert spread == pytest.approx(MOMENT_SIGMA_LN * math.sqrt(2.0 / 3.0), rel=0.05)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"sigma_ln": -0.1}, "non-negative"),
        ({"sigma_ln": math.nan}, "non-negative"),
        ({"sigma_ln": 0.5}, "needs a generator"),
    ],
)
def test_a_departure_without_a_spread_or_generator_is_refused(kwargs, message):
    with pytest.raises(RuptureGeneratorError, match=message):
        split_moment(MOMENT_NM, THREE_COMPONENTS, MEDIUM, **kwargs)


# ------------------------------------------------------------------- the config

FINITE = """
[source]
type = "finite"
magnitude = 6.8
rake_deg = 170.0
"""


def finite_config(tmp_path, apportionment: str = "") -> rupture_config.RuptureConfig:
    """The two-fault example with one magnitude for the event, at 1 km cells.

    ``next`` dips 25 degrees steeper than ``bent`` here, so the two are separate
    components and a departure moves moment between them.
    """
    geojson = (EXAMPLE.parent / "two_faults.geojson").read_text()
    assert '"dip_deg": 75' in geojson
    (tmp_path / "two_faults.geojson").write_text(
        geojson.replace('"dip_deg": 75', '"dip_deg": 85')
    )
    text = EXAMPLE.read_text().replace("spacing_km = 0.1", "spacing_km = 1.0")
    start = text.index("\n[source]\n")
    end = text.index("\n[propagation]\n")
    path = tmp_path / "finite.toml"
    path.write_text(text[:start] + FINITE + apportionment + text[end:])
    return rupture_config.load(path)


def test_a_file_without_an_apportionment_takes_the_reference(tmp_path):
    loaded = finite_config(tmp_path)
    assert loaded.source == rupture_config.Finite(magnitude=6.8, rake_deg=170.0)
    assert isinstance(loaded.source, rupture_config.Finite)
    assert loaded.source.apportionment == rupture_config.ReferenceApportionment()
    scenario = rupture_config.build(loaded)
    reference = split_moment(
        sum(source.moment_nm for source in scenario.sources.values()),
        scenario.realisation,
        scenario.medium,
    )
    assert {
        name: source.moment_nm for name, source in scenario.sources.items()
    } == pytest.approx(reference, rel=1e-12)


def test_a_sampled_apportionment_draws_its_own_stream(tmp_path):
    reference = rupture_config.build(finite_config(tmp_path))
    sampled_config = finite_config(
        tmp_path, '[source.apportionment]\ntype = "sampled"\n'
    )
    assert isinstance(sampled_config.source, rupture_config.Finite)
    assert sampled_config.source.apportionment == rupture_config.SampledApportionment(
        sigma_ln=MOMENT_SIGMA_LN
    )
    sampled = rupture_config.build(sampled_config)
    assert moment_components(sampled.realisation) == [("bent",), ("next",)]
    assert sampled.sources != reference.sources
    # The moment's stream is its own: the tree drawn from the same seed is the same.
    assert sampled.realisation.tree == reference.realisation.tree
    expected = split_moment(
        sum(source.moment_nm for source in reference.sources.values()),
        reference.realisation,
        reference.medium,
        sigma_ln=MOMENT_SIGMA_LN,
        rng=moment_rng(reference.seed),
    )
    assert {
        name: source.moment_nm for name, source in sampled.sources.items()
    } == pytest.approx(expected, rel=1e-12)
    assert rupture_config.build(sampled_config).sources == sampled.sources


def test_a_sampled_apportionment_round_trips(tmp_path):
    loaded = finite_config(
        tmp_path, '[source.apportionment]\ntype = "sampled"\nsigma_ln = 0.4\n'
    )
    again = tmp_path / "again.toml"
    again.write_text(rupture_config.dump(loaded))
    assert rupture_config.load(again) == loaded


@pytest.mark.parametrize(
    ("table", "message"),
    [
        (
            '[source.apportionment]\ntype = "sampled"\nsigma_ln = -1.0\n',
            "sigma_ln is a non-negative number",
        ),
        (
            '[source.apportionment]\ntype = "reference"\nsigma_ln = 0.5\n',
            "unknown keys",
        ),
        ('[source.apportionment]\ntype = "drawn"\n', "has no variant"),
    ],
)
def test_a_bad_apportionment_is_refused(tmp_path, table, message):
    with pytest.raises(RuptureGeneratorError, match=message):
        finite_config(tmp_path, table)
