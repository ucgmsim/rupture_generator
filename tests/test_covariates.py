"""Profiles of covariates other than depth: the covariates, the profiles, the config."""

import dataclasses

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from rupture_generator import (
    COVARIATES,
    SUZUKI_2022,
    CorrelatedRise,
    Geometry,
    NoiseFamily,
    RuptureGeneratorError,
    RuptureModel,
    ShearFractionSpeed,
    chart_field,
    constant_field,
    covariate_profile,
    covariate_ramp,
    embed,
    generate,
    interpolated_field,
    ramp_field,
    read_covariate,
)
from rupture_generator import config as rupture_config

from .conftest import EXAMPLE, holed, tilted

EXAMPLES = EXAMPLE.parent


def _chart(occupied, dip_km=1.0, strike_km=1.0):
    """A flat chart dipping 30 degrees, with cells this long down dip and along strike
    and this occupied mask."""
    rows, columns = occupied.shape
    i, j = np.mgrid[0 : rows + 1, 0 : columns + 1].astype(float)
    down = np.array([0.0, -np.cos(np.radians(30.0)), np.sin(np.radians(30.0))])
    along = np.array([1.0, 0.0, 0.0])
    nodes = (i * dip_km)[..., None] * down + (j * strike_km)[..., None] * along
    return Geometry(nodes, occupied, (columns,))


def _covariates(geometry):
    return {name: read_covariate(geometry, name) for name in COVARIATES}


CELL_KM = st.sampled_from([0.5, 1.0, 2.0])
"""Cell lengths down dip and along strike, which differ in most charts drawn."""


# ------------------------------------------------------------------ covariates


@settings(max_examples=40, deadline=None)
@given(st.integers(1, 7), st.integers(1, 9), CELL_KM, CELL_KM)
def test_a_planes_covariates_are_its_distances(rows, columns, dip_km, strike_km):
    geometry = _chart(np.ones((rows, columns), bool), dip_km, strike_km)
    i, j = np.mgrid[0:rows, 0:columns] + 0.5
    covariates = _covariates(geometry)
    expected = {
        "depth_km": i * dip_km * 0.5,
        "dip_km": i * dip_km,
        "dip_fraction": i / rows,
        "strike_km": j * strike_km,
        "strike_fraction": j / columns,
        "edge_distance_km": np.minimum.reduce(
            [i * dip_km, (rows - i) * dip_km, j * strike_km, (columns - j) * strike_km]
        ),
    }
    for name, values in expected.items():
        np.testing.assert_allclose(covariates[name], values, atol=1e-9, err_msg=name)


def test_a_holed_charts_distances_start_at_its_edges():
    covariates = _covariates(holed())
    # Column 0 starts a row down, and row 0 a column along, each still on its edge.
    np.testing.assert_allclose(
        covariates["dip_km"],
        [[0.5, 0.5, 0.5, 0.5], [0.5, 1.5, 1.5, 1.5], [1.5, 2.5, 2.5, 1.5]],
    )
    np.testing.assert_allclose(covariates["dip_fraction"][:, 0], [1 / 4, 1 / 4, 3 / 4])
    np.testing.assert_allclose(covariates["strike_km"][0, 1:], [0.5, 1.5, 2.5])
    np.testing.assert_allclose(
        covariates["strike_fraction"][2, :3], [1 / 6, 0.5, 5 / 6]
    )


def test_an_oblique_end_is_a_side_and_not_the_up_dip_edge():
    """A fault whose top runs level for 80 km and then steps down two rows a column,
    as an interface's oblique end does: the end's columns start deep, far down dip."""
    columns = np.arange(120)
    top = np.maximum(0, 2 * (columns - 80))
    occupied = np.arange(100)[:, None] >= top[None, :]
    geometry = _chart(occupied)
    dip_km = geometry.dip_km
    assert np.all(dip_km[0, :60] == 0.5)
    # A column on the end starts as far down dip as its distance to the level top.
    for column in range(100, 120):
        first = top[column]
        assert dip_km[first, column] > 0.5 * first
    for column in columns:
        on = occupied[:, column]
        assert np.all(np.diff(dip_km[on, column]) > 0.0)
        assert np.all(np.diff(geometry.dip_fraction[on, column]) > 0.0)


def _segment_distance(points, start, end):
    step = end - start
    t = np.clip(((points - start) @ step) / (step @ step), 0.0, 1.0)
    return np.linalg.norm(points - (start + t[:, None] * step), axis=-1)


def _outline_distance(occupied, cell):
    """Distance from each fault cell's centre to the outline, side by side."""
    n_i, n_j = occupied.shape
    padded = np.pad(occupied, 1)
    sides = [
        (np.array([a, j]), np.array([a, j + 1.0]))
        for a in range(n_i + 1)
        for j in range(n_j)
        if padded[a, j + 1] != padded[a + 1, j + 1]
    ] + [
        (np.array([i, b]), np.array([i + 1.0, b]))
        for i in range(n_i)
        for b in range(n_j + 1)
        if padded[i + 1, b] != padded[i + 1, b + 1]
    ]
    i, j = np.nonzero(occupied)
    centres = np.stack([i + 0.5, j + 0.5], axis=-1) * cell
    return np.min(
        [_segment_distance(centres, a * cell, b * cell) for a, b in sides], axis=0
    )


MASKS = hnp.arrays(bool, st.tuples(st.integers(1, 6), st.integers(1, 6))).filter(np.any)


@settings(max_examples=60, deadline=None)
@given(MASKS, CELL_KM, CELL_KM)
def test_the_edge_distance_is_to_the_nearest_side_of_the_outline(
    occupied, dip_km, strike_km
):
    geometry = _chart(occupied, dip_km, strike_km)
    np.testing.assert_allclose(
        geometry.edge_distance_km[occupied],
        _outline_distance(occupied, np.array([dip_km, strike_km])),
        atol=1e-9,
    )


@settings(max_examples=60, deadline=None)
@given(MASKS, CELL_KM, CELL_KM)
def test_a_cell_off_the_fault_reads_a_nearest_fault_cell(occupied, dip_km, strike_km):
    geometry = _chart(occupied, dip_km, strike_km)
    edge_km = geometry.edge_distance_km
    i, j = np.nonzero(occupied)
    for cell in zip(*np.nonzero(~occupied), strict=True):
        apart_km = np.hypot((i - cell[0]) * dip_km, (j - cell[1]) * strike_km)
        nearest = apart_km <= apart_km.min() + 1e-9
        assert np.any(np.isclose(edge_km[cell], edge_km[i[nearest], j[nearest]]))


@pytest.fixture(scope="module", params=[5.0, 2.0])
def hikurangi(request) -> Geometry:
    loaded = rupture_config.load(EXAMPLES / "hikurangi.toml")
    geometry = dataclasses.replace(loaded.geometry, spacing_km=request.param)
    scenario = rupture_config.build(dataclasses.replace(loaded, geometry=geometry))
    return scenario.realisation["Hikurangi"]


def test_a_curved_interfaces_covariates_follow_its_outline(hikurangi):
    occupied = hikurangi.occupied
    assert not occupied.all()
    covariates = _covariates(hikurangi)
    for name, values in covariates.items():
        assert values.shape == hikurangi.cells, name
        assert np.all(np.isfinite(values)), name
    for name in ("dip_fraction", "strike_fraction"):
        assert np.all((covariates[name] > 0.0) & (covariates[name] < 1.0)), name
    # Down every column of fault the distance and fraction grow.
    for column in np.flatnonzero(occupied.sum(axis=0) > 1):
        on = occupied[:, column]
        assert np.all(np.diff(covariates["dip_km"][on, column]) > 0.0)
        assert np.all(np.diff(covariates["dip_fraction"][on, column]) > 0.0)
    # The interface's southern end is oblique, and its columns start deep: they start
    # far down dip, not on the up-dip edge.
    columns = np.flatnonzero(occupied.any(axis=0))
    first = np.argmax(occupied, axis=0)[columns]
    deep = hikurangi.centres[first, columns, 2] > 20.0
    assert deep.sum() > 5
    assert np.all(covariates["dip_km"][first[deep], columns[deep]] > 40.0)
    shallow = hikurangi.centres[first, columns, 2] < 15.0
    on_edge = (
        covariates["dip_km"][first[shallow], columns[shallow]]
        == 0.5 * (hikurangi.spacing_km[1])
    )
    assert on_edge.mean() > 0.95
    # The up-dip edge is part of the outline, so no cell is farther from the outline
    # than from that edge.
    assert np.all(
        covariates["edge_distance_km"][occupied]
        <= covariates["dip_km"][occupied] + 1e-9
    )


def test_a_covariate_from_elsewhere_is_any_function_of_the_chart():
    geometry = holed()
    coupling = np.linspace(0.0, 1.0, 12).reshape(3, 4)
    profile = covariate_ramp(lambda chart: coupling, 0.5, 0.25, 0.0, 1.0)
    np.testing.assert_allclose(
        profile.on_chart(geometry), np.clip((coupling - 0.25) / 0.5, 0.0, 1.0)
    )
    with pytest.raises(RuptureGeneratorError, match=r"shaped \(2,\)"):
        covariate_ramp(lambda chart: np.zeros(2), 0.5, 0.25, 0.0, 1.0).on_chart(
            geometry
        )
    with pytest.raises(RuptureGeneratorError, match="not finite"):
        read_covariate(geometry, lambda chart: np.full((3, 4), np.nan))


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (
            lambda: covariate_ramp("depth", 1.0, 1.0, 0.0, 1.0),
            "'depth' is not a covariate",
        ),
        (lambda: covariate_ramp("dip_km", 1.0, 0.0, 0.0, 1.0), "is a step"),
        (
            lambda: covariate_ramp("dip_fraction", 6.5, 1.5, 1.0, 0.0),
            "from 5 to 8 lies wholly outside dip_fraction's range, 0 to 1",
        ),
        (
            lambda: covariate_profile("edge_distance_km", [-3.0, -1.0], [1.0, 0.0]),
            "lies wholly outside edge_distance_km's range, 0 to inf",
        ),
        (
            lambda: covariate_profile("dip_km", [1.0, 0.0], [0.0, 1.0]),
            "do not increase",
        ),
        (lambda: covariate_profile("dip_km", [0.0, 1.0], [0.0]), "1 values for 2"),
        (lambda: read_covariate(holed(), "edge_km"), "'edge_km' is not a covariate"),
        (
            lambda: covariate_ramp("dip_fraction", 0.5, 0.1, 0.0, 1.0)(
                np.zeros((2, 3))
            ),
            "read on a chart",
        ),
    ],
)
def test_a_profile_that_is_no_profile_is_refused(build, message):
    with pytest.raises(RuptureGeneratorError, match=message):
        build()


# -------------------------------------------------------------------- profiles


def test_a_depth_profile_of_the_covariate_is_the_depth_field():
    geometry, _ = tilted(30.0, 40.0, cells=(6, 5))
    positions = np.random.default_rng(1).uniform(-5.0, 30.0, (4, 7, 3))
    for ours, theirs in (
        (
            covariate_ramp("depth_km", 2.0, 1.0, 0.5, 0.13),
            ramp_field(2.0, 1.0, 0.5, 0.13),
        ),
        (
            covariate_profile("depth_km", [1.0, 2.0, 3.0], [2.0, 1.0, 2.0]),
            interpolated_field(np.array([1.0, 2.0, 3.0]), np.array([2.0, 1.0, 2.0])),
        ),
    ):
        np.testing.assert_array_equal(ours.on_chart(geometry), theirs(geometry.centres))
        np.testing.assert_array_equal(ours(positions), theirs(positions))


def test_a_slip_blend_on_the_dip_fraction_is_each_family_where_it_is_whole():
    geometry, _ = tilted(0.0, 30.0, cells=(20, 40))
    shallow = covariate_ramp("dip_fraction", 0.5, 0.1, 1.0, 0.0)
    deep = covariate_ramp("dip_fraction", 0.5, 0.1, 0.0, 1.0)
    # The same weights, written out from the fraction a plane's rows have.
    fraction = np.repeat(((np.arange(20) + 0.5) / 20)[:, None], 40, axis=1)
    written = np.clip((fraction - 0.4) / 0.2, 0.0, 1.0)
    families = (NoiseFamily(SUZUKI_2022, weight=shallow), NoiseFamily(weight=deep))
    by_hand = (
        NoiseFamily(SUZUKI_2022, weight=chart_field(lambda chart: 1.0 - written)),
        NoiseFamily(weight=chart_field(lambda chart: written)),
    )
    blend, expected = (embed(noise, geometry, 7.0) for noise in (families, by_hand))
    ours, theirs = (s.draw(np.random.default_rng(5))[0] for s in (blend, expected))
    np.testing.assert_allclose(ours, theirs, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(shallow.on_chart(geometry), 1.0 - written, atol=1e-12)


def _draw(scenario, model, seed=9):
    return generate(
        scenario.realisation,
        scenario.medium,
        scenario.sources,
        model,
        seed=seed,
        jump_propagation_model=scenario.jump_model,
    )


def test_a_speed_ramp_on_the_edge_distance_slows_the_front_at_the_edges(scenario):
    slow = covariate_ramp("edge_distance_km", 1.5, 0.5, 0.5, 1.0)
    plain = _draw(scenario, RuptureModel())
    slowed = _draw(scenario, RuptureModel(speed=ShearFractionSpeed(factor=slow)))
    for name, rupture in slowed.items():
        geometry = rupture.geometry
        edge_km = read_covariate(geometry, "edge_distance_km")
        np.testing.assert_allclose(
            rupture.speed_km_s,
            plain[name].speed_km_s * np.interp(edge_km, [1.0, 2.0], [0.5, 1.0]),
        )
        assert np.any(edge_km < 1.0) and np.any(edge_km > 2.0)
        np.testing.assert_array_equal(rupture.slip_m, plain[name].slip_m)


def test_a_profile_can_read_the_whole_chart(scenario):
    deeper = chart_field(lambda chart: 1.0 + chart.dip_fraction * chart.dip_deg / 90.0)
    plain = _draw(scenario, RuptureModel())
    drawn = _draw(scenario, RuptureModel(rise=CorrelatedRise(factor=deeper)))
    for name, rupture in drawn.items():
        geometry = rupture.geometry
        # Rise time is the same pattern times the factor, scaled to the same mean.
        ratio = rupture.rise_time_s / plain[name].rise_time_s
        expected = 1.0 + geometry.dip_fraction * geometry.dip_deg / 90.0
        scaled = (ratio / expected)[geometry.occupied]
        np.testing.assert_allclose(scaled, scaled[0], rtol=1e-10)
        assert np.ptp(expected[geometry.occupied]) > 0.1


# ------------------------------------------------------------ backward compatibility


def _legacy(value):
    """A model with each config profile as the plain function of position it was
    before covariates."""
    match value:
        case rupture_config.Constant():
            return constant_field(value.value)
        case rupture_config.Ramp(covariate="depth_km"):
            return ramp_field(value.centre, value.half_width, value.below, value.above)
        case rupture_config.Profile(covariate="depth_km"):
            return interpolated_field(np.array(value.at), np.array(value.values))
        case tuple():
            return tuple(map(_legacy, value))
        case _ if dataclasses.is_dataclass(value) and not isinstance(value, type):
            fields = dataclasses.fields(value)
            return dataclasses.replace(
                value, **{f.name: _legacy(getattr(value, f.name)) for f in fields}
            )
    return value


@pytest.mark.parametrize(
    "example",
    ["two_faults", "two_faults_segments", "alpine_hope", "northridge", "puysegur"],
)
def test_every_depth_profile_draws_as_it_did_before_covariates(example):
    """Each example's profiles, as the plain depth fields they were, draw the same bits."""
    loaded = rupture_config.load(EXAMPLES / f"{example}.toml")
    scenario = rupture_config.build(
        dataclasses.replace(
            loaded,
            geometry=dataclasses.replace(
                loaded.geometry, spacing_km=max(loaded.geometry.spacing_km, 1.0)
            ),
        )
    )
    model = scenario.model
    legacy = (
        {name: _legacy(m) for name, m in model.items()}
        if isinstance(model, dict)
        else _legacy(model)
    )
    ours, theirs = (
        _draw(scenario, model, scenario.seed),
        _draw(scenario, legacy, scenario.seed),
    )
    for name in ours:
        for field in (
            "slip_m",
            "rise_time_s",
            "rake_deg",
            "onset_s",
            "speed_km_s",
            "beta",
        ):
            np.testing.assert_array_equal(
                getattr(ours[name], field), getattr(theirs[name], field), err_msg=field
            )


def test_the_old_names_read_as_the_new():
    assert rupture_config.Ramp(
        centre_km=6.5, half_width_km=1.5, shallow=1.0, deep=0.0
    ) == rupture_config.Ramp(centre=6.5, half_width=1.5, below=1.0, above=0.0)
    assert rupture_config.Profile(
        depth_km=(1.0, 2.0), values=(0.0, 1.0)
    ) == rupture_config.Profile(at=(1.0, 2.0), values=(0.0, 1.0), covariate="depth_km")
    positions = np.random.default_rng(2).uniform(0.0, 10.0, (5, 3))
    np.testing.assert_array_equal(
        rupture_config.Ramp(centre_km=2.0, half_width_km=1.0, shallow=0.5, deep=0.1)(
            positions
        ),
        ramp_field(2.0, 1.0, 0.5, 0.1)(positions),
    )


def test_the_old_names_read_and_replace_as_the_new():
    ramp = rupture_config.Ramp(centre_km=6.5, half_width_km=1.5, shallow=1.0, deep=0.0)
    assert (ramp.centre_km, ramp.half_width_km, ramp.shallow, ramp.deep) == (
        6.5,
        1.5,
        1.0,
        0.0,
    )
    moved = dataclasses.replace(ramp, centre_km=3.0)
    assert (moved.centre, moved.centre_km, moved.half_width) == (3.0, 3.0, 1.5)
    assert dataclasses.replace(ramp, centre=4.0).centre_km == 4.0
    profile = rupture_config.Profile(depth_km=(1.0, 2.0), values=(0.0, 1.0))
    assert profile.depth_km == profile.at == (1.0, 2.0)
    assert dataclasses.replace(profile, depth_km=(3.0, 4.0)).at == (3.0, 4.0)


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (
            lambda: rupture_config.Ramp(
                covariate="dip_fraction",
                centre_km=0.5,
                half_width=0.1,
                below=0.0,
                above=1.0,
            ),
            "a ramp of dip_fraction takes centre; centre_km names kilometres",
        ),
        (
            lambda: rupture_config.Profile(
                covariate="dip_km", depth_km=(1.0, 2.0), values=(0.0, 1.0)
            ),
            "a profile of dip_km takes at; depth_km names depths",
        ),
        (
            lambda: rupture_config.Ramp(
                covariate="dip_fraction",
                centre=6.5,
                half_width=1.5,
                below=1.0,
                above=0.0,
            ),
            "wholly outside dip_fraction's range",
        ),
    ],
)
def test_a_profile_whose_units_contradict_its_covariate_is_refused(build, message):
    with pytest.raises(RuptureGeneratorError, match=message):
        build()


def test_an_old_name_in_kilometres_reads_on_a_covariate_in_kilometres():
    ramp = rupture_config.Ramp(
        covariate="dip_km", centre_km=10.0, half_width_km=5.0, shallow=1.0, deep=0.0
    )
    assert (ramp.centre, ramp.half_width) == (10.0, 5.0)


# ---------------------------------------------------------------------- config

BASE = EXAMPLE.read_text().replace(
    "two_faults.geojson", str(EXAMPLE.with_name("two_faults.geojson"))
)

BLEND = """
[[slip.noise]]
relation = { strike_offset = 1.67, dip_exponent = 0.5, dip_offset = 1.69, dip_saturation = 6.3 }
weight = { type = "ramp", covariate = "dip_fraction", centre = 0.3, half_width = 0.1, below = 1.0, above = 0.0 }

[[slip.noise]]
weight = { type = "ramp", covariate = "dip_fraction", centre = 0.3, half_width = 0.1, below = 0.0, above = 1.0 }
"""


def test_a_file_can_blend_slips_noise_on_the_dip_fraction(tmp_path):
    path = tmp_path / "blend.toml"
    path.write_text(BASE + BLEND)
    loaded = rupture_config.load(path)
    shallow, deep = loaded.slip.noise
    assert shallow.weight == rupture_config.Ramp(
        covariate="dip_fraction", centre=0.3, half_width=0.1, below=1.0, above=0.0
    )
    again = tmp_path / "again.toml"
    again.write_text(rupture_config.dump(loaded))
    assert rupture_config.load(again) == loaded
    scenario = rupture_config.build(loaded)
    for geometry in scenario.realisation.values():
        np.testing.assert_allclose(
            shallow.weight.on_chart(geometry) + deep.weight.on_chart(geometry), 1.0
        )
    depth = rupture_config.build(rupture_config.load(EXAMPLE))
    ours, theirs = _draw(scenario, scenario.model), _draw(depth, depth.model)
    assert not np.allclose(ours["bent"].slip_m, theirs["bent"].slip_m)


def test_a_file_in_the_old_names_loads_as_in_the_new(tmp_path):
    assert "centre_km = 2.0" in BASE
    renamed = (
        BASE.replace("centre_km", "centre")
        .replace("half_width_km", "half_width")
        .replace("shallow =", "below =")
        .replace("deep =", "above =")
        .replace("depth_km =", "at =")
    )
    old, new = tmp_path / "old.toml", tmp_path / "new.toml"
    old.write_text(BASE)
    new.write_text(renamed)
    assert rupture_config.load(old) == rupture_config.load(new)


@pytest.mark.parametrize(
    ("beta", "message"),
    [
        (
            'covariate = "depth", centre = 2.0, half_width = 1.0, below = 0.5, above = 0.1',
            (
                "pulse.beta: 'depth' is not a covariate; the covariates are depth_km, "
                "dip_km, dip_fraction, strike_km, strike_fraction, edge_distance_km"
            ),
        ),
        (
            "centre = 2.0, centre_km = 2.0, half_width = 1.0, below = 0.5, above = 0.1",
            "pulse.beta: a ramp takes centre or its old name centre_km, not both",
        ),
        (
            "half_width = 1.0, below = 0.5, above = 0.1",
            "pulse.beta: a ramp needs centre",
        ),
        (
            (
                'covariate = "dip_fraction", centre_km = 0.3, half_width = 0.1, '
                "below = 0.5, above = 0.1"
            ),
            (
                "pulse.beta: a ramp of dip_fraction takes centre; centre_km names "
                "kilometres, which dip_fraction is not in"
            ),
        ),
        (
            (
                'covariate = "dip_fraction", centre = 6.5, half_width = 1.5, '
                "below = 0.5, above = 0.1"
            ),
            (
                "pulse.beta: a profile from 5 to 8 lies wholly outside dip_fraction's "
                "range, 0 to 1, so it is one value over the whole fault"
            ),
        ),
    ],
)
def test_a_bad_ramp_gives_exactly_this_message(tmp_path, beta, message):
    constant = 'beta = { type = "constant", value = 0.2 }'
    assert constant in BASE
    path = tmp_path / "bad.toml"
    path.write_text(BASE.replace(constant, f'beta = {{ type = "ramp", {beta} }}'))
    with pytest.raises(RuptureGeneratorError) as raised:
        rupture_config.load(path)
    assert str(raised.value) == message
