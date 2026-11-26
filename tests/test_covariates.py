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


def _occupied_chart(occupied):
    """A flat chart of 1 km cells dipping 30 degrees, with this occupied mask."""
    return tilted(0.0, 30.0, cells=occupied.shape, occupied=occupied)[0]


def _covariates(geometry):
    return {name: read_covariate(geometry, name) for name in COVARIATES}


# ------------------------------------------------------------------ covariates


@settings(max_examples=40, deadline=None)
@given(
    st.floats(0.0, 359.0),
    st.floats(5.0, 90.0),
    st.integers(1, 7),
    st.integers(1, 9),
)
def test_a_planes_covariates_are_its_distances(strike_deg, dip_deg, rows, columns):
    geometry, _ = tilted(strike_deg, dip_deg, cells=(rows, columns))
    i, j = np.mgrid[0:rows, 0:columns] + 0.5
    covariates = _covariates(geometry)
    expected = {
        "depth_km": i * np.sin(np.radians(dip_deg)),
        "dip_km": i,
        "dip_fraction": i / rows,
        "strike_km": j,
        "strike_fraction": j / columns,
        "edge_distance_km": np.minimum.reduce([i, rows - i, j, columns - j]),
    }
    for name, values in expected.items():
        np.testing.assert_allclose(covariates[name], values, atol=1e-9, err_msg=name)


def test_a_holed_charts_distances_start_at_its_own_edges():
    covariates = _covariates(holed())
    # Column 0 starts a row down, and column 3 ends a row up.
    np.testing.assert_allclose(covariates["dip_km"][1:, 0], [0.5, 1.5])
    np.testing.assert_allclose(covariates["dip_fraction"][1:, 0], [0.25, 0.75])
    np.testing.assert_allclose(covariates["dip_fraction"][:2, 3], [0.25, 0.75])
    np.testing.assert_allclose(covariates["strike_km"][0, 1:], [0.5, 1.5, 2.5])
    np.testing.assert_allclose(
        covariates["strike_fraction"][2, :3], [1 / 6, 0.5, 5 / 6]
    )
    # A cell off the fault takes its nearest fault cell's value.
    for values in covariates.values():
        assert np.all(np.isfinite(values))
    assert covariates["dip_km"][0, 0] in (
        covariates["dip_km"][1, 0],
        covariates["dip_km"][0, 1],
    )


def _segment_distance(points, start, end):
    step = end - start
    t = np.clip(((points - start) @ step) / (step @ step), 0.0, 1.0)
    return np.linalg.norm(points - (start + t[:, None] * step), axis=-1)


def _outline_distance(occupied):
    """Distance from each fault cell's centre to the outline, side by side, in cells."""
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
    centres = np.stack([i + 0.5, j + 0.5], axis=-1)
    return np.min([_segment_distance(centres, *side) for side in sides], axis=0)


@settings(max_examples=60, deadline=None)
@given(hnp.arrays(bool, st.tuples(st.integers(1, 6), st.integers(1, 6))))
def test_the_edge_distance_is_to_the_nearest_side_of_the_outline(occupied):
    if not occupied.any():
        return
    geometry = _occupied_chart(occupied)
    np.testing.assert_allclose(
        geometry.edge_distance_km[occupied], _outline_distance(occupied), atol=1e-9
    )


@pytest.fixture(scope="module")
def hikurangi() -> Geometry:
    loaded = rupture_config.load(EXAMPLES / "hikurangi.toml")
    geometry = dataclasses.replace(loaded.geometry, spacing_km=10.0)
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
    # Down every column of fault the distance grows from the trench, and the bottom
    # is deeper than the top, though the interface near the trench is not monotone.
    dip_km = np.where(occupied, covariates["dip_km"], np.nan)
    depth_km = np.where(occupied, covariates["depth_km"], np.nan)
    for column in range(hikurangi.cells[1]):
        on = occupied[:, column]
        if on.sum() > 1:
            assert np.all(np.diff(dip_km[on, column]) > 0.0)
            assert depth_km[on, column][-1] > depth_km[on, column][0]
    # Each column's fraction runs from its own top edge to its own bottom edge,
    # wherever along strike they lie.
    columns = np.flatnonzero(occupied.sum(axis=0) > 4)
    first = np.argmax(occupied, axis=0)[columns]
    last = occupied.shape[0] - 1 - np.argmax(occupied[::-1], axis=0)[columns]
    fraction = covariates["dip_fraction"]
    assert np.all(fraction[first, columns] < 0.15)
    assert np.all(fraction[last, columns] > 0.85)
    # The distance over the fraction is the column's length, all the way down it.
    for column in columns:
        on = occupied[:, column]
        length_km = dip_km[on, column] / fraction[on, column]
        np.testing.assert_allclose(length_km, length_km[0], rtol=1e-12)
    # The top edge is part of the outline, so no cell is farther from the outline
    # than from its column's top, but for the chart's 1-2% stretch.
    on = occupied
    assert np.all(
        covariates["edge_distance_km"][on] <= 1.05 * covariates["dip_km"][on] + 1e-9
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
    by_dip = chart_field(lambda chart: 1.0 + (chart.dip_deg > 0.0))
    drawn = _draw(scenario, RuptureModel(rise=CorrelatedRise(factor=by_dip)))
    assert all(np.all(r.rise_time_s >= 0.0) for r in drawn.values())


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
