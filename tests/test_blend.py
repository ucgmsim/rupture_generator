"""Noise families: blended sampling, the models that draw noise, and the config."""

import dataclasses

import numpy as np
import pytest

from rupture_generator import (
    STATIONARY,
    SUZUKI_2022,
    CorrelatedPerturbation,
    DrawnSlip,
    GaussianRake,
    NoiseFamily,
    RuptureGeneratorError,
    RuptureModel,
    generate,
)
from rupture_generator import config as rupture_config
from rupture_generator.rupture.source import MAI_BEROZA
from rupture_generator.sampling import Covariance, Grid, VonKarman, sampler

from .conftest import EXAMPLE

GRID = Grid((24, 60), (0.5, 0.5))
SHORT = Covariance(VonKarman(), (2.0, 3.0))
LONG = Covariance(VonKarman(), (4.0, 12.0))
DEEP = np.repeat(np.clip((np.arange(24) - 8.0) / 8.0, 0.0, 1.0)[:, None], 60, axis=1)
"""The deep family's weight: none above row 8, all below row 16, linear between."""


def test_saturation_stops_one_axis_growing():
    below, above = SUZUKI_2022.lengths_km(6.3), SUZUKI_2022.lengths_km(7.0)
    assert above[0] == pytest.approx(below[0])
    assert above[1] > below[1]
    assert MAI_BEROZA.lengths_km(7.0) == pytest.approx(
        (10.0 ** (7.0 / 3.0 - 1.5), 10.0 ** (3.5 - 2.5))
    )


def test_covariances_share_the_widest_padded_shape():
    across = Covariance(VonKarman(), (12.0, 2.0))
    own = [sampler(GRID, [c]).padded for c in (across, SHORT, LONG)]
    widest = (max(p[0] for p in own), max(p[1] for p in own))
    assert widest not in own
    assert sampler(GRID, [across, SHORT, LONG]).padded == widest


def test_a_cell_of_one_family_is_that_familys_draw():
    blend = sampler(GRID, [LONG, SHORT], [1.0 - DEEP, DEEP])
    alone = sampler(GRID, [LONG])
    assert blend.padded == alone.padded
    mixed, alone_draw = (s.draw(np.random.default_rng(3)) for s in (blend, alone))
    for ours, theirs in zip(mixed, alone_draw, strict=True):
        np.testing.assert_allclose(ours[:8], theirs[:8], rtol=1e-12)
        assert not np.allclose(ours[16:], theirs[16:])


def test_draws_are_standard_normal_and_the_pair_independent():
    blend = sampler(GRID, [LONG, SHORT], [1.0 - DEEP, DEEP])
    rng = np.random.default_rng(11)
    pairs = np.array([blend.draw(rng) for _ in range(600)])
    real, imaginary = pairs[:, 0], pairs[:, 1]
    # Through the crossfade, rows 8 to 16, as well as either side of it.
    assert np.abs(real.var(axis=0) - 1.0).max() < 0.25
    assert abs(float(real.var() - 1.0)) < 0.05
    assert abs(float(np.mean(real * imaginary))) < 0.02


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        ([np.zeros(GRID.shape), DEEP], "weigh every covariance at zero"),
        ([1.0 - DEEP, 2.0 * DEEP], r"outside \[0, 1\]"),
        ([DEEP], "need weights shaped"),
    ],
)
def test_bad_weights_are_refused(weights, message):
    with pytest.raises(RuptureGeneratorError, match=message):
        sampler(GRID, [LONG, SHORT], weights)


# ----------------------------------------------------------------- the models


def _deep(positions):
    return np.clip((positions[..., 2] - 4.0) / 4.0, 0.0, 1.0)


def _shallow(positions):
    return 1.0 - _deep(positions)


HYBRID = (NoiseFamily(SUZUKI_2022, 0.6, _shallow), NoiseFamily(weight=_deep))


def _draw(scenario, model):
    return generate(
        scenario.realisation, scenario.medium, scenario.sources, model, seed=9
    )


def test_slip_draws_its_noise_and_rake_keeps_its_own(scenario):
    plain = _draw(scenario, RuptureModel())
    hybrid = _draw(scenario, RuptureModel(slip=DrawnSlip(noise=HYBRID)))
    rake = _draw(scenario, RuptureModel(rake=GaussianRake(noise=HYBRID)))
    for name in scenario.realisation:
        assert not np.allclose(hybrid[name].slip_m, plain[name].slip_m)
        np.testing.assert_array_equal(hybrid[name].rake_deg, plain[name].rake_deg)
        np.testing.assert_array_equal(rake[name].slip_m, plain[name].slip_m)
        assert not np.allclose(rake[name].rake_deg, plain[name].rake_deg)


def test_the_perturbation_can_draw_its_own_noise(scenario):
    slip = DrawnSlip(noise=HYBRID)
    own, slips = (
        _draw(
            scenario,
            RuptureModel(slip=slip, perturbation=CorrelatedPerturbation(noise=noise)),
        )
        for noise in (STATIONARY, None)
    )
    for name in scenario.realisation:
        assert not np.allclose(own[name].onset_s, slips[name].onset_s)


def test_a_weight_out_of_range_is_refused(scenario):
    noise = (NoiseFamily(weight=lambda positions: np.full(positions.shape[:-1], 2.0)),)
    with pytest.raises(RuptureGeneratorError, match=r"\[0, 1\]"):
        _draw(scenario, RuptureModel(slip=DrawnSlip(noise=noise)))


# ----------------------------------------------------------------- the config

NOISE = """
[[slip.noise]]
relation = { strike_offset = 1.67, dip_exponent = 0.5, dip_offset = 1.69, dip_saturation = 6.3 }
hurst = 0.6
weight = { type = "ramp", centre_km = 6.5, half_width_km = 1.5, shallow = 1.0, deep = 0.0 }

[[slip.noise]]
weight = { type = "ramp", centre_km = 6.5, half_width_km = 1.5, shallow = 0.0, deep = 1.0 }

[[perturbation.noise]]
"""


def _hybrid_file(tmp_path, extra=""):
    path = tmp_path / "hybrid.toml"
    path.write_text(
        EXAMPLE.read_text().replace(
            "two_faults.geojson", str(EXAMPLE.with_name("two_faults.geojson"))
        )
        + NOISE
        + extra
    )
    return path


def test_noise_loads_round_trips_and_draws(tmp_path):
    loaded = rupture_config.load(_hybrid_file(tmp_path))
    shallow, deep = loaded.slip.noise
    assert shallow.relation == rupture_config.Correlation(
        strike_offset=1.67, dip_exponent=0.5, dip_offset=1.69, dip_saturation=6.3
    )
    assert (shallow.hurst, deep.relation) == (0.6, rupture_config.Correlation())
    assert loaded.perturbation.noise == rupture_config.STATIONARY
    assert loaded.rake.noise == rupture_config.STATIONARY
    again = tmp_path / "again.toml"
    again.write_text(rupture_config.dump(loaded))
    assert rupture_config.load(again) == loaded
    scenario = rupture_config.build(loaded)
    _draw(scenario, scenario.model)


def test_a_segments_noise_replaces_the_global_list(tmp_path):
    extra = "\n[segments.next.slip]\nnoise = [{ hurst = 0.5 }]\n"
    loaded = rupture_config.load(_hybrid_file(tmp_path, extra))
    segment = loaded.segments["next"].slip
    assert segment is not None and len(segment.noise) == 1
    assert dataclasses.replace(segment, noise=loaded.slip.noise) == loaded.slip
