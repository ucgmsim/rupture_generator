"""Slip noise families: the blended sampler, the models that use it, the config."""

import dataclasses

import numpy as np
import pytest

from rupture_generator import RuptureGeneratorError, generate
from rupture_generator import config as rupture_config
from rupture_generator.rupture.models import (
    CorrelatedPerturbation,
    DrawnSlip,
    NoiseFamily,
    RuptureModel,
    SegmentContext,
)
from rupture_generator.rupture.source import MAI_BEROZA, SUZUKI_2022
from rupture_generator.sampling import Blend, Covariance, Grid, VonKarman, sampler
from rupture_generator.sampling.field import _mirror_weights

from .conftest import EXAMPLE

GRID = Grid((24, 60), (0.5, 0.5))
SHORT = Covariance(VonKarman(), (2.0, 3.0))
LONG = Covariance(VonKarman(), (4.0, 12.0))


def _ramp(rows: int = GRID.shape[0]) -> np.ndarray:
    """Weight of the deep family: 0 above row 8, 1 below row 16, linear between."""
    column = np.clip((np.arange(rows) - 8.0) / 8.0, 0.0, 1.0)
    return np.repeat(column[:, None], GRID.shape[1], axis=1)


def _blend() -> Blend:
    deep = _ramp()
    return Blend((LONG, SHORT), (1.0 - deep, deep))


def _cross(samplers) -> np.ndarray:
    padded = samplers[0].padded
    weights = np.outer(*(_mirror_weights(extent) for extent in padded))
    return np.array(
        [
            [np.sum(weights * a.amplitudes * b.amplitudes) for b in samplers]
            for a in samplers
        ]
    ) / (padded[0] * padded[1])


def test_saturation_stops_one_axis_growing():
    below, above = SUZUKI_2022.lengths_km(6.3), SUZUKI_2022.lengths_km(7.0)
    assert above[0] == pytest.approx(below[0])
    assert above[1] > below[1]
    assert MAI_BEROZA.lengths_km(7.0) == pytest.approx(
        (10.0 ** (7.0 / 3.0 - 1.5), 10.0 ** (3.5 - 2.5))
    )


def test_a_floor_only_ever_widens_the_embedding():
    alone = sampler(GRID, SHORT)
    widened = sampler(GRID, SHORT, (alone.padded[0] * 2, alone.padded[1]))
    assert widened.padded == (alone.padded[0] * 2, alone.padded[1])
    assert sampler(GRID, SHORT, (2, 2)).padded == alone.padded


def test_the_components_share_the_widest_padded_shape():
    across = Covariance(VonKarman(), (12.0, 2.0))
    blended = Blend((across, SHORT, LONG), (np.full(GRID.shape, 1.0),) * 3).sampler(
        GRID
    )
    own = [sampler(GRID, c).padded for c in (across, SHORT, LONG)]
    widest = tuple(max(axis) for axis in zip(*own))
    assert all(s.padded == widest for s in blended.samplers)


def test_every_cell_has_unit_variance_exactly():
    blended = _blend().sampler(GRID)
    variance = np.einsum(
        "ikl,ij,jkl->kl", blended.loadings, _cross(blended.samplers), blended.loadings
    )
    np.testing.assert_allclose(variance, 1.0, rtol=1e-12)


def test_a_cell_of_one_family_is_that_familys_draw():
    blended = _blend().sampler(GRID)
    real, imaginary = blended.draw(np.random.default_rng(3))
    # The same seed draws the same noise for every component.
    seed = int(
        np.random.default_rng(3).integers(0, np.iinfo(np.uint64).max, dtype=np.uint64)
    )
    from rupture_generator._kernels import circulant_draw

    long_draw = circulant_draw(
        blended.samplers[0].amplitudes, blended.samplers[0].padded, GRID.shape, seed
    )
    scale = np.sqrt(_cross(blended.samplers)[0, 0])
    np.testing.assert_allclose(real[:8], long_draw[0][:8] / scale, rtol=1e-12)
    np.testing.assert_allclose(imaginary[:8], long_draw[1][:8] / scale, rtol=1e-12)


def test_draws_are_standard_normal_and_the_pair_independent():
    blended = _blend().sampler(GRID)
    rng = np.random.default_rng(11)
    pairs = np.array([blended.draw(rng) for _ in range(600)])
    real, imaginary = pairs[:, 0], pairs[:, 1]
    # Through the crossfade, rows 8 to 16, as well as either side of it.
    assert np.abs(real.var(axis=0) - 1.0).max() < 0.25
    assert abs(float(real.var() - 1.0)) < 0.05
    assert abs(float(np.mean(real * imaginary))) < 0.02


def test_a_cell_with_no_weight_is_refused():
    deep = _ramp()
    with pytest.raises(RuptureGeneratorError, match="no positive weight"):
        Blend((LONG, SHORT), (np.zeros(GRID.shape), deep)).sampler(GRID)


# ----------------------------------------------------------------- the models


def _hybrid_slip() -> DrawnSlip:
    def deep(positions):
        return np.clip((positions[..., 2] - 4.0) / 4.0, 0.0, 1.0)

    return DrawnSlip(
        noise=(
            NoiseFamily(lambda p: 1.0 - deep(p), SUZUKI_2022, hurst=0.6),
            NoiseFamily(deep),
        )
    )


def _context(scenario, name="bent"):
    chart = scenario.realisation[name]
    return SegmentContext(
        chart,
        scenario.sources[name],
        scenario.medium.shear_speed_km_s(chart.centres),
        scenario.medium.density_g_cm3(chart.centres),
    )


def test_slip_draws_a_blend_its_partners_follow(scenario):
    context = _context(scenario)
    slip = _hybrid_slip()(context, np.random.default_rng(5))
    blend = slip.latent_covariance
    assert isinstance(blend, Blend)
    long, short = blend.components
    assert long.correlation.target == VonKarman(0.6)
    assert short.lengths_km == scenario.sources["bent"].covariance.lengths_km
    assert context.moment_nm(slip.slip_m) == pytest.approx(context.source.moment_nm)
    assert np.all(np.isfinite(slip.latent))


def test_a_weight_out_of_range_is_refused(scenario):
    slip = DrawnSlip(noise=(NoiseFamily(lambda p: np.full(p.shape[:-1], 2.0)),))
    with pytest.raises(RuptureGeneratorError, match=r"weights in \[0, 1\]"):
        slip(_context(scenario), np.random.default_rng(0))


def test_the_perturbation_can_keep_the_sources_noise(scenario):
    def onset(independent):
        model = RuptureModel(
            slip=_hybrid_slip(),
            perturbation=CorrelatedPerturbation(independent=independent),
        )
        drawn = generate(
            scenario.realisation, scenario.medium, scenario.sources, model, seed=9
        )
        return drawn["bent"].onset_s

    assert not np.allclose(onset("slip"), onset("source"))


# ----------------------------------------------------------------- the config

NOISE = """
[[slip.noise]]
relation = { strike_offset = 1.67, dip_exponent = 0.5, dip_offset = 1.69, dip_saturation = 6.3 }
weight = { type = "ramp", centre_km = 6.5, half_width_km = 1.5, shallow = 1.0, deep = 0.0 }

[[slip.noise]]
weight = { type = "ramp", centre_km = 6.5, half_width_km = 1.5, shallow = 0.0, deep = 1.0 }
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


def test_noise_families_load_and_round_trip(tmp_path):
    loaded = rupture_config.load(_hybrid_file(tmp_path))
    shallow, deep = loaded.slip.noise
    assert shallow.relation == rupture_config.Correlation(
        strike_offset=1.67, dip_exponent=0.5, dip_offset=1.69, dip_saturation=6.3
    )
    assert deep.relation is None and deep.hurst is None
    again = tmp_path / "again.toml"
    again.write_text(rupture_config.dump(loaded))
    assert rupture_config.load(again) == loaded


def test_a_segments_noise_replaces_the_global_list(tmp_path):
    extra = '\n[segments.next.slip]\nnoise = [{ weight = { type = "constant", value = 1.0 } }]\n'
    loaded = rupture_config.load(_hybrid_file(tmp_path, extra))
    assert len(loaded.slip.noise) == 2
    segment = loaded.segments["next"].slip
    assert segment is not None and len(segment.noise) == 1
    assert dataclasses.replace(segment, noise=loaded.slip.noise) == loaded.slip
