"""The hot paths, each timed alone on a production-sized grid.

Every case is a 0.1 km grid as large as one segment of the examples: a 25 km wide,
60 km long fault for the field and front kernels, and the same fault's subfaults for
the pulses. Each module draws its inputs once, outside the timing.
"""

import numpy as np
import pytest

from rupture_generator._kernels import (
    circulant_draw,
    eikonal_solve,
    synthesise_pulses,
)
from rupture_generator.rupture.source import segment_source
from rupture_generator.sampling import Grid
from rupture_generator.sampling.field import _embed

SPACING_KM = 0.1
SHAPE = (250, 600)
"""Cells down dip and along strike: a 25 x 60 km fault at 0.1 km."""

MAGNITUDES = (6.5, 7.5)
"""Correlation lengths grow with magnitude, and the embedding's padding with them."""

pytestmark = pytest.mark.benchmark(group="kernels")


@pytest.fixture(scope="module")
def grid():
    return Grid(SHAPE, (SPACING_KM, SPACING_KM))


@pytest.fixture(scope="module", params=MAGNITUDES, ids=lambda m: f"Mw{m}")
def covariance(request):
    return segment_source(request.param, 90.0).covariance


def test_embed(benchmark, grid, covariance):
    amplitudes, padded = benchmark(_embed, grid, covariance)
    benchmark.extra_info["padded_cells"] = padded[0] * padded[1]
    assert amplitudes.shape == (padded[0] // 2 + 1, padded[1] // 2 + 1)


def test_circulant_draw(benchmark, grid, covariance):
    amplitudes, padded = _embed(grid, covariance)
    real, _ = benchmark(circulant_draw, amplitudes, padded, grid.shape, 7)
    assert real.shape == grid.shape


def test_eikonal_solve(benchmark):
    rng = np.random.default_rng(0)
    # A slowness field with the speed field's contrast, 0.25 to 0.6 s/km.
    slowness = np.ascontiguousarray(rng.uniform(0.25, 0.6, SHAPE))
    times = benchmark(
        eikonal_solve,
        slowness,
        (SPACING_KM, SPACING_KM),
        [(SHAPE[0] // 2, SHAPE[1] // 4, 0.0)],
    )
    assert np.all(np.isfinite(times))


@pytest.mark.parametrize("shape", ["impulse", "liu"])
def test_synthesise_pulses(benchmark, shape):
    rng = np.random.default_rng(1)
    count = SHAPE[0] * SHAPE[1]
    slip_m = rng.gamma(2.0, 1.0, count)
    rise_s = rng.uniform(1.0, 4.0, count)
    beta = None if shape == "impulse" else np.full(count, 0.2)
    offsets, _ = benchmark(synthesise_pulses, slip_m, rise_s, 0.005, beta)
    assert offsets.size == count + 1
