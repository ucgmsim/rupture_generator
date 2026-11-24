import io
import itertools
import json

import numpy as np
import pyproj
import pytest
import scipy as sp
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from rupture_generator import Geometry, geometry_from_geojson, moment_from_magnitude
from rupture_generator.geometry.crs import true_azimuth_deg
from rupture_generator.geometry.geometry import bordering
from rupture_generator.rupture.source import MAI_BEROZA
from rupture_generator.sampling import (
    Covariance,
    Grid,
    Marginal,
    PreCorrected,
    VonKarman,
)
from rupture_generator.sampling.field import (
    MAXIMUM_VARIANCE_DEFICIT,
    WRAP_TOLERANCE,
    _decay_length,
    _embed,
    sampler,
)

from .conftest import NZTM, ORIGIN_KM, holed, tilted

TRACE = [[172.0, -42.0], [172.3, -41.8], [172.7, -41.7]]

SECTION = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": TRACE,
            },
            "properties": {
                "name": "bent",
                "dip_deg": 60,
                "dip_direction_deg": 300,
                "upper_depth_km": 0,
                "lower_depth_km": 15,
            },
        }
    ],
}


def _bent():
    return geometry_from_geojson(
        io.StringIO(json.dumps(SECTION)), pyproj.CRS("EPSG:2193")
    )["bent"]


def test_a_bent_trace_hangs_as_one_translated_sheet():
    coarse = _bent()
    # Every node drops by the same vector, so neighbouring planes share their
    # junction all the way down and no seam opens.
    down = coarse.nodes[1] - coarse.nodes[0]
    np.testing.assert_allclose(down, down[:1].repeat(len(down), axis=0), atol=1e-12)
    reach_km = 15.0 / np.tan(np.radians(60.0))
    assert np.linalg.norm(down[0, :2]) == pytest.approx(reach_km, rel=1e-12)
    assert down[0, 2] == pytest.approx(15.0)


def test_a_plane_off_square_to_the_dip_direction_dips_more_steeply():
    trace = np.array(TRACE)
    strikes_deg, _, _ = pyproj.Geod(ellps="WGS84").inv(
        trace[:-1, 0], trace[:-1, 1], trace[1:, 0], trace[1:, 1]
    )
    across = np.abs(np.sin(np.radians(300.0 - strikes_deg)))
    expected = np.degrees(np.arctan(np.tan(np.radians(60.0)) / across))
    # The trace dips to the left. The section reverses it, and its planes run backwards.
    # The 0.05 degrees is the projection's grid convergence, which the geodesic
    # strikes here don't see.
    np.testing.assert_allclose(_bent().dip_deg[0], expected[::-1], atol=0.05)


def test_subdivided_planes_keep_their_area_and_dip():
    coarse = _bent()
    fine = coarse.subdivide(0.5)
    assert fine.areas_km2.sum() == pytest.approx(coarse.areas_km2.sum(), rel=1e-12)
    for plane, (start, stop) in enumerate(
        itertools.pairwise(np.cumsum([0, *fine.plane_cells]))
    ):
        np.testing.assert_allclose(
            fine.dip_deg[:, start:stop], coarse.dip_deg[0, plane], atol=1e-9
        )
    assert fine.plane_cells == (70, 67)


def _reversed(chart):
    """The same cells with ``j`` running the other way."""
    return Geometry(
        chart.nodes[:, ::-1].copy(), chart.occupied[:, ::-1], (chart.cells[1],)
    )


@given(st.floats(0.0, 360.0), st.floats(1.0, 89.0))
def test_strike_keeps_the_fault_to_its_right_whichever_way_the_grid_runs(
    strike_deg, dip_deg
):
    chart, along = tilted(strike_deg, dip_deg)
    for grid in (chart, _reversed(chart)):
        np.testing.assert_allclose(
            grid.strike, np.broadcast_to(along, (3, 4, 2)), atol=1e-9
        )


@pytest.mark.parametrize("dip_deg", [90.0, 90.0 - 1e-9, 0.0])
def test_a_cell_that_dips_to_no_side_takes_its_strike_from_its_j_edges(dip_deg):
    chart, along = tilted(30.0, dip_deg)
    np.testing.assert_allclose(
        chart.strike, np.broadcast_to(along, (3, 4, 2)), atol=1e-9
    )
    np.testing.assert_allclose(
        _reversed(chart).strike, np.broadcast_to(-along, (3, 4, 2)), atol=1e-9
    )


def test_a_degenerate_cell_strikes_east():
    nodes = np.broadcast_to(ORIGIN_KM, (2, 2, 3)).copy()
    chart = Geometry(nodes, np.ones((1, 1), bool), (1,))
    np.testing.assert_array_equal(chart.strike, [[[1.0, 0.0]]])
    assert chart.dip_deg[0, 0] == 0.0


def _wobbled(offsets_km):
    """A chart of 1 km cells, each node moved by its offset."""
    rows, columns = offsets_km.shape[0] - 1, offsets_km.shape[1] - 1
    i, j = np.mgrid[0 : rows + 1, 0 : columns + 1].astype(float)
    nodes = ORIGIN_KM + np.stack([i, j, i], axis=-1) + offsets_km
    return Geometry(nodes, np.ones((rows, columns), bool), (columns,))


OFFSETS = hnp.arrays(
    np.float64,
    st.tuples(st.integers(2, 6), st.integers(2, 6), st.just(3)),
    elements=st.floats(-0.3, 0.3),
)


@given(OFFSETS)
def test_a_chart_that_is_fault_everywhere_averages_every_step(offsets_km):
    chart = _wobbled(offsets_km)
    strike = np.linalg.norm(np.diff(chart.nodes, axis=1), axis=-1).mean()
    dip = np.linalg.norm(np.diff(chart.nodes, axis=0), axis=-1).mean()
    assert chart.spacing_km == (float(strike), float(dip))


@given(OFFSETS, st.data())
def test_the_spacing_ignores_nodes_off_the_fault(offsets_km, data):
    chart = _wobbled(offsets_km)
    occupied = data.draw(hnp.arrays(bool, chart.cells).filter(np.any))
    touched = bordering(occupied, (0, 1))
    far = np.where(touched[..., None], chart.nodes, chart.nodes * 7.0 + 1.0e3)
    partial = Geometry(chart.nodes, occupied, chart.plane_cells)
    assert Geometry(far, occupied, chart.plane_cells).spacing_km == partial.spacing_km


def test_the_cell_near_a_position_is_the_nearest_occupied_one():
    chart = holed()
    # From the unoccupied top-left cell's centre, the cell below lies 0.71 km away in
    # plan, the 45 degree dip's horizontal step, and the cell beside it 1 km.
    east_km, north_km, _ = chart.centres[0, 0]
    assert chart.cell_near(east_km, north_km) == (1, 0)


def test_the_cell_near_a_position_is_the_cell_its_centre_distances_name():
    chart = holed()
    cell = chart.cell_near(*chart.centres[2, 1, :2])
    assert chart.cell_at(*chart.centre_arc_km(*cell)) == cell == (2, 1)


def test_frame_north_is_true_north_turned_by_the_meridian_convergence():
    to_lon_lat = pyproj.Transformer.from_crs(NZTM, "EPSG:4326", always_xy=True)
    to_frame = pyproj.Transformer.from_crs("EPSG:4326", NZTM, always_xy=True)
    lon, lat = np.array([[168.0, 173.0, 177.0], [-45.0, -41.0, -38.0]])
    east_m, north_m = to_frame.transform(lon, lat)
    positions_km = np.stack([east_m, north_m], axis=-1).reshape(1, 3, 2) / 1000.0
    azimuth_deg = true_azimuth_deg(to_lon_lat, positions_km, np.array([0.0, 1.0]))
    convergence_deg = pyproj.Proj(NZTM).get_factors(lon, lat).meridian_convergence
    assert azimuth_deg.shape == (1, 3)
    # AZIMUTH_STEP_KM's chord error, at up to 400 km from the central meridian.
    np.testing.assert_allclose(
        (azimuth_deg[0] - convergence_deg + 180.0) % 360.0 - 180.0, 0.0, atol=1e-3
    )


def test_the_dct_embedding_is_the_wrapped_fft():
    covariance = Covariance(
        PreCorrected(VonKarman(), Marginal("truncated_exponential", 0.9)), (5.0, 3.0)
    )
    grid = Grid((35, 137), (0.5, 0.5))
    amplitudes, padded = _embed(grid, covariance)
    lags = [
        np.arange(e // 2 + 1) * r / ln
        for e, r, ln in zip(padded, grid.resolution_km, covariance.lengths_km)
    ]
    quadrant = covariance.correlation(np.hypot(lags[0][:, None], lags[1][None, :]))

    def wrap(e):
        return np.minimum(np.arange(e), e - np.arange(e))

    spectrum = np.fft.fft2(quadrant[np.ix_(wrap(padded[0]), wrap(padded[1]))]).real
    np.testing.assert_allclose(
        amplitudes**2,
        np.maximum(spectrum[: padded[0] // 2 + 1, : padded[1] // 2 + 1], 0.0),
        atol=1e-12 * spectrum.max(),
    )


@pytest.mark.parametrize(
    "correlation",
    [lambda lag: np.exp(-10.0 * lag), VonKarman(0.005), VonKarman(0.001)],
    ids=["exp(-10r)", "VonKarman(0.005)", "VonKarman(0.001)"],
)
def test_a_correlation_below_tolerance_by_half_a_length_still_embeds(correlation):
    # Below the tolerance already at half a correlation length, so the bracket for
    # the decay length has to move down from [1/2, 1].
    decay = _decay_length(correlation, WRAP_TOLERANCE)
    assert 0.0 < decay < 0.5
    np.testing.assert_allclose(correlation(np.array([decay]))[0], WRAP_TOLERANCE)

    grid = Grid((32, 32), (0.5, 0.5))
    field = sampler(grid, [Covariance(correlation, (2.0, 2.0))])
    assert all(p >= e for p, e in zip(field.padded, grid.shape))


def _target(covariance, grid, scale=(1.0, 1.0)):
    lags = [
        np.arange(e) * r / (ln * k)
        for e, r, ln, k in zip(
            grid.shape, grid.resolution_km, covariance.lengths_km, scale
        )
    ]
    return covariance.correlation(np.hypot(lags[0][:, None], lags[1][None, :]))


@pytest.mark.parametrize(
    ("shape", "magnitude"),
    [((80, 150), 6.5), ((80, 150), 7.5), ((125, 300), 7.5)],
    ids=["8x15 km Mw6.5", "8x15 km Mw7.5", "12x30 km Mw7.5"],
)
def test_the_embedding_delivers_the_covariance_within_its_tolerances(shape, magnitude):
    # The covariance the draws have is exactly the inverse transform of the clipped
    # spectrum, read at a grid lag h through the wrap as min(h, P - h), with no draws.
    grid = Grid(shape, (0.1, 0.1))
    covariance = Covariance(VonKarman(), MAI_BEROZA.lengths_km(magnitude))
    amplitudes, padded = _embed(grid, covariance)
    quadrant = sp.fft.idctn(amplitudes**2, type=1)
    wrapped = [
        np.minimum(np.arange(e), p - np.arange(e)) for e, p in zip(shape, padded)
    ]
    delivered = quadrant[np.ix_(*wrapped)]

    error = np.abs(delivered - _target(covariance, grid)).max()
    assert error <= WRAP_TOLERANCE + MAXIMUM_VARIANCE_DEFICIT

    # The lengths the delivered covariance corresponds to: what the tolerances are
    # really answerable to, since the regression that sets them scatters by far more.
    fit = sp.optimize.least_squares(
        lambda scale: (_target(covariance, grid, scale) - delivered)[::4, ::4].ravel(),
        [1.0, 1.0],
    )
    np.testing.assert_allclose(fit.x, 1.0, atol=1e-3)


def test_moment_magnitude_is_hanks_kanamori_equation_7():
    assert np.log10(moment_from_magnitude(6.0) * 1e7) == pytest.approx(
        1.5 * 6.0 + 16.05
    )


@pytest.mark.parametrize(
    "marginal",
    [
        Marginal("truncated_exponential", 0.9),
        Marginal("gamma", 0.75),
        Marginal("doubly_truncated_normal", 0.6, 4.3),
    ],
)
def test_the_tabulated_transform_matches_the_quantile(marginal):
    from rupture_generator.sampling.norta import NORTA_TAIL, _distribution

    latent = np.random.default_rng(0).standard_normal(100_000)
    exact = _distribution(marginal).ppf(
        np.clip(sp.stats.norm.cdf(latent), NORTA_TAIL, 1 - NORTA_TAIL)
    )
    np.testing.assert_allclose(marginal.apply(latent), exact, rtol=2e-6)


@pytest.mark.parametrize("maximum", [1.5, 2.0, 4.3, 10.0])
@pytest.mark.parametrize("fraction", [0.05, 0.5, 0.99])
def test_a_doubly_truncated_normal_hits_its_mean_spread_and_maximum(maximum, fraction):
    from rupture_generator.sampling.norta import (
        _distribution,
        _doubly_truncated_normal_ceiling,
    )

    spread = fraction * _doubly_truncated_normal_ceiling(maximum)
    fitted = _distribution(Marginal("doubly_truncated_normal", spread, maximum))
    assert fitted.mean() == pytest.approx(1.0, abs=1e-9)
    assert fitted.std() == pytest.approx(spread, abs=1e-7)
    assert fitted.support() == pytest.approx((0.0, maximum), abs=1e-9)


def test_the_doubly_truncated_normal_approaches_the_truncated_exponential():
    from rupture_generator.sampling.norta import (
        _distribution,
        _doubly_truncated_normal_ceiling,
    )

    # 0.90 puts the truncated exponential's largest slip at about 4.3 mean slips.
    exponential = _distribution(Marginal("truncated_exponential", 0.9))
    maximum = float(exponential.support()[1])
    # The fit stops widening its parent short of the limit, so it falls just below.
    assert 0.0 < 0.9 - _doubly_truncated_normal_ceiling(maximum) < 5e-4
