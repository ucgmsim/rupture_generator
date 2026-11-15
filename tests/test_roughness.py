"""A rough fault moves and turns each SRF point, and leaves every other field be."""

import dataclasses

import numpy as np
import pytest
import srf as srf_format

from rupture_generator import (
    Geometry,
    RuptureGeneratorError,
    SegmentContext,
    SelfAffineRoughness,
    generate,
)
from rupture_generator.formats.srf import _plane_slices, _rough_cells, write_rupture


def _draw(scenario, roughness):
    model = scenario.model
    assert not isinstance(model, dict)
    return generate(
        scenario.realisation,
        scenario.medium,
        scenario.sources,
        dataclasses.replace(model, roughness=roughness),
        seed=scenario.seed,
        jump_propagation_model=scenario.jump_model,
    )


@pytest.fixture(scope="module")
def flat(scenario):
    return _draw(scenario, SelfAffineRoughness(amplitude_ratio=0.0))


def test_roughness_leaves_every_other_field_alone(flat, ruptures):
    for name, rough in ruptures.items():
        assert np.abs(rough.roughness_km).max() > 0.0
        for field in ("slip_m", "rise_time_s", "rake_deg", "onset_s", "speed_km_s"):
            np.testing.assert_array_equal(
                getattr(rough, field), getattr(flat[name], field)
            )


def test_a_flat_fault_writes_the_charts_points(flat, tmp_path):
    path = tmp_path / "flat.srf"
    write_rupture(path, flat)
    parsed = srf_format.read(path)

    def column(pick):
        return np.concatenate(
            [
                pick(rupture.geometry)[:, columns].ravel()
                for rupture in flat.values()
                for columns in _plane_slices(rupture.geometry)
            ]
        )

    np.testing.assert_allclose(
        parsed.points.dip, column(lambda g: g.dip_deg), atol=1e-4
    )
    np.testing.assert_allclose(
        parsed.points.dep, column(lambda g: g.centres[..., 2]), atol=1e-5
    )
    start = 0
    for plane in parsed.planes:
        stop = start + plane.nstk * plane.ndip
        np.testing.assert_allclose(parsed.points.stk[start:stop], plane.stk, atol=1e-4)
        start = stop


def _ramp(flat, strike_deg, dip_deg, *, along_strike, slope):
    """A flat 10 km by 5 km chart in 0.5 km cells, and the plane ``h = slope x``.

    Moving a plane's nodes along its normal by a linear offset gives another plane, so
    every cell's dip and strike are exact.
    """
    phi, delta = np.radians(strike_deg), np.radians(dip_deg)
    along = np.array([np.sin(phi), np.cos(phi), 0.0])
    down = np.array(
        [np.cos(phi) * np.cos(delta), -np.sin(phi) * np.cos(delta), np.sin(delta)]
    )
    i, j = np.meshgrid(np.arange(11), np.arange(21), indexing="ij")
    geometry = Geometry(
        0.5 * (i[..., None] * down + j[..., None] * along),
        np.ones((10, 20), dtype=bool),
        (20,),
    )
    distance_km = 0.5 * (j if along_strike else i)
    offset_km = slope * distance_km[..., None] * geometry.node_normals
    rupture = next(iter(flat.values()))
    return dataclasses.replace(rupture, geometry=geometry, roughness_km=offset_km)


@pytest.mark.parametrize("slope", [-0.3, 0.1, 5.0], ids=["up", "down", "rolled"])
def test_a_dip_ramp_turns_the_dip_about_strike(flat, slope):
    # The down-dip tangent d + h' n has dip D - atan(h') when n points up. The
    # rolled-over sign keeps that continuous past vertical.
    rupture = _ramp(flat, 30.0, 40.0, along_strike=False, slope=slope)
    normal_up = -np.sign(rupture.geometry.normals[0, 0, 2])
    _, turn_deg, dip_deg = _rough_cells(rupture)
    np.testing.assert_allclose(
        dip_deg, 40.0 - normal_up * np.degrees(np.arctan(slope)), atol=1e-9
    )
    np.testing.assert_allclose(turn_deg, 0.0, atol=1e-9)


@pytest.mark.parametrize("dip_deg", [40.0, 90.0])
def test_a_strike_ramp_turns_the_strike(flat, dip_deg):
    # The normal's horizontal part, sin D long, gains h' across it.
    rupture = _ramp(flat, 30.0, dip_deg, along_strike=True, slope=0.1)
    _, turn_deg, _ = _rough_cells(rupture)
    expected = np.degrees(np.arctan(0.1 / np.sin(np.radians(dip_deg))))
    np.testing.assert_allclose(np.abs(turn_deg), expected, atol=1e-9)


@pytest.fixture(scope="module")
def context(scenario):
    name, geometry = next(iter(scenario.realisation.items()))
    return SegmentContext(
        geometry=geometry,
        source=scenario.sources[name],
        shear_speed_km_s=np.full(geometry.cells, 3.5),
        density_g_cm3=np.full(geometry.cells, 2.7),
    )


def _offset_km(context, **settings):
    drawn = SelfAffineRoughness(**settings)(context, np.random.default_rng(7))
    return np.sum(drawn.values * context.geometry.node_normals, axis=-1), drawn


def test_the_offset_scales_with_the_ratio_and_has_no_trend(context):
    single, drawn = _offset_km(context, amplitude_ratio=0.005)
    double, _ = _offset_km(context, amplitude_ratio=0.01)
    np.testing.assert_allclose(double, 2.0 * single)
    i, j = np.indices(single.shape)
    for basis in (np.ones_like(single), i, j):
        assert abs(np.sum(single * basis)) < 1e-9 * single.size
    assert drawn.statistics["rms_km"] == pytest.approx(single.std())


def test_keeping_depth_moves_the_same_distance_off_the_chart(context):
    normal_km, _ = _offset_km(context)
    level_km, drawn = _offset_km(context, keep_depth=True)
    np.testing.assert_allclose(level_km, normal_km)
    np.testing.assert_array_equal(drawn.values[..., 2], 0.0)


@pytest.mark.parametrize(
    "settings",
    [
        {"amplitude_ratio": -0.1},
        {"maximum_wavelength_km": 0.0},
        {"hurst": 0.0},
        {"hurst": 1.5},
    ],
)
def test_a_meaningless_surface_is_refused(settings):
    with pytest.raises(RuptureGeneratorError):
        SelfAffineRoughness(**settings)
