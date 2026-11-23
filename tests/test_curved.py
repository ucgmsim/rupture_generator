"""Ruptures on the NZ Community Fault Model's subduction interfaces, file to file.

Each interface is the example's config cut at :data:`SPACING_KM`, drawn through the
command line to an SRF and to a fields file, and both files read back.
"""

import dataclasses
import json

import numpy as np
import pyproj
import pytest

from rupture_generator import RuptureGeneratorError, generate
from rupture_generator import config as rupture_config
from rupture_generator.cli import main
from rupture_generator.formats.fields import read_fields
from rupture_generator.formats.srf import read_rupture
from rupture_generator.rupture.medium import rigidity_pa
from rupture_generator.rupture.source import moment_from_magnitude

from .conftest import EXAMPLE

EXAMPLES = EXAMPLE.parent
CFM = EXAMPLES / "cfm"

SPACING_KM = 10.0
"""Coarse enough to be cheap: measured at 0.3 s and 140 MB to chart and draw
Hikurangi, the larger interface."""

INTERFACES = {"hikurangi": "Hikurangi", "puysegur": "Puyseguer"}
"""Each example, and the segment the fault model names its surface."""


def _coarse(tmp_path, example, **changes):
    """The example's config at :data:`SPACING_KM`, with ``changes`` to its tables."""
    loaded = rupture_config.load(EXAMPLES / f"{example}.toml")
    geometry = dataclasses.replace(loaded.geometry, spacing_km=SPACING_KM)
    loaded = dataclasses.replace(loaded, geometry=geometry, **changes)
    path = tmp_path / f"{example}.toml"
    path.write_text(rupture_config.dump(loaded))
    return path


@pytest.fixture(scope="module", params=sorted(INTERFACES))
def drawn(request, tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp(request.param)
    path = _coarse(tmp_path, request.param)
    srf_path, fields_path = tmp_path / "out.srf", tmp_path / "out.h5"
    assert main([str(path), str(srf_path)]) == 0
    assert main([str(path), str(fields_path)]) == 0
    loaded = rupture_config.load(path)
    return loaded, read_fields(fields_path), read_rupture(srf_path)


def test_the_moment_delivered_is_the_target(drawn):
    loaded, fields, srf = drawn
    target_nm = moment_from_magnitude(loaded.source.magnitude)
    (rupture,) = fields.values()
    rigidity = rigidity_pa(rupture.shear_speed_km_s, rupture.density_g_cm3)
    fields_nm = np.sum(rigidity * rupture.geometry.areas_km2 * 1e6 * rupture.slip_m)
    srf_nm = np.sum(srf.rigidity_pa * srf.area_m2 * srf.slip_m)
    assert fields_nm == pytest.approx(target_nm, rel=1e-9)
    # The SRF holds single precision.
    assert srf_nm == pytest.approx(target_nm, rel=1e-5)


def test_slip_is_zero_off_the_fault_and_onsets_are_causal_on_it(drawn):
    _, fields, srf = drawn
    (rupture,) = fields.values()
    occupied = rupture.geometry.occupied
    assert 0.3 < occupied.mean() < 0.9
    assert np.all(rupture.slip_m[~occupied] == 0.0)
    assert np.all(rupture.slip_m[occupied] >= 0.0)
    assert rupture.slip_m[occupied].mean() > 0.0
    assert np.all(srf.slip_m[~occupied.ravel()] == 0.0)

    hypocentre = fields.hypocentre
    assert hypocentre is not None
    cell = rupture.geometry.cell_at(hypocentre.strike_km, hypocentre.dip_km)
    onset_s = rupture.onset_s[occupied]
    assert np.all(np.isfinite(onset_s))
    assert rupture.onset_s[cell] == 0.0
    # No subfault starts with or before the one the rupture starts at.
    assert np.count_nonzero(onset_s == 0.0) == 1 and onset_s.min() == 0.0
    np.testing.assert_allclose(srf.onset_s, rupture.onset_s.ravel(), atol=1e-4)


@pytest.mark.parametrize("example", sorted(INTERFACES))
def test_an_unperturbed_front_never_outruns_its_fastest_speed(example, tmp_path):
    # Without the timing perturbation the onset is the eikonal solve's travel time,
    # which no straight line from the hypocentre at the greatest front speed can beat. A
    # chart that let the front cut across its unoccupied cells, or that shrank the
    # fault, would.
    still = rupture_config.Perturbation(offset_s=0.0, coefficient=0.0)
    loaded = rupture_config.load(_coarse(tmp_path, example, perturbation=still))
    scenario = rupture_config.build(loaded)
    (rupture,) = generate(
        scenario.realisation,
        scenario.medium,
        scenario.sources,
        scenario.model,
        seed=scenario.seed,
        jump_propagation_model=scenario.jump_model,
    ).values()
    geometry, hypocentre = rupture.geometry, scenario.realisation.hypocentre
    assert hypocentre is not None
    occupied = geometry.occupied
    start = geometry.centres[geometry.cell_at(hypocentre.strike_km, hypocentre.dip_km)]
    distance_km = np.linalg.norm(geometry.centres - start, axis=-1)[occupied]
    fastest_s = distance_km / rupture.speed_km_s[occupied].max()
    # The chart's cells are true to the mesh to within its 1-2% stretch.
    assert np.all(rupture.onset_s[occupied] >= 0.97 * fastest_s)


@pytest.mark.parametrize("example", sorted(INTERFACES))
def test_a_longitude_and_latitude_lands_on_the_cell_beneath_it(example):
    loaded = rupture_config.load(EXAMPLES / f"{example}.toml")
    loaded = dataclasses.replace(
        loaded,
        geometry=dataclasses.replace(loaded.geometry, spacing_km=SPACING_KM),
    )
    scenario = rupture_config.build(loaded)
    hypocentre = scenario.realisation.hypocentre
    assert hypocentre is not None
    assert hypocentre.segment == INTERFACES[example]
    chart = scenario.realisation[hypocentre.segment]
    cell = chart.cell_at(hypocentre.strike_km, hypocentre.dip_km)
    to_frame = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:2193", always_xy=True)
    east_m, north_m = to_frame.transform(
        loaded.hypocentre.longitude_deg, loaded.hypocentre.latitude_deg
    )
    centre = chart.centres[cell]
    offset_km = np.hypot(centre[0] - east_m / 1000.0, centre[1] - north_m / 1000.0)
    # The point lies on the fault, so within half a cell's diagonal of its cell.
    assert offset_km <= 0.5 * np.hypot(*chart.spacing_km)
    # The examples' headers state the depth at their own 5 km spacing.
    assert 12.0 < centre[2] < 20.0
    # It's the nearest occupied cell in plan.
    plan_km = np.hypot(
        chart.centres[..., 0] - east_m / 1000.0,
        chart.centres[..., 1] - north_m / 1000.0,
    )
    assert offset_km == plan_km[chart.occupied].min()


def test_the_files_read_back_with_the_hypocentre_where_it_was_asked(drawn):
    loaded, fields, srf = drawn
    (rupture,) = fields.values()
    n_i, n_j = rupture.geometry.cells
    (plane,) = srf.planes
    assert (plane.ndip, plane.nstk) == (n_i, n_j)
    assert srf.slip_m.shape == (n_i * n_j,)
    assert srf.hypocentre is not None
    lon, lat, depth_km = srf.hypocentre
    to_frame = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:2193", always_xy=True)
    asked = np.array(
        to_frame.transform(
            loaded.hypocentre.longitude_deg, loaded.hypocentre.latitude_deg
        )
    )
    read = np.array(to_frame.transform(lon, lat))
    assert np.hypot(*(read - asked)) / 1000.0 <= 0.5 * np.hypot(
        *rupture.geometry.spacing_km
    )
    assert 12.0 < depth_km < 20.0
    # The header summarises the fault, not the rectangle around it.
    assert 0.0 <= plane.dtop < 10.0
    assert 5.0 < plane.dip < 30.0


SECTION = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [[176.6, -39.9], [176.9, -39.6]],
            },
            "properties": {
                "name": "upper plate",
                "dip_deg": 60,
                "dip_direction_deg": 300,
                "upper_depth_km": 0,
                "lower_depth_km": 20,
            },
        }
    ],
}
"""A hypothetical planar crustal fault above the Hikurangi interface, in Hawke's Bay."""


def test_a_curved_surface_and_a_planar_section_rupture_together(tmp_path):
    sections = tmp_path / "section.geojson"
    sections.write_text(json.dumps(SECTION))
    loaded = rupture_config.load(EXAMPLES / "hikurangi.toml")
    geometry = dataclasses.replace(
        loaded.geometry,
        path=(CFM / "Hikurangi.ts.gz", sections),
        spacing_km=SPACING_KM,
    )
    loaded = dataclasses.replace(
        loaded,
        geometry=geometry,
        propagation=rupture_config.Predetermined(parents={"upper plate": "Hikurangi"}),
    )
    path = tmp_path / "joint.toml"
    path.write_text(rupture_config.dump(loaded))
    fields_path = tmp_path / "joint.h5"
    assert main([str(path), str(fields_path)]) == 0
    assert main([str(path), str(tmp_path / "joint.srf")]) == 0

    fields = read_fields(fields_path)
    assert list(fields) == ["Hikurangi", "upper plate"]
    section = fields["upper plate"]
    assert section.geometry.occupied.all()
    assert section.onset_s.min() > 0.0
    total_nm = sum(
        np.sum(
            rigidity_pa(r.shear_speed_km_s, r.density_g_cm3)
            * r.geometry.areas_km2
            * 1e6
            * r.slip_m
        )
        for r in fields.values()
    )
    assert total_nm == pytest.approx(
        moment_from_magnitude(loaded.source.magnitude), rel=1e-9
    )
    srf = read_rupture(tmp_path / "joint.srf")
    assert len(srf.planes) == 1 + section.geometry.planes


def test_one_name_in_two_files_is_refused(tmp_path):
    loaded = rupture_config.load(EXAMPLES / "hikurangi.toml")
    surface = CFM / "Hikurangi.ts.gz"
    geometry = dataclasses.replace(
        loaded.geometry, path=(surface, surface), spacing_km=SPACING_KM
    )
    with pytest.raises(
        RuptureGeneratorError,
        match=r"^geometry\.path: .*Hikurangi\.ts\.gz names \['Hikurangi'\], which an "
        "earlier file names too",
    ):
        rupture_config.build(dataclasses.replace(loaded, geometry=geometry))


def test_a_surface_outside_the_frame_is_refused():
    loaded = rupture_config.load(EXAMPLES / "hikurangi.toml")
    # Read in a frame for Utah, NZTM's coordinates land far from its area of use.
    geometry = dataclasses.replace(
        loaded.geometry, crs="EPSG:32612", spacing_km=SPACING_KM
    )
    with pytest.raises(
        RuptureGeneratorError, match=r"EPSG:32612 .* the faults 'Hikurangi' lie outside"
    ):
        rupture_config.build(dataclasses.replace(loaded, geometry=geometry))
