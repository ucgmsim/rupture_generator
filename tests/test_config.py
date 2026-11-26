import dataclasses
import tomllib

import numpy as np
import pytest

from rupture_generator import RuptureGeneratorError
from rupture_generator import config as rupture_config
from rupture_generator.rupture.source import MAI_BEROZA

from .conftest import EXAMPLE

BASE = EXAMPLE.read_text()


def test_dump_round_trips(tmp_path):
    loaded = rupture_config.load(EXAMPLE)
    again = tmp_path / "again.toml"
    again.write_text(rupture_config.dump(loaded))
    assert rupture_config.load(again) == loaded


def test_defaults_are_the_librarys():
    from rupture_generator import (
        CorrelatedPerturbation,
        CorrelatedRise,
        DrawnSlip,
        GaussianRake,
        ShearFractionSpeed,
    )

    loaded = rupture_config.load(EXAMPLE)
    # The config's classes are subclasses, so compare values rather than objects. The
    # example gives its own depth profiles, and the file's profiles are tables where
    # the library's are functions.
    for ours, theirs in (
        (loaded.rise, CorrelatedRise()),
        (loaded.rake, GaussianRake()),
        (loaded.speed, ShearFractionSpeed()),
        (loaded.perturbation, CorrelatedPerturbation()),
    ):
        for field in dataclasses.fields(theirs):
            if field.name not in {"factor", "slip_weight", "noise"}:
                assert getattr(ours, field.name) == getattr(theirs, field.name)
    assert loaded.slip.side_taper == DrawnSlip().side_taper
    for ours, theirs in zip(
        (loaded.slip.noise, loaded.rake.noise),
        (DrawnSlip().noise, GaussianRake().noise),
    ):
        assert [(dataclasses.astuple(f.relation), f.hurst) for f in ours] == [
            (dataclasses.astuple(f.relation), f.hurst) for f in theirs
        ]
    positions = np.zeros((2, 3))
    for profile in (rupture_config.Rise().factor, rupture_config.Speed().factor):
        np.testing.assert_array_equal(
            profile(positions), CorrelatedRise().factor(positions)
        )


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("bottom_taper", "botom_taper", "slip: has unknown keys ['botom_taper']"),
        (
            "bottom_taper = 0.02",
            "bottom_taper = 0.7",
            "slip: bottom_taper is a fraction",
        ),
        (
            'type = "sampled"',
            'type = "sample"',
            "propagation: has no variant with type = 'sample'",
        ),
        ("seed = 7\n", "", "seed: is missing"),
        ("seed = 7\n", "seed = -7\n", "seed: a seed is a non-negative integer"),
        ("seed = 7\n", "seed = 7.5\n", "seed: a seed is a non-negative integer"),
        # An old name reads as its new one, which the error then names.
        (
            "centre_km = 2.0",
            'centre_km = "two"',
            "rise.slip_weight.centre:",
        ),
        (
            "dip_fraction = 0.5\n",
            "",
            "hypocentre: a hypocentre is strike_km and dip_km",
        ),
        ("seed = 7", "seed = = 7", "is not TOML"),
        # The moment's coefficient, from before the mean slip set the rise time.
        (
            "[rise.slip_weight]\n",
            "[rise]\ncoefficient = 1.6\n\n[rise.slip_weight]\n",
            "rise: has unknown keys ['coefficient']",
        ),
        (
            "[rise.slip_weight]\n",
            "[rise]\nslip_coefficient = 0.0\n\n[rise.slip_weight]\n",
            "rise: a rise time is a duration",
        ),
    ],
)
def test_a_bad_file_names_the_key_at_fault(tmp_path, old, new, message):
    assert old in BASE
    path = tmp_path / "bad.toml"
    path.write_text(BASE.replace(old, new))
    with pytest.raises(
        RuptureGeneratorError, match=message.replace("[", r"\[").replace("(", r"\(")
    ):
        rupture_config.load(path)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("next = 6.4", "nxt = 6.4", "source: every segment needs a magnitude"),
        (
            'segment = "bent"',
            'segment = "bnt"',
            "hypocentre.segment: 'bnt' is not one of",
        ),
        (
            'crs = "EPSG:2193"',
            'crs = ""',
            "geometry.crs: a projected CRS in metres is required",
        ),
        (
            'crs = "EPSG:2193"',
            'crs = "EPSG:2227"',
            "geometry.crs: EPSG:2227 .* US survey foot",
        ),
        (
            'crs = "EPSG:2193"',
            'crs = "EPSG:32612"',
            "EPSG:32612 .* the faults 'bent', 'next' lie outside it",
        ),
    ],
)
def test_parts_that_do_not_fit_are_refused_when_built(old, new, message):
    path = EXAMPLE.with_name("_test_build.toml")
    path.write_text(BASE.replace(old, new))
    try:
        with pytest.raises(RuptureGeneratorError, match=message):
            rupture_config.build(rupture_config.load(path))
    finally:
        path.unlink()


def test_the_cli_writes_an_srf_and_its_config(tmp_path):
    import srf

    from rupture_generator.cli import main

    output = tmp_path / "out.srf"
    assert main([str(EXAMPLE), str(output)]) == 0
    assert output.read_text().startswith("2.0\n# seed = 7\n")
    record = tomllib.loads("\n".join(srf.read_comments(output)))
    del record["provenance"]
    assert record == tomllib.loads(rupture_config.dump(rupture_config.load(EXAMPLE)))
    assert list(tmp_path.iterdir()) == [output]


def test_the_cli_reports_a_bad_file(tmp_path, capsys):
    from rupture_generator.cli import main

    bad = tmp_path / "bad.toml"
    bad.write_text(BASE.replace("seed = 7\n", ""))
    assert main([str(bad), str(tmp_path / "out.srf")]) == 2
    assert "seed: is missing" in capsys.readouterr().err


@pytest.mark.parametrize(
    "example",
    sorted(path.name for path in EXAMPLE.parent.glob("*.toml")),
)
def test_every_example_builds(example):
    scenario = rupture_config.build(rupture_config.load(EXAMPLE.parent / example))
    assert set(scenario.sources) == set(scenario.realisation)


def test_slip_can_follow_a_doubly_truncated_normal(tmp_path):
    path = tmp_path / "doubly.toml"
    path.write_text(
        BASE.replace(
            "[slip]\n",
            '[slip]\nfamily = "doubly_truncated_normal"\n'
            "coefficient_of_variation = 0.6\nmaximum = 4.0\n",
        )
    )
    loaded = rupture_config.load(path)
    assert loaded.slip.marginal.family == "doubly_truncated_normal"
    assert loaded.slip.marginal.maximum == 4.0
    again = tmp_path / "again.toml"
    again.write_text(rupture_config.dump(loaded))
    assert rupture_config.load(again) == loaded


SEGMENTS = EXAMPLE.with_name("two_faults_segments.toml")


def test_a_segment_table_overrides_the_global_one():
    loaded = rupture_config.load(SEGMENTS)
    scenario = rupture_config.build(loaded)
    assert isinstance(scenario.model, dict)
    bent, following = scenario.model["bent"], scenario.model["next"]
    assert bent.slip == loaded.slip and bent.speed == loaded.speed
    assert isinstance(bent.slip, rupture_config.Slip)
    assert isinstance(following.slip, rupture_config.Slip)
    assert isinstance(following.speed, rupture_config.Speed)
    assert following.slip.coefficient_of_variation == 0.6
    # The keys the segment's table leaves out are the global table's.
    assert following.slip.bottom_taper == loaded.slip.bottom_taper == 0.02
    assert following.speed.velocity_fraction == 0.7
    assert following.speed.factor == loaded.speed.factor
    assert following.rise == bent.rise
    assert following.perturbation == bent.perturbation

    (longer,) = following.slip.noise
    assert longer.relation.lengths_km(7.0)[1] > MAI_BEROZA.lengths_km(7.0)[1]


def test_a_profile_in_a_segment_table_replaces_the_global_one(tmp_path):
    path = tmp_path / "profiles.toml"
    path.write_text(
        BASE.replace("two_faults.geojson", str(EXAMPLE.with_name("two_faults.geojson")))
        + "\n[segments.next.rise.slip_weight]\n"
        'type = "constant"\nvalue = 0.5\n'
    )
    loaded = rupture_config.load(path)
    rise = loaded.segments["next"].rise
    assert rise is not None
    assert rise.slip_weight == rupture_config.Constant(value=0.5)
    assert rise.factor == loaded.rise.factor
    assert rise.slip_coefficient == loaded.rise.slip_coefficient


def test_segment_tables_survive_the_record(tmp_path):
    import srf

    from rupture_generator.cli import main

    loaded = rupture_config.load(SEGMENTS)
    again = tmp_path / "again.toml"
    again.write_text(rupture_config.dump(loaded))
    assert rupture_config.load(again) == loaded
    assert "[segments.next.slip]" in again.read_text()

    small = tmp_path / "small.toml"
    small.write_text(
        SEGMENTS.read_text()
        .replace("two_faults.geojson", str(EXAMPLE.with_name("two_faults.geojson")))
        .replace("spacing_km = 0.1", "spacing_km = 0.5")
    )
    output = tmp_path / "out.srf"
    assert main([str(small), str(output)]) == 0
    record = tomllib.loads("\n".join(srf.read_comments(output)))
    del record["provenance"]
    assert record == tomllib.loads(rupture_config.dump(rupture_config.load(small)))
    assert "next" in record["segments"]


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (
            "velocity_fraction = 0.7",
            "velocity_fractio = 0.7",
            "segments.next.speed: has unknown keys ['velocity_fractio']",
        ),
        (
            "[segments.next.speed]",
            "[segments.next.sped]",
            "segments.next: has unknown keys ['sped']",
        ),
        (
            "velocity_fraction = 0.7",
            "velocity_fraction = 7.0",
            "segments.next.speed: a velocity fraction lies in",
        ),
    ],
)
def test_a_bad_segment_table_names_the_segment(tmp_path, old, new, message):
    base = SEGMENTS.read_text()
    assert old in base
    path = tmp_path / "bad.toml"
    path.write_text(base.replace(old, new))
    with pytest.raises(
        RuptureGeneratorError, match=message.replace("[", r"\[").replace("(", r"\(")
    ):
        rupture_config.load(path)


def test_a_segment_table_for_no_segment_is_refused(tmp_path):
    path = tmp_path / "unknown.toml"
    path.write_text(
        SEGMENTS.read_text()
        .replace("two_faults.geojson", str(EXAMPLE.with_name("two_faults.geojson")))
        .replace("[segments.next.speed]", "[segments.nxt.speed]")
    )
    with pytest.raises(RuptureGeneratorError, match=r"segments: \['nxt'\] are not"):
        rupture_config.build(rupture_config.load(path))


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        # A key at the top level has no table to name. The error cites the file instead.
        ("seed = 7\n", "seed = 7\nbogus = 1\n", "the file: has unknown keys ['bogus']"),
        # The error shows the type name of a wrong-kind value, not its full repr.
        (
            "magnitudes = { bent = 6.6, next = 6.4 }",
            "magnitudes = 3",
            "source.magnitudes: 3 is not a dict",
        ),
        ("dt_s = 0.005", "dt_s = 0.0", "pulse: a sample interval of 0.0 s is none"),
        ("dt_s = 0.005", "dt_s = -1.0", "pulse: a sample interval of -1.0 s is none"),
        ("seed = 7\n", "seed = 7\nsegments = 3\n", "segments: 3 is not a table"),
        (
            'type = "sampled"\n',
            'type = "sampled"\n\n[propagation.model]\nreach = "mean"\n',
            "propagation.model.reach: 'mean' is not 'sampled' or 'reference'",
        ),
        (
            "seed = 7\n",
            "seed = 7\nsegments = { next = 3 }\n",
            "segments.next: 3 is not a table",
        ),
    ],
)
def test_a_bad_file_gives_exactly_this_message(tmp_path, old, new, message):
    assert old in BASE
    path = tmp_path / "bad.toml"
    path.write_text(BASE.replace(old, new))
    with pytest.raises(RuptureGeneratorError) as raised:
        rupture_config.load(path)
    assert str(raised.value) == message


def test_a_positive_sample_interval_is_kept(tmp_path):
    path = tmp_path / "fine.toml"
    path.write_text(BASE.replace("dt_s = 0.005", "dt_s = 1e-6"))
    assert rupture_config.load(path).pulse.dt_s == 1e-6


@pytest.mark.parametrize(
    ("propagation", "message"),
    [
        (None, "propagation: several segments need a [propagation] table"),
        (
            rupture_config.Predetermined(parents={"next": "bent", "nxt": "bent"}),
            "propagation.parents: ['nxt'] are not segments",
        ),
    ],
)
def test_a_propagation_that_does_not_fit_is_refused(propagation, message):
    loaded = dataclasses.replace(rupture_config.load(EXAMPLE), propagation=propagation)
    with pytest.raises(RuptureGeneratorError) as raised:
        rupture_config.build(loaded)
    assert str(raised.value).startswith(message)


def test_a_predetermined_tree_is_the_one_given():
    loaded = dataclasses.replace(
        rupture_config.load(EXAMPLE),
        propagation=rupture_config.Predetermined(parents={"next": "bent"}),
    )
    scenario = rupture_config.build(loaded)
    parents = {
        name: parent for name, parent, _ in scenario.realisation.in_causal_order()
    }
    assert parents == {"bent": None, "next": "bent"}


def test_a_reference_reach_reaches_the_jump_model(tmp_path):
    path = tmp_path / "reference.toml"
    path.write_text(
        BASE.replace(
            'type = "sampled"\n',
            'type = "sampled"\n\n[propagation.model]\nreach = "reference"\n',
        ).replace("two_faults.geojson", str(EXAMPLE.with_name("two_faults.geojson")))
    )
    loaded = rupture_config.load(path)
    assert loaded.propagation is not None
    assert loaded.propagation.model.reach == "reference"
    again = tmp_path / "again.toml"
    again.write_text(rupture_config.dump(loaded))
    assert rupture_config.load(again) == loaded
    assert rupture_config.build(loaded).jump_model.reach == "reference"


def test_the_reach_is_sampled_by_default():
    propagation = rupture_config.load(EXAMPLE).propagation
    assert propagation is not None
    assert propagation.model.reach == "sampled"


# ------------------------------------------------- geometry files and hypocentres

ONE_PAIR = (
    "hypocentre: a hypocentre is strike_km and dip_km, strike_fraction and "
    "dip_fraction, or longitude_deg and latitude_deg: one pair, complete"
)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (
            'path = "two_faults.geojson"',
            "path = 3",
            "geometry.path: 3 is not a path or a list of paths",
        ),
        (
            'path = "two_faults.geojson"',
            "path = []",
            "geometry.path: [] is not a path or a list of paths",
        ),
        (
            'path = "two_faults.geojson"',
            'path = ["two_faults.geojson", 3]',
            "geometry.path: ['two_faults.geojson', 3] is not a path or a list of paths",
        ),
        (
            "dip_fraction = 0.5\n",
            "dip_fraction = 0.5\nlongitude_deg = 172.3\nlatitude_deg = -41.8\n",
            ONE_PAIR,
        ),
        (
            "strike_fraction = 0.3\ndip_fraction = 0.5\n",
            "longitude_deg = 172.3\n",
            ONE_PAIR,
        ),
    ],
)
def test_a_bad_geometry_or_hypocentre_gives_exactly_this_message(
    tmp_path, old, new, message
):
    assert old in BASE
    path = tmp_path / "bad.toml"
    path.write_text(BASE.replace(old, new))
    with pytest.raises(RuptureGeneratorError) as raised:
        rupture_config.load(path)
    assert str(raised.value) == message


def test_a_path_is_one_file_or_a_list_resolved_against_the_config(tmp_path):
    one = rupture_config.load(EXAMPLE)
    path = EXAMPLE.with_name("_test_list.toml")
    path.write_text(BASE.replace('"two_faults.geojson"', '["two_faults.geojson"]'))
    try:
        listed = rupture_config.load(path)
    finally:
        path.unlink()
    assert listed.geometry.path == one.geometry.path
    assert one.geometry.path == (EXAMPLE.with_name("two_faults.geojson").resolve(),)
    again = tmp_path / "again.toml"
    again.write_text(rupture_config.dump(listed))
    assert rupture_config.load(again) == listed


def _with_hypocentre(**position) -> rupture_config.RuptureConfig:
    loaded = rupture_config.load(EXAMPLE)
    hypocentre = rupture_config.HypocentreConfig(segment="bent", **position)
    return dataclasses.replace(loaded, hypocentre=hypocentre)


def test_a_longitude_and_latitude_on_a_planar_section_names_its_cell():
    # A point of the trace, at the top of the 60 degree section.
    scenario = rupture_config.build(
        _with_hypocentre(longitude_deg=172.3, latitude_deg=-41.8)
    )
    hypocentre = scenario.realisation.hypocentre
    assert hypocentre is not None
    chart = scenario.realisation["bent"]
    i, j = chart.cell_at(hypocentre.strike_km, hypocentre.dip_km)
    assert i == 0
    # The recorded distances are the cell centre's.
    assert hypocentre.strike_km == pytest.approx(
        0.5 * (chart.strike_arc_km[j] + chart.strike_arc_km[j + 1])
    )
    assert hypocentre.dip_km == pytest.approx(0.5 * chart.dip_arc_km[1])


@pytest.mark.parametrize(
    ("position", "message"),
    [
        (
            {"longitude_deg": 175.0, "latitude_deg": -40.0},
            r"^hypocentre: \(.*\) km lies .* km in plan from the nearest fault cell",
        ),
        (
            {"strike_km": 1000.0, "dip_km": 1.0},
            r"^hypocentre: 1000\.0 km is off the fault along strike",
        ),
        (
            {"strike_fraction": 1.5, "dip_fraction": 0.5},
            r"^hypocentre: a hypocentre at fractions \(1\.5, 0\.5\) is off the fault",
        ),
    ],
)
def test_a_hypocentre_off_the_fault_is_refused_when_built(position, message):
    with pytest.raises(RuptureGeneratorError, match=message):
        rupture_config.build(_with_hypocentre(**position))


def test_a_hypocentre_in_a_cell_off_a_curved_fault_is_refused_when_built():
    loaded = rupture_config.load(EXAMPLE.with_name("hikurangi.toml"))
    loaded = dataclasses.replace(
        loaded,
        geometry=dataclasses.replace(loaded.geometry, spacing_km=10.0),
        hypocentre=rupture_config.HypocentreConfig(
            segment="Hikurangi", strike_fraction=0.0, dip_fraction=0.0
        ),
    )
    with pytest.raises(
        RuptureGeneratorError,
        match=r"^hypocentre: 0\.0 km along strike and 0\.0 km down dip is cell "
        r"\(0, 0\), which is off the fault",
    ):
        rupture_config.build(loaded)


def _tsurf(name: str, corners_m: list[tuple[float, float, float]]) -> str:
    """A TSurf of two triangles on four corners, in NZTM metres, elevation up."""
    vertices = "".join(
        f"VRTX {k} {x} {y} {z}\n" for k, (x, y, z) in enumerate(corners_m, start=1)
    )
    return (
        f"GOCAD TSurf 1\nHEADER {{\nname:{name}\n}}\n"
        'AXIS_UNIT "m" "m" "m"\nZPOSITIVE Elevation\nTFACE\n'
        f"{vertices}TRGL 1 2 3\nTRGL 2 4 3\nEND\n"
    )


EAST_M, NORTH_M = 1_570_000.0, 5_180_000.0
"""A point in NZTM near Christchurch, where the made-up surfaces lie."""


def _surface_config(path) -> rupture_config.RuptureConfig:
    loaded = rupture_config.load(EXAMPLE)
    return dataclasses.replace(
        loaded,
        geometry=dataclasses.replace(loaded.geometry, path=(path,), spacing_km=2.0),
        hypocentre=rupture_config.HypocentreConfig(
            segment="plane", strike_fraction=0.5, dip_fraction=0.5
        ),
        source=rupture_config.Finite(magnitude=6.0, rake_deg=90.0),
        propagation=None,
    )


def test_an_uncompressed_tsurf_builds_a_surface_chart(tmp_path):
    # 20 km along strike and 10 km down a 45 degree dip: 10 by 5 cells of 2 km.
    path = tmp_path / "plane.ts"
    step_m = 10_000.0 / np.sqrt(2.0)
    path.write_text(
        _tsurf(
            "plane",
            [
                (EAST_M, NORTH_M, 0.0),
                (EAST_M + 20_000.0, NORTH_M, 0.0),
                (EAST_M, NORTH_M + step_m, -step_m),
                (EAST_M + 20_000.0, NORTH_M + step_m, -step_m),
            ],
        )
    )
    scenario = rupture_config.build(_surface_config(path))
    chart = scenario.realisation["plane"]
    assert chart.plane_cells == (chart.cells[1],)
    assert chart.occupied.all() and chart.cells == (5, 10)
    assert chart.areas_km2.sum() == pytest.approx(200.0)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("GOCAD Voxet 1\n", r"bad\.ts: line 1: 'GOCAD Voxet 1' is not a TSurf"),
        (
            _tsurf(
                "plane",
                [
                    (EAST_M, NORTH_M, -5_000.0),
                    (EAST_M + 20_000.0, NORTH_M, -5_000.0),
                    (EAST_M, NORTH_M + 10_000.0, -5_000.0),
                    (EAST_M + 20_000.0, NORTH_M + 10_000.0, -5_000.0),
                ],
            ),
            r"bad\.ts: plane: the mesh is level",
        ),
    ],
)
def test_a_surface_file_that_does_not_chart_names_the_file(tmp_path, text, message):
    path = tmp_path / "bad.ts"
    path.write_text(text)
    with pytest.raises(RuptureGeneratorError, match=message):
        rupture_config.build(_surface_config(path))
