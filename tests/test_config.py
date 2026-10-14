import dataclasses
import tomllib

import numpy as np
import pytest

from rupture_generator import RuptureGeneratorError
from rupture_generator import config as rupture_config

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
            if field.name not in {"factor", "slip_weight"}:
                assert getattr(ours, field.name) == getattr(theirs, field.name)
    assert loaded.slip.side_taper == DrawnSlip().side_taper
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
        (
            "centre_km = 2.0",
            'centre_km = "two"',
            "rise.slip_weight.centre_km:",
        ),
        (
            "dip_fraction = 0.5\n",
            "",
            "hypocentre: a hypocentre is strike_km and dip_km",
        ),
        ("seed = 7", "seed = = 7", "is not TOML"),
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
    assert isinstance(following.slip, rupture_config.Slip)
    assert isinstance(following.speed, rupture_config.Speed)
    assert following.slip.coefficient_of_variation == 0.6
    # The keys the segment's table leaves out are the global table's.
    assert following.slip.bottom_taper == loaded.slip.bottom_taper == 0.02
    assert following.speed.velocity_fraction == 0.7
    assert following.speed.factor == loaded.speed.factor
    assert following.rise == bent.rise
    assert following.perturbation == bent.perturbation

    default = rupture_config.build(rupture_config.load(EXAMPLE))
    assert scenario.sources["bent"] == default.sources["bent"]
    assert (
        scenario.sources["next"].covariance.lengths_km[1]
        > default.sources["next"].covariance.lengths_km[1]
    )


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
    assert rise.coefficient == loaded.rise.coefficient


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
