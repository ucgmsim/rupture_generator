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
    from rupture_generator import RuptureSettings

    loaded = rupture_config.load(EXAMPLE)
    library = RuptureSettings()
    # The config's classes are subclasses, so compare values rather than objects.
    assert dataclasses.astuple(loaded.rise) == dataclasses.astuple(library.rise)
    assert dataclasses.astuple(loaded.timing) == dataclasses.astuple(library.timing)
    assert loaded.slip.side_taper == library.slip.side_taper


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
            "profiles.rise_time_slip_weight.centre_km:",
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
    assert isinstance(scenario.settings, dict)
    bent, following = scenario.settings["bent"], scenario.settings["next"]
    assert bent.slip == loaded.slip and bent.timing == loaded.timing
    assert following.slip.coefficient_of_variation == 0.6
    # The keys the segment's table leaves out are the global table's.
    assert following.slip.bottom_taper == loaded.slip.bottom_taper == 0.02
    assert following.timing.velocity_fraction == 0.7
    assert following.timing.correlation == loaded.timing.correlation
    assert following.rise == bent.rise

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
        + "\n[segments.next.profiles.rise_time_slip_weight]\n"
        'type = "constant"\nvalue = 0.5\n'
    )
    loaded = rupture_config.load(path)
    profiles = loaded.segments["next"].profiles
    assert profiles is not None
    assert profiles.rise_time_slip_weight == rupture_config.Constant(value=0.5)
    assert profiles.rupture_speed_factor == loaded.profiles.rupture_speed_factor


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
            "segments.next.timing: has unknown keys ['velocity_fractio']",
        ),
        (
            "[segments.next.timing]",
            "[segments.next.timin]",
            "segments.next: has unknown keys ['timin']",
        ),
        (
            "velocity_fraction = 0.7",
            "velocity_fraction = 7.0",
            "segments.next.timing: a velocity fraction lies in",
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
        .replace("[segments.next.timing]", "[segments.nxt.timing]")
    )
    with pytest.raises(RuptureGeneratorError, match=r"segments: \['nxt'\] are not"):
        rupture_config.build(rupture_config.load(path))


def _coarse(tmp_path, extra=""):
    path = tmp_path / "coarse.toml"
    path.write_text(
        BASE.replace("two_faults.geojson", str(EXAMPLE.with_name("two_faults.geojson")))
        .replace("spacing_km = 0.1", "spacing_km = 0.5")
        .replace("[slip]\n", f"{extra}\n[slip]\n")
    )
    return path


def test_a_slip_file_keeps_its_slip_on_its_segment(tmp_path, ruptures):
    from rupture_generator.cli import main

    slip_m = ruptures["next"].slip_m[:, ::-1]
    np.save(tmp_path / "next.npy", slip_m)
    path = _coarse(tmp_path, '[segments.next]\nslip_path = "next.npy"\n')
    loaded = rupture_config.load(path)
    assert loaded.segments["next"].slip_path == tmp_path / "next.npy"
    slips = rupture_config.build(loaded).slips
    assert set(slips) == {"next"}
    assert np.array_equal(slips["next"], slip_m)

    output = tmp_path / "out.srf"
    assert main([str(path), str(output)]) == 0
    assert rupture_config.load(output.with_name("out.srf.toml")) == loaded


def test_a_slip_file_is_read_as_data_alone(tmp_path):
    np.save(tmp_path / "next.npy", np.array([{"a": 1}], dtype=object))
    path = _coarse(tmp_path, '[segments.next]\nslip_path = "next.npy"\n')
    with pytest.raises(RuptureGeneratorError, match="not a .npy array"):
        rupture_config.build(rupture_config.load(path))
