import dataclasses

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
        (
            "centre_km = 2.0",
            'centre_km = "two"',
            "materials.rise_time_slip_weight.centre_km:",
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
    from rupture_generator.cli import main

    output = tmp_path / "out.srf"
    assert main([str(EXAMPLE), str(output)]) == 0
    assert output.read_text().startswith("2.0\n")
    assert rupture_config.load(output.with_name("out.srf.toml")) == rupture_config.load(
        EXAMPLE
    )


def test_the_cli_reports_a_bad_file(tmp_path, capsys):
    from rupture_generator.cli import main

    bad = tmp_path / "bad.toml"
    bad.write_text(BASE.replace("seed = 7\n", ""))
    assert main([str(bad), str(tmp_path / "out.srf")]) == 2
    assert "seed: is missing" in capsys.readouterr().err
