import hashlib
import importlib.metadata
import shutil
import subprocess
import tomllib
from pathlib import Path

import numpy as np
import pytest

import rupture_generator
from rupture_generator import RuptureGeneratorError, generate, read_fields
from rupture_generator import config as rupture_config
from rupture_generator import record as run_record
from rupture_generator.cli import main
from rupture_generator.formats.fields import write_fields
from rupture_generator.formats.srf import write_rupture

from .conftest import EXAMPLE, srf_body


@pytest.fixture
def copied_example(tmp_path):
    """The example and its geometry, copied beside where the records go."""
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    shutil.copy(EXAMPLE, inputs)
    shutil.copy(EXAMPLE.with_suffix(".geojson"), inputs)
    (tmp_path / "out").mkdir()
    return inputs / EXAMPLE.name


def test_the_version_is_the_installed_one():
    assert rupture_generator.__version__ == importlib.metadata.version(
        "rupture-generator"
    )


def test_a_record_round_trips(tmp_path, copied_example):
    loaded = rupture_config.load(copied_example)
    location = tmp_path / "out" / "rupture.toml"
    location.write_text(run_record.run_record(copied_example, loaded, location))
    config, provenance = run_record.read_record(location)
    assert config == loaded
    assert provenance.seed == loaded.seed
    assert provenance.rupture_generator == rupture_generator.__version__
    assert set(provenance.dependencies) == set(run_record.DEPENDENCIES)
    read = [copied_example, loaded.geometry.path]
    for stated, path in zip(provenance.inputs, read, strict=True):
        assert stated.path == f"../inputs/{path.name}"
        assert stated.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_a_record_moved_with_its_inputs_still_runs(tmp_path, copied_example):
    location = tmp_path / "out" / "rupture.toml"
    loaded = rupture_config.load(copied_example)
    location.write_text(run_record.run_record(copied_example, loaded, location))
    stated = tomllib.loads(location.read_text())["geometry"]["path"]
    assert stated == "../inputs/two_faults.geojson"
    moved = shutil.copytree(tmp_path, tmp_path.with_name(tmp_path.name + "_moved"))
    config, _ = run_record.read_record(moved / "out" / location.name)
    assert config.geometry.path == moved / "inputs" / "two_faults.geojson"


def test_a_path_sharing_only_the_root_stays_absolute():
    assert run_record._relative(Path("/a/b"), Path("/c")) == Path("/a/b")
    assert run_record._relative(Path("/a/b"), Path("/a/c")) == Path("../b")


def test_outside_a_checkout_there_is_no_commit(monkeypatch):
    def refuse(*arguments):
        raise subprocess.CalledProcessError(128, ["git", *arguments])

    monkeypatch.setattr(run_record, "_git", refuse)
    provenance = run_record.provenance(EXAMPLE, rupture_config.load(EXAMPLE))
    assert provenance.commit is None and provenance.modified is None
    assert "commit" not in run_record._ENCODER.encode(run_record._Table(provenance))


def test_rerunning_an_srf_writes_the_same_srf(tmp_path, coarse_example):
    first, again = tmp_path / "first.srf", tmp_path / "again.srf"
    assert main([str(coarse_example), str(first)]) == 0
    assert main([str(first), str(again)]) == 0
    assert srf_body(again) == srf_body(first)
    config, provenance = run_record.read_record(again)
    assert config == run_record.read_record(first)[0]
    # The rerun read the first SRF, and states it as its rupture file.
    assert provenance.inputs[0].path == "first.srf"
    assert provenance.inputs[0].sha256 == hashlib.sha256(first.read_bytes()).hexdigest()


def test_the_srf_of_a_fields_file_keeps_the_draws_record(tmp_path, coarse_example):
    fields = tmp_path / "fields" / "rupture.h5"
    fields.parent.mkdir()
    assert main([str(coarse_example), str(fields)]) == 0
    made = tmp_path / "srfs" / "rupture.srf"
    made.parent.mkdir()
    assert main([str(fields), str(made)]) == 0
    from_fields, from_srf = run_record.read_record(fields), run_record.read_record(made)
    assert from_srf == from_fields
    # The paths moved with the record, so the SRF's still reach the inputs.
    assert from_srf[1].inputs[0].path == f"../{coarse_example.name}"
    assert from_srf[0].geometry.path.exists()


def test_an_srf_without_comments_has_no_record(tmp_path, ruptures):
    path = tmp_path / "bare.srf"
    write_rupture(path, ruptures)
    with pytest.raises(RuptureGeneratorError, match="no header comments"):
        run_record.read_record(path)
    assert run_record.relocated(path, tmp_path / "other.srf") is None


def test_rerunning_a_fields_record_draws_the_same_fields(tmp_path, coarse_example):
    path = tmp_path / "rupture.h5"
    assert main([str(coarse_example), str(path)]) == 0
    config, provenance = run_record.read_record(path)
    assert provenance.seed == config.seed
    assert config == rupture_config.load(coarse_example)
    scenario = rupture_config.build(config)
    drawn = generate(
        scenario.realisation,
        scenario.medium,
        scenario.sources,
        scenario.settings,
        seed=scenario.seed,
        jump_propagation_model=scenario.jump_model,
    )
    stored = read_fields(path)
    assert list(stored) == list(drawn)
    for name, rupture in drawn.items():
        for field in ("slip_m", "rise_time_s", "rake_deg", "onset_s"):
            np.testing.assert_array_equal(
                getattr(stored[name], field), getattr(rupture, field)
            )


def test_a_file_without_a_record_is_refused(tmp_path, ruptures):
    fields = tmp_path / "bare.h5"
    write_fields(fields, ruptures)
    with pytest.raises(RuptureGeneratorError, match=run_record.ATTRIBUTE):
        run_record.read_record(fields)
    with pytest.raises(RuptureGeneratorError, match="provenance"):
        run_record.read_record(EXAMPLE)
