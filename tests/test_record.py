import hashlib
import importlib.metadata
import tomllib

import srf
import xarray as xr

import rupture_generator
from rupture_generator import config as rupture_config
from rupture_generator import record as run_record
from rupture_generator.cli import main
from rupture_generator.formats.fields import (
    RECORD_ATTRIBUTE,
    fields_to_srf,
    write_fields,
)

from .conftest import EXAMPLE


def test_the_version_is_the_installed_one():
    assert rupture_generator.__version__ == importlib.metadata.version(
        "rupture-generator"
    )


def test_a_record_states_the_config_and_what_ran_it():
    loaded = rupture_config.load(EXAMPLE)
    stated = tomllib.loads(run_record.run_record(EXAMPLE, loaded))
    provenance = stated.pop("provenance")
    assert stated == tomllib.loads(rupture_config.dump(loaded))
    assert provenance["seed"] == loaded.seed
    assert provenance["rupture_generator"] == rupture_generator.__version__
    assert set(provenance["dependencies"]) == set(run_record.DEPENDENCIES)
    read = [EXAMPLE, loaded.geometry.path]
    for given, path in zip(provenance["inputs"], read, strict=True):
        assert given["path"] == str(path.resolve())
        assert given["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_srf_of_a_fields_file_keeps_the_draws_record(tmp_path, coarse_example):
    fields, made = tmp_path / "rupture.h5", tmp_path / "rupture.srf"
    assert main([str(coarse_example), str(fields)]) == 0
    assert main([str(fields), str(made)]) == 0
    with xr.open_datatree(fields, engine="h5netcdf") as tree:
        record = tree.attrs[RECORD_ATTRIBUTE]
    assert list(srf.read_comments(made)) == record.splitlines()


def test_a_fields_file_without_a_record_makes_an_srf_without_one(tmp_path, ruptures):
    fields, made = tmp_path / "bare.h5", tmp_path / "bare.srf"
    write_fields(fields, ruptures)
    fields_to_srf(fields, made)
    assert srf.read_comments(made) == ()
