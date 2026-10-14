import dataclasses
import subprocess
import sys

import numpy as np
import pytest
import xarray as xr

from rupture_generator import Realisation, RuptureGeneratorError
from rupture_generator import config as rupture_config
from rupture_generator.cli import main
from rupture_generator.formats.fields import fields_to_srf, read_fields, write_fields

from .conftest import EXAMPLE, SPACING_KM


@pytest.fixture(scope="module")
def fields_path(tmp_path_factory, ruptures):
    path = tmp_path_factory.mktemp("fields") / "rupture.h5"
    write_fields(path, ruptures)
    return path


@pytest.fixture
def coarse_example(tmp_path):
    loaded = rupture_config.load(EXAMPLE)
    geometry = dataclasses.replace(loaded.geometry, spacing_km=SPACING_KM)
    path = tmp_path / "coarse.toml"
    path.write_text(rupture_config.dump(dataclasses.replace(loaded, geometry=geometry)))
    return path


def test_the_srf_from_stored_fields_is_the_srf_from_the_draw(
    tmp_path, fields_path, srf_path
):
    again = tmp_path / "again.srf"
    fields_to_srf(fields_path, again)
    assert again.read_bytes() == srf_path.read_bytes()


def test_the_reader_restores_the_realisation(fields_path, ruptures):
    restored = read_fields(fields_path)
    assert list(restored) == list(ruptures)
    assert dict(restored.tree) == dict(ruptures.tree)
    assert restored.hypocentre == ruptures.hypocentre
    assert restored.crs == ruptures.crs
    for name, rupture in ruptures.items():
        back = restored[name]
        np.testing.assert_array_equal(back.geometry.nodes, rupture.geometry.nodes)
        assert back.geometry.plane_cells == rupture.geometry.plane_cells
        np.testing.assert_array_equal(back.slip_m, rupture.slip_m)
        np.testing.assert_array_equal(back.speed_km_s, rupture.speed_km_s)
        assert back.beta is not None
        np.testing.assert_array_equal(back.beta, rupture.beta)
        assert back.alpha_t == rupture.alpha_t
        assert back.dt_s == rupture.dt_s


def test_each_segment_is_a_group_with_units(fields_path, ruptures):
    with xr.open_datatree(fields_path, engine="h5netcdf") as tree:
        assert list(tree.dataset["segment"].values) == list(ruptures)
        for name, rupture in ruptures.items():
            group = tree[name]
            assert group["slip_m"].dims == ("dip", "strike")
            assert group["slip_m"].shape == rupture.geometry.cells
            assert group["slip_m"].attrs["units"] == "m"
            np.testing.assert_array_equal(
                group["centres_km"].values, rupture.geometry.centres
            )


def test_a_lone_segment_without_a_tree_or_pulse_shape_round_trips(tmp_path, ruptures):
    name = ruptures.hypocentre.segment
    impulses = dataclasses.replace(ruptures[name], beta=None)
    alone = Realisation({name: impulses}, ruptures.crs, ruptures.hypocentre)
    path = tmp_path / "alone.nc"
    write_fields(path, alone)
    stored = read_fields(path)
    assert stored[name].beta is None
    assert not stored.tree
    np.testing.assert_array_equal(stored[name].onset_s, alone[name].onset_s)


def test_a_segment_name_with_a_slash_is_refused(tmp_path, ruptures):
    renamed = {f"a/{name}": rupture for name, rupture in ruptures.items()}
    with pytest.raises(RuptureGeneratorError, match="group"):
        write_fields(tmp_path / "slash.h5", Realisation(renamed, ruptures.crs))


def test_a_file_without_the_fields_is_a_rupture_error(tmp_path):
    path = tmp_path / "other.h5"
    xr.Dataset({"x": ("y", [1.0])}).to_netcdf(path, engine="h5netcdf")
    with pytest.raises(RuptureGeneratorError, match="not a fields file"):
        read_fields(path)


def test_the_cli_writes_fields_that_make_its_srf(tmp_path, coarse_example):
    direct, fields, made = (tmp_path / name for name in ("a.srf", "a.h5", "b.srf"))
    assert main([str(coarse_example), str(direct)]) == 0
    assert main([str(coarse_example), str(fields)]) == 0
    assert fields.with_name("a.h5.toml").exists()
    assert main([str(fields), str(made)]) == 0
    assert made.read_bytes() == direct.read_bytes()


@pytest.mark.parametrize(
    ("source", "output"), [("coarse.toml", "out.txt"), ("in.h5", "out.h5")]
)
def test_the_cli_refuses_an_output_it_cannot_write(
    tmp_path, capsys, coarse_example, source, output
):
    assert main([str(tmp_path / source), str(tmp_path / output)]) == 2
    assert str(tmp_path / output) in capsys.readouterr().err


HEAVY_IMPORTS_PROBE = (
    "import sys, rupture_generator.cli; "
    "print(sorted({'xarray', 'pandas', 'h5netcdf'} & set(sys.modules)))"
)


def test_drawing_an_srf_does_not_import_xarray():
    # xarray and pandas take about a third of a second and 85 MiB to import, and an
    # SRF run never needs them. Only reading or writing a fields file does.
    loaded = subprocess.run(
        [
            sys.executable,
            "-c",
            HEAVY_IMPORTS_PROBE,
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert loaded == "[]"
