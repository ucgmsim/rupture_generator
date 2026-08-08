"""``rupture-generator mesh``, the mesh file, and what a bad geometry looks like.

Three things, and the third is the one a test suite usually skips.

**The file round-trips.** Property-tested over extents and plane counts, in *both*
containers, because a format that is lossless in HDF5 and lossy in Zarr is a format that
silently depends on which one you picked.

**The rounding is the one the summary claims.** A config asks for a subfault size and
gets whole cells; the table prints what it actually used, and if that number is not the
one in the file then the table is lying about the thing it exists to show.

**The errors are usable.** A CLI's error path is the part a user meets first and the part
nothing normally exercises. Each kind of broken config gets a test asserting the panel
*names the key* -- not that it failed, which is easy, but that it says which word to
change.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyproj
import pytest
import xarray as xr
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from typer.testing import CliRunner

from rupture_generator.formats import Format, from_path
from rupture_generator.formats.mesh import read_mesh, write_mesh
from rupture_generator.scripts.cli import app
from rupture_generator.scripts.mesh_cli import build_surface, cell_counts

runner = CliRunner()

SETTINGS = settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

GEOMETRY = """
schema_version = 1
crs = "EPSG:2193"

[[surfaces]]
type = "fault"
name = "kaikoura"
origin = {{ longitude_deg = 173.00, latitude_deg = -42.60 }}
top_depth_km = 0.0

[[surfaces.planes]]
end = {{ longitude_deg = 173.40, latitude_deg = -42.40 }}
dip_deg = 70.0
bottom_depth_km = 15.0
discretisation = {{ {first} }}

[[surfaces.planes]]
end = {{ longitude_deg = 173.90, latitude_deg = -42.10 }}
dip_deg = 55.0
bottom_depth_km = 12.0
discretisation = {{ {second} }}
"""


def a_geometry_file(
    tmp_path: Path,
    first: str = "subfault_size_km = 2.0",
    second: str = "subfault_size_km = 2.0",
) -> Path:
    path = tmp_path / "geometry.toml"
    path.write_text(GEOMETRY.format(first=first, second=second))
    return path


def run(*arguments: str) -> object:
    return runner.invoke(app, [str(argument) for argument in arguments])


class TestTheCommandRuns:
    @pytest.mark.parametrize("suffix", [".h5", ".zarr"])
    def test_it_writes_a_mesh_in_either_container(
        self, tmp_path: Path, suffix: str
    ) -> None:
        output = tmp_path / f"mesh{suffix}"
        result = run("mesh", a_geometry_file(tmp_path), output)

        assert result.exit_code == 0, result.output
        assert output.exists()
        meshes, crs = read_mesh(output)
        assert set(meshes) == {"kaikoura"}
        assert meshes["kaikoura"].patch_count == 2
        assert crs.to_string() == "EPSG:2193"

    def test_the_summary_names_the_surface_and_its_planes(self, tmp_path: Path) -> None:
        result = run("mesh", a_geometry_file(tmp_path), tmp_path / "mesh.h5")
        assert "kaikoura" in result.output
        # Two planes with different dips, which is what a fused single grid could not do.
        assert "70.0" in result.output
        assert "55.0" in result.output

    def test_quiet_says_nothing_but_still_writes(self, tmp_path: Path) -> None:
        output = tmp_path / "mesh.h5"
        result = run("mesh", a_geometry_file(tmp_path), output, "--quiet")
        assert result.exit_code == 0
        assert "kaikoura" not in result.output
        assert output.exists()

    def test_a_point_source_becomes_one_cell(self, tmp_path: Path) -> None:
        path = tmp_path / "point.toml"
        path.write_text(
            """
            schema_version = 1
            crs = "EPSG:2193"
            [[surfaces]]
            type = "point"
            name = "aftershock"
            centre = { longitude_deg = 173.5, latitude_deg = -42.3 }
            depth_km = 8.0
            strike_deg = 55.0
            dip_deg = 60.0
            size_km = 0.5
            """
        )
        output = tmp_path / "point.h5"
        assert run("mesh", path, output).exit_code == 0

        meshes, _ = read_mesh(output)
        assert meshes["aftershock"].cell_extents(0) == (1, 1)

    def test_the_geometry_travels_with_the_mesh(self, tmp_path: Path) -> None:
        """The config is stored verbatim, so a mesh can say what it was built from.

        Provenance is the thing an SRF has never had: a file arrives and there is no way
        to tell which inputs produced it.
        """
        geometry = a_geometry_file(tmp_path)
        output = tmp_path / "mesh.h5"
        run("mesh", geometry, output)

        with xr.open_datatree(output, engine="h5netcdf") as tree:
            assert tree.attrs["geometry_config"] == geometry.read_text()
            assert json.loads(tree.attrs["surfaces"]) == ["kaikoura"]


class TestTheDiscretisationIsWhatTheSummarySays:
    @pytest.mark.parametrize(
        ("length_km", "width_km", "size_km", "expected"),
        [
            (20.0, 12.0, 1.0, (20, 12)),
            (20.0, 12.0, 2.0, (10, 6)),
            # Rounds to nearest, not down: 39.7 / 1.0 is 40 cells of 0.99 km, which is
            # closer to what was asked for than 39 of 1.02.
            (39.7, 16.0, 1.0, (40, 16)),
            (39.7, 16.0, 4.0, (10, 4)),
            # A plane smaller than the size asked for is still a plane.
            (0.4, 0.3, 1.0, (1, 1)),
        ],
    )
    def test_a_size_becomes_whole_cells(
        self,
        length_km: float,
        width_km: float,
        size_km: float,
        expected: tuple[int, int],
    ) -> None:
        from rupture_generator.config.geometry import Discretisation

        cuts = cell_counts(
            Discretisation(subfault_size_km=size_km), length_km, width_km
        )
        assert (cuts.strike_count, cuts.dip_count) == expected

    def test_explicit_counts_are_used_as_given(self) -> None:
        from rupture_generator.config.geometry import Discretisation

        cuts = cell_counts(Discretisation(strike_count=7, dip_count=3), 20.0, 12.0)
        assert (cuts.strike_count, cuts.dip_count) == (7, 3)

    @given(
        size_km=st.floats(min_value=0.1, max_value=10.0),
        length_km=st.floats(min_value=0.5, max_value=200.0),
        width_km=st.floats(min_value=0.5, max_value=60.0),
    )
    @SETTINGS
    def test_the_cell_is_never_more_than_half_a_size_from_the_request(
        self, size_km: float, length_km: float, width_km: float
    ) -> None:
        """Rounding to nearest, stated as a property.

        The actual cell is `length / round(length / size)`. Rounding to nearest means the
        count is within half of the exact ratio, so the cell is within a factor set by
        that -- and never zero cells, which is not a surface.
        """
        from rupture_generator.config.geometry import Discretisation

        cuts = cell_counts(
            Discretisation(subfault_size_km=size_km), length_km, width_km
        )
        assert cuts.strike_count >= 1
        assert cuts.dip_count >= 1
        assert (
            abs(cuts.strike_count - length_km / size_km) <= 0.5
            or cuts.strike_count == 1
        )

    def test_the_size_in_the_summary_is_the_size_in_the_file(
        self, tmp_path: Path
    ) -> None:
        """Otherwise the table is lying about the one thing it exists to show."""
        output = tmp_path / "mesh.h5"
        result = run(
            "mesh",
            a_geometry_file(
                tmp_path, "subfault_size_km = 3.0", "subfault_size_km = 3.0"
            ),
            output,
        )
        assert result.exit_code == 0

        meshes, _ = read_mesh(output)
        strike_km, dip_km = meshes["kaikoura"].spacing(0)
        assert f"{strike_km:.2f}x{dip_km:.2f}" in result.output


class TestTheFileRoundTrips:
    @given(
        strike_count=st.integers(min_value=1, max_value=20),
        dip_count=st.integers(min_value=1, max_value=14),
    )
    @SETTINGS
    @pytest.mark.parametrize("suffix", [".h5", ".zarr"])
    def test_a_mesh_survives_both_containers_exactly(
        self, tmp_path: Path, suffix: str, strike_count: int, dip_count: int
    ) -> None:
        """Node positions come back bit-identical, so everything derived does too.

        In both containers, because a format that is lossless in one and lossy in the
        other silently depends on which was picked.
        """
        from rupture_generator.config.geometry import (
            Discretisation,
            FaultConfig,
            LonLat,
            PlaneConfig,
        )

        surface = FaultConfig(
            name="fault",
            origin=LonLat(longitude_deg=173.0, latitude_deg=-42.6),
            planes=[
                PlaneConfig(
                    end=LonLat(longitude_deg=173.4, latitude_deg=-42.4),
                    dip_deg=70.0,
                    bottom_depth_km=15.0,
                    discretisation=Discretisation(
                        strike_count=strike_count, dip_count=dip_count
                    ),
                )
            ],
        )
        crs = pyproj.CRS("EPSG:2193")
        mesh = build_surface(surface, crs)

        # A fresh name per example: hypothesis reuses the tmp_path fixture.
        path = tmp_path / f"mesh_{strike_count}_{dip_count}{suffix}"
        write_mesh({"fault": mesh}, crs, path)
        back, back_crs = read_mesh(path)

        assert back_crs.to_string() == crs.to_string()
        assert back["fault"].origin == mesh.origin
        assert back["fault"].cell_extents(0) == (strike_count, dip_count)
        for rebuilt, original in zip(
            back["fault"].node_positions(0), mesh.node_positions(0), strict=True
        ):
            assert np.array_equal(rebuilt, original)

    @pytest.mark.parametrize("suffix", [".h5", ".zarr"])
    def test_many_planes_come_back_in_order(self, tmp_path: Path, suffix: str) -> None:
        """Eleven planes, in both containers, because **Zarr does not preserve order**.

        Not defensive. Measured: asked for eleven groups written `plane_0` through
        `plane_10`, Zarr hands them back as

            plane_10, plane_8, plane_5, plane_7, plane_9, plane_6, plane_4, ...

        which is neither insertion nor lexicographic order. HDF5 preserves insertion, so
        a suite that only tested HDF5 would be green while every Zarr mesh silently
        reordered its planes -- and with two planes the wrong order comes up about half
        the time, so it would be an intermittent failure somewhere else entirely.

        That is why `from_datatree` keys on the stored `plane` attribute. Each plane here
        is given a dip that identifies it, so a reordering cannot hide.
        """
        from rupture_generator.config.geometry import (
            Discretisation,
            FaultConfig,
            LonLat,
            PlaneConfig,
        )

        dips = [20.0 + 5.0 * index for index in range(11)]
        surface = FaultConfig(
            name="many",
            origin=LonLat(longitude_deg=173.0, latitude_deg=-42.6),
            planes=[
                PlaneConfig(
                    end=LonLat(
                        longitude_deg=173.0 + 0.05 * (index + 1), latitude_deg=-42.6
                    ),
                    dip_deg=dip,
                    bottom_depth_km=10.0,
                    discretisation=Discretisation(strike_count=3, dip_count=2),
                )
                for index, dip in enumerate(dips)
            ],
        )
        crs = pyproj.CRS("EPSG:2193")
        mesh = build_surface(surface, crs)
        assert mesh.patch_count == 11

        path = tmp_path / f"many{suffix}"
        write_mesh({"many": mesh}, crs, path)
        back, _ = read_mesh(path)

        recovered = [float(back["many"].dip_deg(patch)[0, 0]) for patch in range(11)]
        assert recovered == pytest.approx(dips, abs=1e-9)

    def test_two_surfaces_stay_separate(self, tmp_path: Path) -> None:
        from rupture_generator.config.geometry import (
            Discretisation,
            FaultConfig,
            LonLat,
            PlaneConfig,
        )

        crs = pyproj.CRS("EPSG:2193")

        def a_fault(name: str, dip_deg: float) -> FaultConfig:
            return FaultConfig(
                name=name,
                origin=LonLat(longitude_deg=173.0, latitude_deg=-42.6),
                planes=[
                    PlaneConfig(
                        end=LonLat(longitude_deg=173.3, latitude_deg=-42.4),
                        dip_deg=dip_deg,
                        bottom_depth_km=12.0,
                        discretisation=Discretisation(strike_count=5, dip_count=3),
                    )
                ],
            )

        meshes = {
            name: build_surface(a_fault(name, dip), crs)
            for name, dip in (("alpine", 60.0), ("hope", 80.0))
        }
        path = tmp_path / "two.h5"
        write_mesh(meshes, crs, path)
        back, _ = read_mesh(path)

        assert set(back) == {"alpine", "hope"}
        assert float(back["alpine"].dip_deg(0)[0, 0]) == pytest.approx(60.0, abs=1e-9)
        assert float(back["hope"].dip_deg(0)[0, 0]) == pytest.approx(80.0, abs=1e-9)


class TestFormatInference:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("mesh.h5", Format.NETCDF),
            ("mesh.hdf5", Format.NETCDF),
            ("mesh.nc", Format.NETCDF),
            ("mesh.zarr", Format.ZARR),
            ("rupture.srf", Format.SRF),
            # The two-suffix check: `.srf.h5` is SW4's and has to beat `.h5`, or a
            # native file goes out wearing someone else's layout.
            ("rupture.srf.h5", Format.SRF_HDF5),
            ("rupture.srf.hdf5", Format.SRF_HDF5),
            ("a.long.name.with.dots.h5", Format.NETCDF),
        ],
    )
    def test_an_extension_names_a_format(self, name: str, expected: Format) -> None:
        assert from_path(Path(name)) == expected

    @pytest.mark.parametrize("name", ["mesh", "mesh.txt", "mesh.tar.gz"])
    def test_an_unknown_extension_is_refused_rather_than_guessed(
        self, name: str
    ) -> None:
        with pytest.raises(ValueError, match="no format for"):
            from_path(Path(name))

    def test_a_mesh_cannot_be_written_as_an_srf(self, tmp_path: Path) -> None:
        """An SRF holds a rupture. There is nothing to put in its slip columns."""
        result = run("mesh", a_geometry_file(tmp_path), tmp_path / "mesh.srf")
        assert result.exit_code != 0


class TestBadGeometriesAreExplained:
    """The panel has to name the key, not merely fail.

    Each case asserts the *word the reader must change* appears in the output. A test
    that only checked the exit code would pass on a traceback.
    """

    def test_a_value_out_of_range_names_its_key(self, tmp_path: Path) -> None:
        path = tmp_path / "geometry.toml"
        path.write_text(
            GEOMETRY.format(
                first="subfault_size_km = 2.0", second="subfault_size_km = 2.0"
            ).replace("dip_deg = 70.0", "dip_deg = 91.0")
        )
        result = run("mesh", path, tmp_path / "mesh.h5")

        assert result.exit_code == 1
        assert "dip_deg" in result.output
        assert "91" in result.output

    def test_a_misspelt_key_is_named_and_a_correction_suggested(
        self, tmp_path: Path
    ) -> None:
        """The near miss is the common case, and naming the intended key turns a
        rejection into an instruction."""
        path = tmp_path / "geometry.toml"
        path.write_text(
            GEOMETRY.format(
                first="subfault_size_km = 2.0", second="subfault_size_km = 2.0"
            ).replace("dip_deg = 70.0", "dipp_deg = 70.0")
        )
        result = run("mesh", path, tmp_path / "mesh.h5")

        assert result.exit_code == 1
        assert "dipp_deg" in result.output
        assert "dip_deg" in result.output, "no suggestion offered"

    def test_a_syntax_error_shows_the_line(self, tmp_path: Path) -> None:
        path = tmp_path / "geometry.toml"
        path.write_text('schema_version = 1\ncrs = "EPSG:2193"\n\n[[surfaces]\n')
        result = run("mesh", path, tmp_path / "mesh.h5")

        assert result.exit_code == 1
        assert "syntax" in result.output.lower()
        assert "4" in result.output, "the line number is not shown"

    def test_a_missing_key_is_named(self, tmp_path: Path) -> None:
        path = tmp_path / "geometry.toml"
        path.write_text(
            GEOMETRY.format(
                first="subfault_size_km = 2.0", second="subfault_size_km = 2.0"
            ).replace("bottom_depth_km = 15.0\n", "")
        )
        result = run("mesh", path, tmp_path / "mesh.h5")

        assert result.exit_code == 1
        assert "bottom_depth_km" in result.output

    def test_a_geographic_crs_is_refused_with_the_right_advice(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "geometry.toml"
        path.write_text(
            GEOMETRY.format(
                first="subfault_size_km = 2.0", second="subfault_size_km = 2.0"
            ).replace('crs = "EPSG:2193"', 'crs = "EPSG:4326"')
        )
        result = run("mesh", path, tmp_path / "mesh.h5")

        assert result.exit_code == 1
        assert "crs" in result.output
        assert "2193" in result.output, "the message does not say what to use instead"

    def test_a_missing_file_is_refused_by_typer(self, tmp_path: Path) -> None:
        """`exists=True` on the argument, so the check is the framework's and the
        message is the framework's."""
        result = run("mesh", tmp_path / "absent.toml", tmp_path / "mesh.h5")
        assert result.exit_code != 0


class TestHelpReadsLikeProse:
    @pytest.mark.parametrize("arguments", [[], ["mesh"]])
    def test_help_exits_cleanly(self, arguments: list[str]) -> None:
        result = runner.invoke(app, [*arguments, "--help"])
        assert result.exit_code == 0

    def test_the_root_lists_its_subcommands(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert "mesh" in result.output

    def test_no_arguments_shows_help_rather_than_failing(self) -> None:
        result = runner.invoke(app, [])
        assert "Usage" in result.output


class TestTheShippedExampleWorks:
    def test_the_example_geometry_builds(self, tmp_path: Path) -> None:
        """`examples/kaikoura.geometry.toml` is documentation that runs.

        An example that has drifted from the schema is worse than none: it is the first
        thing anyone copies.
        """
        example = Path(__file__).parent.parent / "examples" / "kaikoura.geometry.toml"
        output = tmp_path / "mesh.h5"
        result = run("mesh", example, output)

        assert result.exit_code == 0, result.output
        meshes, _ = read_mesh(output)
        assert meshes["kaikoura"].patch_count == 2
