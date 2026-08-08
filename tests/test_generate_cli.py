"""``rupture-generator generate``, and the rupture file it writes.

Four output formats, and the interesting question is whether they agree. Two of them are
other people's layouts written through `rupture_generator.srf`, and two are this
package's own -- so a slip field that differs between them means one path is lying, and
the SRF is the one with an outside consumer.

The other thing tested here is the line between a **bent fault** and a **multi-segment**
one. A fault whose trace bends is one continuous surface, and its planes are fused into a
single grid whose strike varies along it -- genslip's `bent` corpus case. A fault whose
dip, dip direction or width changes between planes is two surfaces that touch, and the
generator has no rupture front that crosses between them.

The test for which is geometric: the planes' shared column of nodes either coincides or
it does not. `examples/hope.geometry.toml` is the first kind and
`examples/kaikoura.geometry.toml` the second, so both are exercised on files that ship.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from typer.testing import CliRunner

from rupture_generator.formats.rupture import planes_in, read_rupture
from rupture_generator.moment import cumulative_moment, moment_rate, rigidity_dyne_cm2
from rupture_generator.scripts.cli import app
from rupture_generator.srf import read_srf

runner = CliRunner()

EXAMPLES = Path(__file__).parent.parent / "examples"

SETTINGS = settings(
    max_examples=10,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

ONE_PLANE = """
schema_version = 1
crs = "EPSG:2193"
[[surfaces]]
type = "fault"
name = "alpine"
origin = {{ longitude_deg = 172.00, latitude_deg = -43.50 }}
[[surfaces.planes]]
end = {{ longitude_deg = 172.30, latitude_deg = -43.35 }}
dip_deg = 70.0
bottom_depth_km = 14.0
discretisation = {{ subfault_size_km = {size} }}
"""


def run(*arguments: object) -> object:
    return runner.invoke(app, [str(argument) for argument in arguments])


@pytest.fixture
def mesh(tmp_path: Path) -> Path:
    """A single-plane mesh, which is what `generate` accepts without being told."""
    geometry = tmp_path / "geometry.toml"
    geometry.write_text(ONE_PLANE.format(size=2.0))
    output = tmp_path / "mesh.h5"
    assert run("mesh", geometry, output, "--quiet").exit_code == 0
    return output


@pytest.fixture
def config() -> Path:
    return EXAMPLES / "crustal.toml"


class TestItGenerates:
    @pytest.mark.parametrize("suffix", [".h5", ".zarr", ".srf", ".srf.h5"])
    def test_it_writes_every_format(
        self, tmp_path: Path, mesh: Path, config: Path, suffix: str
    ) -> None:
        output = tmp_path / f"rupture{suffix}"
        result = run("generate", config, mesh, output, "--quiet")

        assert result.exit_code == 0, result.output
        assert output.exists()

    def test_the_summary_reports_what_ran(
        self, tmp_path: Path, mesh: Path, config: Path
    ) -> None:
        result = run("generate", config, mesh, tmp_path / "rupture.h5")

        assert "alpine" in result.output
        assert "6.20" in result.output, "the magnitude is not reported"
        assert "pcg" in result.output, "the engine that ran is not reported"

    @pytest.mark.parametrize(
        ("option", "value"), [("--seed", 4321), ("--realisation", 3)]
    )
    def test_the_command_line_overrides_the_config(
        self, tmp_path: Path, mesh: Path, config: Path, option: str, value: int
    ) -> None:
        """And what actually ran is what gets recorded, not what the file said."""
        output = tmp_path / "rupture.h5"
        assert (
            run("generate", config, mesh, output, option, value, "--quiet").exit_code
            == 0
        )

        with read_rupture(output) as tree:
            assert tree.attrs[option.removeprefix("--")] == value

    def test_a_point_source_generates_without_a_seed(
        self, tmp_path: Path, mesh: Path
    ) -> None:
        """It draws nothing, so the same inputs give the same output every time."""
        config = tmp_path / "point.toml"
        config.write_text(
            """
            schema_version = 1
            [hypocentre]
            strike_km = 12.0
            dip_km = 6.0
            [velocity_model]
            bottom_depth_km  = [1.0, 5.0, 12.0, 1000.0]
            shear_speed_km_s = [1.8, 3.2, 3.5, 4.6]
            density_g_cm3    = [2.1, 2.5, 2.7, 3.2]
            [source]
            type = "point"
            magnitude = 5.0
            rise_time_s = 0.8
            average_dip_deg = 70.0
            average_rake_deg = 175.0
            [timing]
            rupture_time_scale = -0.35
            rise_time_blend   = { centre_km = 2.0,  half_width_km = 1.0 }
            shallow_ramp      = { centre_km = 6.5,  half_width_km = 1.5 }
            deep_ramp         = { centre_km = 17.5, half_width_km = 2.5 }
            beta_shallow_ramp = { centre_km = 2.0,  half_width_km = 1.0 }
            beta_mid_ramp     = { centre_km = 6.5,  half_width_km = 1.5 }
            """
        )
        first, second = tmp_path / "a.h5", tmp_path / "b.h5"
        assert run("generate", config, mesh, first, "--quiet").exit_code == 0
        assert run("generate", config, mesh, second, "--quiet").exit_code == 0

        with read_rupture(first) as one, read_rupture(second) as two:
            slip = lambda tree: planes_in(tree)[0][2]["slip_cm"].to_numpy()
            assert np.array_equal(slip(one), slip(two))


class TestTheFormatsAgree:
    """A slip field that differs between outputs means one path is lying."""

    def test_the_srf_and_the_native_file_carry_the_same_rupture(
        self, tmp_path: Path, mesh: Path, config: Path
    ) -> None:
        native, srf_path = tmp_path / "r.h5", tmp_path / "r.srf"
        run("generate", config, mesh, native, "--quiet")
        run("generate", config, mesh, srf_path, "--quiet")

        srf = read_srf(srf_path)
        with read_rupture(native) as tree:
            ((_, _, plane),) = planes_in(tree)

            for name, from_srf in (
                ("slip_cm", srf.points.slip_cm),
                ("rake_deg", srf.points.rake_deg),
                ("onset_s", srf.points.onset_s),
                ("centre_depth_km", srf.points.depth_km),
            ):
                # The SRF is float32 -- six significant figures is the format's own
                # resolution rather than a shortcut, so this is the tightest honest
                # bound rather than a slack one.
                assert plane[name].to_numpy().ravel() == pytest.approx(
                    from_srf, rel=1e-6
                ), name

    def test_the_pulses_survive_the_srf(
        self, tmp_path: Path, mesh: Path, config: Path
    ) -> None:
        """Compared as **pulse lengths**, not as rise times.

        `README.md`'s first trap, and this test walked into it before reading it: the
        SRF does not store a rise time. `crates/srf/src/srf_parser.rs:178` derives one
        as `nt1 * dt`, and *"nt1 is not rise_time / dt -- it is what the slip-rate
        generator returned"*. Comparing the two produces a bounded, systematic-looking
        offset that reads exactly like an off-by-one and is not one.

        So the comparison is the thing the format actually carries.
        """
        native, srf_path = tmp_path / "r.h5", tmp_path / "r.srf"
        run("generate", config, mesh, native, "--quiet")
        run("generate", config, mesh, srf_path, "--quiet")

        srf = read_srf(srf_path)
        with read_rupture(native) as tree:
            ((_, _, plane),) = planes_in(tree)
            offsets = plane["slip_rate_offset"].to_numpy()
            native_lengths = np.diff(offsets)
            srf_lengths = np.diff(srf.slip_rate.indptr)

            assert np.array_equal(native_lengths, srf_lengths)
            assert plane["slip_rate"].to_numpy() == pytest.approx(
                srf.slip_rate.data, rel=1e-5
            )

    def test_the_srf_carries_the_true_strike(
        self, tmp_path: Path, mesh: Path, config: Path
    ) -> None:
        """Not the grid strike the mesh works in.

        The plane header is where a consumer reads the fault's orientation, and the two
        differ by up to five degrees. An SRF stores whole degrees, so a five-degree
        error is five counts in a field that has no room for it.
        """
        srf_path, native = tmp_path / "r.srf", tmp_path / "r.h5"
        run("generate", config, mesh, srf_path, "--quiet")
        run("generate", config, mesh, native, "--quiet")
        srf = read_srf(srf_path)

        with read_rupture(native) as tree:
            ((_, _, plane),) = planes_in(tree)
            assert srf.planes[0].strike_deg == pytest.approx(
                float(plane["strike_deg"].to_numpy()[0, 0]), abs=1e-3
            )

    @pytest.mark.parametrize("suffix", [".h5", ".zarr"])
    def test_both_native_containers_carry_the_same_rupture(
        self, tmp_path: Path, mesh: Path, config: Path, suffix: str
    ) -> None:
        reference = tmp_path / "reference.h5"
        run("generate", config, mesh, reference, "--quiet")
        other = tmp_path / f"other{suffix}"
        run("generate", config, mesh, other, "--quiet")

        with read_rupture(reference) as one, read_rupture(other) as two:
            ((_, _, first),) = planes_in(one)
            ((_, _, second),) = planes_in(two)
            for name in ("slip_cm", "rake_deg", "onset_s", "rise_time_s"):
                assert np.array_equal(
                    first[name].to_numpy(), second[name].to_numpy()
                ), name


class TestTheFileIsUsable:
    """What `FORMAT.md` tells a reader to do actually works."""

    def test_the_pulses_reconstruct_as_a_csr_matrix(
        self, tmp_path: Path, mesh: Path, config: Path
    ) -> None:
        """The snippet in `FORMAT.md`, run.

        Documentation that has drifted is worse than none, and CSR is the one part of
        this format a reader has to assemble rather than read.
        """
        output = tmp_path / "r.h5"
        run("generate", config, mesh, output, "--quiet")

        with read_rupture(output) as tree:
            ((_, _, plane),) = planes_in(tree)
            offsets = plane["slip_rate_offset"].to_numpy()
            subfaults = plane["slip_cm"].to_numpy().size

            pulses = sp.csr_array(
                (
                    plane["slip_rate"].to_numpy(),
                    plane["slip_rate_column"].to_numpy(),
                    offsets,
                ),
                shape=(subfaults, int(np.diff(offsets).max())),
            )
            assert pulses.shape[0] == subfaults

            # Each pulse integrates to its subfault's slip -- which is what a slip-rate
            # function *is*, and is the check that the CSR was assembled right rather
            # than merely assembled.
            interval_s = plane.attrs["sample_interval_s"]
            integrals = np.asarray(pulses.sum(axis=1)).ravel() * interval_s
            slip = plane["slip_cm"].to_numpy().ravel()
            slipping = np.diff(offsets) > 0
            assert integrals[slipping] == pytest.approx(slip[slipping], rel=1e-6)

    def test_the_moment_rate_integrates_to_the_recorded_moment(
        self, tmp_path: Path, mesh: Path, config: Path
    ) -> None:
        """The identity, end to end through the file rather than in memory.

        `tests/test_moment.py` asserts it on a rupture object. This asserts the file
        preserved everything that identity depends on -- the pulses, their offsets, the
        areas, and the moment attribute.
        """
        output = tmp_path / "r.h5"
        run("generate", config, mesh, output, "--quiet")

        with read_rupture(output) as tree:
            ((_, _, plane),) = planes_in(tree)

            class FromFile:
                shape = (
                    plane.attrs["strike_count"],
                    plane.attrs["dip_count"],
                )
                sample_interval_s = plane.attrs["sample_interval_s"]
                slip_rate = plane["slip_rate"].to_numpy()
                slip_rate_offsets = plane["slip_rate_offset"].to_numpy()
                onset_s = plane["onset_s"].to_numpy().ravel()
                rise_time_s = plane["rise_time_s"].to_numpy().ravel()
                slip_cm = plane["slip_cm"].to_numpy().ravel()
                moment_dyne_cm = plane.attrs["moment_dyne_cm"]

            # The materials the generator sampled, from the config's own model.
            import tomllib

            model = tomllib.loads(config.read_text())["velocity_model"]
            bottoms = np.asarray(model["bottom_depth_km"])
            depth = plane["centre_depth_km"].to_numpy().ravel()
            layer = np.minimum(
                np.searchsorted(bottoms, depth, side="left"), len(bottoms) - 1
            )
            rigidity = rigidity_dyne_cm2(
                np.asarray(model["shear_speed_km_s"])[layer],
                np.asarray(model["density_g_cm3"])[layer],
            )

            times_s, rate = moment_rate(
                FromFile(), plane["area_cm2"].to_numpy().ravel(), rigidity
            )
            assert cumulative_moment(times_s, rate)[-1] == pytest.approx(
                FromFile.moment_dyne_cm, rel=1e-3
            )

    def test_the_config_travels_with_the_rupture(
        self, tmp_path: Path, mesh: Path, config: Path
    ) -> None:
        """Provenance an SRF has never had: a file that says what produced it."""
        output = tmp_path / "r.h5"
        run("generate", config, mesh, output, "--quiet")

        with read_rupture(output) as tree:
            assert tree.attrs["config"] == config.read_text()
            assert tree.attrs["rng_engine"] == "pcg"
            assert tree.attrs["seed"] == 1234

    def test_the_nodes_are_there_so_the_file_stands_alone(
        self, tmp_path: Path, mesh: Path, config: Path
    ) -> None:
        """A slip field without its geometry is a grid of numbers."""
        output = tmp_path / "r.h5"
        run("generate", config, mesh, output, "--quiet")

        with read_rupture(output) as tree:
            ((_, _, plane),) = planes_in(tree)
            strike_count = plane.attrs["strike_count"]
            dip_count = plane.attrs["dip_count"]

            assert plane["node_east_km"].shape == (dip_count + 1, strike_count + 1)
            assert plane["slip_cm"].shape == (dip_count, strike_count)

    def test_every_variable_carries_a_unit(
        self, tmp_path: Path, mesh: Path, config: Path
    ) -> None:
        """`README.md`'s argument: this is what stopped shear speed being written in
        km/s where the SRF wants cm/s."""
        output = tmp_path / "r.h5"
        run("generate", config, mesh, output, "--quiet")

        with read_rupture(output) as tree:
            ((_, _, plane),) = planes_in(tree)
            for name, variable in plane.data_vars.items():
                if name.startswith("slip_rate_"):
                    continue  # index arrays, which have no unit
                assert "units" in variable.attrs, name
                assert "long_name" in variable.attrs, name


class TestABentFaultIsOneRupture:
    """Two planes, one surface, one grid -- which is genslip's `bent` case.

    `examples/hope.geometry.toml` turns 20 degrees with the same dip, dip direction and
    depth range either side, so the planes share their whole column of nodes and fuse.
    """

    @pytest.fixture
    def bent(self, tmp_path: Path) -> Path:
        output = tmp_path / "hope.h5"
        assert (
            run("mesh", EXAMPLES / "hope.geometry.toml", output, "--quiet").exit_code
            == 0
        )
        return output

    def test_its_planes_share_their_column_exactly(self, bent: Path) -> None:
        """The criterion, measured. Zero, not merely small.

        A conforming bend places its shared column down the *bisector* of the two
        bearings, stretched by `1 / cos(half the deflection)`, which is the one line that
        lies in both planes at once. Without that the planes diverge below the vertex --
        by 1.285 km on this very fault, which is what this asserts is not happening.
        """
        from rupture_generator.formats.mesh import read_mesh
        from rupture_generator.scripts.generate_cli import seam_gap_km

        meshes, _ = read_mesh(bent)
        assert seam_gap_km(meshes["hope"], 0, 1) == 0.0

    def test_it_generates_on_one_fused_grid(
        self, tmp_path: Path, bent: Path, config: Path
    ) -> None:
        output = tmp_path / "r.h5"
        result = run("generate", config, bent, output)

        assert result.exit_code == 0, result.output
        assert "2 planes fused" in result.output

        with read_rupture(output) as tree:
            planes = planes_in(tree)
            assert len(planes) == 2
            # The columns add up: the fused grid was split back, not regenerated.
            total = sum(plane["slip_cm"].shape[1] for _, _, plane in planes)
            assert total == 56

    def test_the_rupture_front_crosses_the_bend(
        self, tmp_path: Path, bent: Path, config: Path
    ) -> None:
        """The point of fusing. Two separate generations would each start at zero.

        The hypocentre is on the first plane, so the second plane's earliest onset must
        be *later* than zero -- the front arrived there by travelling across the seam.
        """
        output = tmp_path / "r.h5"
        run("generate", config, bent, output, "--quiet")

        with read_rupture(output) as tree:
            (_, _, first), (_, _, second) = planes_in(tree)
            assert float(first["onset_s"].min()) == pytest.approx(0.0, abs=1e-9)
            assert float(second["onset_s"].min()) > 1.0

    def test_each_plane_carries_its_own_strike(
        self, tmp_path: Path, bent: Path, config: Path
    ) -> None:
        """Which is why the strike is a field rather than a header value."""
        output = tmp_path / "r.h5"
        run("generate", config, bent, output, "--quiet")

        with read_rupture(output) as tree:
            (_, _, first), (_, _, second) = planes_in(tree)
            assert (
                abs(
                    float(first["strike_deg"][0, 0]) - float(second["strike_deg"][0, 0])
                )
                > 10.0
            )

    def test_every_plane_gets_a_self_contained_csr(
        self, tmp_path: Path, bent: Path, config: Path
    ) -> None:
        """Splitting the fused pulses means re-concatenating them, not slicing offsets.

        A plane whose offsets still indexed into the *surface's* samples would look
        right until something read one.
        """
        output = tmp_path / "r.h5"
        run("generate", config, bent, output, "--quiet")

        with read_rupture(output) as tree:
            for _, _, plane in planes_in(tree):
                offsets = plane["slip_rate_offset"].to_numpy()
                assert offsets[0] == 0
                assert offsets[-1] == len(plane["slip_rate"])
                assert len(offsets) == plane["slip_cm"].size + 1

    def test_naming_a_plane_generates_on_it_alone(
        self, tmp_path: Path, bent: Path, config: Path
    ) -> None:
        """The opt-out, for a study that wants one plane of a bent fault."""
        output = tmp_path / "r.h5"
        result = run("generate", config, bent, output, "--plane", 0)

        assert result.exit_code == 0
        assert "plane 0" in result.output
        with read_rupture(output) as tree:
            assert len(planes_in(tree)) == 1


class TestAMultiSegmentFaultIsRefused:
    """Planes that do not form one surface, refused by name.

    `examples/kaikoura.geometry.toml` changes dip from 70 to 55 degrees and its bottom
    depth from 15 to 12 km between planes, so the two surfaces touch along the trace and
    diverge below it. There is no rupture front that crosses that.
    """

    @pytest.fixture
    def segmented(self, tmp_path: Path) -> Path:
        output = tmp_path / "kaikoura.h5"
        run("mesh", EXAMPLES / "kaikoura.geometry.toml", output, "--quiet")
        return output

    def test_it_is_refused_with_the_reason(
        self, tmp_path: Path, segmented: Path, config: Path
    ) -> None:
        result = run("generate", config, segmented, tmp_path / "r.h5")

        assert result.exit_code == 1
        assert "kaikoura" in result.output
        assert "--plane" in result.output, "the message does not say what to do instead"

    def test_naming_a_plane_resolves_it(
        self, tmp_path: Path, segmented: Path, config: Path
    ) -> None:
        result = run(
            "generate", config, segmented, tmp_path / "r.h5", "--plane", 1, "--quiet"
        )
        assert result.exit_code == 0, result.output

    def test_planes_cut_at_different_resolutions_are_refused(
        self, tmp_path: Path, config: Path
    ) -> None:
        """A 10% spread is rounding; a factor of four is two requests.

        The bound is what rounding *one* requested size can produce -- see
        `SPACING_SPREAD`. genslip averages its per-subfault `ds` the same way, so a
        small spread is normal and a large one is a different question.
        """
        # Explicit counts, so the *dip* discretisation still matches and this isolates
        # the strike spacing -- changing the requested size would change both, and the
        # dip-count check would fire first.
        geometry = tmp_path / "uneven.toml"
        text = EXAMPLES.joinpath("hope.geometry.toml").read_text()
        first, second = text.rsplit("discretisation = { subfault_size_km = 1.0 }", 1)
        geometry.write_text(
            first.replace(
                "discretisation = { subfault_size_km = 1.0 }",
                "discretisation = { strike_count = 7, dip_count = 14 }",
            )
            + "discretisation = { strike_count = 29, dip_count = 14 }"
            + second
        )
        mesh_path = tmp_path / "uneven.h5"
        run("mesh", geometry, mesh_path, "--quiet")
        result = run("generate", config, mesh_path, tmp_path / "r.h5")

        assert result.exit_code == 1
        assert "spread" in result.output

    def test_a_bent_fault_cannot_be_written_to_an_srf_yet(
        self, tmp_path: Path, config: Path
    ) -> None:
        """One PLANE record per plane is what an SRF wants, and `assemble` builds one.

        Refused by name rather than writing a single-plane header over a two-plane
        rupture, which would put every subfault of the second plane in the wrong place.
        """
        mesh_path = tmp_path / "hope.h5"
        run("mesh", EXAMPLES / "hope.geometry.toml", mesh_path, "--quiet")
        result = run("generate", config, mesh_path, tmp_path / "r.srf")

        assert result.exit_code == 1
        assert "PLANE" in result.output


class TestItRefusesAmbiguity:
    def test_a_plane_that_does_not_exist_is_refused(
        self, tmp_path: Path, mesh: Path, config: Path
    ) -> None:
        result = run("generate", config, mesh, tmp_path / "r.h5", "--plane", 7)
        assert result.exit_code == 1
        assert "0..0" in result.output

    def test_a_surface_that_does_not_exist_is_refused(
        self, tmp_path: Path, mesh: Path, config: Path
    ) -> None:
        result = run("generate", config, mesh, tmp_path / "r.h5", "--surface", "hope")
        assert result.exit_code == 1
        assert "alpine" in result.output, "the message does not say what there was"

    def test_several_surfaces_need_naming(self, tmp_path: Path, config: Path) -> None:
        """Two surfaces in one geometry file are two ruptures -- the gap case."""
        geometry = tmp_path / "both.toml"
        geometry.write_text(
            EXAMPLES.joinpath("hope.geometry.toml").read_text()
            + EXAMPLES.joinpath("kaikoura.geometry.toml")
            .read_text()
            .split("[[surfaces]]", 1)[1]
            .join(["[[surfaces]]", ""])
        )
        mesh_path = tmp_path / "both.h5"
        if run("mesh", geometry, mesh_path, "--quiet").exit_code != 0:
            pytest.skip("the combined geometry did not build")

        result = run("generate", config, mesh_path, tmp_path / "r.h5")
        assert result.exit_code == 1
        assert "--surface" in result.output

    @pytest.mark.parametrize(
        ("strike_km", "dip_km"), [(-1.0, 5.0), (500.0, 5.0), (5.0, -1.0), (5.0, 500.0)]
    )
    def test_a_hypocentre_off_the_plane_is_refused(
        self, tmp_path: Path, mesh: Path, strike_km: float, dip_km: float
    ) -> None:
        """`DEFECTS.md` 17 was this arithmetic wrong by a cell, silently. Off the plane
        entirely is the loud case, and it stays loud."""
        config = tmp_path / "config.toml"
        config.write_text(
            (EXAMPLES / "crustal.toml")
            .read_text()
            .replace("strike_km = 12.0", f"strike_km = {strike_km}")
            .replace("dip_km = 6.0", f"dip_km = {dip_km}")
        )
        result = run("generate", config, mesh, tmp_path / "r.h5")

        assert result.exit_code == 1
        assert "hypocentre" in result.output.lower()

    def test_a_broken_config_names_its_key(self, tmp_path: Path, mesh: Path) -> None:
        config = tmp_path / "config.toml"
        config.write_text(
            (EXAMPLES / "crustal.toml")
            .read_text()
            .replace("magnitude = 6.2", "magnitude = 99.0")
        )
        result = run("generate", config, mesh, tmp_path / "r.h5")

        assert result.exit_code == 1
        assert "magnitude" in result.output


class TestReproducibility:
    @given(seed=st.integers(min_value=1, max_value=9999))
    @SETTINGS
    def test_a_seed_reproduces_a_rupture_through_the_file(
        self, tmp_path: Path, mesh: Path, config: Path, seed: int
    ) -> None:
        """The whole pipeline, not just the generator: config, mesh, projection, write.

        A step that lost determinism -- an unordered dict, a timestamp in a field --
        would show here and nowhere upstream.
        """
        first, second = tmp_path / f"a{seed}.h5", tmp_path / f"b{seed}.h5"
        run("generate", config, mesh, first, "--seed", seed, "--quiet")
        run("generate", config, mesh, second, "--seed", seed, "--quiet")

        with read_rupture(first) as one, read_rupture(second) as two:
            ((_, _, a),) = planes_in(one)
            ((_, _, b),) = planes_in(two)
            assert np.array_equal(a["slip_cm"].to_numpy(), b["slip_cm"].to_numpy())
            assert np.array_equal(a["onset_s"].to_numpy(), b["onset_s"].to_numpy())

    def test_different_seeds_give_different_ruptures(
        self, tmp_path: Path, mesh: Path, config: Path
    ) -> None:
        first, second = tmp_path / "a.h5", tmp_path / "b.h5"
        run("generate", config, mesh, first, "--seed", 1, "--quiet")
        run("generate", config, mesh, second, "--seed", 2, "--quiet")

        with read_rupture(first) as one, read_rupture(second) as two:
            ((_, _, a),) = planes_in(one)
            ((_, _, b),) = planes_in(two)
            assert not np.array_equal(a["slip_cm"].to_numpy(), b["slip_cm"].to_numpy())


class TestTheShippedExampleWorks:
    def test_the_example_config_generates(self, tmp_path: Path, mesh: Path) -> None:
        """`examples/crustal.toml` is documentation that runs."""
        result = run(
            "generate", EXAMPLES / "crustal.toml", mesh, tmp_path / "r.h5", "--quiet"
        )
        assert result.exit_code == 0, result.output
