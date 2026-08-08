"""Tests that actually run genslip.

Skipped unless `GENSLIP_BINARY` points at one, because most of the suite does not need
it and a missing binary is not a failing port. What these pin is the *harness*: that
the arguments it renders are a set genslip accepts, that the file it writes is a file
genslip reads, and that a seed reproduces a rupture through the whole invocation.

Nothing here compares the port against the reference yet. That needs every getpar name
mapped onto the port's five spec groups, and a wrong mapping is indistinguishable from
a wrong port, so it is its own commit.
"""

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from rupture_generator import VelocityModel1D
from tests.harness import gsf
from tests.harness.genslip_reference import (
    generate_segment_rupture,
    write_velocity_model,
)
from tests.harness.test_unroll import _make_minimal_params

STRIKE, DIP = 20, 12
SUBFAULT_KM = 0.5
TOP_DEPTH_KM = 1.0
SEED = 20260807

# Four crustal layers, as the arrays both sides are handed.
BOTTOM_DEPTH_KM = np.array([1.0, 5.0, 12.0, 30.0], dtype=np.float64)
SHEAR_SPEED_KM_S = np.array([1.8, 2.6, 3.2, 3.6], dtype=np.float64)
DENSITY_G_CM3 = np.array([2.1, 2.4, 2.6, 2.7], dtype=np.float64)

genslip = pytest.mark.skipif(
    not os.environ.get("GENSLIP_BINARY"),
    reason="set GENSLIP_BINARY to a genslip v5.6.2 built with -std=gnu17",
)


def binary() -> Path:
    return Path(os.environ["GENSLIP_BINARY"])


def geometry() -> gsf.GsfSubfaults:
    """A 10 x 6 km plane dipping 80 degrees, its top edge a kilometre down.

    The top edge is deliberately deeper than half a subfault's vertical extent, so the
    `dtop` genslip derives is a real number rather than the surface it clamps to.
    """
    return gsf.on_a_plane(
        strike_count=STRIKE,
        dip_count=DIP,
        along_strike_km=SUBFAULT_KM,
        down_dip_km=SUBFAULT_KM,
        centre_longitude_deg=172.0,
        centre_latitude_deg=-43.5,
        strike_deg=45.0,
        dip_deg=80.0,
        top_depth_km=TOP_DEPTH_KM,
        rake_deg=175.0,
    )


def run(**overrides):
    """Drive genslip once, with the configured defaults and version 2.0 output.

    Returns the whole `ReferenceRun` -- the SRF and the quantities genslip reported
    deriving on stderr. `test_mapping.py` checks the mapping against the latter.
    """
    parameters = _make_minimal_params(
        read_gsf=True,
        read_erf=False,
        # The configured production value. With roughness on, the two spectral fields
        # PRUNED.md describes stop being numerically inert.
        alpha_rough=0.0,
        **overrides.pop("parameters", {}),
    )
    return generate_segment_rupture(
        geometry(),
        parameters,
        binary(),
        magnitude=6.2,
        strike_count=STRIKE,
        dip_count=DIP,
        seed=overrides.pop("seed", SEED),
        hypocentre_strike_km=0.0,
        hypocentre_dip_km=3.0,
        velocity_model=velocity_model(),
    )


def velocity_model() -> VelocityModel1D:
    """The four layers both sides run on."""
    return VelocityModel1D(BOTTOM_DEPTH_KM, SHEAR_SPEED_KM_S, DENSITY_G_CM3)


class TestTheVelocityModelFile:
    def test_bottoms_become_thicknesses(self, tmp_path: Path) -> None:
        path = tmp_path / "velocity_model.1d"
        write_velocity_model(velocity_model(), path)
        lines = path.read_text().splitlines()
        assert lines[0] == "4"
        thicknesses = [float(line.split()[0]) for line in lines[1:]]
        assert thicknesses == pytest.approx([1.0, 4.0, 7.0, 18.0])

    def test_the_shear_speeds_are_the_third_column(self, tmp_path: Path) -> None:
        # genslip reads `th vp vs den`, so a transposed vp/vs would silently change
        # every rupture speed in the model.
        path = tmp_path / "velocity_model.1d"
        write_velocity_model(velocity_model(), path)
        speeds = [float(line.split()[2]) for line in path.read_text().splitlines()[1:]]
        assert speeds == pytest.approx(SHEAR_SPEED_KM_S)


@genslip
class TestTheReferenceRuns:
    def test_it_produces_one_point_per_subfault(self) -> None:
        reference = run()
        assert len(reference.srf.points) == STRIKE * DIP
        assert len(reference.srf.planes) == 1

    def test_the_header_is_the_geometry_it_was_given(self) -> None:
        reference = run()
        plane = reference.srf.planes[0]
        assert plane.strike_count == STRIKE
        assert plane.dip_count == DIP
        assert plane.length_km == pytest.approx(STRIKE * SUBFAULT_KM)
        assert plane.width_km == pytest.approx(DIP * SUBFAULT_KM)
        assert plane.strike_deg == pytest.approx(45.0)
        assert plane.dip_deg == pytest.approx(80.0)

    def test_the_top_depth_is_the_one_the_gsf_implies(self) -> None:
        # genslip derives dtop from the GSF rather than being told it, so this is the
        # check that `GsfSubfaults.top_depth_km` computes what the binary computes --
        # including the truncated radians constant and the float32 dip average.
        reference = run()
        assert reference.srf.planes[0].top_depth_km == pytest.approx(
            geometry().top_depth_km, abs=1e-4
        )

    def test_it_ruptures(self) -> None:
        # Not a physics claim, a "the arguments were accepted and something happened"
        # claim: a run that silently produced zeros would pass every assertion above.
        reference = run()
        assert reference.srf.points.slip_cm.max() > 0.0
        assert reference.srf.points.onset_s.min() == pytest.approx(0.0)
        assert reference.srf.points.onset_s.max() > 0.0
        assert reference.srf.slip_rate.data.max() > 0.0

    def test_a_seed_reproduces_a_rupture(self) -> None:
        first, second = run(), run()
        assert np.array_equal(first.srf.points.slip_cm, second.srf.points.slip_cm)
        assert np.array_equal(first.srf.points.onset_s, second.srf.points.onset_s)
        assert np.array_equal(first.srf.slip_rate.data, second.srf.slip_rate.data)

    def test_a_different_seed_is_a_different_rupture(self) -> None:
        assert not np.array_equal(
            run(seed=SEED).srf.points.slip_cm, run(seed=SEED + 1).srf.points.slip_cm
        )


@genslip
def test_without_ns_and_nh_genslip_writes_nothing_and_says_nothing(
    tmp_path: Path,
) -> None:
    """The reason `_DEFAULT_GEOMETRY_PARAMETERS` sets `ns` and `nh`, pinned.

    This asserts genslip's behaviour, not the harness's: asked for its default -1
    realisations it runs the whole model, reports its progress on stderr, exits **0**
    and writes a **zero-byte** SRF. Anything driving this binary has to check the
    output size, because the exit code will not tell it.
    """
    gsf_path = tmp_path / "geometry.gsf"
    gsf.write_gsf(geometry(), gsf_path)
    velocity_path = tmp_path / "velocity_model.1d"
    write_velocity_model(
        BOTTOM_DEPTH_KM, SHEAR_SPEED_KM_S, DENSITY_G_CM3, velocity_path
    )
    srf_path = tmp_path / "rupture.srf"

    with srf_path.open("wb") as output:
        completed = subprocess.run(
            [
                str(binary()),
                "read_gsf=1",
                "read_erf=0",
                f"infile={gsf_path}",
                f"velfile={velocity_path}",
                "mag=6.2",
                f"nstk={STRIKE}",
                f"ndip={DIP}",
                f"seed={SEED}",
                "write_srf=1",
            ],
            stdout=output,
            stderr=subprocess.PIPE,
            check=False,
        )

    assert completed.returncode == 0
    assert srf_path.stat().st_size == 0
    # It got as far as reporting the grid, so this is not an early bail-out.
    assert b"nslip= -1 nhypo= -1" in completed.stderr
