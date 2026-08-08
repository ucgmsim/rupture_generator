"""What the point-source path differs from `generic_slip2srf` by, measured.

Skipped unless `GENERIC_SLIP2SRF` points at a build of it. **Nothing here asserts
agreement.** The two disagree on purpose in two places, and an assertion would be
encoding one of those choices twice: once in the code and once as a tolerance that
has to be widened whenever the choice is revisited.

What this does instead is record the sizes, so the choices are decidable rather than
arguable -- `ENGINEERING_RULES.md` rule 4. The table `test_the_divergence_table`
prints is the artefact; the assertions that do exist are about the *comparison* being
meaningful, not about the answer.

# Why the shapes can be compared at all and the durations cannot

The port's rise time is `factor_at(depth) / mean(factor_at) * risetime`; the C's is
`factor_at(depth) * risetime`. On a single-depth fixture the port's mean is that
subfault's own factor, so the two differ by exactly `factor_at(depth)` -- a clean
scalar, recorded rather than absorbed.

That makes the shapes comparable by handing the port the C's *realised* duration.
Decomposing the divergence before measuring it is the same move that found
`DEFECTS.md` 17 and 18: "the pulses differ" says nothing, "the pulses differ because
the durations differ, and here is the ratio" says what to do.

# What was found on the first run

`SlipRateShape::Urs` extrapolated its depth ramp instead of clamping at the ends.
`DepthRamp` interpolates and does not clamp; every other caller in the crate brackets
it with a three-branch `if` and this one did not. At 7 km, one kilometre below the
ramp, the tail height came out at 0.05 against the C's 0.2 and half the pulse was
wrong. Fixed, and pinned in `slip_rate_contract.rs::the_urs_tail_ramp_has_ends`.

Worth stating plainly, because it is the argument for this file existing: eleven Rust
tests and eight Python ones covered the point-source path and none of them caught it.
Every one of them was written by the same reader of the same source.
"""

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from rupture_generator import (
    FaultGrid,
    PointSourceSpec,
    Ramp,
    SlipRateShape,
    TimingSpec,
    VelocityModel1D,
    generate_point_source,
)
from rupture_generator.srf import read_srf

reference_binary = pytest.mark.skipif(
    not os.environ.get("GENERIC_SLIP2SRF"),
    reason="set GENERIC_SLIP2SRF to a build of EMOD3D's generic_slip2srf",
)

# One subfault, at one depth, so the rise-time ratio is a scalar rather than a field.
DEPTH_KM = 7.0
SLIP_CM = 12.5
DT = 0.005
RISETIME_S = 0.35
RISETIME_FAC = 2.0
RISETIME_DEP = 6.5
RISETIME_DEP_RANGE = 1.5

# The C's ramp at DEPTH_KM: 1 below `dep + range`, `risetimefac` above `dep - range`.
DEPTH_FACTOR = 1.0 + (RISETIME_FAC - 1.0) * (
    (RISETIME_DEP + RISETIME_DEP_RANGE) - DEPTH_KM
) / (2.0 * RISETIME_DEP_RANGE)

SHAPES = {
    "ucsb": SlipRateShape.ucsb(),
    "ucsb2": SlipRateShape.ucsb2(),
    "ucsb-T2": SlipRateShape.ucsb_t(2.0),
    "ucsb-varT1": SlipRateShape.ucsb_var_t1(0.13),
    "brune": SlipRateShape.brune(),
    "urs": SlipRateShape.urs(),
    "esg2006": SlipRateShape.esg2006(),
    "cos": SlipRateShape.cos(),
    "seki": SlipRateShape.seki(),
    "delta": SlipRateShape.delta(),
}

# `brune` is the one shape whose *duration* the port defines differently, so its
# samples are not a comparison of shapes at all. Named rather than silently skipped.
DURATION_DIFFERS = {"brune"}


@pytest.fixture(scope="module")
def workspace() -> Path:
    return Path(tempfile.mkdtemp(prefix="point-source-reference-"))


@pytest.fixture(scope="module")
def gsf_file(workspace: Path) -> Path:
    """One subfault, half a kilometre square, in the format the C reads.

    `lon lat dep ds dw stk dip rake slip tinit segno` -- eleven columns, which is the
    branch the C takes when `risetime` is given.
    """
    path = workspace / "point.gsf"
    path.write_text(
        f"1\n172.6 -43.5 {DEPTH_KM:.4f} 0.5 0.5 45.0 60.0 175.0 {SLIP_CM:.4f} 0.0 0\n"
    )
    return path


def reference_pulse(gsf_file: Path, workspace: Path, stype: str) -> np.ndarray:
    """Run `generic_slip2srf` and read back the one subfault's slip-rate function.

    `plane_header=1` because the SRF reader wants a PLANE section, and because it is
    what the workflow passes.
    """
    output = workspace / f"{stype}.srf"
    subprocess.run(
        [
            os.environ["GENERIC_SLIP2SRF"],
            f"infile={gsf_file}",
            f"outfile={output}",
            "outbin=0",
            "plane_header=1",
            f"stype={stype}",
            f"dt={DT}",
            f"risetime={RISETIME_S}",
            f"risetimefac={RISETIME_FAC}",
            f"risetimedep={RISETIME_DEP}",
            f"risetimedep_range={RISETIME_DEP_RANGE}",
        ],
        check=True,
        capture_output=True,
    )
    row = read_srf(output).slip_rate[[0]].toarray().ravel()
    return np.trim_zeros(row, "b")


def port_pulse(shape: SlipRateShape, duration_s: float) -> np.ndarray:
    """The same subfault through the port, rescaled to the reference's slip.

    Rescaled because the port derives slip from the moment while the C is *given* it,
    so comparing the raw amplitudes would measure the magnitude the fixture happens
    to name. Both pulses integrate to their own slip, so one factor makes them
    comparable and leaves the shape untouched.
    """
    grid = FaultGrid(
        1,
        1,
        2,
        2,
        0.5,
        0.5,
        depth_km=np.array([DEPTH_KM], dtype=np.float64),
        base_rake_deg=np.array([175.0], dtype=np.float64),
        velocity_fraction=np.array([0.8], dtype=np.float64),
    )
    rupture = generate_point_source(
        grid,
        VelocityModel1D(
            np.array([1.0, 5.0, 12.0, 30.0], dtype=np.float64),
            np.array([1.8, 2.6, 3.2, 3.6], dtype=np.float64),
            np.array([2.1, 2.4, 2.6, 2.7], dtype=np.float64),
        ),
        PointSourceSpec(5.2, duration_s, average_dip_deg=60.0, average_rake_deg=175.0),
        TimingSpec(
            rupture_time_scale=-0.35,
            rise_time_blend=Ramp(2.0, 1.0),
            shallow_ramp=Ramp(RISETIME_DEP, RISETIME_DEP_RANGE),
            deep_ramp=Ramp(17.5, 2.5),
            beta_shallow_ramp=Ramp(2.0, 1.0),
            beta_mid_ramp=Ramp(6.5, 1.5),
            slip_rate_shape=shape,
            sample_interval_s=DT,
        ),
        hypocentre_strike=0,
        hypocentre_dip=0,
    )
    count = int(rupture.slip_rate_offsets[1])
    samples = np.asarray(rupture.slip_rate[:count], dtype=np.float64)
    return samples * (SLIP_CM / float(rupture.slip_cm[0]))


@reference_binary
class TestTheDivergence:
    """Measured, printed, and not asserted."""

    def test_the_divergence_table(self, gsf_file: Path, workspace: Path) -> None:
        """Print what each shape differs by. Run with `-s` to read it."""
        print(
            f"\n  depth {DEPTH_KM} km, risetime {RISETIME_S} s, "
            f"depth factor {DEPTH_FACTOR:.4f}"
        )
        print(f"  the C's realised duration: {DEPTH_FACTOR * RISETIME_S:.6f} s\n")
        print(f"  {'stype':<12} {'C nt1':>6} {'port nt1':>9} {'worst relative':>15}")
        print(f"  {'-' * 12} {'-' * 6} {'-' * 9} {'-' * 15}")

        for name, shape in SHAPES.items():
            theirs = reference_pulse(gsf_file, workspace, name)
            ours = port_pulse(shape, DEPTH_FACTOR * RISETIME_S)
            overlap = min(len(theirs), len(ours))
            scale = max(float(np.abs(theirs).max()), 1e-30) if len(theirs) else 1.0
            worst = (
                float(np.abs(theirs[:overlap] - ours[:overlap]).max()) / scale
                if overlap
                else float("nan")
            )
            note = "  <- duration differs by design" if name in DURATION_DIFFERS else ""
            print(f"  {name:<12} {len(theirs):>6} {len(ours):>9} {worst:>15.4e}{note}")
        print()

    @pytest.mark.parametrize("stype", sorted(set(SHAPES) - DURATION_DIFFERS))
    def test_the_shapes_agree_to_the_file_s_own_precision(
        self, stype: str, gsf_file: Path, workspace: Path
    ) -> None:
        """Nine of the ten agree as closely as an SRF file can express.

        This *is* asserted, because it is not a tolerance -- it is the resolution of
        the format. An SRF writes `%13.5e`, six significant figures, so 1e-5 relative
        is the finest disagreement the comparison can see. Everything but `brune`
        lands there.

        A bound tight enough to be meaningful and loose enough to be honest: the
        `urs` bug this file found sat at 5.2e-01, five orders past it.
        """
        theirs = reference_pulse(gsf_file, workspace, stype)
        ours = port_pulse(SHAPES[stype], DEPTH_FACTOR * RISETIME_S)

        # One extra sample, always: `oliu_p` and `sampled` both close the pulse with a
        # forced zero where the C stops at the last computed sample.
        assert len(ours) - len(theirs) in (0, 1), (
            f"{stype}: {len(ours)} samples against the reference's {len(theirs)}"
        )

        overlap = min(len(theirs), len(ours))
        scale = float(np.abs(theirs).max())
        worst = float(np.abs(theirs[:overlap] - ours[:overlap]).max()) / scale
        assert worst < 1e-5, f"{stype} diverges by {worst:.3e}"

    def test_brune_differs_by_its_duration_and_says_so(
        self, gsf_file: Path, workspace: Path
    ) -> None:
        """The one deliberate shape difference, measured rather than described.

        `generic_slip2srf` sets brune's time constant from the subfault's *slip* --
        `0.1*e^-1*sqrt(slip)/1.2`, then multiplied by the depth factor -- where the
        port uses the rise time like every other shape. So the two pulses are the same
        function of different arguments, and the ratio of their lengths is the ratio
        of those arguments.

        Asserted as a *positive* claim about the size of the difference. A regression
        that quietly adopted the C's rule would make this test fail, which is the
        point: the choice should be hard to undo by accident.
        """
        theirs = reference_pulse(gsf_file, workspace, "brune")
        ours = port_pulse(SHAPES["brune"], DEPTH_FACTOR * RISETIME_S)

        c_time_constant = 0.1 * np.exp(-1.0) * np.sqrt(SLIP_CM) / 1.2 * DEPTH_FACTOR
        port_time_constant = DEPTH_FACTOR * RISETIME_S
        expected_ratio = port_time_constant / c_time_constant

        assert len(ours) / len(theirs) == pytest.approx(expected_ratio, rel=0.02), (
            f"brune's lengths are in the ratio {len(ours) / len(theirs):.3f}, not the "
            f"{expected_ratio:.3f} the two time constants imply"
        )

    def test_the_rise_times_differ_by_the_depth_factor(
        self, gsf_file: Path, workspace: Path
    ) -> None:
        """The other deliberate difference, and the one that is a pure scalar.

        The C treats `risetime` as the unstretched value, so its ramp only lengthens.
        The port treats it as the fault-wide average, so the ramp redistributes. On a
        single-depth fixture that is exactly `factor_at(depth)`, which is what this
        measures -- by handing the port the plain `risetime` and comparing lengths.
        """
        theirs = reference_pulse(gsf_file, workspace, "ucsb")
        ours = port_pulse(SHAPES["ucsb"], RISETIME_S)

        assert len(theirs) / len(ours) == pytest.approx(DEPTH_FACTOR, rel=0.02), (
            f"the C's pulse is {len(theirs) / len(ours):.4f} times longer, not the "
            f"{DEPTH_FACTOR:.4f} the depth factor implies"
        )


# Deliberately not asserted:
#
# - Onset. The C writes `inittime` at every subfault and the port solves for a front;
#   on a one-subfault fixture both are zero, so there is nothing to measure here, and
#   on a plane there is nothing to compare against. `point_source.rs` asserts the
#   agreement at one subfault and the front across a plane, which is the whole claim.
# - That `esg2006` agrees *because it is right*. It agrees on this build, and that is
#   all this can say: `gen_esg2006_stf` folds its normalisation into an uninitialised
#   `float` (`slip.c:342`), which on this compilation happened to start at zero. That
#   is a property of the build, not of the program -- see `DEFECTS.md` 20. A future
#   gcc, a different optimisation level or a different call site may all disagree, and
#   the port would still be right.
