"""Tests for the compiled boundary, `rupture_generator._core`.

The Rust side is pinned against genslip function by function; none of that is
repeated here. What these cover is what only exists at the boundary: that arrays
cross it with the right shape and dtype, that a seed reproduces a model *through*
it, that bad input is refused rather than mistranslated, and that the hand-written
stub still describes what it claims to.
"""

import ast
from pathlib import Path

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from rupture_generator import _core as core

STRIKE, DIP = 16, 10


def fault_grid(strike: int = STRIKE, dip: int = DIP) -> core.FaultGrid:
    """Build a fault whose depths span both rise-time ramps.

    The padded extents are rounded up to even, as genslip rounds its own.
    """
    return core.FaultGrid(
        strike,
        dip,
        2 * ((strike + 4) // 2 + 1),
        2 * ((dip + 2) // 2 + 1),
        1.0,
        1.0,
        depth_km=np.array([0.5 + i * 2.0 for i in range(dip)], dtype=np.float32),
        base_rake_deg=np.full(strike * dip, 175.0, dtype=np.float32),
        velocity_fraction=np.full(strike * dip, 0.8, dtype=np.float32),
    )


def velocity_model() -> core.VelocityModel1D:
    """A four-layer crustal model."""
    return core.VelocityModel1D(
        np.array([1.0, 5.0, 12.0, 30.0], dtype=np.float32),
        np.array([1.8, 2.6, 3.2, 3.6], dtype=np.float32),
        np.array([2.1, 2.4, 2.6, 2.7], dtype=np.float32),
    )


def specs() -> tuple[core.SourceSpec, core.SlipSpec, core.TimingSpec]:
    """The configured defaults, as `root/defaults.yaml` sets them."""
    shallow, deep = core.Ramp(6.5, 1.5), core.Ramp(17.5, 2.5)
    return (
        core.SourceSpec(
            6.5,
            core.SpectrumModel.Mai,
            2.50,
            1.50,
            average_dip_deg=60.0,
            average_rake_deg=175.0,
        ),
        core.SlipSpec(core.SpectrumModel.Mai),
        core.TimingSpec(
            rupture_time_scale=-0.35,
            rise_time_blend=core.Ramp(2.0, 1.0),
            shallow_ramp=shallow,
            deep_ramp=deep,
            beta_shallow_ramp=core.Ramp(2.0, 1.0),
            beta_mid_ramp=core.Ramp(6.5, 1.5),
        ),
    )


def generate(seed: int = 20260807, realisation: int = 0, **kwargs) -> core.GeneratedRupture:
    """Generate a model with the configured defaults."""
    source, slip, timing = specs()
    return core.generate_rupture(
        kwargs.pop("grid", fault_grid()),
        velocity_model(),
        source,
        slip,
        timing,
        seed=seed,
        realisation=realisation,
        hypocentre_strike=STRIKE // 2,
        hypocentre_dip=DIP // 2,
        **kwargs,
    )


class TestArraysCrossTheBoundary:
    def test_every_field_is_one_value_per_subfault(self) -> None:
        rupture = generate()
        for name in ("slip_cm", "rake_deg", "onset_s", "rise_time_s"):
            values = getattr(rupture, name)
            assert values.shape == (STRIKE * DIP,), name
            assert values.dtype == np.float32, name

    def test_the_shape_is_reported_and_reshapes(self) -> None:
        rupture = generate()
        assert rupture.shape == (STRIKE, DIP)
        # Along-strike index fastest, so the grid is (dip, strike) in C order.
        assert rupture.slip_grid().shape == (DIP, STRIKE)
        assert rupture.slip_grid().flatten() == pytest.approx(rupture.slip_cm)

    def test_the_slip_rate_offsets_index_the_pulses(self) -> None:
        rupture = generate()
        offsets = rupture.slip_rate_offsets
        assert offsets.dtype == np.uint64
        # One longer than the subfault count, starting at zero and ending at the
        # total: that is what makes it a csr_array indptr.
        assert offsets.shape == (STRIKE * DIP + 1,)
        assert offsets[0] == 0
        assert offsets[-1] == len(rupture.slip_rate)
        assert np.all(np.diff(offsets.astype(np.int64)) >= 0)

    def test_each_pulse_integrates_to_its_subfault_slip(self) -> None:
        # The one physical claim that spans the whole boundary: what came back as a
        # pulse and what came back as a slip value describe the same subfault.
        rupture = generate()
        offsets = rupture.slip_rate_offsets
        for index in range(0, STRIKE * DIP, 7):
            start, end = int(offsets[index]), int(offsets[index + 1])
            if start == end:
                continue
            integral = rupture.slip_rate[start:end].sum() * rupture.sample_interval_s
            assert integral == pytest.approx(rupture.slip_cm[index], rel=1e-3)


class TestReproducibility:
    def test_a_seed_reproduces_a_model(self) -> None:
        first, second = generate(seed=4242), generate(seed=4242)
        assert np.array_equal(first.slip_cm, second.slip_cm)
        assert np.array_equal(first.onset_s, second.onset_s)
        assert np.array_equal(first.slip_rate, second.slip_rate)

    def test_the_realisation_index_selects_a_different_model(self) -> None:
        assert not np.array_equal(
            generate(seed=4242, realisation=0).slip_cm,
            generate(seed=4242, realisation=1).slip_cm,
        )

    def test_a_realisation_does_not_depend_on_the_ones_before_it(self) -> None:
        # What makes a campaign restartable: realisation 7 is the same whether or not
        # 0 through 6 were generated first.
        direct = generate(seed=99, realisation=7).slip_cm
        for index in range(7):
            generate(seed=99, realisation=index)
        assert np.array_equal(direct, generate(seed=99, realisation=7).slip_cm)


class TestRefusesBadInput:
    def test_an_odd_padded_extent_is_refused(self) -> None:
        # It would otherwise panic three layers down, inside the spectrum
        # constructor, with a Rust PanicException rather than a ValueError.
        with pytest.raises(ValueError, match="must be even"):
            core.FaultGrid(
                STRIKE,
                DIP,
                STRIKE + 5,
                DIP + 2,
                1.0,
                1.0,
                depth_km=np.zeros(DIP, dtype=np.float32),
                base_rake_deg=np.zeros(STRIKE * DIP, dtype=np.float32),
                velocity_fraction=np.full(STRIKE * DIP, 0.8, dtype=np.float32),
            )

    def test_a_fault_larger_than_its_padding_is_refused(self) -> None:
        with pytest.raises(ValueError, match="does not fit"):
            core.FaultGrid(
                STRIKE,
                DIP,
                4,
                4,
                1.0,
                1.0,
                depth_km=np.zeros(DIP, dtype=np.float32),
                base_rake_deg=np.zeros(STRIKE * DIP, dtype=np.float32),
                velocity_fraction=np.full(STRIKE * DIP, 0.8, dtype=np.float32),
            )

    def test_a_short_depth_array_is_refused(self) -> None:
        with pytest.raises(ValueError, match="needs one per row"):
            core.FaultGrid(
                STRIKE,
                DIP,
                STRIKE + 4,
                DIP + 2,
                1.0,
                1.0,
                depth_km=np.zeros(DIP - 1, dtype=np.float32),
                base_rake_deg=np.zeros(STRIKE * DIP, dtype=np.float32),
                velocity_fraction=np.full(STRIKE * DIP, 0.8, dtype=np.float32),
            )

    def test_a_short_per_subfault_array_is_refused(self) -> None:
        with pytest.raises(ValueError, match="base_rake_deg"):
            core.FaultGrid(
                STRIKE,
                DIP,
                STRIKE + 4,
                DIP + 2,
                1.0,
                1.0,
                depth_km=np.zeros(DIP, dtype=np.float32),
                base_rake_deg=np.zeros(3, dtype=np.float32),
                velocity_fraction=np.full(STRIKE * DIP, 0.8, dtype=np.float32),
            )

    def test_a_hypocentre_off_the_fault_is_refused(self) -> None:
        source, slip, timing = specs()
        with pytest.raises(ValueError, match="outside a"):
            core.generate_rupture(
                fault_grid(),
                velocity_model(),
                source,
                slip,
                timing,
                seed=1,
                hypocentre_strike=STRIKE,
                hypocentre_dip=0,
            )

    def test_a_ragged_velocity_model_is_refused(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            core.VelocityModel1D(
                np.array([1.0, 5.0], dtype=np.float32),
                np.array([1.8], dtype=np.float32),
                np.array([2.1, 2.4], dtype=np.float32),
            )

    def test_an_empty_velocity_model_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one layer"):
            core.VelocityModel1D(
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
                np.array([], dtype=np.float32),
            )

    def test_a_non_positive_wavelength_is_refused(self) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            core.SlipSpec(core.SpectrumModel.Mai, min_wavelength_km=0.0)


class TestStubMatchesTheExtension:
    """The stub is hand-written, so nothing keeps it honest but this."""

    @staticmethod
    def stub_names() -> set[str]:
        source = (Path(core.__file__).parent / "_core.pyi").read_text()
        tree = ast.parse(source)
        return {
            node.name
            for node in tree.body
            if isinstance(node, ast.ClassDef | ast.FunctionDef)
        }

    def test_every_public_member_is_in_the_stub(self) -> None:
        exported = {name for name in dir(core) if not name.startswith("_")}
        missing = exported - self.stub_names()
        assert not missing, f"exported but undocumented: {sorted(missing)}"

    def test_the_stub_describes_nothing_that_is_not_exported(self) -> None:
        # Type aliases are stub-only by nature and are excluded by name.
        described = self.stub_names()
        exported = {name for name in dir(core) if not name.startswith("_")}
        extra = described - exported
        assert not extra, f"documented but not exported: {sorted(extra)}"

    def test_every_enum_variant_exists(self) -> None:
        for enum_name, variants in [
            (
                "SpectrumModel",
                ["Somerville", "Mai", "Frankel", "MaiSomerville", "Suzuki", "InputCorners"],
            ),
            ("RiseTimeWeighting", ["Uniform", "BySlip", "BySlipAndRuptureSpeed"]),
        ]:
            cls = getattr(core, enum_name)
            for variant in variants:
                assert hasattr(cls, variant), f"{enum_name}.{variant}"


@given(
    strike=st.integers(min_value=2, max_value=20),
    dip=st.integers(min_value=2, max_value=20),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_any_fault_gives_one_value_per_subfault(strike: int, dip: int, seed: int) -> None:
    """Shape is a function of the fault alone, whatever the seed."""
    source, slip, timing = specs()
    rupture = core.generate_rupture(
        fault_grid(strike, dip),
        velocity_model(),
        source,
        slip,
        timing,
        seed=seed,
        hypocentre_strike=strike // 2,
        hypocentre_dip=dip // 2,
    )
    assert rupture.slip_cm.shape == (strike * dip,)
    assert rupture.shape == (strike, dip)
    assert rupture.slip_rate_offsets.shape == (strike * dip + 1,)


@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_any_seed_reproduces(seed: int) -> None:
    assert np.array_equal(generate(seed=seed).slip_cm, generate(seed=seed).slip_cm)


# Deliberately not asserted:
#
# - Any of the physics. Every stage is pinned against genslip on the Rust side, and
#   repeating a weaker version of that here would be coverage theatre.
# - That runtime signatures match the stub. PyO3 does not expose real signatures for
#   keyword-only arguments, so the check would be against `(*args, **kwargs)` and
#   would pass whatever the stub said.
