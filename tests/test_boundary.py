"""Tests for the compiled boundary, `rupture_generator._core`.

The Rust side is pinned against genslip function by function; none of that is
repeated here. What these cover is what only exists at the boundary: that arrays
cross it with the right shape and dtype, that a seed reproduces a model *through*
it, that bad input is refused rather than mistranslated, and that the hand-written
stub still describes what it claims to.
"""

import ast
import enum
from pathlib import Path

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from rupture_generator import _core as core

STRIKE, DIP = 16, 10


def padded(count: int) -> int:
    """genslip's padded extent for a single-plane fault.

    `(int)(flen_max*extend_fac/flen * nstk)`, rounded up to even. On one plane
    `flen_max` clamps up to `flen`, so the ratio is exactly `extend_fac` and the whole
    rule collapses to this. `mapping.padded_extents` is the general form, checked
    against the binary; `test_mapping.py` asserts the two agree here, so this stays a
    simplification rather than becoming a second unchecked copy.

    It replaced `2 * ((strike + 4) // 2 + 1)`, which was not genslip's rule at all --
    it gave 26 for a 20-subfault fault where genslip reports `nstk2 = 22`. Nothing
    failed, because `FaultGrid` takes the padding as an argument, so it misled a
    reader rather than breaking anything.
    """
    scaled = int(1.10 * count)
    return scaled + 1 if scaled % 2 else scaled


def fault_grid(strike: int = STRIKE, dip: int = DIP) -> core.FaultGrid:
    """Build a fault whose depths span both rise-time ramps."""
    return core.FaultGrid(
        strike,
        dip,
        padded(strike),
        padded(dip),
        1.0,
        1.0,
        # One depth per subfault, along-strike fastest. This fault is a plane, so
        # every subfault in a dip row is at the same depth.
        depth_km=np.repeat(
            np.array([0.5 + i * 2.0 for i in range(dip)], dtype=np.float64), strike
        ),
        base_rake_deg=np.full(strike * dip, 175.0, dtype=np.float64),
        velocity_fraction=np.full(strike * dip, 0.8, dtype=np.float64),
    )


def velocity_model() -> core.VelocityModel1D:
    """A four-layer crustal model."""
    return core.VelocityModel1D(
        np.array([1.0, 5.0, 12.0, 30.0], dtype=np.float64),
        np.array([1.8, 2.6, 3.2, 3.6], dtype=np.float64),
        np.array([2.1, 2.4, 2.6, 2.7], dtype=np.float64),
    )


def timing_spec(**overrides) -> core.TimingSpec:
    """The configured timing defaults, with anything a test wants to move."""
    return core.TimingSpec(
        rupture_time_scale=-0.35,
        rise_time_blend=core.Ramp(2.0, 1.0),
        shallow_ramp=core.Ramp(6.5, 1.5),
        deep_ramp=core.Ramp(17.5, 2.5),
        beta_shallow_ramp=core.Ramp(2.0, 1.0),
        beta_mid_ramp=core.Ramp(6.5, 1.5),
        **overrides,
    )


def specs() -> tuple[core.SourceSpec, core.SlipSpec, core.TimingSpec]:
    """The configured defaults, as `root/defaults.yaml` sets them."""
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
        timing_spec(),
    )


def generate(
    seed: int = 20260807, realisation: int = 0, **kwargs
) -> core.GeneratedRupture:
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
            assert values.dtype == np.float64, name

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
                depth_km=np.zeros(STRIKE * DIP, dtype=np.float64),
                base_rake_deg=np.zeros(STRIKE * DIP, dtype=np.float64),
                velocity_fraction=np.full(STRIKE * DIP, 0.8, dtype=np.float64),
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
                depth_km=np.zeros(STRIKE * DIP, dtype=np.float64),
                base_rake_deg=np.zeros(STRIKE * DIP, dtype=np.float64),
                velocity_fraction=np.full(STRIKE * DIP, 0.8, dtype=np.float64),
            )

    def test_a_short_depth_array_is_refused(self) -> None:
        # Depth is one value per subfault now, not one per dip row, so what is short
        # here is a whole fault's worth rather than a column's.
        with pytest.raises(ValueError, match="depth_km"):
            core.FaultGrid(
                STRIKE,
                DIP,
                STRIKE + 4,
                DIP + 2,
                1.0,
                1.0,
                depth_km=np.zeros(STRIKE * DIP - 1, dtype=np.float64),
                base_rake_deg=np.zeros(STRIKE * DIP, dtype=np.float64),
                velocity_fraction=np.full(STRIKE * DIP, 0.8, dtype=np.float64),
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
                depth_km=np.zeros(STRIKE * DIP, dtype=np.float64),
                base_rake_deg=np.zeros(3, dtype=np.float64),
                velocity_fraction=np.full(STRIKE * DIP, 0.8, dtype=np.float64),
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
                np.array([1.0, 5.0], dtype=np.float64),
                np.array([1.8], dtype=np.float64),
                np.array([2.1, 2.4], dtype=np.float64),
            )

    def test_an_empty_velocity_model_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one layer"):
            core.VelocityModel1D(
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
            )

    def test_a_non_positive_wavelength_is_refused(self) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            core.SlipSpec(core.SpectrumModel.Mai, min_wavelength_km=0.0)


class TestTheVelocityModelReadsBack:
    """A model hands back the layers it was given.

    The property that makes the getters worth having: anything needing the layers can
    take the model rather than the model *and* its three arrays. The harness used to
    take both, and two descriptions of the same layers are two things that can drift.
    """

    LAYERS = (
        np.array([1.0, 5.0, 12.0, 30.0], dtype=np.float64),
        np.array([1.8, 2.6, 3.2, 3.6], dtype=np.float64),
        np.array([2.1, 2.4, 2.6, 2.7], dtype=np.float64),
    )

    def test_the_arrays_come_back_unchanged(self) -> None:
        model = core.VelocityModel1D(*self.LAYERS)
        for built_from, read_back in zip(
            self.LAYERS,
            (model.bottom_depth_km, model.shear_speed_km_s, model.density_g_cm3),
            strict=True,
        ):
            assert np.array_equal(read_back, built_from)
            assert read_back.dtype == np.float64

    def test_a_model_round_trips_through_its_own_constructor(self) -> None:
        first = core.VelocityModel1D(*self.LAYERS)
        second = core.VelocityModel1D(
            first.bottom_depth_km, first.shear_speed_km_s, first.density_g_cm3
        )
        assert np.array_equal(second.bottom_depth_km, first.bottom_depth_km)
        assert np.array_equal(second.shear_speed_km_s, first.shear_speed_km_s)
        assert np.array_equal(second.density_g_cm3, first.density_g_cm3)

    def test_writing_to_what_comes_back_does_not_reach_the_model(self) -> None:
        # A view would let this corrupt a model that is validated on construction.
        model = core.VelocityModel1D(*self.LAYERS)
        model.bottom_depth_km[0] = -99.0
        assert model.bottom_depth_km[0] == pytest.approx(1.0)

    def test_the_length_is_the_layer_count(self) -> None:
        assert len(core.VelocityModel1D(*self.LAYERS)) == 4


class TestThePointSourcePath:
    """The other entry point, which draws nothing.

    The physics is pinned on the Rust side in `point_source.rs`; what these cover is
    what only exists at the boundary. The one exception is determinism, which is
    asserted on both sides deliberately: it is the property the whole path is built
    around, and a boundary that quietly reintroduced a seed would satisfy every Rust
    test.
    """

    @staticmethod
    def spec() -> core.PointSourceSpec:
        return core.PointSourceSpec(
            5.2, 0.35, average_dip_deg=60.0, average_rake_deg=175.0
        )

    @classmethod
    def generate(cls, shape: core.SlipRateShape | None = None) -> core.GeneratedRupture:
        _, _, timing = specs()
        if shape is not None:
            timing = timing_spec(slip_rate_shape=shape)
        return core.generate_point_source(
            fault_grid(),
            velocity_model(),
            cls.spec(),
            timing,
            hypocentre_strike=STRIKE // 2,
            hypocentre_dip=DIP // 2,
        )

    def test_the_arrays_have_one_value_per_subfault(self) -> None:
        rupture = self.generate()
        assert rupture.slip_cm.shape == (STRIKE * DIP,)
        assert rupture.rake_deg.shape == (STRIKE * DIP,)
        assert rupture.onset_s.shape == (STRIKE * DIP,)
        assert rupture.shape == (STRIKE, DIP)
        assert rupture.slip_rate_offsets.shape == (STRIKE * DIP + 1,)

    def test_there_is_no_seed_to_pass(self) -> None:
        # The signature is the claim. A point source that accepted a seed would be
        # advertising randomness it does not have.
        with pytest.raises(TypeError):
            core.generate_point_source(
                fault_grid(),
                velocity_model(),
                self.spec(),
                specs()[2],
                seed=1,
                hypocentre_strike=0,
                hypocentre_dip=0,
            )

    def test_it_is_deterministic_across_the_boundary(self) -> None:
        first, second = self.generate(), self.generate()
        for field in ("slip_cm", "rake_deg", "onset_s", "rise_time_s", "slip_rate"):
            assert np.array_equal(getattr(first, field), getattr(second, field))

    def test_a_hypocentre_off_the_fault_is_refused(self) -> None:
        with pytest.raises(ValueError, match="outside"):
            core.generate_point_source(
                fault_grid(),
                velocity_model(),
                self.spec(),
                specs()[2],
                hypocentre_strike=STRIKE,
                hypocentre_dip=0,
            )

    def test_a_non_positive_rise_time_is_refused(self) -> None:
        with pytest.raises(ValueError, match="rise_time_s"):
            core.PointSourceSpec(5.2, 0.0, average_dip_deg=60.0, average_rake_deg=175.0)

    @pytest.mark.parametrize(
        "shape",
        [
            core.SlipRateShape.oliu_p2(),
            core.SlipRateShape.ucsb(),
            core.SlipRateShape.ucsb2(),
            core.SlipRateShape.ucsb_t(2.0),
            core.SlipRateShape.ucsb_var_t1(0.2),
            core.SlipRateShape.brune(),
            core.SlipRateShape.urs(),
            core.SlipRateShape.esg2006(),
            core.SlipRateShape.cos(),
            core.SlipRateShape.seki(),
            core.SlipRateShape.delta(),
        ],
    )
    def test_every_shape_crosses_the_boundary(self, shape: core.SlipRateShape) -> None:
        rupture = self.generate(shape)
        assert rupture.slip_rate_offsets[-1] > 0
        assert np.all(np.isfinite(rupture.slip_rate))


class TestTheStypeVocabulary:
    """`generic_slip2srf`'s option strings, parsed in one place.

    The C dispatches with `strncmp`, so `ucsb-T` takes a numeric suffix and anything
    unrecognised falls through to `brune` -- silently generating a different rupture
    from the one that was asked for. Parsing here raises instead.
    """

    @pytest.mark.parametrize(
        "stype",
        [
            "brune",
            "delta",
            "esg2006",
            "urs",
            "ucsb",
            "ucsb2",
            "ucsb-varT1",
            "cos",
            "seki",
        ],
    )
    def test_every_name_the_c_accepts_parses(self, stype: str) -> None:
        assert core.SlipRateShape.from_stype(stype) is not None

    def test_the_ucsb_t_suffix_is_a_number(self) -> None:
        assert core.SlipRateShape.from_stype("ucsb-T0.2") == core.SlipRateShape.ucsb_t(
            0.2
        )
        # Bare `ucsb-T` is a stretch of one, which is plain `ucsb`.
        assert core.SlipRateShape.from_stype("ucsb-T") == core.SlipRateShape.ucsb_t(1.0)

    def test_an_unrecognised_name_is_refused_rather_than_defaulted(self) -> None:
        with pytest.raises(ValueError, match="not a slip-rate function"):
            core.SlipRateShape.from_stype("OliuP")

    def test_a_non_numeric_suffix_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must be a number"):
            core.SlipRateShape.from_stype("ucsb-Twelve")

    def test_a_non_positive_stretch_is_refused(self) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            core.SlipRateShape.ucsb_t(0.0)


class TestStubMatchesTheExtension:
    """The stub is hand-written, so nothing keeps it honest but this.

    Names *and* members. Checking only the top-level names let a class gain a getter
    in Rust and keep a stub that did not describe it, which is exactly what happened
    to `VelocityModel1D`: it had a constructor and nothing else for long enough that
    the harness grew a workaround for reading a model back.
    """

    @staticmethod
    def stub_tree() -> ast.Module:
        return ast.parse((Path(core.__file__).parent / "_core.pyi").read_text())

    @classmethod
    def stub_names(cls) -> set[str]:
        return {
            node.name
            for node in cls.stub_tree().body
            if isinstance(node, ast.ClassDef | ast.FunctionDef)
        }

    @classmethod
    def stub_members(cls, class_name: str) -> set[str]:
        """Public attributes, methods and annotated fields a stub class declares."""
        for node in cls.stub_tree().body:
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                members = set()
                for statement in node.body:
                    if isinstance(statement, ast.FunctionDef):
                        members.add(statement.name)
                    elif isinstance(statement, ast.AnnAssign) and isinstance(
                        statement.target, ast.Name
                    ):
                        members.add(statement.target.id)
                    elif isinstance(statement, ast.Assign):
                        # `Somerville = ...`, how the stub spells an enum variant.
                        members.update(
                            target.id
                            for target in statement.targets
                            if isinstance(target, ast.Name)
                        )
                return {name for name in members if not name.startswith("_")}
        raise AssertionError(f"{class_name} is not in the stub")

    @staticmethod
    def runtime_members(compiled: type) -> set[str]:
        """The same, from the compiled class.

        `dir()` on a PyO3 type carries `object`'s inheritance, and on an enum carries
        `enum`'s machinery, so both are subtracted rather than filtered by name --
        a getter called `name` or `value` would otherwise be invisible.
        """
        inherited = set(dir(object))
        if isinstance(compiled, enum.EnumMeta):
            inherited |= set(dir(enum.Enum))
        return {
            name
            for name in dir(compiled)
            if not name.startswith("_") and name not in inherited
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

    @pytest.mark.parametrize(
        "class_name",
        [
            "Ramp",
            "FaultGrid",
            "VelocityModel1D",
            "SourceSpec",
            "PointSourceSpec",
            "SlipSpec",
            "TimingSpec",
            "GeneratedRupture",
            "SpectrumModel",
            "RiseTimeWeighting",
            "SlipRateShape",
            "Projected",
            "Plane",
            "Fault",
            "PointSource",
            "Cuts",
            "RefinedMesh",
        ],
    )
    def test_every_class_describes_the_members_it_has(self, class_name: str) -> None:
        described = self.stub_members(class_name)
        actual = self.runtime_members(getattr(core, class_name))
        assert described == actual, (
            f"{class_name}: stub-only {sorted(described - actual)}, "
            f"undocumented {sorted(actual - described)}"
        )

    def test_the_parametrisation_covers_every_stub_class(self) -> None:
        """So a new class cannot be added to the stub and skip the member check."""
        checked = {
            case
            for mark in self.test_every_class_describes_the_members_it_has.pytestmark
            for case in mark.args[1]
        }
        stub_classes = {
            node.name
            for node in self.stub_tree().body
            if isinstance(node, ast.ClassDef)
        }
        assert checked == stub_classes

    def test_every_enum_variant_exists(self) -> None:
        for enum_name, variants in [
            (
                "SpectrumModel",
                [
                    "Somerville",
                    "Mai",
                    "Frankel",
                    "MaiSomerville",
                    "Suzuki",
                    "InputCorners",
                ],
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
def test_any_fault_gives_one_value_per_subfault(
    strike: int, dip: int, seed: int
) -> None:
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
