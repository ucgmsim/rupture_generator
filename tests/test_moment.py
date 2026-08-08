"""The moment rate function, against the one identity that makes it checkable.

Moment is not an emergent quantity here. The generator *scales the slip field to hit a
target*, so the moment a rupture carries is a number that was chosen, and
`ENGINEERING_RULES.md` classes it as **exact, to the f64 fold** -- the only entry in the
tolerance table that is not a bound.

That gives this module a reference no plotting code could have: the integral of the
moment rate must be the moment the generator was told to produce. Nothing here compares
against a stored curve or a second transcription of the sum.

# What the tolerance is, and why it is not exact

Two things stand between the identity and equality, and both are quantisation rather
than error.

Onsets are placed at the sample nearest them, so a pulse can move by up to half a sample
-- which shifts the curve without changing its area, and so does not enter here at all.

The pulses themselves are what the slip-rate generator returned, sampled at `dt`, and
each integrates to its subfault's slip only to the accuracy of a Riemann sum over its own
length. That is a property of the discretisation the SRF stores, not of this code, and it
is what the 1e-3 relative bound covers -- against a slip bound of 1e-2 and a moment that
is otherwise exact.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from rupture_generator import _core
from rupture_generator.moment import (
    cumulative_moment,
    moment_rate,
    rigidity_dyne_cm2,
)
from rupture_generator.units import CM2_PER_KM2

STRIKE, DIP = 16, 10
SUBFAULTS = STRIKE * DIP


def padded_extent(count: int) -> int:
    """A wraparound margin the spectral generators accept: bigger, and even."""
    return count + 4 + (count % 2)


SETTINGS = settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

INTEGRAL = 1e-3
"""How close the integral gets. See the module note: pulse quantisation, not error."""


def fault_grid(strike: int = STRIKE, dip: int = DIP) -> _core.FaultGrid:
    depths = np.repeat(
        np.array([0.5 + index * 2.0 for index in range(dip)], dtype=np.float64), strike
    )
    return _core.FaultGrid(
        strike,
        dip,
        padded_extent(strike),
        padded_extent(dip),
        1.0,
        1.0,
        depth_km=depths,
        base_rake_deg=np.full(strike * dip, 175.0, dtype=np.float64),
        velocity_fraction=np.full(strike * dip, 0.8, dtype=np.float64),
    )


SHEAR_SPEED_KM_S = np.array([1.8, 2.6, 3.2, 3.6], dtype=np.float64)
DENSITY_G_CM3 = np.array([2.1, 2.4, 2.6, 2.7], dtype=np.float64)


def velocity_model() -> _core.VelocityModel1D:
    return _core.VelocityModel1D(
        np.array([1.0, 5.0, 12.0, 30.0], dtype=np.float64),
        SHEAR_SPEED_KM_S,
        DENSITY_G_CM3,
    )


def materials(strike: int = STRIKE, dip: int = DIP) -> tuple[np.ndarray, np.ndarray]:
    """Area and rigidity per subfault, sampled the way the core samples them.

    A 1 km by 1 km subfault, and the layer each depth falls in -- the same clamping
    `VelocityModel1D.layer_at` does, transcribed here because the boundary does not
    expose it. Deliberately a transcription and not a second implementation of the
    *moment*: what it computes is the material, and the identity under test is the sum.
    """
    bottoms = np.array([1.0, 5.0, 12.0, 30.0])
    depths = np.repeat(
        np.array([0.5 + index * 2.0 for index in range(dip)], dtype=np.float64), strike
    )
    layer = np.minimum(np.searchsorted(bottoms, depths, side="left"), len(bottoms) - 1)
    area_cm2 = np.full(strike * dip, 1.0 * 1.0 * CM2_PER_KM2)
    return area_cm2, rigidity_dyne_cm2(SHEAR_SPEED_KM_S[layer], DENSITY_G_CM3[layer])


def specs(
    magnitude: float = 6.8, shape: _core.SlipRateShape | None = None
) -> tuple[_core.SourceSpec, _core.SlipSpec, _core.TimingSpec]:
    ramp = _core.Ramp
    return (
        _core.SourceSpec(
            magnitude,
            _core.SpectrumModel.Mai,
            2.50,
            1.50,
            average_dip_deg=60.0,
            average_rake_deg=175.0,
        ),
        _core.SlipSpec(_core.SpectrumModel.Mai),
        _core.TimingSpec(
            rupture_time_scale=-0.35,
            rise_time_blend=ramp(2.0, 1.0),
            shallow_ramp=ramp(6.5, 1.5),
            deep_ramp=ramp(17.5, 2.5),
            beta_shallow_ramp=ramp(2.0, 1.0),
            beta_mid_ramp=ramp(6.5, 1.5),
            slip_rate_shape=shape,
        ),
    )


def pulse_lengths(rupture: _core.GeneratedRupture) -> np.ndarray:
    """How many samples each subfault's pulse has. Zero means no pulse at all."""
    return np.diff(np.asarray(rupture.slip_rate_offsets, dtype=np.int64))


def a_rupture(
    magnitude: float = 6.8,
    seed: int = 1234,
    shape: _core.SlipRateShape | None = None,
    strike: int = STRIKE,
    dip: int = DIP,
) -> _core.GeneratedRupture:
    source, slip, timing = specs(magnitude, shape)
    return _core.generate_rupture(
        fault_grid(strike, dip),
        velocity_model(),
        source,
        slip,
        timing,
        seed=seed,
        hypocentre_strike=strike // 2,
        hypocentre_dip=dip // 2,
    )


class TestTheIntegralIsTheMoment:
    """The identity the whole module rests on."""

    def test_it_recovers_the_moment_the_generator_was_scaled_to(self) -> None:
        rupture = a_rupture()
        area_cm2, rigidity = materials()
        times_s, rate = moment_rate(rupture, area_cm2, rigidity)

        assert cumulative_moment(times_s, rate)[-1] == pytest.approx(
            rupture.moment_dyne_cm, rel=INTEGRAL
        )

    @pytest.mark.parametrize("magnitude", [5.5, 6.0, 6.8, 7.5, 8.2])
    def test_it_holds_across_four_orders_of_moment(self, magnitude: float) -> None:
        """A relative bound has to hold where the number is large as well as small.

        M5.5 to M8.2 is a factor of 1e4 in moment. A bug that added a constant, or that
        lost a subfault, would show at one end and not the other.
        """
        rupture = a_rupture(magnitude=magnitude)
        area_cm2, rigidity = materials()
        times_s, rate = moment_rate(rupture, area_cm2, rigidity)

        assert cumulative_moment(times_s, rate)[-1] == pytest.approx(
            rupture.moment_dyne_cm, rel=INTEGRAL
        )

    @given(seed=st.integers(min_value=1, max_value=100_000))
    @SETTINGS
    def test_it_holds_for_any_realisation(self, seed: int) -> None:
        """Every drawn slip field, not the one that happened to be checked."""
        rupture = a_rupture(seed=seed)
        area_cm2, rigidity = materials()
        times_s, rate = moment_rate(rupture, area_cm2, rigidity)

        assert cumulative_moment(times_s, rate)[-1] == pytest.approx(
            rupture.moment_dyne_cm, rel=INTEGRAL
        )

    @pytest.mark.parametrize(
        "shape_name",
        [
            "oliu_p2",
            "ucsb",
            "ucsb2",
            "brune",
            "urs",
            "esg2006",
            "seki",
            "delta",
        ],
    )
    def test_it_holds_for_every_pulse_shape(self, shape_name: str) -> None:
        """The pulse shapes differ in length, peak and where their area sits.

        `seki` shifts its own onset and `delta` puts everything in one sample; if the sum
        depended on the shape rather than on the area, those two would say so.

        `cos` is absent and that is not an oversight -- it drops subfaults it cannot
        represent, which is `DEFECTS.md` 20, and
        `TestCosDropsSubfaultsItCannotRepresent` measures exactly how much.
        """
        shape = getattr(_core.SlipRateShape, shape_name)()
        rupture = a_rupture(shape=shape)
        area_cm2, rigidity = materials()
        times_s, rate = moment_rate(rupture, area_cm2, rigidity)

        assert cumulative_moment(times_s, rate)[-1] == pytest.approx(
            rupture.moment_dyne_cm, rel=INTEGRAL
        )

    @given(
        strike=st.integers(min_value=2, max_value=20),
        dip=st.integers(min_value=2, max_value=12),
    )
    @SETTINGS
    def test_it_holds_at_any_extent(self, strike: int, dip: int) -> None:
        """Including non-square faults, where a transposed index would show."""
        rupture = a_rupture(strike=strike, dip=dip)
        area_cm2, rigidity = materials(strike, dip)
        times_s, rate = moment_rate(rupture, area_cm2, rigidity)

        assert cumulative_moment(times_s, rate)[-1] == pytest.approx(
            rupture.moment_dyne_cm, rel=INTEGRAL
        )


class TestTheCurveIsPhysical:
    def test_it_is_non_negative_everywhere(self) -> None:
        """Slip rate is a speed in the rake direction. Nothing runs backwards."""
        rupture = a_rupture()
        _, rate = moment_rate(rupture, *materials())
        assert np.all(rate >= 0.0)

    def test_it_starts_and_ends_at_nothing(self) -> None:
        """A rupture that has not begun releases no moment, and one that has finished
        releases no more."""
        rupture = a_rupture()
        _, rate = moment_rate(rupture, *materials())
        assert rate[0] == pytest.approx(0.0, abs=1e-6 * rate.max())
        assert rate[-1] == pytest.approx(0.0, abs=1e-6 * rate.max())

    def test_the_cumulative_curve_never_falls(self) -> None:
        rupture = a_rupture()
        times_s, rate = moment_rate(rupture, *materials())
        assert np.all(np.diff(cumulative_moment(times_s, rate)) >= 0.0)

    def test_it_starts_at_the_first_onset(self) -> None:
        """Rather than at zero, so a delayed rupture has no run of leading zeros."""
        rupture = a_rupture()
        times_s, _ = moment_rate(rupture, *materials())
        assert times_s[0] == pytest.approx(rupture.onset_s.min(), abs=1e-12)

    def test_it_lasts_about_as_long_as_the_rupture(self) -> None:
        """The timeline covers every pulse and does not run far past the last one."""
        rupture = a_rupture()
        times_s, _ = moment_rate(rupture, *materials())
        last_pulse_s = rupture.onset_s.max() + rupture.rise_time_s.max()

        assert times_s[-1] >= rupture.onset_s.max()
        assert times_s[-1] < last_pulse_s + 10.0


class TestScaling:
    """What the sum is linear in, asserted rather than assumed."""

    @pytest.mark.parametrize("factor", [0.5, 2.0, 10.0])
    def test_it_is_linear_in_area(self, factor: float) -> None:
        rupture = a_rupture()
        area_cm2, rigidity = materials()
        _, plain = moment_rate(rupture, area_cm2, rigidity)
        _, scaled = moment_rate(rupture, area_cm2 * factor, rigidity)
        assert scaled == pytest.approx(plain * factor, rel=1e-12)

    @pytest.mark.parametrize("factor", [0.5, 2.0, 10.0])
    def test_it_is_linear_in_rigidity(self, factor: float) -> None:
        rupture = a_rupture()
        area_cm2, rigidity = materials()
        _, plain = moment_rate(rupture, area_cm2, rigidity)
        _, scaled = moment_rate(rupture, area_cm2, rigidity * factor)
        assert scaled == pytest.approx(plain * factor, rel=1e-12)

    def test_rigidity_is_about_thirty_gigapascals_for_crustal_rock(self) -> None:
        """The check a reader can make without trusting any of the arithmetic.

        3.2 km/s and 2.6 g/cm^3 is 27 GPa, which is what crustal rock is. Getting the
        exponent wrong by one is a factor of ten in the moment -- a whole magnitude
        unit, from a constant nobody would look at twice. `units.py` makes the same
        argument about the same number.
        """
        pascals = float(rigidity_dyne_cm2(np.array([3.2]), np.array([2.6]))[0]) * 0.1
        assert pascals == pytest.approx(27e9, rel=0.05)


class TestRefusals:
    @pytest.mark.parametrize("name", ["area_cm2", "rigidity_dyne_cm2"])
    def test_a_material_array_of_the_wrong_length_is_refused(self, name: str) -> None:
        """A short array would silently sum the wrong subfaults."""
        rupture = a_rupture()
        area_cm2, rigidity = materials()
        arrays = {"area_cm2": area_cm2, "rigidity_dyne_cm2": rigidity}
        arrays[name] = arrays[name][:-1]

        with pytest.raises(ValueError, match=name):
            moment_rate(rupture, arrays["area_cm2"], arrays["rigidity_dyne_cm2"])


class TestSilentSubfaultsContributeNothing:
    def test_a_fault_with_tapered_edges_still_integrates(self) -> None:
        """Edge subfaults are tapered to nothing and get no pulse at all.

        `nt1 = 0` and no samples, which is not the same as a pulse of zeros -- and on a
        tapered fault it is every edge subfault. A sum that treated a missing pulse as a
        present one would read past the end of the concatenated samples.
        """
        rupture = a_rupture()
        lengths = np.diff(np.asarray(rupture.slip_rate_offsets, dtype=np.int64))
        assert (lengths == 0).any(), "no silent subfaults, so this proves nothing"

        times_s, rate = moment_rate(rupture, *materials())
        assert cumulative_moment(times_s, rate)[-1] == pytest.approx(
            rupture.moment_dyne_cm, rel=INTEGRAL
        )


class TestCosDropsSubfaultsItCannotRepresent:
    """`DEFECTS.md` 20, pinned so its size cannot move quietly.

    A raised cosine is `1 - cos(2*pi*t/T)`, which is **zero at t = 0**. A pulse whose
    duration is the one-sample rise-time floor therefore has a single sample of zero,
    `normalise` sees an integral of zero, and the subfault gets no pulse at all --
    indistinguishable downstream from one that did not slip.

    Found by the integral identity above: every other shape closes to 1e-3 and `cos`
    was 1.0% short. The shortfall is not noise, and this says what it is.
    """

    def _dropped(self) -> tuple[_core.GeneratedRupture, np.ndarray]:
        """The rupture, and which subfaults `cos` silences that `oliu_p2` does not."""
        reference = a_rupture(shape=_core.SlipRateShape.oliu_p2())
        cosine = a_rupture(shape=_core.SlipRateShape.cos())
        assert np.array_equal(reference.slip_cm, cosine.slip_cm), (
            "the shapes drew different slip, so nothing below compares like with like"
        )

        return cosine, np.where(
            (pulse_lengths(reference) > 0) & (pulse_lengths(cosine) == 0)
        )[0]

    def test_it_drops_subfaults_that_slip_substantially(self) -> None:
        """Not the `MINSLIP` guard of defect 16: these subfaults slip metres."""
        cosine, dropped = self._dropped()
        assert len(dropped) > 0, (
            "cos dropped nothing, so defect 20 is gone -- update it"
        )
        assert cosine.slip_cm[dropped].min() > 100.0

    def test_the_dropped_subfaults_are_at_the_rise_time_floor(self) -> None:
        """Which is the mechanism: one sample, at t = 0, where the cosine is zero."""
        cosine, dropped = self._dropped()
        assert cosine.rise_time_s[dropped] == pytest.approx(
            cosine.sample_interval_s, rel=1e-9
        )

    def test_the_missing_moment_is_exactly_what_they_carry(self) -> None:
        """The shortfall is accounted for, not merely bounded.

        If the integral were short for any *other* reason, this would fail -- which is
        what makes it a measurement of the defect rather than a tolerance around it.
        """
        cosine, dropped = self._dropped()
        area_cm2, rigidity = materials()
        times_s, rate = moment_rate(cosine, area_cm2, rigidity)

        integrated = cumulative_moment(times_s, rate)[-1]
        carried = float(
            (cosine.slip_cm[dropped] * area_cm2[dropped] * rigidity[dropped]).sum()
        )

        assert integrated + carried == pytest.approx(
            cosine.moment_dyne_cm, rel=INTEGRAL
        )

    def test_it_is_about_one_percent_on_this_fixture(self) -> None:
        """The number `DEFECTS.md` quotes. If it drifts, the entry is stale."""
        cosine, dropped = self._dropped()
        area_cm2, rigidity = materials()
        carried = float(
            (cosine.slip_cm[dropped] * area_cm2[dropped] * rigidity[dropped]).sum()
        )

        assert carried / cosine.moment_dyne_cm == pytest.approx(0.00996, abs=1e-4)

    @pytest.mark.parametrize(
        "shape_name",
        ["oliu_p2", "ucsb", "ucsb2", "brune", "urs", "esg2006", "seki", "delta"],
    )
    def test_no_other_shape_drops_anything(self, shape_name: str) -> None:
        """`cos` is alone in being zero at t = 0, so it should be alone in this."""
        reference = a_rupture(shape=_core.SlipRateShape.oliu_p2())
        other = a_rupture(shape=getattr(_core.SlipRateShape, shape_name)())

        dropped = np.where(
            (pulse_lengths(reference) > 0) & (pulse_lengths(other) == 0)
        )[0]
        assert len(dropped) == 0, f"{shape_name} also drops {dropped.tolist()}"
