//! What a slip-rate pulse must be, rather than what its samples happen to be.
//!
//! These were the non-parity half of `slip_rate_parity.rs`, which compared samples
//! against the C bit for bit and said nothing about what the shape is *for*. The
//! parity half is gone; this is what it was protecting.
//!
//! Liu, Archuleta & Hartzell (2006): a piecewise sinusoid whose rising limb occupies
//! a `beta` fraction of the duration, normalised so its integral is the slip. Nothing
//! below depends on the pulse being *that* function — a different source-time
//! function that conserves moment, starts and ends at rest, and gets more impulsive
//! as `beta` falls would satisfy all of it. That is the point: the contract is the
//! physics, not the formula.

use genslip::slip_rate::{SlipRateShape, oliu_p};

mod common;
use common::tolerance::pulse_round_trip;

/// genslip's `nt_max`. Large enough never to bind here.
const MAX_SAMPLES: usize = 100_000;

/// The integral is the slip. This is what makes the moment come out right.
///
/// Held to a *derived* bound rather than a chosen one: the normalisation is a
/// single-precision fold over the samples, so the round-trip error grows with the
/// pulse length. The detection floor is about 3e-6 relative — one wrong sample in a
/// thousand of a smooth pulse.
#[test]
fn the_integral_is_the_slip() {
    let slip = 250.0_f32;
    let dt = 0.005_f32;

    for beta in [0.5_f32, 0.13] {
        for duration_s in [0.2_f32, 1.0, 4.0] {
            let pulse = oliu_p(slip, duration_s, beta, dt, MAX_SAMPLES);
            let integral: f64 = pulse
                .as_slice()
                .iter()
                .map(|value| f64::from(*value) * f64::from(dt))
                .sum();

            let bound = f64::from(slip) * pulse_round_trip(pulse.len());
            assert!(
                (integral - f64::from(slip)).abs() < bound,
                "beta {beta}, duration {duration_s}: integral {integral} against slip \
                 {slip}, past a bound of {bound}"
            );
        }
    }
}

/// A pulse begins and ends at rest.
///
/// Not decoration: a source-time function that starts at a non-zero rate is a step in
/// velocity, which radiates energy at every frequency and is not what the model means.
#[test]
fn the_pulse_starts_and_ends_at_zero() {
    let pulse = oliu_p(150.0, 1.0, 0.3, 0.005, MAX_SAMPLES);
    assert_eq!(pulse.as_slice().first(), Some(&0.0));
    assert_eq!(pulse.as_slice().last(), Some(&0.0));
}

/// A smaller `beta` peaks earlier.
///
/// What the parameter means physically: it sets how much of the duration the rising
/// limb occupies, so a smaller value concentrates the slip earlier and makes the
/// pulse more impulsive. Nothing in a parity test says this.
#[test]
fn a_smaller_beta_gives_a_more_impulsive_pulse() {
    let dt = 0.005_f32;
    let peak_time = |beta: f32| {
        let pulse = oliu_p(150.0, 1.0, beta, dt, MAX_SAMPLES);
        let (index, _) = pulse
            .as_slice()
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .expect("a non-empty pulse");
        #[expect(clippy::cast_precision_loss, reason = "small sample indices")]
        let time = index as f32 * dt;
        time
    };

    assert!(
        peak_time(0.13) < peak_time(0.5),
        "a smaller beta should peak earlier"
    );
}

/// A pulse shorter than half a sample produces nothing at all.
///
/// The subfault contributes no moment. Reproduced rather than rounded up to one
/// sample, because rounding up would add moment the model did not ask for — and the
/// alternative, a pulse whose integral is the slip crammed into one sample, is a
/// spike of arbitrary amplitude.
#[test]
fn a_pulse_shorter_than_half_a_sample_produces_nothing() {
    assert!(oliu_p(150.0, 0.002, 0.5, 0.005, MAX_SAMPLES).is_empty());
}

/// Around one sample, the shape is a fixed spike rather than a computed curve.
///
/// Too short to resolve a sinusoid, so the original substitutes three points. Pinned
/// because it is a discontinuity in behaviour: a rewrite that computed the shape here
/// instead would change the shortest pulses on every fault without changing anything
/// a smooth-field test looks at.
#[test]
fn a_pulse_of_about_one_sample_is_a_fixed_spike() {
    let pulse = oliu_p(150.0, 0.006, 0.5, 0.005, MAX_SAMPLES);
    assert_eq!(pulse.len(), 3, "expected the three-point substitute");
    assert_eq!(pulse.as_slice().first(), Some(&0.0));
    assert_eq!(pulse.as_slice().last(), Some(&0.0));
}

/// Longer pulses carry the same slip at a lower rate.
///
/// The scaling that ties rise time to amplitude. A pulse twice as long must peak at
/// roughly half the rate, or the moment would not be conserved across the fault's
/// depth-dependent rise times.
#[test]
fn a_longer_pulse_is_a_gentler_one() {
    let peak = |duration_s: f32| {
        oliu_p(150.0, duration_s, 0.3, 0.005, MAX_SAMPLES)
            .as_slice()
            .iter()
            .copied()
            .fold(0.0_f32, f32::max)
    };

    let short = peak(0.5);
    let long = peak(2.0);
    assert!(
        long < short,
        "a four-times-longer pulse peaked at {long}, not below {short}"
    );
    // Roughly inverse: four times the duration, about a quarter the rate. Loose,
    // because the shape is not a rectangle.
    assert!(
        (short / long) > 2.0 && (short / long) < 8.0,
        "the peak ratio {} is not close to the duration ratio of 4",
        short / long
    );
}

/// The ucsb family **is** `oliu_p`, and the claim is an identity rather than an
/// agreement.
///
/// `generic_slip2srf` writes the same three-piece sinusoid four times over 220 lines
/// (`slip.c:114-336`), moving the breakpoints each time. Each is `oliu_p` at a
/// different `(duration, beta)`, so the port has one function and four aliases. That
/// is only worth doing if it is exactly true, so it is asserted exactly: every
/// computed sample equal bit for bit, not within a tolerance.
///
/// `oliu_p` appends one closing zero the C does not, which is the whole of the
/// difference and is checked here rather than glossed. A trailing zero contributes
/// nothing to the integral, so the normalisation -- and therefore every sample before
/// it -- is untouched.
mod the_ucsb_family_is_oliu_p {
    use super::*;

    /// Slips and durations spanning the range a subfault sees, and then some.
    const SLIPS: [f32; 4] = [0.5, 12.0, 250.0, 4000.0];
    const DURATIONS: [f32; 5] = [0.09, 0.2, 1.0, 4.0, 30.0];
    const DT: f32 = 0.005;

    /// `pulse` is `expected` with one zero appended, and nothing else.
    fn assert_alias(shape: SlipRateShape, slip: f32, duration_s: f32, expected_beta: f32) {
        let pulse = shape.pulse(slip, duration_s, f32::NAN, DT, MAX_SAMPLES);
        let expected = oliu_p(slip, duration_s, expected_beta, DT, MAX_SAMPLES);

        // `beta` is passed as NaN deliberately: only `OliuP2` reads it, and a shape
        // that leaked it into its arithmetic would produce NaN samples here rather
        // than plausible ones.
        assert_eq!(
            pulse.as_slice(),
            expected.as_slice(),
            "{shape:?} at slip {slip}, duration {duration_s} is not oliu_p at \
             beta {expected_beta}"
        );
    }

    #[test]
    fn ucsb_is_beta_of_thirteen_hundredths() {
        for slip in SLIPS {
            for duration_s in DURATIONS {
                assert_alias(SlipRateShape::Ucsb, slip, duration_s, 0.13);
            }
        }
    }

    #[test]
    fn ucsb2_doubles_the_duration_and_halves_beta() {
        // The C's comment is "keep peak at same place": `tau = 2*t0` with
        // `tau1 = 0.5*0.13*tau` puts `tau1` back at `0.13*t0`.
        for slip in SLIPS {
            for duration_s in DURATIONS {
                let pulse = SlipRateShape::Ucsb2.pulse(slip, duration_s, f32::NAN, DT, MAX_SAMPLES);
                let expected = oliu_p(slip, 2.0 * duration_s, 0.065, DT, MAX_SAMPLES);
                assert_eq!(pulse.as_slice(), expected.as_slice());
            }
        }
    }

    #[test]
    fn ucsb_t_stretches_the_duration_and_leaves_the_peak() {
        for stretch in [0.5_f32, 1.0, 2.0, 3.7] {
            for duration_s in DURATIONS {
                let pulse = SlipRateShape::UcsbT { stretch }.pulse(
                    250.0,
                    duration_s,
                    f32::NAN,
                    DT,
                    MAX_SAMPLES,
                );
                let expected = oliu_p(250.0, stretch * duration_s, 0.13 / stretch, DT, MAX_SAMPLES);
                assert_eq!(pulse.as_slice(), expected.as_slice());
            }
        }
    }

    #[test]
    fn a_stretch_of_one_is_plain_ucsb() {
        // Not redundant with the two above: it is the claim that the `ucsb-T` option
        // string degenerates to the shape it is a generalisation of, which is what
        // makes one function rather than two correct.
        for duration_s in DURATIONS {
            let stretched =
                SlipRateShape::UcsbT { stretch: 1.0 }.pulse(250.0, duration_s, f32::NAN, DT, 4096);
            let plain = SlipRateShape::Ucsb.pulse(250.0, duration_s, f32::NAN, DT, 4096);
            assert_eq!(stretched.as_slice(), plain.as_slice());
        }
    }

    #[test]
    fn var_t1_is_beta_straight_through_and_defaults_to_ucsb() {
        for tau1_ratio in [0.05_f32, 0.13, 0.3, 0.5] {
            for duration_s in DURATIONS {
                let pulse = SlipRateShape::UcsbVarT1 { tau1_ratio }.pulse(
                    250.0,
                    duration_s,
                    f32::NAN,
                    DT,
                    MAX_SAMPLES,
                );
                let expected = oliu_p(250.0, duration_s, tau1_ratio, DT, MAX_SAMPLES);
                assert_eq!(pulse.as_slice(), expected.as_slice());
            }
        }

        // The C reads `tau1_ratio` from the input file's thirteenth column and
        // substitutes 0.13 when it is absent, which is `ucsb`.
        let defaulted =
            SlipRateShape::UcsbVarT1 { tau1_ratio: 0.13 }.pulse(250.0, 1.0, f32::NAN, DT, 4096);
        let ucsb = SlipRateShape::Ucsb.pulse(250.0, 1.0, f32::NAN, DT, 4096);
        assert_eq!(defaulted.as_slice(), ucsb.as_slice());
    }

    #[test]
    fn oliu_p2_is_the_one_shape_that_reads_beta() {
        // The other four carry their own, which is why the field value can be NaN
        // above. This is the negative half of that claim.
        let from_field = SlipRateShape::OliuP2.pulse(250.0, 1.0, 0.4, DT, MAX_SAMPLES);
        let direct = oliu_p(250.0, 1.0, 0.4, DT, MAX_SAMPLES);
        assert_eq!(from_field.as_slice(), direct.as_slice());

        let poisoned = SlipRateShape::OliuP2.pulse(250.0, 1.0, f32::NAN, DT, MAX_SAMPLES);
        assert!(
            poisoned.as_slice().iter().any(|value| value.is_nan()),
            "OliuP2 ignored the beta it is supposed to read"
        );
    }

    /// The one place the port and `generic_slip2srf` differ, stated as a fact.
    ///
    /// Recorded rather than hidden inside a tolerance: `oliu_p` is genslip's
    /// `gen_OliuP_stf`, which closes the pulse with a forced zero; the ucsb family
    /// stops at the last computed sample. So a port pulse is one sample longer, and
    /// that sample is zero.
    #[test]
    fn the_extra_sample_the_c_does_not_write_is_a_zero() {
        for duration_s in DURATIONS {
            let pulse = SlipRateShape::Ucsb.pulse(250.0, duration_s, f32::NAN, DT, MAX_SAMPLES);
            let samples = pulse.as_slice();

            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "matching the C's truncation of a small non-negative count"
            )]
            let c_count = (duration_s / DT + 0.5) as usize;
            assert_eq!(samples.len(), c_count + 1, "at duration {duration_s}");
            assert_eq!(
                samples.last(),
                Some(&0.0),
                "the extra sample is not zero, so it is not free"
            );
        }
    }
}

// Deliberately not asserted:
//
// - That the pulse is non-negative everywhere. It nearly is, but the piecewise
//   sinusoid is not constrained to be, and asserting it would be asserting something
//   about the shape the original does not guarantee.
// - That the ucsb aliases agree with `generic_slip2srf`'s *output*. They agree with
//   `oliu_p`, which is a claim this file can settle on its own; agreeing with the C
//   is a claim about a binary, and it lives in the reference comparison instead.
