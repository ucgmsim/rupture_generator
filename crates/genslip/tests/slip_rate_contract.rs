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

use genslip::slip_rate::oliu_p;

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

// Deliberately not asserted:
//
// - That the pulse is non-negative everywhere. It nearly is, but the piecewise
//   sinusoid is not constrained to be, and asserting it would be asserting something
//   about the shape the original does not guarantee.
// - Anything about the other seven `stype` generators. Only `OliuP2` is configured
//   and the rest are unported -- see `PRUNED.md`.
