//! The slip-rate function reproduces genslip's, including its degenerate cases.
#![cfg(feature = "wavefront-compat")]

use genslip::slip_rate::oliu_p;
use genslip_oracle::field as oracle;
use proptest::prelude::*;

const MAX_SAMPLES: usize = 100_000;

fn check(slip: f32, duration_s: f32, beta: f32, dt: f32, label: &str) {
    let expected = oracle::oliu_p_slip_rate(slip, duration_s, beta, dt, MAX_SAMPLES);
    let produced = oliu_p(slip, duration_s, beta, dt, MAX_SAMPLES);

    assert_eq!(
        produced.len(),
        expected.len(),
        "{label}: {} samples vs {}",
        produced.len(),
        expected.len()
    );
    for (index, (got, want)) in produced.as_slice().iter().zip(&expected).enumerate() {
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "{label}: mismatch at sample {index}: {got} vs {want}"
        );
    }
}

#[test]
fn the_default_pulse_matches() {
    // beta 0.5 shallow through 0.13 deep, rise times spanning what the depth
    // stretch produces, at the configured 5 ms sampling.
    for beta in [0.5_f32, 0.3, 0.13, 0.1] {
        for duration_s in [0.2_f32, 0.5, 1.0, 2.4, 8.0] {
            check(
                150.0,
                duration_s,
                beta,
                0.005,
                &format!("beta {beta}, duration {duration_s}"),
            );
        }
    }
}

#[test]
fn a_pulse_shorter_than_half_a_sample_produces_nothing() {
    // The subfault contributes no moment at all. Reproduced rather than rounded up,
    // because rounding up would add moment the model did not ask for.
    let produced = oliu_p(150.0, 0.002, 0.5, 0.005, MAX_SAMPLES);
    assert!(produced.is_empty());

    let expected = oracle::oliu_p_slip_rate(150.0, 0.002, 0.5, 0.005, MAX_SAMPLES);
    assert!(expected.is_empty());
}

#[test]
fn a_pulse_of_about_one_sample_is_a_fixed_spike() {
    // Too short to resolve a shape, so the original substitutes a three-point
    // triangle rather than computing one. Checked against the C, which is the only
    // way to be sure of the sample count.
    check(150.0, 0.006, 0.5, 0.005, "one-sample pulse");

    let produced = oliu_p(150.0, 0.006, 0.5, 0.005, MAX_SAMPLES);
    assert_eq!(produced.len(), 3);
}

#[test]
fn the_integral_is_the_slip() {
    // The property that makes the moment come out right whatever the shape. Held to
    // a relative tolerance because the normalisation is a single-precision fold.
    for beta in [0.5_f32, 0.13] {
        for duration_s in [0.2_f32, 1.0, 4.0] {
            let slip = 250.0_f32;
            let dt = 0.005_f32;
            let produced = oliu_p(slip, duration_s, beta, dt, MAX_SAMPLES);
            let integral: f32 = produced.as_slice().iter().map(|value| value * dt).sum();
            assert!(
                (integral - slip).abs() < slip * 1e-4,
                "beta {beta}, duration {duration_s}: integral {integral}, slip {slip}"
            );
        }
    }
}

#[test]
fn the_pulse_starts_and_ends_at_zero() {
    let produced = oliu_p(150.0, 1.0, 0.3, 0.005, MAX_SAMPLES);
    assert_eq!(produced.as_slice().first(), Some(&0.0));
    assert_eq!(produced.as_slice().last(), Some(&0.0));
}

#[test]
fn a_smaller_beta_gives_a_more_impulsive_pulse() {
    // What beta means physically: it sets how much of the duration the rising limb
    // occupies, so a smaller value concentrates the slip earlier. Stated as a
    // property because the parity tests say nothing about what the shape is FOR.
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

proptest! {
    #[test]
    fn the_pulse_matches_for_arbitrary_parameters(
        slip in 0.1f32..1000.0,
        duration_s in 0.001f32..20.0,
        beta in 0.05f32..0.95,
        dt in 0.001f32..0.05,
    ) {
        let expected = oracle::oliu_p_slip_rate(slip, duration_s, beta, dt, MAX_SAMPLES);
        let produced = oliu_p(slip, duration_s, beta, dt, MAX_SAMPLES);

        prop_assert_eq!(produced.len(), expected.len());
        for (index, (got, want)) in produced.as_slice().iter().zip(&expected).enumerate() {
            prop_assert_eq!(got.to_bits(), want.to_bits(), "at sample {}", index);
        }
    }
}

// Deliberately not asserted:
//
// - That the pulse is non-negative everywhere. It nearly is, but the piecewise
//   sinusoid is not constrained to be, and asserting it would be asserting something
//   about the shape that the original does not guarantee.
// - Anything about the other seven `stype` generators. Only `OliuP2` is configured,
//   and the rest are unported -- see `PRUNED.md`.
