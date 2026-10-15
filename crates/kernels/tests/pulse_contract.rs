//! What a slip-rate pulse must be, rather than what its samples happen to be.
//!
//! Liu, Archuleta & Hartzell (2006): a piecewise sinusoid whose rising limb occupies
//! a `beta` fraction of the duration, normalised so its integral is the slip. Nothing
//! below depends on the pulse being *that* function. A different source-time function
//! would satisfy all of it if it conserved slip and started and ended at rest, and if
//! it grew more impulsive as `beta` fell. The properties test the physics, not the
//! formula. They quantify over generated `(slip, rise time, dt, beta)`, including rise
//! times *below* what `dt` can represent, because what happens to those, an
//! instantaneous slip within one sample, belongs to the contract as much as the pulse
//! does.

mod common;

use _kernels::pulse::{self, CsrPulses, Error, MIN_SLIP_M, Shape};
use common::{Subfaults, adversarial_subfaults, exact, resolvable_subfaults};
use proptest::prelude::*;

fn synthesise(subfaults: &Subfaults) -> Result<CsrPulses, Error> {
    pulse::synthesise_pulses(
        &subfaults.slip_m,
        &subfaults.rise_time_s,
        Shape::OliuP {
            beta: &subfaults.beta,
        },
        subfaults.dt_s,
    )
}

fn row(csr: &CsrPulses, subfault: usize) -> &[f64] {
    &csr.samples[csr.offsets[subfault]..csr.offsets[subfault + 1]]
}

/// Relative bound on `|dt·Σ − slip|` for a renormalised pulse of `n` samples.
///
/// The renormalisation makes the integral the slip *by construction*, and only
/// rounding remains. The rounding comes from the `f64` fold (each add rounds at ≤ half
/// an ulp, and modelled as independent that's ~`ε·√n/3` relative) and from the common
/// scale factor (~2ε, systematic, which doesn't average away). The factor of four
/// covers the tails; at 400 samples the bound is ~2e-14, so "exactly" is the right
/// word for it.
fn integral_round_trip(samples: usize) -> f64 {
    4.0 * f64::EPSILON * (exact(samples).sqrt() + 2.0)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(96))]

    /// The CSR is well-formed, empty exactly where the slip is negligible, and every
    /// pulse integrates to its subfault's slip.
    ///
    /// `dt·Σ = slip` is the one thing both shapes promise, and it's what makes the
    /// moment come out right whichever shape the caller picks.
    #[test]
    fn every_slipping_subfault_conserves_its_slip(subfaults in resolvable_subfaults()) {
        let csr = synthesise(&subfaults).expect("finite rise times never refuse");

        prop_assert_eq!(csr.offsets.len(), subfaults.slip_m.len() + 1);
        prop_assert_eq!(csr.offsets[0], 0);
        prop_assert_eq!(*csr.offsets.last().unwrap(), csr.samples.len());
        prop_assert!(csr.offsets.windows(2).all(|pair| pair[0] <= pair[1]));

        for (subfault, &slip) in subfaults.slip_m.iter().enumerate() {
            let samples = row(&csr, subfault);
            if slip.abs() <= MIN_SLIP_M {
                prop_assert!(
                    samples.is_empty(),
                    "subfault {} does not slip but got {} samples",
                    subfault, samples.len()
                );
                continue;
            }
            prop_assert!(!samples.is_empty());
            let integral: f64 = samples.iter().map(|sample| sample * subfaults.dt_s).sum();
            prop_assert!(
                (integral - slip).abs() <= integral_round_trip(samples.len()) * slip,
                "subfault {} integrates to {} m, not {} m",
                subfault, integral, slip
            );
        }
    }

    /// A pulse's support is its rise time, to the half-sample the rounding allows.
    ///
    /// The closing zero isn't support (it exists so the pulse ends at rest), and the
    /// support is `(len − 1)·dt`. The sample count rounds `rise/dt` to nearest, which
    /// gives the half-sample tolerance.
    #[test]
    fn the_support_is_the_rise_time(subfaults in resolvable_subfaults()) {
        let csr = synthesise(&subfaults).expect("resolvable rise times never refuse");
        for (subfault, &slip) in subfaults.slip_m.iter().enumerate() {
            if slip.abs() <= MIN_SLIP_M {
                continue;
            }
            let support = exact(row(&csr, subfault).len() - 1) * subfaults.dt_s;
            let rise = subfaults.rise_time_s[subfault];
            prop_assert!(
                (support - rise).abs() <= 0.5 * subfaults.dt_s + 1e-9 * rise,
                "subfault {}: support {} s against a rise time of {} s at dt {}",
                subfault, support, rise, subfaults.dt_s
            );
        }
    }

    /// A pulse starts and ends at rest.
    ///
    /// Not decoration: a source-time function that starts or ends at a non-zero
    /// rate is a step in velocity, which radiates energy at every frequency and is
    /// not what the model means.
    #[test]
    fn every_pulse_starts_and_ends_at_rest(subfaults in resolvable_subfaults()) {
        let csr = synthesise(&subfaults).expect("resolvable rise times never refuse");
        for (subfault, &slip) in subfaults.slip_m.iter().enumerate() {
            if slip.abs() <= MIN_SLIP_M {
                continue;
            }
            let samples = row(&csr, subfault);
            prop_assert_eq!(samples.first(), Some(&0.0), "subfault {} starts moving", subfault);
            prop_assert_eq!(samples.last(), Some(&0.0), "subfault {} never stops", subfault);
        }
    }

    /// Every subfault whose slip exceeds the guard keeps all of it, whatever its rise
    /// time, and a rise time under half a sample slips instantaneously.
    ///
    /// genslip floors every rise time under `dt` to `dt`, a one-sample pulse. Here a
    /// rise time that rounds to no samples gets the same three-point spike as one that
    /// rounds to one. The kernel neither drops nor refuses a slipping subfault. Over
    /// rise times generated down to zero, each slipping subfault conserves its slip,
    /// and one safely under half a sample (0.4 of one, clear of the round-to-nearest
    /// boundary) is exactly the spike.
    #[test]
    fn a_slipping_subfault_is_never_dropped(subfaults in adversarial_subfaults()) {
        let csr = synthesise(&subfaults).expect("finite rise times never refuse");
        for (subfault, &slip) in subfaults.slip_m.iter().enumerate() {
            let samples = row(&csr, subfault);
            prop_assert_eq!(
                samples.is_empty(),
                slip.abs() <= MIN_SLIP_M,
                "subfault {} with slip {} m and rise {} s at dt {}",
                subfault, slip, subfaults.rise_time_s[subfault], subfaults.dt_s
            );
            if slip.abs() <= MIN_SLIP_M {
                continue;
            }
            let integral: f64 = samples.iter().map(|sample| sample * subfaults.dt_s).sum();
            prop_assert!(
                (integral - slip).abs() <= integral_round_trip(samples.len()) * slip.abs(),
                "subfault {} integrates to {} m, not {} m",
                subfault, integral, slip
            );
            if subfaults.rise_time_s[subfault] < 0.4 * subfaults.dt_s {
                prop_assert_eq!(
                    samples,
                    &[0.0, slip / subfaults.dt_s, 0.0][..],
                    "subfault {} at rise {} s, dt {} s",
                    subfault, subfaults.rise_time_s[subfault], subfaults.dt_s
                );
            }
        }
    }

    /// `delta` is the impulse `[0, slip/dt, 0]`, whatever the rise time says.
    ///
    /// Exactly the spike `oliu_p_into` substitutes for a pulse too short to resolve, and
    /// this is that branch under its own name. It never reads the rise time, so it
    /// never refuses one.
    #[test]
    fn delta_is_the_impulse_whatever_the_rise_time(subfaults in adversarial_subfaults()) {
        let csr = pulse::synthesise_pulses(
            &subfaults.slip_m,
            &subfaults.rise_time_s,
            Shape::Delta,
            subfaults.dt_s,
        )
        .expect("delta represents every rise time");

        let ones = vec![1.0; subfaults.slip_m.len()];
        let ignoring_rise =
            pulse::synthesise_pulses(&subfaults.slip_m, &ones, Shape::Delta, subfaults.dt_s)
                .expect("delta represents every rise time");
        prop_assert_eq!(&csr, &ignoring_rise, "delta read the rise time");

        for (subfault, &slip) in subfaults.slip_m.iter().enumerate() {
            if slip.abs() > MIN_SLIP_M {
                prop_assert_eq!(
                    row(&csr, subfault),
                    &[0.0, slip / subfaults.dt_s, 0.0][..],
                    "subfault {}",
                    subfault
                );
            } else {
                prop_assert!(row(&csr, subfault).is_empty());
            }
        }
    }
}

/// A smaller `beta` peaks earlier.
///
/// The parameter sets how much of the duration the rising limb occupies. A smaller
/// value concentrates the slip earlier and makes the pulse more impulsive. Deterministic because discrete peak positions move in
/// sample-sized steps: the claim needs betas far enough apart to be visible at the
/// fixture's resolution.
#[test]
fn a_smaller_beta_gives_a_more_impulsive_pulse() {
    let dt_s = 0.005;
    let peak_time = |beta: f64| {
        let csr = pulse::synthesise_pulses(&[1.5], &[1.0], Shape::OliuP { beta: &[beta] }, dt_s)
            .expect("a one-second rise time at 5 ms is comfortably representable");
        let (index, _) = csr
            .samples
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .expect("a non-empty pulse");
        exact(index) * dt_s
    };
    assert!(
        peak_time(0.13) < peak_time(0.5),
        "a smaller beta should peak earlier"
    );
}

/// Around one sample and below it, the shape is a fixed spike rather than a computed
/// curve, and the spike is exactly what `delta` produces.
///
/// Pinned because it's a discontinuity in behaviour: computing the sinusoid here
/// instead would change the shortest pulses on every fault without changing
/// anything a smooth-field test looks at.
#[test]
fn a_pulse_of_about_one_sample_is_the_delta_spike() {
    let (slip, dt_s) = (2.5, 0.005);
    let oliu = pulse::synthesise_pulses(&[slip], &[0.006], Shape::OliuP { beta: &[0.3] }, dt_s)
        .expect("one sample is representable — as the spike");
    let delta = pulse::synthesise_pulses(&[slip], &[0.006], Shape::Delta, dt_s)
        .expect("delta represents every rise time");
    assert_eq!(oliu.samples, delta.samples);
    assert_eq!(oliu.samples, vec![0.0, slip / dt_s, 0.0]);
}

/// A rise time under half a sample, down to exactly zero, slips instantaneously: the
/// subfault gets the one-sample spike and keeps all of its slip, as genslip's floor
/// to `dt` gives it. The subfaults around it keep their computed pulses.
#[test]
fn a_rise_time_under_half_a_sample_slips_instantaneously() {
    let dt_s = 0.005;
    for rise in [0.2 * dt_s, 0.0] {
        let slip = 3.18;
        let csr = pulse::synthesise_pulses(
            &[1.0, slip, 1.0],
            &[1.0, rise, 1.0],
            Shape::OliuP {
                beta: &[0.2, 0.2, 0.2],
            },
            dt_s,
        )
        .expect("a short rise time is an instantaneous slip, not an error");
        let spike = row(&csr, 1);
        assert_eq!(spike, &[0.0, slip / dt_s, 0.0][..], "rise {rise} s");
        let integral: f64 = spike.iter().map(|sample| sample * dt_s).sum();
        assert!(
            (integral - slip).abs() <= integral_round_trip(spike.len()) * slip,
            "rise {rise} s integrates to {integral} m, not {slip} m"
        );
        for subfault in [0, 2] {
            assert_eq!(row(&csr, subfault).len(), 201, "subfault {subfault}");
        }
    }
}

/// The integral keeps the slip's sign: a back-slipping subfault conserves its slip
/// too, without rectification.
#[test]
fn a_negative_slip_is_conserved_with_its_sign() {
    let (slip, dt_s) = (-2.0, 0.005);
    let csr = pulse::synthesise_pulses(&[slip], &[1.0], Shape::OliuP { beta: &[0.2] }, dt_s)
        .expect("the sign of slip is not the pulse's business");
    let integral: f64 = csr.samples.iter().map(|sample| sample * dt_s).sum();
    assert!(
        (integral - slip).abs() <= integral_round_trip(csr.samples.len()) * slip.abs(),
        "{integral} m against {slip} m"
    );
}

// ---------------------------------------------------------------------------------
// Refusals: the kernel names bad inputs rather than synthesising around them
// ---------------------------------------------------------------------------------

/// A slipping subfault's rise time must be a duration. A negative or non-finite one
/// means something failed upstream, and the error gives the subfault's index rather
/// than slipping it instantaneously. The kernel ignores a subfault that doesn't slip.
#[test]
fn a_rise_time_that_is_no_duration_is_refused_by_subfault() {
    for bad in [-0.001, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let error = pulse::synthesise_pulses(
            &[0.0, 1.0, 1.0],
            &[bad, 1.0, bad],
            Shape::OliuP {
                beta: &[0.2, 0.2, 0.2],
            },
            0.005,
        )
        .expect_err("a slipping subfault's rise time must be a duration");
        assert!(
            matches!(error, Error::RiseTimeOutOfRange { subfault: 2, .. }),
            "rise = {bad} gave {error}"
        );
        assert!(error.to_string().contains("subfault 2"), "{error}");
    }
}

#[test]
fn a_degenerate_sample_interval_is_refused() {
    for bad in [0.0, -0.005, f64::NAN, f64::INFINITY] {
        let error = pulse::synthesise_pulses(&[1.0], &[1.0], Shape::OliuP { beta: &[0.2] }, bad)
            .expect_err("dt is not an interval");
        assert!(
            matches!(error, Error::NonPositiveSampleInterval { .. }),
            "dt = {bad} gave {error}"
        );
    }
}

#[test]
fn mismatched_arrays_are_refused_by_name() {
    let error = pulse::synthesise_pulses(
        &[1.0, 2.0],
        &[1.0],
        Shape::OliuP { beta: &[0.2, 0.2] },
        0.005,
    )
    .expect_err("one rise time for two subfaults");
    assert_eq!(
        error,
        Error::MismatchedLengths {
            field: "rise_time_s",
            expected: 2,
            got: 1
        }
    );

    let error = pulse::synthesise_pulses(
        &[1.0, 2.0],
        &[1.0, 1.0],
        Shape::OliuP { beta: &[0.2] },
        0.005,
    )
    .expect_err("one beta for two subfaults");
    assert!(matches!(
        error,
        Error::MismatchedLengths { field: "beta", .. }
    ));
}

/// `beta` beyond a half would let the sinusoid's second piece overrun the duration.
/// Zero would divide the rising limb by nothing. The kernel refuses both by subfault,
/// even when that subfault doesn't slip: a bad parameter is a bad parameter.
#[test]
fn a_beta_outside_its_range_is_refused_by_subfault() {
    for bad in [0.0, -0.1, 0.51, 1.0, f64::NAN] {
        let error = pulse::synthesise_pulses(
            &[1.0, 1.0],
            &[1.0, 1.0],
            Shape::OliuP { beta: &[0.2, bad] },
            0.005,
        )
        .expect_err("beta must be in (0, 0.5]");
        assert!(
            matches!(error, Error::BetaOutOfRange { subfault: 1, .. }),
            "beta = {bad} gave {error}"
        );
    }
}

#[test]
fn a_non_finite_slip_is_refused_by_subfault() {
    for bad in [f64::NAN, f64::INFINITY] {
        let error = pulse::synthesise_pulses(
            &[1.0, bad],
            &[1.0, 1.0],
            Shape::OliuP { beta: &[0.2, 0.2] },
            0.005,
        )
        .expect_err("slip must be finite");
        assert!(
            matches!(error, Error::NonFiniteSlip { subfault: 1, .. }),
            "slip = {bad} gave {error}"
        );
    }
}
