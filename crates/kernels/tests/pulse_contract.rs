//! What a slip-rate pulse must be, rather than what its samples happen to be.
//!
//! Liu, Archuleta & Hartzell (2006): a piecewise sinusoid whose rising limb occupies
//! a `beta` fraction of the duration, normalised so its integral is the slip. Nothing
//! below depends on the pulse being *that* function — a different source-time
//! function that conserves slip, starts and ends at rest, and gets more impulsive as
//! `beta` falls would satisfy all of it. That is the point: the contract is the
//! physics, not the formula. Quantified over generated `(slip, rise time, dt, beta)`
//! per `PLAN.md` §6, including rise times *below* what `dt` can represent, because
//! the refusal is as much the contract as the pulse.

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
/// The renormalisation makes the integral the slip *by construction*; what is left
/// is rounding. Two sources: the `f64` fold (each add rounds at ≤ half an ulp;
/// modelled as independent that is ~`ε·√n/3` relative) and the common scale factor
/// (~2ε, systematic — it does not average away). The factor of four covers the
/// tails; at 400 samples the bound is ~2e-14, eleven orders below the 1% slip
/// tolerance of the old bounds table, so "exactly" is the right word for it.
fn integral_round_trip(samples: usize) -> f64 {
    4.0 * f64::EPSILON * (exact(samples).sqrt() + 2.0)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(96))]

    /// The CSR is well-formed, empty exactly where the slip is negligible, and every
    /// pulse integrates to its subfault's slip.
    ///
    /// `dt·Σ = slip` is the one thing both shapes promise, and it is what makes the
    /// moment come out right whichever is chosen.
    #[test]
    fn every_slipping_subfault_conserves_its_slip(subfaults in resolvable_subfaults()) {
        let csr = synthesise(&subfaults).expect("resolvable rise times never refuse");

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
    /// The closing zero is not support — it exists so the pulse ends at rest — so
    /// the support is `(len − 1)·dt`, and the sample count rounds `rise/dt` to
    /// nearest, hence the half-sample tolerance.
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

    /// Every subfault with slip above the guard has a non-empty pulse — or the call
    /// refuses, naming a subfault whose rise time genuinely rounds to zero samples.
    ///
    /// **`DEFECTS.md` 21.** genslip silently dropped subfaults whose rise time
    /// rounded to a single zero sample — 0.63% of the moment on the corpus fixture,
    /// gone without a word — and "unrepresentable rise time is an error naming the
    /// subfault, never a silent zero" is one of the four wrong numbers the rewrite
    /// fixes. This property is what keeps it fixed: over rise times generated down
    /// to zero, a slipping subfault either gets samples or gets named, and the
    /// margins (0.6/0.4 of a sample) pin *when* each outcome is required, clear of
    /// the round-to-nearest boundary at half a sample.
    #[test]
    fn a_slipping_subfault_is_never_silently_dropped(subfaults in adversarial_subfaults()) {
        match synthesise(&subfaults) {
            Ok(csr) => {
                for (subfault, &slip) in subfaults.slip_m.iter().enumerate() {
                    prop_assert_eq!(
                        row(&csr, subfault).is_empty(),
                        slip.abs() <= MIN_SLIP_M,
                        "subfault {} with slip {} m and rise {} s at dt {}",
                        subfault, slip, subfaults.rise_time_s[subfault], subfaults.dt_s
                    );
                }
                // If any slipping subfault's rise time were safely below half a
                // sample the call was required to refuse instead.
                for (subfault, &slip) in subfaults.slip_m.iter().enumerate() {
                    prop_assert!(
                        slip.abs() <= MIN_SLIP_M
                            || subfaults.rise_time_s[subfault] >= 0.4 * subfaults.dt_s,
                        "subfault {} slips {} m with a rise time of {} s at dt {} \
                         and was not refused",
                        subfault, slip, subfaults.rise_time_s[subfault], subfaults.dt_s
                    );
                }
            }
            Err(Error::UnrepresentableRiseTime { subfault, rise_time_s, dt_s }) => {
                prop_assert!(
                    subfaults.slip_m[subfault].abs() > MIN_SLIP_M,
                    "the refusal names subfault {}, which does not slip",
                    subfault
                );
                prop_assert_eq!(rise_time_s.to_bits(), subfaults.rise_time_s[subfault].to_bits());
                prop_assert_eq!(dt_s.to_bits(), subfaults.dt_s.to_bits());
                prop_assert!(
                    rise_time_s < 0.6 * dt_s,
                    "subfault {} was refused at rise {} s, dt {} s — a representable \
                     rise time",
                    subfault, rise_time_s, dt_s
                );
            }
            Err(other) => prop_assert!(false, "unexpected refusal: {}", other),
        }
    }

    /// `delta` is the impulse `[0, slip/dt, 0]`, whatever the rise time says.
    ///
    /// Exactly the spike `oliu_p` substitutes for a pulse too short to resolve, so
    /// it is that branch under its own name — and it is the reason the shape can
    /// never refuse a rise time.
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
/// What the parameter means physically: it sets how much of the duration the rising
/// limb occupies, so a smaller value concentrates the slip earlier and makes the
/// pulse more impulsive. Deterministic because discrete peak positions move in
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

/// Around one sample, the shape is a fixed spike rather than a computed curve —
/// and the spike is exactly what `delta` produces.
///
/// Pinned because it is a discontinuity in behaviour: a rewrite that computed the
/// sinusoid here instead would change the shortest pulses on every fault without
/// changing anything a smooth-field test looks at.
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

/// The integral carries the slip's sign: a back-slipping subfault is conserved too,
/// not rectified.
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
// Refusals: bad inputs are named, not synthesised around
// ---------------------------------------------------------------------------------

/// The refusal names everything the caller needs: which subfault, its rise time,
/// and the interval that cannot represent it (`DEFECTS.md` 21).
#[test]
fn the_unrepresentable_refusal_names_the_subfault() {
    let error = pulse::synthesise_pulses(
        &[1.0, 3.18, 1.0],
        &[1.0, 0.002, 1.0],
        Shape::OliuP {
            beta: &[0.2, 0.2, 0.2],
        },
        0.005,
    )
    .expect_err("subfault 1 slips but cannot be sampled");
    assert_eq!(
        error,
        Error::UnrepresentableRiseTime {
            subfault: 1,
            rise_time_s: 0.002,
            dt_s: 0.005
        }
    );
    let message = error.to_string();
    for needed in ["subfault 1", "0.002", "0.005"] {
        assert!(message.contains(needed), "missing {needed}: {message}");
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

/// `beta` beyond a half would let the sinusoid's second piece overrun the duration;
/// zero would divide the rising limb by nothing. Both are refused by subfault, and
/// refused even when that subfault does not slip — a bad parameter is a bad
/// parameter.
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
