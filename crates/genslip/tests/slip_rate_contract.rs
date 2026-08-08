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

/// What every shape has to be, whichever it is.
///
/// `generic_slip2srf` writes the sampling, the truncation, the fold and the rescale
/// out separately in each of its ten generators. They are one helper here, so these
/// are properties of that helper as much as of the shapes -- which is the point:
/// "conserves slip" should be one line of code, not a thing each shape remembers.
mod every_shape {
    use super::*;

    const DT: f32 = 0.005;

    /// All eleven, at parameters that exercise each one's own branch.
    fn all() -> Vec<SlipRateShape> {
        vec![
            SlipRateShape::OliuP2,
            SlipRateShape::Ucsb,
            SlipRateShape::Ucsb2,
            SlipRateShape::UcsbT { stretch: 2.0 },
            SlipRateShape::UcsbVarT1 { tau1_ratio: 0.2 },
            SlipRateShape::Brune,
            SlipRateShape::Urs,
            SlipRateShape::Esg2006,
            SlipRateShape::Cos,
            SlipRateShape::Seki,
            SlipRateShape::Delta,
        ]
    }

    fn integral(pulse: &genslip::slip_rate::SlipRate) -> f64 {
        pulse
            .as_slice()
            .iter()
            .map(|value| f64::from(*value) * f64::from(DT))
            .sum()
    }

    /// The integral is the slip. The only thing all eleven promise, and the reason
    /// the moment comes out right whichever is chosen.
    #[test]
    fn conserves_slip() {
        for shape in all() {
            for slip in [0.5_f32, 250.0, 4000.0] {
                for duration_s in [0.05_f32, 0.2, 1.0, 4.0, 20.0] {
                    let pulse = shape.pulse(slip, duration_s, 0.35, DT, MAX_SAMPLES);
                    assert!(
                        !pulse.is_empty(),
                        "{shape:?} produced nothing at slip {slip}, duration {duration_s}"
                    );
                    let bound = pulse_round_trip(pulse.len()) * f64::from(slip);
                    assert!(
                        (integral(&pulse) - f64::from(slip)).abs() <= bound,
                        "{shape:?} at duration {duration_s} integrates to {}, not {slip}",
                        integral(&pulse)
                    );
                }
            }
        }
    }

    /// A pulse ends at rest. Not every one *starts* at rest -- `seki` does not, by
    /// construction -- so that half is asserted separately, on the shapes that claim it.
    #[test]
    fn ends_at_rest() {
        for shape in all() {
            let pulse = shape.pulse(250.0, 1.0, 0.35, DT, MAX_SAMPLES);
            assert_eq!(
                pulse.as_slice().last(),
                Some(&0.0),
                "{shape:?} does not return to zero"
            );
        }
    }

    /// Doubling the duration roughly halves the peak, because the area is fixed.
    ///
    /// The weakest statement that still says these are *pulses* rather than arbitrary
    /// sample runs. Loose, because none of them is a rectangle and `delta` is
    /// duration-independent by definition and so excluded.
    #[test]
    fn a_longer_pulse_is_a_gentler_one() {
        let peak = |shape: SlipRateShape, duration_s: f32| {
            shape
                .pulse(250.0, duration_s, 0.35, DT, MAX_SAMPLES)
                .as_slice()
                .iter()
                .copied()
                .fold(0.0_f32, f32::max)
        };
        for shape in all() {
            if shape == SlipRateShape::Delta {
                continue;
            }
            let ratio = peak(shape, 0.5) / peak(shape, 2.0);
            assert!(
                (2.0..8.0).contains(&ratio),
                "{shape:?}: quadrupling the duration changed the peak by {ratio}x, \
                 not by something near 4"
            );
        }
    }

    /// A duration too short to resolve collapses to the shortest thing the shape can
    /// be, rather than to a long run of garbage.
    ///
    /// Two different floors, both deliberate. Most shapes give nothing or `oliu_p`'s
    /// three-sample spike. `urs` gives five, because its triangles have sample floors
    /// of their own -- the spike is at least two samples and the whole pulse at least
    /// four (`slip.c:65-70`), since a triangle below that is not a triangle. Asserting
    /// a single number here would have meant deleting one of the two.
    #[test]
    fn an_unresolvable_duration_collapses_to_the_shortest_pulse_there_is() {
        for shape in all() {
            let pulse = shape.pulse(250.0, 1e-6, 0.35, DT, MAX_SAMPLES);
            let floor = if shape == SlipRateShape::Urs { 5 } else { 3 };
            assert!(
                pulse.len() <= floor,
                "{shape:?} produced {} samples for a 1 microsecond pulse",
                pulse.len()
            );
        }
    }

    /// The cap binds, and binding does not produce a pulse that lies about its area.
    #[test]
    fn the_sample_cap_binds() {
        for shape in all() {
            let pulse = shape.pulse(250.0, 100.0, 0.35, DT, 64);
            assert!(pulse.len() <= 64, "{shape:?} ignored the cap");
        }
    }
}

/// The six shapes that are not `oliu_p`, each checked against what it is *for*.
mod the_closed_form_shapes {
    use super::*;

    const DT: f32 = 0.001;

    fn samples(shape: SlipRateShape, duration_s: f32, parameter: f32) -> Vec<f32> {
        shape
            .pulse(100.0, duration_s, parameter, DT, MAX_SAMPLES)
            .as_slice()
            .to_vec()
    }

    fn peak_at_s(values: &[f32]) -> f32 {
        #[expect(clippy::cast_precision_loss, reason = "sample indices are small")]
        let index = values
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map_or(0, |(index, _)| index) as f32;
        index * DT
    }

    /// `brune`: peaks at exactly one time constant, and decays from there.
    ///
    /// `(t/T)exp(-t/T)` has its maximum at `t = T`. That is the definition, and it is
    /// what says the duration handed in is being used as the time constant rather
    /// than as something else.
    #[test]
    fn brune_peaks_at_one_time_constant() {
        for duration_s in [0.05_f32, 0.2, 1.0] {
            let peak = peak_at_s(&samples(SlipRateShape::Brune, duration_s, 0.0));
            assert!(
                (peak - duration_s).abs() <= 2.0 * DT,
                "brune with T = {duration_s} peaked at {peak}"
            );
        }
    }

    /// `brune` is causal and one-sided: it starts at zero and never rises again.
    #[test]
    fn brune_rises_once_and_decays() {
        let values = samples(SlipRateShape::Brune, 0.2, 0.0);
        assert_eq!(values.first(), Some(&0.0));
        let peak_index = values
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map_or(0, |(index, _)| index);
        assert!(
            values[..peak_index]
                .windows(2)
                .all(|pair| pair[0] <= pair[1]),
            "brune is not monotone up to its peak"
        );
        assert!(
            values[peak_index..values.len() - 1]
                .windows(2)
                .all(|pair| pair[0] >= pair[1]),
            "brune is not monotone after its peak"
        );
    }

    /// `esg2006`: a Gaussian centred at twice the duration and symmetric about it.
    #[test]
    fn esg2006_is_a_symmetric_gaussian() {
        let duration_s = 0.25_f32;
        let values = samples(SlipRateShape::Esg2006, duration_s, 0.0);
        let peak = peak_at_s(&values);
        assert!(
            (peak - 2.0 * duration_s).abs() <= 2.0 * DT,
            "the Gaussian peaked at {peak}, not at {}",
            2.0 * duration_s
        );

        // Symmetry about the centre, which no other shape here has.
        let centre = values.len() / 2;
        for offset in 1..centre.min(200) {
            let (left, right) = (values[centre - offset], values[centre + offset]);
            assert!(
                (left - right).abs() <= 1e-3 * values[centre],
                "asymmetric at offset {offset}: {left} against {right}"
            );
        }
    }

    /// `cos`: a full raised cosine, so it is zero at both ends and peaks in the middle.
    #[test]
    fn cos_is_a_full_raised_cosine() {
        let duration_s = 0.4_f32;
        let values = samples(SlipRateShape::Cos, duration_s, 0.0);
        assert_eq!(values.first(), Some(&0.0));
        let peak = peak_at_s(&values);
        assert!(
            (peak - 0.5 * duration_s).abs() <= 2.0 * DT,
            "the cosine peaked at {peak}, not at {}",
            0.5 * duration_s
        );
    }

    /// `seki`: **does not start at rest**, and the onset shift is what admits it.
    ///
    /// `sech²(2(2t/T - 1))` is at `sech²(-2)`, about 7% of its peak, at `t = 0`. The
    /// original moves the subfault's onset back a quarter of the duration to
    /// compensate; this asserts both halves, because a shift without the discontinuity
    /// would be unmotivated and a discontinuity without the shift is a pulse arriving
    /// early.
    #[test]
    fn seki_starts_abruptly_and_pays_for_it_with_an_onset_shift() {
        let duration_s = 0.5_f32;
        let values = samples(SlipRateShape::Seki, duration_s, 0.0);
        let peak = values.iter().copied().fold(0.0_f32, f32::max);
        let ratio = values[0] / peak;
        assert!(
            (0.05..0.10).contains(&ratio),
            "seki starts at {ratio} of its peak, not the ~7% sech^2(-2) gives"
        );

        assert!(
            (SlipRateShape::Seki.onset_shift_s(duration_s) - (-0.25 * duration_s)).abs() < 1e-9,
            "seki does not move the onset back a quarter of its duration"
        );

        // And it is the only one that does.
        for shape in [
            SlipRateShape::OliuP2,
            SlipRateShape::Ucsb,
            SlipRateShape::Brune,
            SlipRateShape::Urs,
            SlipRateShape::Esg2006,
            SlipRateShape::Cos,
            SlipRateShape::Delta,
        ] {
            assert!(
                shape.onset_shift_s(duration_s).abs() < f32::EPSILON,
                "{shape:?} moves the onset by {} and has no reason to",
                shape.onset_shift_s(duration_s)
            );
        }
    }

    /// `urs`: a narrow spike then a long tail, with the tail's height the parameter.
    #[test]
    fn urs_is_a_spike_then_a_tail() {
        let duration_s = 1.0_f32;
        for tail in [0.2_f32, 0.35, 0.5] {
            let values = samples(SlipRateShape::Urs, duration_s, tail);
            assert_eq!(values.first(), Some(&0.0));

            let peak_index = values
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map_or(0, |(index, _)| index);
            let peak_at = peak_at_s(&values);
            assert!(
                (peak_at - 0.1 * duration_s).abs() <= 3.0 * DT,
                "the spike peaked at {peak_at}, not a tenth of the way in"
            );

            // The shoulder where the second triangle starts sits at the tail's height.
            let shoulder = (f32::from(2u8) - tail) * peak_at_s(&values) / DT;
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "a small non-negative sample index"
            )]
            let shoulder = shoulder as usize;
            let ratio = values[shoulder.min(values.len() - 1)] / values[peak_index];
            assert!(
                (ratio - tail).abs() < 0.1,
                "the shoulder is at {ratio} of the peak, not the {tail} asked for"
            );
        }
    }

    /// `urs`'s tail height is a ramp with *ends*, and the ends are where it went wrong.
    ///
    /// [`shape_parameter_field`] applied `DepthRamp::scaled_from_deep` without
    /// bracketing it. `DepthRamp` interpolates and does not clamp -- every other
    /// caller in the crate brackets it with an explicit three-branch `if` -- so a
    /// subfault below 6 km got an extrapolated tail rather than `betadeep`. At 7 km,
    /// one kilometre past the ramp, that was 0.05 against the C's 0.2, and half the
    /// pulse was wrong.
    ///
    /// The reference comparison found it; this is what keeps it found. Both plateaux
    /// and both breakpoints, because an unbracketed ramp is right in the middle and
    /// wrong at every depth outside it -- which is most of a fault.
    #[test]
    fn the_urs_tail_ramp_has_ends() {
        use genslip::slip_rate::shape_parameter_field;

        let profile = genslip::slip_rate::BetaProfile {
            shallow_ramp: genslip::rise_time::DepthRamp {
                centre_km: 2.0,
                half_width_km: 1.0,
            },
            shallow: 0.5,
            mid_ramp: genslip::rise_time::DepthRamp {
                centre_km: 6.5,
                half_width_km: 1.5,
            },
            mid: 0.13,
            deep: 0.13,
        };
        let depths = [0.0_f32, 2.0, 3.9, 4.0, 5.0, 6.0, 6.1, 7.0, 30.0];
        let field = shape_parameter_field(SlipRateShape::Urs, 1, &depths, profile);

        // betashal above the ramp, betadeep below it, and the midpoint between.
        let expected = [0.5_f32, 0.5, 0.5, 0.5, 0.35, 0.2, 0.2, 0.2, 0.2];
        for (index, (depth, want)) in depths.iter().zip(expected).enumerate() {
            let got = field[[index, 0]];
            assert!(
                (got - want).abs() < 1e-6,
                "at {depth} km the tail is {got}, not {want}"
            );
        }
    }

    /// A taller tail carries more of the slip late.
    ///
    /// What the depth ramp on this parameter is *for*: shallow subfaults get a longer,
    /// less impulsive release.
    #[test]
    fn a_taller_urs_tail_delays_the_slip() {
        let centroid = |tail: f32| {
            let values = samples(SlipRateShape::Urs, 1.0, tail);
            #[expect(clippy::cast_precision_loss, reason = "sample indices are small")]
            let weighted: f32 = values
                .iter()
                .enumerate()
                .map(|(index, value)| index as f32 * value)
                .sum();
            weighted / values.iter().sum::<f32>()
        };
        assert!(
            centroid(0.5) > centroid(0.2),
            "a taller tail did not move the slip later"
        );
    }

    /// `delta` is the spike `oliu_p` already falls back to, under its own name.
    ///
    /// An identity, so asserted as one -- it is why there is no eleventh generator.
    #[test]
    fn a_delta_is_the_spike_oliu_p_falls_back_to() {
        for slip in [0.5_f32, 250.0, 4000.0] {
            let delta = SlipRateShape::Delta.pulse(slip, 1.0, 0.0, DT, MAX_SAMPLES);
            // A duration of about one sample is what triggers `oliu_p`'s fallback.
            let fallback = oliu_p(slip, DT, 0.13, DT, MAX_SAMPLES);
            assert_eq!(delta.as_slice(), fallback.as_slice());
            assert_eq!(delta.as_slice(), [0.0, slip / DT, 0.0]);
        }
    }

    /// `delta` does not care about the duration. Nothing else here can say that.
    #[test]
    fn a_delta_is_the_same_pulse_at_every_duration() {
        let first = samples(SlipRateShape::Delta, 0.01, 0.0);
        for duration_s in [0.5_f32, 5.0, 50.0] {
            assert_eq!(samples(SlipRateShape::Delta, duration_s, 0.0), first);
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
// - That `brune`'s duration is the C's. It is not, deliberately: `generic_slip2srf`
//   derives a time constant from the subfault's slip, and this uses the rise time
//   like every other shape. See `SlipRateShape::Brune`.
// - That `esg2006` agrees with the C at all. It cannot: `gen_esg2006_stf` folds its
//   normalisation into an uninitialised `sum` (`slip.c:342`), so its output is
//   whatever was on the stack. `DEFECTS.md` 20.
