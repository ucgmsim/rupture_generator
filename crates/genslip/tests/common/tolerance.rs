//! How close is close enough, derived rather than chosen.
//!
//! `ENGINEERING_RULES.md` rule 1: a tolerance nobody can reconstruct is a tolerance
//! nobody can defend, and it rots the moment a grid size changes. So every bound here
//! is a function of sample size, and every one states what it can **see** — a
//! tolerance without a detection floor is half a specification, because it says what
//! passes and not what would fail, and so cannot be told apart from a vacuous one.

/// Unit roundoff for `f32`: half an ulp at 1.0, `2^-24`.
///
/// Not `f32::EPSILON`, which is `2^-23` — the *gap* between representable numbers
/// rather than the worst rounding error, which is half of it. Getting this wrong
/// doubles every bound below.
///
/// **The crate computes in `f64` now, and this is still the constant most bounds are
/// built from.** That is not an oversight. Every comparison against genslip is a
/// comparison against numbers genslip computed in `float` and wrote as six
/// significant figures, so the *reference* sets the resolution, not the port. A bound
/// derived from [`U_F64`] would be asserting that the C is more precise than it is.
pub const U_F32: f64 = 5.960_464_477_539_063e-8;

/// Unit roundoff for `f64`: half an ulp at 1.0, `2^-53`.
///
/// For claims about the port against *itself* — two spellings of one formula, a
/// refactor that should have moved nothing — where the reference is this crate rather
/// than genslip and its resolution is the one that binds.
pub const U_F64: f64 = 1.110_223_024_625_157e-16;

/// Standard deviations allowed on any single statistical assertion.
///
/// Bonferroni over the suite: at roughly 300 statistical assertions and a family-wise
/// false-failure budget of 1e-3, the per-assertion two-tailed bound is z = 4.65.
/// Rounded up, and fixed here rather than argued per test.
///
/// With fixed seeds nothing flakes on a rerun; this governs the risk when a *new*
/// seed, grid or kernel enters the matrix.
pub const Z: f64 = 5.0;

/// The physical bounds from `ENGINEERING_RULES.md`. Two ruptures agreeing within
/// these are the same rupture.
pub mod acceptable {
    /// Relative, on slip in centimetres.
    pub const SLIP: f64 = 1.0e-2;
    /// Absolute, in seconds. About 18 degrees of phase at 1 Hz.
    pub const ONSET_S: f64 = 5.0e-2;
    /// Absolute, in degrees. The SRF stores whole degrees, so this is its floor.
    pub const RAKE_DEG: f64 = 1.0;
    /// Relative, on rise time in seconds.
    pub const RISE: f64 = 1.0e-2;
}

/// Relative error a left-to-right `f32` sum of `n` comparable positive terms accrues.
///
/// The worst case `(n-1)*u` is useless — 6e-3 at n = 1e5 — because it assumes every
/// rounding goes the same way. Modelling them as independent gives
/// `sigma = u*sqrt(n)/3`, and this returns `9 sigma`, which measured about five times
/// the worst of sixty actual folds at each of n = 100 … 100 000.
///
/// **Detection floor**: `3*u*sqrt(n)` relative. At n = 1e5 that is 5.7e-5, so a
/// moment sum accumulated in `f32` cannot detect fewer than about six missing
/// subfaults out of a hundred thousand. That is a fact about the accumulator, not
/// about the test — widen it and the floor drops to one subfault.
///
/// Only valid when the terms share a sign; a sum that cancels has an unbounded
/// condition number and this bound says nothing.
#[must_use]
pub fn f32_sum_relative(count: usize) -> f64 {
    let n = genslip::units::exact(count);
    3.0 * U_F32 * n.sqrt()
}

/// Scale-relative difference two implementations of the same arithmetic may show.
///
/// `u*log2(n)` is the standard growth of FFT rounding over `n` points; the factor of
/// eight is the one dimensionless headroom in the suite, and it is about sixty times
/// the 7.06e-8 measured for swapping FFT engines.
///
/// **Detection floor**: about 4e-6 of the field's own scale on a 500-point grid. The
/// smallest physically meaningful change is four orders of magnitude above that — a
/// one-cell hypocentre shift moves onset by 2e-2 of its range — so this separates
/// "different arithmetic" from "different rupture" with room to spare.
#[must_use]
pub fn engine_drift(points: usize) -> f64 {
    let n = genslip::units::exact(points.max(2));
    8.0 * U_F32 * n.log2()
}

/// Round-trip error of a pulse normalised so `dt * sum == slip`, over `n` samples.
///
/// Three sources: the `f32` fold of the original (`u*sqrt(n)/3`), the rounding of the
/// common scale factor (about `2u`, systematic — it does not average away), and the
/// per-sample scaling (negligible). **Detection floor**: about 3e-6 relative, so one
/// wrong sample in a thousand of a smooth pulse is caught.
#[must_use]
pub fn pulse_round_trip(samples: usize) -> f64 {
    let n = genslip::units::exact(samples);
    4.0 * U_F32 * (n.sqrt() + 2.0)
}

/// Acceptance band for a pooled power ratio, which is exactly `chi2_nu / nu`.
///
/// Returns `(low, high)` for the **cube root** of the ratio, via Wilson–Hilferty:
/// `(chi2_nu/nu)^(1/3)` is close to normal with mean `1 - 2/(9nu)` and standard
/// deviation `sqrt(2/(9nu))`, far closer than the ratio itself is. Checked against
/// 200k draws of the exact law: at z = 5 the exceedance rate is 5e-6 for nu of a few
/// hundred, i.e. the band is honest rather than merely conservative.
///
/// `nu` is a count of degrees of freedom, not of grid points: a generic wavenumber
/// contributes two — a real and an imaginary Gaussian — and each of the four
/// self-conjugate points contributes one, because those are forced real.
///
/// **Detection floor**: a power error of about `2 * 3*sqrt(2/(9nu))`. At nu = 9000
/// that is 10%, which catches every missing band-pass factor and every misplaced
/// `1/sqrt(2)`.
#[must_use]
pub fn wilson_hilferty_band(degrees_of_freedom: usize, z: f64) -> (f64, f64) {
    assert!(
        degrees_of_freedom > 0,
        "a band needs at least one degree of freedom"
    );
    let nu = genslip::units::exact(degrees_of_freedom);
    let centre = 1.0 - 2.0 / (9.0 * nu);
    let spread = (2.0 / (9.0 * nu)).sqrt();
    (centre - z * spread, centre + z * spread)
}

/// Standard error of a regression slope whose residuals are `ln` of an exponential.
///
/// `ln R` for `R ~ Exp(1)` has variance `pi^2/6` regardless of wavenumber, so the fit
/// is homoscedastic and ordinary least squares is the right estimator. Pass the
/// design matrix's total squared deviation, `sum (X - Xbar)^2`.
///
/// **Detection floor**: about `2*z` of this. On the fixture grid, eight realisations
/// resolve a change in the von Karman exponent of about 0.05.
#[must_use]
pub fn log_spectrum_slope_error(sum_squared_deviation: f64) -> f64 {
    assert!(
        sum_squared_deviation > 0.0,
        "a slope needs spread in the regressor"
    );
    (std::f64::consts::PI * std::f64::consts::PI / 6.0 / sum_squared_deviation).sqrt()
}

/// How far `exp(k * ln x)` may differ from the direct power of `x`.
///
/// The port writes `cbrt` where the original writes `exp(ln(M0)/3)`, and `powf` where
/// it writes `exp(n*ln(x))`. Same function, two evaluations — but the gap is **not** a
/// couple of ulps, and assuming it was is how the first attempt at this bound failed.
///
/// `ln(x)` is computed to within `u` *relative*, so its absolute error is
/// `|ln x| * u`. Scaling by `k` and exponentiating turns that straight back into a
/// relative error of `k * |ln x| * u` on the result. On a seismic moment of 1e26,
/// `|ln x|` is about 60, so at `k = 1/3` the two spellings differ by around
/// `20u ≈ 2e-15` — twenty times what a naive ulp count predicts, and exactly what is
/// measured. The `+ 1` covers the final `exp`'s own rounding.
///
/// The factor of 16 is headroom for what happens *after*. In the rise-time field the
/// power law is not the last step: the field is renormalised by its own mean over
/// every subfault, which neither cancels the difference nor bounds it more tightly,
/// so the value that reaches the comparison has been carried through two more stages.
/// Sixteen is the smallest power of two that covers the worst of them.
///
/// **They used to agree to the bit, and that was an artefact.** While the port
/// computed in `f32`, every one of these was narrowed to `float` on the way out and
/// the store rounded the difference away. `float_identities.rs` asserted the same
/// pairs were *not* equal at full width the whole time; the parity tests asserted
/// they were. Both were right about their own width, and only the move to `f64` made
/// the two statements meet.
///
/// **Detection floor**: about 4e-14 relative on a magnitude-7.5 moment, and 7e-15 on
/// a unit-scale field value. Eight orders below what an `f32` could express, so it
/// cannot hide a wrong constant — which is the only thing a transcription of the same
/// formula was ever able to catch.
/// A zero argument or a zero exponent means no logarithm was taken, so there is no
/// amplification to allow for — `ln(0)` is `-inf` and would otherwise make the bound
/// `NaN`, which fails every comparison including the ones that should pass.
#[must_use]
pub fn transcendental_spelling(argument: f64, exponent: f64) -> f64 {
    let amplification = if argument == 0.0 || exponent == 0.0 {
        0.0
    } else {
        exponent.abs() * argument.abs().ln().abs()
    };
    16.0 * U_F64 * (amplification + 1.0)
}

/// Whether two values agree, counting two NaNs as agreeing.
///
/// **Not a convenience.** A rise-time field on a 1x1 fault is NaN on both sides, and
/// for the same reason: `rescale_about_unit_mean` divides by the field's own
/// coefficient of variation, which is zero when there is one subfault, so the
/// prescribed spread is undefined. The port and the C degenerate identically, which
/// is a real property of the transcription and worth asserting — but `NaN == NaN` is
/// false and `(NaN - NaN).abs() <= bound` is false, so saying it takes this.
///
/// It says *"both defined and close, or both undefined"*. A NaN on one side only is a
/// failure, which is the case that matters.
#[must_use]
pub fn agree(got: f64, want: f64, bound: f64) -> bool {
    if got.is_nan() || want.is_nan() {
        return got.is_nan() && want.is_nan();
    }
    (got - want).abs() <= bound * want.abs().max(1.0)
}

/// How far a value moves when a sum over `count` terms is **reassociated**.
///
/// Not an error bound on either sum — both are correct. It is the gap between two
/// correct answers, which is what a comparison across a refactor actually needs.
///
/// `ndarray`'s `.mean()` and `.sum()` fold pairwise where a hand-written
/// `for value in field { total += value }` folds left to right. Pairwise is the
/// *more* accurate of the two, growing as `u*sqrt(log n)` against the sequential
/// fold's `u*sqrt(n)`, so the sequential one bounds the difference: `2u*sqrt(n)`.
///
/// **Detection floor**: 1.3e-15 relative on a 32-subfault field, 8e-15 on a thousand.
/// Below the point where two spellings of the same arithmetic can be told apart, and
/// eight orders below anything `f32` could express.
#[must_use]
pub fn fold_reorder(count: usize) -> f64 {
    2.0 * U_F64 * genslip::units::exact(count).sqrt()
}

/// What a whole elementwise *pipeline* may differ by across a reassociation.
///
/// Deliberately not derived, and that is the point worth writing down. The rise-time
/// field is five stages deep — blend, normalise, rescale to a prescribed spread,
/// power law, normalise again — and each stage's difference feeds the next. The
/// rescale in particular multiplies by `sigma/variation`, so a `sigma` of 2 turns a
/// 1e-15 difference in the variation into a several-times-larger one in the output.
/// A tight bound on that chain is a paper, not a test.
///
/// So this is **chosen**, and its justification is the gap it sits in rather than its
/// derivation:
///
/// * reassociating the folds moves the answer by ~1e-14 at the worst `sigma` the
///   proptest reaches — measured, not assumed;
/// * a wrong *constant* — a coefficient, an exponent, a ramp bound, which is the only
///   thing a transcription of the same formulas can catch — moves it by 1e-3 or more.
///
/// Eleven orders of clear air between the two. A bound anywhere in it does the same
/// job, and one that tracked the reassociation exactly would be a more precise answer
/// to a question nobody asked.
pub const PIPELINE_REASSOCIATION: f64 = 1.0e-10;
