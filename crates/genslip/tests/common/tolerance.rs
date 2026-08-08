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
pub const U_F32: f64 = 5.960_464_477_539_063e-8;

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
    #[expect(clippy::cast_precision_loss, reason = "counts are far below 2^53")]
    let n = count as f64;
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
    #[expect(clippy::cast_precision_loss, reason = "grid sizes are small")]
    let n = (points.max(2)) as f64;
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
    #[expect(clippy::cast_precision_loss, reason = "pulse lengths are small")]
    let n = samples as f64;
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
    #[expect(clippy::cast_precision_loss, reason = "dof counts are small")]
    let nu = degrees_of_freedom as f64;
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
