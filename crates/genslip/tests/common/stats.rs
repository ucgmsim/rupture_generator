//! How two fields differ: in level, in spread, or in shape.
//!
//! # Why the decomposition rather than one number
//!
//! A single worst-case difference says only "something moved", and that is the least
//! useful thing a comparison can say. Splitting it into three said, in one run, what
//! two real defects actually were:
//!
//! | | mean ratio | sigma ratio | correlation | what it was |
//! | --- | --- | --- | --- | --- |
//! | `DEFECTS.md` 18 | 0.98 | **1.63** | 0.993 | one affine transform where another belonged |
//! | `DEFECTS.md` 14 | 1.00 | **0.05** | ~1.0 | a dimensionless spread used where degrees belonged |
//!
//! Both look like "nearly right" under a worst-case norm and like a specific,
//! findable mistake under this one. `ENGINEERING_RULES.md` records the technique;
//! this is it.

/// Widen a field for comparison. Fields are `f32`; travel times are already `f64`.
#[must_use]
pub fn widen(values: &[f32]) -> Vec<f64> {
    values.iter().copied().map(f64::from).collect()
}

#[must_use]
pub fn mean(values: &[f64]) -> f64 {
    assert!(!values.is_empty(), "the mean of nothing is not a number");
    #[expect(clippy::cast_precision_loss, reason = "test-sized fields")]
    let count = values.len() as f64;
    values.iter().sum::<f64>() / count
}

/// Population standard deviation — divides by `n`, as the original's does.
#[must_use]
pub fn population_sigma(values: &[f64]) -> f64 {
    let centre = mean(values);
    #[expect(clippy::cast_precision_loss, reason = "test-sized fields")]
    let count = values.len() as f64;
    (values.iter().map(|v| (v - centre).powi(2)).sum::<f64>() / count).sqrt()
}

/// Pearson correlation. Returns 1.0 for two constant fields, which is the useful
/// convention here: it means "the same shape", and a constant field has no shape to
/// disagree about.
#[must_use]
pub fn pearson(first: &[f64], second: &[f64]) -> f64 {
    assert_eq!(first.len(), second.len(), "correlating unequal lengths");
    let (a, b) = (mean(first), mean(second));
    let covariance: f64 = first
        .iter()
        .zip(second)
        .map(|(x, y)| (x - a) * (y - b))
        .sum();
    let spread_a: f64 = first.iter().map(|x| (x - a).powi(2)).sum::<f64>().sqrt();
    let spread_b: f64 = second.iter().map(|y| (y - b).powi(2)).sum::<f64>().sqrt();
    if spread_a == 0.0 || spread_b == 0.0 {
        return 1.0;
    }
    covariance / (spread_a * spread_b)
}

/// Three numbers that say *how* two fields differ, not merely that they do.
#[derive(Clone, Copy, Debug)]
pub struct Decomposition {
    /// Ratio of means. Off 1 means the whole field is scaled or shifted.
    pub mean_ratio: f64,
    /// Ratio of population sigmas. Off 1 means the *contrast* is wrong while the
    /// level may be right — the shape `DEFECTS.md` 14 and 18 both took.
    pub sigma_ratio: f64,
    /// Off 1 means the *pattern* is wrong: subfaults reordered, an index shifted, a
    /// different draw stream.
    pub correlation: f64,
    /// Worst absolute difference over the reference's own largest magnitude.
    ///
    /// Scale-relative rather than element-relative, deliberately: a field that
    /// crosses zero has cells where any absolute drift is an unbounded *relative*
    /// error, and a per-element norm reports that as a catastrophe.
    pub worst_scale_relative: f64,
}

/// Compare a produced field against a reference.
///
/// # Panics
///
/// If the two disagree about how many subfaults there are.
#[must_use]
pub fn decompose(produced: &[f64], reference: &[f64]) -> Decomposition {
    assert_eq!(
        produced.len(),
        reference.len(),
        "comparing fields of different extent"
    );
    let scale = reference
        .iter()
        .fold(0.0_f64, |worst, value| worst.max(value.abs()))
        .max(f64::MIN_POSITIVE);
    let worst = produced
        .iter()
        .zip(reference)
        .fold(0.0_f64, |worst, (a, b)| worst.max((a - b).abs()));

    let (reference_mean, reference_sigma) = (mean(reference), population_sigma(reference));
    Decomposition {
        mean_ratio: if reference_mean == 0.0 {
            1.0
        } else {
            mean(produced) / reference_mean
        },
        sigma_ratio: if reference_sigma == 0.0 {
            1.0
        } else {
            population_sigma(produced) / reference_sigma
        },
        correlation: pearson(produced, reference),
        worst_scale_relative: worst / scale,
    }
}

impl std::fmt::Display for Decomposition {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "mean x{:.6}, sigma x{:.6}, correlation {:.7}, worst {:.3e} of scale",
            self.mean_ratio, self.sigma_ratio, self.correlation, self.worst_scale_relative
        )
    }
}

/// Autocorrelation at one subfault's lag along strike.
///
/// A cheap stand-in for a correlation length: a field correlated over more subfaults
/// along an axis has a higher lag-one autocorrelation along it. Enough to say *which
/// axis is smoother*, which is all the fast-axis contract needs, and it avoids
/// fitting anything.
#[must_use]
pub fn lag_one_along_strike(field: &[f64], strike_count: usize, dip_count: usize) -> f64 {
    let pairs = (0..dip_count).flat_map(|dip| {
        (0..strike_count - 1)
            .map(move |strike| (strike + dip * strike_count, strike + 1 + dip * strike_count))
    });
    lag_correlation(field, pairs)
}

/// Autocorrelation at one subfault's lag down dip.
#[must_use]
pub fn lag_one_along_dip(field: &[f64], strike_count: usize, dip_count: usize) -> f64 {
    let pairs = (0..dip_count - 1).flat_map(|dip| {
        (0..strike_count).map(move |strike| {
            (
                strike + dip * strike_count,
                strike + (dip + 1) * strike_count,
            )
        })
    });
    lag_correlation(field, pairs)
}

fn lag_correlation(field: &[f64], pairs: impl Iterator<Item = (usize, usize)>) -> f64 {
    let (here, there): (Vec<f64>, Vec<f64>) = pairs.map(|(a, b)| (field[a], field[b])).unzip();
    pearson(&here, &there)
}
