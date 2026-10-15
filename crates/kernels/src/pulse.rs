//! Pulse synthesis for point-sources.

use std::f64::consts::PI;

use rayon::prelude::*;

/// The slip below which a subfault doesn't get a pulse at all.
pub const MIN_SLIP_M: f64 = 1.0e-4;

/// Dispatchable enum of shape functions
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Shape<'a> {
    /// The Liu-Archuleta-Hartzell piecewise sinusoid, `beta` per subfault.
    ///
    /// `beta` is the fraction of the rise time spent in the rising limb. Larger is
    /// smoother, and shallow subfaults get the largest values. It's an array because
    /// the depth profile produces one. genslip's `ucsb` shape, the same sinusoid with
    /// one `beta` for every subfault, is a constant array.
    OliuP { beta: &'a [f64] },
    /// A single-sample impulse: `[0, slip/dt, 0]`.
    ///
    /// Exactly what [`Shape::OliuP`] substitutes for a pulse of a sample or less,
    /// rise times under half a sample included, and this is that branch under its
    /// own name rather than a second spelling.
    Delta,
}

/// What pulse synthesis refuses, in its own vocabulary.
#[derive(Clone, Copy, Debug, PartialEq, thiserror::Error)]
pub enum Error {
    /// The per-subfault arrays must agree on how many subfaults there are.
    #[error(
        "{field} has {got} values for {expected} subfaults; the per-subfault arrays must agree"
    )]
    MismatchedLengths {
        field: &'static str,
        expected: usize,
        got: usize,
    },
    /// A sample interval that isn't a positive, finite time samples nothing.
    #[error("a sample interval of {dt_s} s samples nothing; dt must be positive and finite")]
    NonPositiveSampleInterval { dt_s: f64 },
    /// Slip must be finite. NaN slip means something failed upstream, and a quiet zero
    /// would hide it.
    #[error("subfault {subfault} has a slip of {slip_m} m")]
    NonFiniteSlip { subfault: usize, slip_m: f64 },
    /// `beta` must be in `(0, 0.5]`: the sinusoid's second piece spans
    /// `beta·T .. 2·beta·T`, so beyond a half the pieces overrun the duration, and at
    /// zero the rising limb divides by nothing.
    #[error(
        "subfault {subfault} has beta = {beta}; the rising limb's fraction must be in (0, 0.5]"
    )]
    BetaOutOfRange { subfault: usize, beta: f64 },
    /// A slipping subfault's rise time must be a finite, non-negative duration. A
    /// short one is fine, since under half a sample it slips instantaneously, but
    /// NaN, infinity or a negative time means something failed upstream.
    #[error("subfault {subfault} slips, but has a rise time of {rise_time_s} s")]
    RiseTimeOutOfRange { subfault: usize, rise_time_s: f64 },
}

/// Every subfault's pulse, as compressed sparse rows.
///
/// `offsets` has one entry per subfault plus one; subfault `k`'s samples are
/// `samples[offsets[k]..offsets[k + 1]]`, in m/s at the sample interval of the
/// synthesis. An empty row is a subfault that doesn't slip, which the format keeps
/// distinct from a pulse whose samples happen to be zero.
#[derive(Clone, Debug, PartialEq)]
pub struct CsrPulses {
    pub offsets: Vec<usize>,
    pub samples: Vec<f64>,
}

/// Slip-normalised pulse synthesis.
///
/// A slipping subfault's pulse integrates to its slip, however short its rise time.
/// A rise time of about a sample or less, down to zero, gets the three-point spike
/// `[0, slip/dt, 0]`, and the subfault slips instantaneously within one sample.
/// genslip floors rise times to `dt` the same way.
///
/// # Errors
///
/// [`Error`]: mismatched array lengths, a non-positive `dt_s`, non-finite slip, a
/// `beta` outside `(0, 0.5]`, or, for [`Shape::OliuP`], a slipping subfault whose
/// rise time is negative or not finite.
pub fn synthesise_pulses(
    slip_m: &[f64],
    rise_time_s: &[f64],
    shape: Shape<'_>,
    dt_s: f64,
) -> Result<CsrPulses, Error> {
    if !dt_s.is_finite() || dt_s <= 0.0 {
        return Err(Error::NonPositiveSampleInterval { dt_s });
    }
    if rise_time_s.len() != slip_m.len() {
        return Err(Error::MismatchedLengths {
            field: "rise_time_s",
            expected: slip_m.len(),
            got: rise_time_s.len(),
        });
    }
    if let Shape::OliuP { beta } = shape {
        if beta.len() != slip_m.len() {
            return Err(Error::MismatchedLengths {
                field: "beta",
                expected: slip_m.len(),
                got: beta.len(),
            });
        }
        for (subfault, &value) in beta.iter().enumerate() {
            if !value.is_finite() || value <= 0.0 || value > 0.5 {
                return Err(Error::BetaOutOfRange {
                    subfault,
                    beta: value,
                });
            }
        }
    }
    for (subfault, &value) in slip_m.iter().enumerate() {
        if !value.is_finite() {
            return Err(Error::NonFiniteSlip {
                subfault,
                slip_m: value,
            });
        }
    }

    // Pass 1: how long every pulse is, without evaluating one.
    let mut offsets = Vec::with_capacity(slip_m.len() + 1);
    offsets.push(0);
    let mut total = 0;
    for (subfault, &slip) in slip_m.iter().enumerate() {
        let length = if slip.abs() > MIN_SLIP_M {
            match shape {
                Shape::Delta => SPIKE_SAMPLES,
                Shape::OliuP { .. } => {
                    let rise = rise_time_s[subfault];
                    if !rise.is_finite() || rise < 0.0 {
                        return Err(Error::RiseTimeOutOfRange {
                            subfault,
                            rise_time_s: rise,
                        });
                    }
                    match samples(rise, dt_s) {
                        // Too short to resolve, under half a sample included: a fixed
                        // spike replaces the shape, and the subfault slips at once.
                        0 | 1 => SPIKE_SAMPLES,
                        // One more sample than the duration covers, to close the pulse.
                        count => count + 1,
                    }
                }
            }
        } else {
            0
        };
        total += length;
        offsets.push(total);
    }

    let mut all_samples = vec![0.0_f64; total];
    let job = Job {
        slip_m,
        rise_time_s,
        offsets: &offsets,
        shape,
        dt_s,
    };
    fill(job, &mut all_samples);
    Ok(CsrPulses {
        offsets,
        samples: all_samples,
    })
}

/// What every pulse in a fill needs and none of it varies by subfault.
#[derive(Clone, Copy)]
struct Job<'a> {
    slip_m: &'a [f64],
    rise_time_s: &'a [f64],
    offsets: &'a [usize],
    shape: Shape<'a>,
    dt_s: f64,
}

/// The samples in a pulse too short for its shape to mean anything: rise, peak, fall.
const SPIKE_SAMPLES: usize = 3;

/// How many samples a duration covers, rounded to nearest. Pass 1 refuses a negative
/// or non-finite duration before calling this.
fn samples(duration_s: f64, dt_s: f64) -> usize {
    (duration_s / dt_s).round() as usize
}

/// Subfaults per parallel task: enough that scheduling takes less time than the
/// pulses' samples, few enough that work stealing evens out the long pulses.
const SUBFAULTS_PER_TASK: usize = 64;

/// Fill every pulse into its own slice of `out`, in parallel.
///
/// The offsets cut `out` into one disjoint `&mut [f64]` per run of
/// [`SUBFAULTS_PER_TASK`] subfaults. The tasks share only immutable inputs and
/// don't need locks. Rise time varies by an order of magnitude across a fault, and
/// rayon's work stealing evens out the uneven runs. Pass 1 checked every input, so
/// nothing here can fail.
fn fill(job: Job<'_>, out: &mut [f64]) {
    let subfaults = job.slip_m.len();
    let mut tasks = Vec::with_capacity(subfaults.div_ceil(SUBFAULTS_PER_TASK));
    let mut rest = out;
    for start in (0..subfaults).step_by(SUBFAULTS_PER_TASK) {
        let end = (start + SUBFAULTS_PER_TASK).min(subfaults);
        let (run, tail) =
            std::mem::take(&mut rest).split_at_mut(job.offsets[end] - job.offsets[start]);
        tasks.push((start, end, run));
        rest = tail;
    }
    tasks
        .into_par_iter()
        .for_each(|(start, end, run)| fill_run(job, start, end, run));
}

/// Subfaults `start..end`, whose samples fill exactly `run`, in order.
fn fill_run(job: Job<'_>, start: usize, end: usize, run: &mut [f64]) {
    let base = job.offsets[start];
    for subfault in start..end {
        let pulse = &mut run[job.offsets[subfault] - base..job.offsets[subfault + 1] - base];
        if !pulse.is_empty() {
            write(job, subfault, pulse);
        }
    }
}

/// One subfault's pulse, into the slice pass 1 sized for it.
fn write(job: Job<'_>, subfault: usize, pulse: &mut [f64]) {
    let slip = job.slip_m[subfault];
    match job.shape {
        Shape::OliuP { beta } => oliu_p_into(
            pulse,
            slip,
            job.rise_time_s[subfault],
            beta[subfault],
            job.dt_s,
        ),
        Shape::Delta => pulse.copy_from_slice(&[0.0, slip / job.dt_s, 0.0]),
    }
}

/// The `OliuP` slip-rate function: a piecewise sinusoid after Liu, Archuleta &
/// Hartzell (2006).
///
/// Three pieces, of which `beta` sets the first two's extent. Writing `tau1` for
/// `beta * duration`:
///
/// * `0 .. tau1`: the rising limb, a raised cosine plus a half-period sine that
///   makes the rise sharper than the fall;
/// * `tau1 .. 2*tau1`: the peak and the start of the decay;
/// * `2*tau1 .. duration`: the tail, a quarter cosine to zero.
///
/// `normalise` scales the result so `dt * sum` is `slip`, and one trailing zero closes
/// the pulse. A source-time function that ends at a non-zero rate is a step in
/// velocity, which radiates at every frequency and isn't what the model means.
///
/// A duration of about one sample or less, down to zero, gives a fixed three-point
/// spike rather than anything computed. The shape means nothing at that resolution,
/// and a triangle replaces it, which is also what [`Shape::Delta`] is. A subfault
/// whose rise time is under half a sample slips instantaneously, as genslip's floor
/// to `dt` makes it.
///
/// `values` is the caller's slice of the output buffer, already the length pass 1
/// worked out for this subfault, and already zero. Writing into it rather than
/// returning a `Vec` limits the whole synthesis to one allocation, rather than one
/// per subfault and a copy of every sample out of each.
///
/// (orig. `gen_OliuP_stf`, `gslip_sliprate_subs.c`)
fn oliu_p_into(values: &mut [f64], slip: f64, duration_s: f64, beta: f64, dt_s: f64) {
    let count = samples(duration_s, dt_s);
    if count <= 1 {
        // Too short to resolve: use a fixed spike in place of a computed shape.
        values.copy_from_slice(&[0.0, 1.0, 0.0]);
        return normalise(values, slip, dt_s);
    }

    let rise_end = beta * duration_s;
    let peak_end = 2.0 * rise_end;
    let decay_span = duration_s - rise_end;

    // One more sample than the duration covers, left at the zero the buffer arrived
    // with. That sample closes the pulse.
    for (index, value) in values.iter_mut().enumerate().take(count).skip(1) {
        let time = index as f64 * dt_s;
        *value = if time < rise_end {
            let arg = PI * time / rise_end;
            0.7 - 0.7 * arg.cos() + 0.6 * (0.5 * arg).sin()
        } else if time < peak_end {
            let rising = PI * time / rise_end;
            let decaying = PI * (time - rise_end) / decay_span;
            1.0 - 0.7 * rising.cos() + 0.3 * decaying.cos()
        } else if time < duration_s {
            let decaying = PI * (time - rise_end) / decay_span;
            0.3 + 0.3 * decaying.cos()
        } else {
            0.0
        };
    }
    normalise(values, slip, dt_s);
}

/// Scale so `dt · Σ values` is `slip`.
///
/// Shared so that "conserves slip" is one line of code rather than a property each
/// shape has to remember to have. The spike's middle sample is 1. A computed shape
/// writes positive samples before its duration ends, because with `beta <= 0.5`
/// no piece's cosine argument gets as far as making it zero. Either way the
/// integral is positive.
fn normalise(values: &mut [f64], slip: f64, dt_s: f64) {
    let integral: f64 = values.iter().map(|value| dt_s * value).sum();
    debug_assert!(
        integral.is_finite() && integral > 0.0,
        "a pulse shape integrates to {integral}"
    );
    let scale = slip / integral;
    for value in values {
        *value *= scale;
    }
}
