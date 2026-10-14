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
    /// the depth profile produces one. A constant-`beta` parametrisation (the old
    /// `ucsb` family) is a constant array.
    OliuP { beta: &'a [f64] },
    /// A single-sample impulse: `[0, slip/dt, 0]`.
    ///
    /// Exactly what [`Shape::OliuP`] substitutes for a pulse too short to resolve,
    /// and this is that branch under its own name rather than a second spelling.
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
    /// A slipping subfault whose rise time rounds to zero samples at this interval.
    ///
    /// Refused rather than dropped: the caller can lower `dt_s` or floor the rise
    /// time, but it can't lose the moment.
    #[error(
        "subfault {subfault} slips, but its rise time of {rise_time_s} s rounds to zero \
         samples at dt = {dt_s} s; refusing rather than silently dropping its moment"
    )]
    UnrepresentableRiseTime {
        subfault: usize,
        rise_time_s: f64,
        dt_s: f64,
    },
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
/// # Errors
///
/// [`Error`]: mismatched array lengths, a non-positive `dt_s`, non-finite slip, a
/// `beta` outside `(0, 0.5]`, or a slipping subfault whose rise time
/// is unrepresentable at `dt_s`.
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
                Shape::OliuP { .. } => match samples(rise_time_s[subfault], dt_s) {
                    0 => {
                        return Err(Error::UnrepresentableRiseTime {
                            subfault,
                            rise_time_s: rise_time_s[subfault],
                            dt_s,
                        });
                    }
                    // Too short to resolve: a fixed spike replaces the shape.
                    1 => SPIKE_SAMPLES,
                    // One more sample than the duration covers, to close the pulse.
                    count => count + 1,
                },
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
    fill(job, &mut all_samples)?;
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

/// How many samples a duration covers, rounded to nearest. A negative or NaN duration
/// saturates to zero, an empty pulse.
fn samples(duration_s: f64, dt_s: f64) -> usize {
    (duration_s / dt_s).round() as usize
}

/// Subfaults per parallel task: enough that an impulse's three samples aren't
/// swamped by scheduling, few enough that work stealing evens out the long pulses.
const SUBFAULTS_PER_TASK: usize = 64;

/// Fill every pulse into its own slice of `out`, in parallel.
///
/// The offsets cut `out` into one disjoint `&mut [f64]` per run of
/// [`SUBFAULTS_PER_TASK`] subfaults, so the tasks share only immutable inputs and
/// need no locks. Rise time varies by an order of magnitude across a fault, and
/// rayon's work stealing evens out the uneven runs. A failure is reported at the
/// lowest failing subfault, however the work was scheduled.
fn fill(job: Job<'_>, out: &mut [f64]) -> Result<(), Error> {
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
        .find_map_first(|(start, end, run)| fill_run(job, start, end, run).err())
        .map_or(Ok(()), Err)
}

/// Subfaults `start..end`, whose samples fill exactly `run`, in order.
fn fill_run(job: Job<'_>, start: usize, end: usize, run: &mut [f64]) -> Result<(), Error> {
    let base = job.offsets[start];
    for subfault in start..end {
        let pulse = &mut run[job.offsets[subfault] - base..job.offsets[subfault + 1] - base];
        if !pulse.is_empty() {
            write(job, subfault, pulse)?;
        }
    }
    Ok(())
}

/// One subfault's pulse, into the slice pass 1 sized for it.
fn write(job: Job<'_>, subfault: usize, pulse: &mut [f64]) -> Result<(), Error> {
    let slip = job.slip_m[subfault];
    match job.shape {
        Shape::OliuP { beta } => oliu_p_into(
            pulse,
            subfault,
            slip,
            job.rise_time_s[subfault],
            beta[subfault],
            job.dt_s,
        ),
        Shape::Delta => {
            pulse.copy_from_slice(&[0.0, slip / job.dt_s, 0.0]);
            Ok(())
        }
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
/// A duration of about one sample gives a fixed three-point spike rather than
/// anything computed. The shape means nothing at that resolution, and a triangle
/// replaces it, which is also what [`Shape::Delta`] is.
///
/// `values` is the caller's slice of the output buffer, already the length pass 1
/// worked out for this subfault, and already zero. Writing into it rather than
/// returning a `Vec` limits the whole synthesis to one allocation: a fault of two
/// million subfaults was two million heap allocations and a copy of every sample out
/// of each.
///
/// # Errors
///
/// [`Error::UnrepresentableRiseTime`] at `subfault` if the shape can't be normalised.
///
/// (orig. `gen_OliuP_stf`, `gslip_sliprate_subs.c`)
fn oliu_p_into(
    values: &mut [f64],
    subfault: usize,
    slip: f64,
    duration_s: f64,
    beta: f64,
    dt_s: f64,
) -> Result<(), Error> {
    let count = samples(duration_s, dt_s);
    if count == 1 {
        // Too short to resolve: use a fixed spike in place of a computed shape.
        values.copy_from_slice(&[0.0, 1.0, 0.0]);
        return normalise(values, subfault, slip, duration_s, dt_s);
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
    normalise(values, subfault, slip, duration_s, dt_s)
}

/// Scale so `dt · Σ values` is `slip`, or refuse if no scale can do it.
///
/// Shared so that "conserves slip" is one line of code rather than a property each
/// shape has to remember to have.
///
/// # Errors
///
/// [`Error::UnrepresentableRiseTime`] at `subfault` if the integral isn't positive,
/// which means the shape degenerated at this resolution.
fn normalise(
    values: &mut [f64],
    subfault: usize,
    slip: f64,
    rise_time_s: f64,
    dt_s: f64,
) -> Result<(), Error> {
    let integral: f64 = values.iter().map(|value| dt_s * value).sum();
    if !integral.is_finite() || integral <= 0.0 {
        return Err(Error::UnrepresentableRiseTime {
            subfault,
            rise_time_s,
            dt_s,
        });
    }
    let scale = slip / integral;
    for value in values {
        *value *= scale;
    }
    Ok(())
}
