//! Pulse synthesis for point-sources.

use crate::counts::{exact, samples};
use std::f64::consts::PI;

/// The slip below which a subfault gets no pulse at all.
pub const MIN_SLIP_M: f64 = 1.0e-4;

/// Dispatchable enum of shape functions
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Shape<'a> {
    /// The Liu–Archuleta–Hartzell piecewise sinusoid, `beta` per subfault.
    ///
    /// `beta` is the fraction of the rise time spent in the rising limb — larger is
    /// smoother, and shallow subfaults get the largest values. It is an array
    /// because that is what the depth profile produces; a constant-`beta`
    /// parametrisation (the old `ucsb` family) is a constant array.
    OliuP { beta: &'a [f64] },
    /// A single-sample impulse: `[0, slip/dt, 0]`.
    ///
    /// Exactly what [`Shape::OliuP`] substitutes for a pulse too short to resolve,
    /// so it is that branch under its own name rather than a second spelling.
    Delta,
}

/// What pulse synthesis refuses, in its own vocabulary.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Error {
    /// The per-subfault arrays must agree on how many subfaults there are.
    MismatchedLengths {
        field: &'static str,
        expected: usize,
        got: usize,
    },
    /// A sample interval that is not a positive, finite time samples nothing.
    NonPositiveSampleInterval { dt_s: f64 },
    /// Slip must be finite; NaN slip is an upstream failure, not a quiet zero.
    NonFiniteSlip { subfault: usize, slip_m: f64 },
    /// `beta` must be in `(0, 0.5]`: the sinusoid's second piece spans
    /// `beta·T .. 2·beta·T`, so beyond a half the pieces overrun the duration, and at
    /// zero the rising limb divides by nothing.
    BetaOutOfRange { subfault: usize, beta: f64 },
    /// A slipping subfault whose rise time rounds to zero samples at this interval.
    ///
    /// The refusal that replaces the silent drop of `DEFECTS.md` 21. The caller can
    /// lower `dt_s` or floor the rise time; what it cannot do is lose the moment.
    UnrepresentableRiseTime {
        subfault: usize,
        rise_time_s: f64,
        dt_s: f64,
    },
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::MismatchedLengths {
                field,
                expected,
                got,
            } => write!(
                f,
                "{field} has {got} values for {expected} subfaults; the per-subfault \
                 arrays must agree"
            ),
            Self::NonPositiveSampleInterval { dt_s } => write!(
                f,
                "a sample interval of {dt_s} s samples nothing; dt must be positive \
                 and finite"
            ),
            Self::NonFiniteSlip { subfault, slip_m } => {
                write!(f, "subfault {subfault} has a slip of {slip_m} m")
            }
            Self::BetaOutOfRange { subfault, beta } => write!(
                f,
                "subfault {subfault} has beta = {beta}; the rising limb's fraction \
                 must be in (0, 0.5]"
            ),
            Self::UnrepresentableRiseTime {
                subfault,
                rise_time_s,
                dt_s,
            } => write!(
                f,
                "subfault {subfault} slips, but its rise time of {rise_time_s} s \
                 rounds to zero samples at dt = {dt_s} s; refusing rather than \
                 silently dropping its moment"
            ),
        }
    }
}

impl std::error::Error for Error {}

/// Every subfault's pulse, as compressed sparse rows.
///
/// `offsets` has one entry per subfault plus one; subfault `k`'s samples are
/// `samples[offsets[k]..offsets[k + 1]]`, in m/s at the sample interval the pulses
/// were synthesised at. An empty row is a subfault that does not slip — a fact the
/// format keeps, distinct from a pulse whose samples happen to be zero.
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
/// is unrepresentable at `dt_s` (`DEFECTS.md` 21).
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
                    // Too short to resolve; a fixed spike stands in for the shape.
                    1 => SPIKE_SAMPLES,
                    // One more sample than the duration covers, so the pulse closes.
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
    if let Some(subfault) = fill(job, &mut all_samples) {
        return Err(Error::UnrepresentableRiseTime {
            subfault,
            rise_time_s: rise_time_s[subfault],
            dt_s,
        });
    }
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

/// Below this many samples the threads cost more than they save.
///
/// Spawning is tens of microseconds; a hundred thousand samples is under a
/// millisecond of work. Small faults -- and every test in this crate -- take the
/// serial path, which is also the one that stays debuggable.
const PARALLEL_FROM_SAMPLES: usize = 100_000;

/// The samples in a pulse too short for its shape to mean anything: rise, peak, fall.
const SPIKE_SAMPLES: usize = 3;

/// Fill every pulse into its own slice of `out`, over as many threads as there are
/// cores. Returns the lowest subfault whose pulse could not be normalised, if any.
///
/// **The split is by sample count, not by subfault count.** Rise time varies by an
/// order of magnitude across a fault -- deep subfaults slip for far longer than
/// shallow ones -- so equal shares of subfaults are unequal shares of work, and the
/// slowest thread sets the time. `offsets` is already the prefix sum of the work, so
/// the boundary that divides it evenly is one `partition_point` away.
///
/// Each thread then owns a contiguous, disjoint `&mut [f64]`, handed out by
/// `split_at_mut`. No locking, no atomics, and nothing shared but immutable inputs.
fn fill(job: Job<'_>, out: &mut [f64]) -> Option<usize> {
    let subfaults = job.slip_m.len();
    let threads = if out.len() < PARALLEL_FROM_SAMPLES {
        1
    } else {
        std::thread::available_parallelism().map_or(1, std::num::NonZero::get)
    };

    let mut bounds = Vec::with_capacity(threads + 1);
    bounds.push(0);
    for thread in 1..threads {
        let target = out.len() * thread / threads;
        let at = job.offsets.partition_point(|&offset| offset < target);
        // Monotone and in range whatever `partition_point` says: empty ranges are
        // fine, overlapping ones would alias the output.
        bounds.push(at.clamp(bounds[thread - 1], subfaults));
    }
    bounds.push(subfaults);

    let mut remaining = out;
    let mut pieces = Vec::with_capacity(threads);
    for window in bounds.windows(2) {
        let (start, end) = (window[0], window[1]);
        let (mine, rest) = remaining.split_at_mut(job.offsets[end] - job.offsets[start]);
        pieces.push((start, end, mine));
        remaining = rest;
    }

    std::thread::scope(|scope| {
        let workers: Vec<_> = pieces
            .into_iter()
            .map(|(start, end, mine)| scope.spawn(move || fill_range(job, mine, start, end)))
            .collect();
        workers
            .into_iter()
            .filter_map(|worker| worker.join().expect("a pulse worker panicked"))
            .min()
    })
}

/// One thread's share: subfaults `start..end`, whose samples are exactly `mine`.
fn fill_range(job: Job<'_>, mine: &mut [f64], start: usize, end: usize) -> Option<usize> {
    let base = job.offsets[start];
    for subfault in start..end {
        let pulse = &mut mine[job.offsets[subfault] - base..job.offsets[subfault + 1] - base];
        if pulse.is_empty() {
            continue;
        }
        let slip = job.slip_m[subfault];
        let written = match job.shape {
            Shape::OliuP { beta } => oliu_p_into(
                pulse,
                slip,
                job.rise_time_s[subfault],
                beta[subfault],
                job.dt_s,
            ),
            Shape::Delta => {
                pulse.copy_from_slice(&[0.0, slip / job.dt_s, 0.0]);
                true
            }
        };
        if !written {
            return Some(subfault);
        }
    }
    None
}

/// The `OliuP` slip-rate function: a piecewise sinusoid after Liu, Archuleta &
/// Hartzell (2006).
///
/// Three pieces, of which `beta` sets the first two's extent. Writing `tau1` for
/// `beta * duration`:
///
/// * `0 .. tau1` — the rising limb, a raised cosine plus a half-period sine that
///   makes the rise sharper than the fall;
/// * `tau1 .. 2*tau1` — the peak and the start of the decay;
/// * `2*tau1 .. duration` — the tail, a quarter cosine to zero.
///
/// The result is normalised so `dt * sum` is `slip`, and one trailing zero closes the
/// pulse — a source-time function that ends at a non-zero rate is a step in velocity,
/// which radiates at every frequency and is not what the model means.
///
/// A duration of about one sample gives a fixed three-point spike rather than
/// anything computed — the shape is meaningless at that resolution, so a triangle
/// stands in, which is also what [`Shape::Delta`] is.
///
/// `values` is the caller's slice of the output buffer, already the length pass 1
/// worked out for this subfault, and already zero. Writing into it rather than
/// returning a `Vec` is what keeps the whole synthesis to one allocation: a fault of
/// two million subfaults was two million heap allocations and a copy of every sample
/// out of each. `false` means the pulse could not be normalised.
///
/// (orig. `gen_OliuP_stf`, `gslip_sliprate_subs.c`)
fn oliu_p_into(values: &mut [f64], slip: f64, duration_s: f64, beta: f64, dt_s: f64) -> bool {
    let count = samples(duration_s, dt_s);
    if count == 1 {
        // Too short to resolve. A fixed spike, not a computed shape.
        values.copy_from_slice(&[0.0, 1.0, 0.0]);
        return normalise(values, slip, dt_s);
    }

    let rise_end = beta * duration_s;
    let peak_end = 2.0 * rise_end;
    let decay_span = duration_s - rise_end;

    // One more sample than the duration covers, left at the zero the buffer arrived
    // with, so the pulse closes.
    for (index, value) in values.iter_mut().enumerate().take(count).skip(1) {
        let time = exact(index) * dt_s;
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
    normalise(values, slip, dt_s)
}

/// Scale so `dt · Σ values` is `slip`, or report that no scale can do it.
///
/// Shared so that "conserves slip" is one line of code rather than a property each
/// shape has to remember to have. A non-positive integral means the shape
/// degenerated at this resolution; `false` here becomes
/// [`Error::UnrepresentableRiseTime`] at the subfault that owns the pulse, because
/// only the caller knows which one that is.
fn normalise(values: &mut [f64], slip: f64, dt_s: f64) -> bool {
    let integral: f64 = values.iter().map(|value| dt_s * value).sum();
    if !integral.is_finite() || integral <= 0.0 {
        return false;
    }
    let scale = slip / integral;
    for value in values {
        *value *= scale;
    }
    true
}
