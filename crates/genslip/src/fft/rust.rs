//! The `rustfft` engine — no C library, and where this is going.
//!
//! `rustfft` matches FFTW's conventions in the two ways that matter here: it is
//! unnormalised in both directions, and its inverse is the conjugate-transpose of
//! its forward rather than a scaled one. So substituting it changes no convention,
//! only the rounding.
//!
//! It was not bit-identical to the FFTW engine it replaced, and no test asked it
//! to be: they agreed to 7.06e-08 relative, which is about an f64 ulp.
//! What `fft_contract.rs` asks is that both satisfy the same properties, and it
//! measures how far apart they are so the Stage 3 swap has a number to be judged
//! against.

use std::sync::Arc;

use num_complex::Complex64;
use rustfft::FftPlanner;

use super::{Direction, Fft};

/// A planned transform of one length and direction.
struct Plan {
    fft: Arc<dyn rustfft::Fft<f64>>,
    length: usize,
    direction: Direction,
    scratch: Vec<Complex64>,
}

/// `rustfft`, planning lazily and caching one plan per (length, direction).
///
/// Unlike FFTW, a `rustfft` plan is not tied to a buffer address, so this transforms
/// the caller's slice in place with no copying.
pub struct RustFft {
    planner: FftPlanner<f64>,
    plans: Vec<Plan>,
}

impl RustFft {
    /// A new engine with no plans yet.
    #[must_use]
    pub fn new() -> Self {
        Self {
            planner: FftPlanner::new(),
            plans: Vec::new(),
        }
    }

    fn plan_index(&mut self, length: usize, direction: Direction) -> usize {
        if let Some(index) = self
            .plans
            .iter()
            .position(|plan| plan.length == length && plan.direction == direction)
        {
            return index;
        }

        let fft = match direction {
            Direction::Forward => self.planner.plan_fft_forward(length),
            Direction::Inverse => self.planner.plan_fft_inverse(length),
        };
        let scratch = vec![Complex64::default(); fft.get_inplace_scratch_len()];
        self.plans.push(Plan {
            fft,
            length,
            direction,
            scratch,
        });
        self.plans.len() - 1
    }
}

impl Default for RustFft {
    fn default() -> Self {
        Self::new()
    }
}

/// Reports the cache state; `FftPlanner` is not `Debug` and the plans are opaque.
#[expect(
    clippy::missing_fields_in_debug,
    reason = "rustfft's planner and plans do not implement Debug"
)]
impl std::fmt::Debug for RustFft {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RustFft")
            .field("plans", &self.plans.len())
            .finish()
    }
}

impl Fft for RustFft {
    fn transform(&mut self, values: &mut [Complex64], direction: Direction) {
        let length = values.len();
        if length == 0 {
            return;
        }

        let index = self.plan_index(length, direction);
        let plan = &mut self.plans[index];
        plan.fft.process_with_scratch(values, &mut plan.scratch);
    }
}
