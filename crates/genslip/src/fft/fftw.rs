//! The FFTW engine — what genslip uses, and the reason bit-parity is achievable.
//!
//! **Stage 1 only.** Replaced by [`super::RustFft`] once the scientific suite is the
//! gate; nothing about the rest of the port depends on which engine is in use.
//!
//! # Why this is reproducible at all
//!
//! genslip plans with `FFTW_ESTIMATE`, which picks a plan from a cost model rather
//! than by timing candidates. `FFTW_MEASURE` would not be reproducible even against
//! itself — it benchmarks at plan time, so the same binary can pick different plans
//! on a loaded machine. Had genslip used it, bit-parity would have been off the table
//! entirely.
//!
//! # Alignment is part of the contract
//!
//! An `FFTW_ESTIMATE` plan records the alignment of the buffers it was planned with
//! and selects SIMD codelets accordingly, so a differently-aligned buffer can take a
//! different code path and round differently. genslip plans on a `malloc`ed buffer,
//! which glibc aligns to 16 bytes for these sizes, so this allocates to 16 too.
//!
//! # FFTW's planner is global mutable state, and is not thread-safe
//!
//! Planning mutates a process-wide wisdom table. Two threads planning at once
//! corrupt it, and FFTW notices: `planner.c:261: assertion failed: SLVNDX(slot) ==
//! slvndx`, then abort. genslip is one process doing one thing and never meets this;
//! a Rust test binary runs its tests in parallel and meets it immediately.
//!
//! `fftwf_make_planner_thread_safe` installs a lock inside FFTW around every
//! planner entry point, and this engine calls it once before planning anything. That
//! is better than a lock of our own for a reason worth knowing: the lock lives inside
//! libfftw3f, so it also covers planner calls made by *other* code linking the same
//! library — which during Stage 1 means the C oracle. A mutex on this side would
//! have protected only this side, and the two race.
//!
//! Execution is re-entrant and is not serialised; it is where the time goes.
//!
//! This is a real constraint on any caller wanting several faults in flight at once,
//! and it is the second reason (after the C dependency) that `RustFft` is where this
//! is going: `rustfft`'s planner needs no lock at all.

use std::alloc::{Layout, alloc, dealloc};
use std::ffi::{c_int, c_uint, c_void};
use std::ptr::NonNull;

use num_complex::Complex32;

use super::{Direction, Fft};

/// `fftwf_complex` is `float[2]`, which is `Complex32`'s layout.
type FftwComplex = [f32; 2];

#[expect(
    non_camel_case_types,
    reason = "opaque C handle, named as FFTW names it"
)]
enum fftwf_plan_s {}
type FftwPlan = *mut fftwf_plan_s;

const FFTW_FORWARD: c_int = -1;
const FFTW_BACKWARD: c_int = 1;
/// `1 << 6`. Plan from a cost model rather than by timing — see the module note.
const FFTW_ESTIMATE: c_uint = 1 << 6;

/// glibc's `malloc` guarantees this for any allocation of 16 bytes or more, and that
/// is what genslip's buffer gets.
const MALLOC_ALIGNMENT: usize = 16;

/// Installs FFTW's internal planner lock, once per process.
static MAKE_PLANNER_THREAD_SAFE: std::sync::Once = std::sync::Once::new();

unsafe extern "C" {
    fn fftwf_plan_dft_1d(
        n: c_int,
        input: *mut FftwComplex,
        output: *mut FftwComplex,
        sign: c_int,
        flags: c_uint,
    ) -> FftwPlan;
    fn fftwf_execute(plan: FftwPlan);
    fn fftwf_make_planner_thread_safe();
    fn fftwf_destroy_plan(plan: FftwPlan);
}

/// A planned transform of one length and direction.
struct Plan {
    handle: FftwPlan,
    length: usize,
    direction: Direction,
}

impl Drop for Plan {
    fn drop(&mut self) {
        // SAFETY: `plan` came from `fftwf_plan_dft_1d` and is destroyed once.
        // Destruction touches the same planner state as creation and is covered by
        // the same internal lock.
        //
        // The original leaks its first plan -- it overwrites the handle with the
        // second before destroying anything. Not reproduced: it is a resource leak,
        // not a numerical behaviour, and nothing observable depends on it.
        unsafe { fftwf_destroy_plan(self.handle) }
    }
}

/// FFTW, planning lazily and caching one plan per (length, direction).
///
/// The buffer is owned and reused: FFTW's `fftwf_execute` transforms the arrays the
/// plan was built with, so values are copied in and out around each call. genslip
/// does the same, which is why the copies are here rather than avoided.
pub struct FftwFft {
    buffer: NonNull<FftwComplex>,
    capacity: usize,
    plans: Vec<Plan>,
}

impl FftwFft {
    /// A new engine with no plans yet.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            buffer: NonNull::dangling(),
            capacity: 0,
            plans: Vec::new(),
        }
    }

    fn layout(capacity: usize) -> Layout {
        Layout::from_size_align(capacity * size_of::<FftwComplex>(), MALLOC_ALIGNMENT)
            .expect("FFT buffer layout is valid for any realistic length")
    }

    /// Ensure the scratch buffer holds at least `length` points.
    ///
    /// Reallocating invalidates every plan, because a plan is tied to the addresses
    /// it was built with. They are dropped rather than reused.
    fn reserve(&mut self, length: usize) {
        if length <= self.capacity {
            return;
        }

        self.plans.clear();
        let new = Self::layout(length);
        // SAFETY: `new` has non-zero size because `length > self.capacity >= 0`.
        #[expect(
            clippy::cast_ptr_alignment,
            reason = "the layout requests MALLOC_ALIGNMENT, far above [f32; 2]'s 4"
        )]
        let pointer = unsafe { alloc(new) }.cast::<FftwComplex>();
        let pointer = NonNull::new(pointer).expect("allocation failed");

        if self.capacity > 0 {
            // SAFETY: the old buffer was allocated with this exact layout and has
            // not been freed; no plan refers to it any more, they were just dropped.
            unsafe {
                dealloc(
                    self.buffer.as_ptr().cast::<u8>(),
                    Self::layout(self.capacity),
                );
            }
        }

        self.buffer = pointer;
        self.capacity = length;
    }

    /// The plan for this length and direction, building it if needed.
    fn plan_for(&mut self, length: usize, direction: Direction) -> FftwPlan {
        if let Some(existing) = self
            .plans
            .iter()
            .find(|plan| plan.length == length && plan.direction == direction)
        {
            return existing.handle;
        }

        let sign = match direction {
            Direction::Forward => FFTW_FORWARD,
            Direction::Inverse => FFTW_BACKWARD,
        };
        let n = c_int::try_from(length).expect("transform length must fit in a C int");

        // SAFETY: idempotent, and FFTW requires it before any planner call.
        MAKE_PLANNER_THREAD_SAFE.call_once(|| unsafe { fftwf_make_planner_thread_safe() });

        // SAFETY: the buffer holds at least `length` elements -- `reserve` ran
        // first -- and FFTW only reads its addresses at plan time. The planner's
        // own lock, installed above, covers the global state this mutates.
        let plan = unsafe {
            fftwf_plan_dft_1d(
                n,
                self.buffer.as_ptr(),
                self.buffer.as_ptr(),
                sign,
                FFTW_ESTIMATE,
            )
        };
        assert!(
            !plan.is_null(),
            "FFTW could not plan a length-{length} transform"
        );

        self.plans.push(Plan {
            handle: plan,
            length,
            direction,
        });
        plan
    }
}

impl Default for FftwFft {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for FftwFft {
    fn drop(&mut self) {
        // Plans first: they refer to the buffer.
        self.plans.clear();
        if self.capacity > 0 {
            // SAFETY: allocated with this layout, freed once, no plan outlives it.
            unsafe {
                dealloc(
                    self.buffer.as_ptr().cast::<u8>(),
                    Self::layout(self.capacity),
                );
            }
        }
    }
}

// SAFETY: `FftwFft` owns its buffer and plans exclusively and hands out no interior
// references. FFTW's planner is not thread-safe, but planning happens behind `&mut`.
unsafe impl Send for FftwFft {}

impl Fft for FftwFft {
    fn transform(&mut self, values: &mut [Complex32], direction: Direction) {
        let length = values.len();
        if length == 0 {
            return;
        }

        self.reserve(length);
        let plan = self.plan_for(length, direction);

        // SAFETY: the buffer holds at least `length` elements and `Complex32` has
        // the same layout as `fftwf_complex` -- two contiguous f32, real first.
        let scratch = unsafe { std::slice::from_raw_parts_mut(self.buffer.as_ptr(), length) };
        for (slot, value) in scratch.iter_mut().zip(values.iter()) {
            *slot = [value.re, value.im];
        }

        // SAFETY: `plan` was built for this buffer, this length and this direction,
        // and the buffer has not moved since -- `reserve` clears the plans when it
        // reallocates.
        unsafe { fftwf_execute(plan) };

        for (value, slot) in values.iter_mut().zip(scratch.iter()) {
            *value = Complex32::new(slot[0], slot[1]);
        }
    }
}

/// Reports the cache state rather than the buffer contents, which are scratch.
#[expect(
    clippy::missing_fields_in_debug,
    reason = "the buffer is transient scratch; printing it would be noise"
)]
impl std::fmt::Debug for FftwFft {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("FftwFft")
            .field("capacity", &self.capacity)
            .field("plans", &self.plans.len())
            .finish()
    }
}

const _: () = {
    assert!(size_of::<Complex32>() == size_of::<FftwComplex>());
    assert!(align_of::<Complex32>() == align_of::<FftwComplex>());
    // Silences the unused-import warning for `c_void` if the FFI shrinks.
    let _ = size_of::<*mut c_void>();
};
