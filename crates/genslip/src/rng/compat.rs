//! genslip v5.6.2's generator, reproduced exactly.
//!
//! A 31-bit truncated linear congruential generator, with normal deviates formed by
//! the Irwin-Hall construction — the sum of twelve uniforms, shifted and scaled.
//!
//! Neither choice is one you would make today, and neither is defended here. They
//! are reproduced because the *stream* is the compatibility contract: a rupture
//! model matches the original only if every consumer draws the same quantity of
//! randomness in the same order. Use [`super::Pcg`] for new work.
//!
//! (orig. misc.c:48)

use super::{DrawSource, Realisations};

/// Multiplier and increment of the ANSI C `rand()` recurrence.
const MULTIPLIER: i64 = 1_103_515_245;
const INCREMENT: i64 = 12_345;

/// The recurrence is truncated to 31 bits.
///
/// The C evaluates it in `long` — 64-bit on every platform this has run on — but
/// that turns out not to matter: the mask sits below the word size, and the low
/// bits of a product depend only on the low bits of its operands, so 32-bit and
/// 64-bit evaluation give the same sequence. Verified, not assumed.
const STATE_MASK: i64 = 0x7fff_ffff;

/// The state spans `[0, 2^31)` but is divided by `2^30`, which after the shift maps
/// it onto `[-1, 1]` — closed at the top, not half-open.
const HALF_STATE_RANGE: f64 = 1_073_741_824.0;

/// Uniforms summed to form one normal deviate.
///
/// Twelve is not arbitrary: a sum of twelve uniforms on `[0, 1)` has variance
/// exactly 1, so subtracting the mean of 6 gives a unit-variance approximate normal
/// with no scaling constant. It is also the draw cost of every normal in the
/// program, and so part of the stream contract.
const UNIFORMS_PER_GAUSSIAN: usize = 12;

/// Draws burned to reach each successive slip realisation.
///
/// genslip rewinds to the starting seed and skips this many draws per realisation
/// rather than generating the preceding ones (`genslip_v5.6.2.c:1667-1669`).
const DRAWS_PER_REALISATION: usize = 10;

/// genslip's linear congruential generator, bit-compatible with the C.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GenslipLcg {
    /// The seed this stream started from, kept so realisations can rewind to it.
    starting_seed: i64,
    state: i64,
}

impl GenslipLcg {
    /// Start a stream at `seed`.
    #[must_use]
    pub const fn new(seed: i64) -> Self {
        Self {
            starting_seed: seed,
            state: seed,
        }
    }

    /// The current state, in the representation the C writes out.
    ///
    /// genslip can dump its final seed to a file (`dump_last_seed`), so comparing
    /// this against that file is how draw consumption is audited against the
    /// original.
    #[must_use]
    pub const fn seed(self) -> i64 {
        self.state
    }

    /// Advance by `count` uniforms, discarding them.
    pub const fn discard(&mut self, count: usize) {
        let mut remaining = count;
        while remaining > 0 {
            self.uniform_deviate();
            remaining -= 1;
        }
    }

    /// The recurrence itself, as a `const fn` so [`discard`](Self::discard) can be
    /// one too.
    const fn uniform_deviate(&mut self) -> f64 {
        self.state = (self.state * MULTIPLIER + INCREMENT) & STATE_MASK;
        // The state is 31 bits by `STATE_MASK`, so this is exact -- and it is an
        // `i64` rather than a count, so it does not go through `units::exact`.
        #[expect(
            clippy::cast_precision_loss,
            reason = "31 bits of state, exactly representable in f64"
        )]
        let state = self.state as f64;
        state / HALF_STATE_RANGE - 1.0
    }
}

impl DrawSource for GenslipLcg {
    fn uniform(&mut self) -> f64 {
        self.uniform_deviate()
    }

    fn gaussian(&mut self, sigma: f64, mean: f64) -> f64 {
        let mut sum = 0.0;
        for _ in 0..UNIFORMS_PER_GAUSSIAN {
            sum += 0.5 * (1.0 + self.uniform_deviate());
        }
        (sum - 6.0) * sigma + mean
    }
}

impl Realisations for GenslipLcg {
    fn realisation(&self, index: u64) -> Self {
        let mut stream = Self::new(self.starting_seed);
        stream.discard(DRAWS_PER_REALISATION * usize::try_from(index).unwrap_or(usize::MAX));
        stream
    }
}
