//! The fault grid every field lives on.
//!
//! Every field — slip, rake, the rupture-time and rise-time perturbations, the
//! roughness — is generated in the wavenumber domain on a rectangular grid, filtered
//! there, and transformed back. Both domains use the same shape, so they use the same
//! type.
//!
//! # The axis convention, written down once
//!
//! **A fault grid is `Array2` of shape `(dip_count, strike_count)`, and is indexed
//! `field[[dip, strike]]`.**
//!
//! That is the along-strike index varying fastest in memory, which is not an
//! implementation detail: it is the order the generators consume randomness in, the
//! order an SRF stores points in, and the layout the FFT expects. ndarray is
//! row-major, so "strike fastest" means strike is the *last* index — hence dip first.
//!
//! Two habits follow from it, and both exist because a transposition compiles.
//!
//! * **Index with `[[dip, strike]]`, never `[(dip, strike)]`.** ndarray accepts the
//!   tuple form too, and the old hand-rolled types used a tuple meaning
//!   `(strike, dip)`. The two are the same syntax with reversed meaning, so the
//!   bracket form is used everywhere to make a stale tuple a compile error rather
//!   than a silent transpose.
//! * **Build with [`zeros`] and [`from_values`], not `Array2::zeros`.** They take
//!   `(strike_count, dip_count)`, in that order, so a call site reads the way the
//!   rest of the crate does and the reversal happens in exactly one place.

use ndarray::{Array2, ArrayBase, Data, DataMut, Ix2};
use num_complex::Complex64;

/// A real field on the fault: slip, rake, rise time, a perturbation.
pub type SlipField = Array2<f64>;

/// A complex field in the wavenumber domain.
///
/// The same shape and layout as a [`SlipField`] — the two differ in what a value
/// *means*, and in the spectrum being padded past the fault's own extent.
pub type Spectrum = Array2<Complex64>;

/// Naming the axes of a fault grid.
///
/// `.nrows()` and `.ncols()` are correct and say nothing. On a fault the rows are dip
/// and the columns are strike, and every loop in this crate cares which — so the
/// accessors say so, and a reader does not have to remember the convention above to
/// follow a line of code.
pub trait FaultAxes {
    /// The element type, so [`flat`](FaultAxes::flat) can name it.
    type Value;

    /// Number of subfaults along strike. The fast axis.
    fn strike_count(&self) -> usize;
    /// Number of subfaults down dip. The slow axis.
    fn dip_count(&self) -> usize;

    /// `(strike_count, dip_count)`, in the order the rest of the crate writes them.
    fn extent(&self) -> (usize, usize) {
        (self.strike_count(), self.dip_count())
    }

    /// The grid as one contiguous slice, along-strike index fastest.
    ///
    /// What the FFT, the RNG and the SRF writer all consume. ndarray returns an
    /// `Option` here because a view can be strided; every grid this crate builds is
    /// owned and standard-layout, so a `None` is a bug rather than a case.
    fn flat(&self) -> &[Self::Value];
}

/// The mutable half, separate because it needs `DataMut` where reading does not.
pub trait FaultAxesMut: FaultAxes {
    /// The grid as one contiguous mutable slice, along-strike index fastest.
    fn flat_mut(&mut self) -> &mut [Self::Value];
}

impl<S: Data<Elem = A>, A> FaultAxes for ArrayBase<S, Ix2> {
    type Value = A;

    fn strike_count(&self) -> usize {
        self.ncols()
    }

    fn dip_count(&self) -> usize {
        self.nrows()
    }

    fn flat(&self) -> &[A] {
        self.as_slice()
            .expect("a fault grid is owned and standard-layout")
    }
}

impl<S: DataMut<Elem = A>, A> FaultAxesMut for ArrayBase<S, Ix2> {
    fn flat_mut(&mut self) -> &mut [A] {
        self.as_slice_mut()
            .expect("a fault grid is owned and standard-layout")
    }
}

/// A grid of zeros, `strike_count` across and `dip_count` down.
///
/// The argument order is the crate's, not ndarray's. See the module note.
#[must_use]
pub fn zeros<A: Clone + num_traits::Zero>(strike_count: usize, dip_count: usize) -> Array2<A> {
    Array2::zeros((dip_count, strike_count))
}

/// A grid from values already in along-strike-fastest order.
///
/// # Panics
///
/// If `values` does not hold exactly `strike_count * dip_count` entries.
#[must_use]
pub fn from_values<A>(strike_count: usize, dip_count: usize, values: Vec<A>) -> Array2<A> {
    let expected = strike_count * dip_count;
    assert_eq!(
        values.len(),
        expected,
        "got {} values for a {strike_count}x{dip_count} grid",
        values.len()
    );
    Array2::from_shape_vec((dip_count, strike_count), values)
        .expect("length checked immediately above")
}

/// A spectrum of zeros, checked for the extents the generators require.
///
/// # Panics
///
/// If either extent is zero or odd. The generators address the Nyquist row and column
/// directly (`strike_count / 2`, `dip_count / 2`) and enforce Hermitian symmetry about
/// them, which only makes sense for an even extent.
///
/// An **assertion rather than an error**, because the extents reach here from a
/// `FaultGrid` that `realisation::check` has already validated — see `error.rs` on
/// where that line sits. A failure here is this crate's bug, not a caller's.
///
/// This is not a restriction the port invents. genslip sizes all four of its padded
/// grids and then rounds each up to even (`if(nstk2%2) nstk2++;` and its three
/// siblings, `genslip_v5.6.2.c:1471-1490`), so an odd extent means the sizing is wrong
/// rather than that the grid should cope with it.
#[must_use]
pub fn spectrum(strike_count: usize, dip_count: usize) -> Spectrum {
    assert!(
        strike_count > 0 && dip_count > 0,
        "spectrum extents must be non-zero, got {strike_count}x{dip_count}"
    );
    assert!(
        strike_count.is_multiple_of(2) && dip_count.is_multiple_of(2),
        "spectrum extents must be even, got {strike_count}x{dip_count}"
    );
    zeros(strike_count, dip_count)
}

/// Impose Hermitian symmetry, so the inverse transform is real-valued.
///
/// The generators fill only the non-negative dip half of the grid; this reflects it
/// into the negative half with conjugation. A real field has `F(-k) = conj(F(k))`,
/// so without this the inverse transform would carry an imaginary part and the
/// resulting slip distribution would be complex.
///
/// The three passes handle, in order: the zero and Nyquist dip rows; the zero and
/// Nyquist strike columns; and the interior. The interior pass reflects both
/// diagonals — `(strike, dip)` into `(-strike, -dip)` *and* `(-strike, dip)` into
/// `(strike, -dip)` — because both are needed to fill the negative-dip half.
///
/// (orig. slip.c:2003)
pub fn impose_hermitian_symmetry(spectrum: &mut Spectrum) {
    let strike_count = spectrum.strike_count();
    let dip_count = spectrum.dip_count();
    let strike_nyquist = strike_count / 2;
    let dip_nyquist = dip_count / 2;

    // The zero-dip and Nyquist-dip rows.
    for strike in 1..strike_nyquist {
        spectrum[[0, strike_count - strike]] = spectrum[[0, strike]].conj();
        spectrum[[dip_nyquist, strike_count - strike]] = spectrum[[dip_nyquist, strike]].conj();
    }

    // The zero-strike and Nyquist-strike columns.
    for dip in 1..dip_nyquist {
        spectrum[[dip_count - dip, 0]] = spectrum[[dip, 0]].conj();
        spectrum[[dip_count - dip, strike_nyquist]] = spectrum[[dip, strike_nyquist]].conj();
    }

    // The interior, both diagonals.
    for dip in 1..dip_nyquist {
        for strike in 1..strike_nyquist {
            spectrum[[dip_count - dip, strike_count - strike]] = spectrum[[dip, strike]].conj();
            spectrum[[dip_count - dip, strike]] = spectrum[[dip, strike_count - strike]].conj();
        }
    }
}
