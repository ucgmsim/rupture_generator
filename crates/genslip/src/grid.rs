//! The wavenumber grid the stochastic fields live on.
//!
//! Every field — slip, rake, the rupture-time and rise-time perturbations, the
//! roughness — is generated in the wavenumber domain on a rectangular grid, filtered
//! there, and transformed back. The grid is stored with the along-strike index
//! varying fastest, which is also the order deviates are drawn in, so iteration
//! order and draw order are the same thing.

use num_complex::Complex32;

/// A complex field on a rectangular wavenumber grid.
///
/// Stored row-major with the along-strike index fastest: element `(strike, dip)`
/// lives at `strike + dip * strike_count`. That layout is not an implementation
/// detail — it is the order the generators consume randomness in, and it is the
/// layout the FFT expects.
#[derive(Clone, Debug, PartialEq)]
pub struct Spectrum {
    strike_count: usize,
    dip_count: usize,
    values: Vec<Complex32>,
}

impl Spectrum {
    /// A grid of zeros.
    ///
    /// # Panics
    ///
    /// If either dimension is zero, or is odd. The generators address the Nyquist
    /// row and column directly (`strike_count / 2`, `dip_count / 2`) and enforce
    /// Hermitian symmetry about them, which only makes sense for even extents.
    ///
    /// This is not a restriction the port invents. genslip sizes all four of its
    /// padded grids and then rounds each up to even (`if(nstk2%2) nstk2++;` and its
    /// three siblings, `genslip_v5.6.2.c:1471-1490`), so an odd extent means the
    /// sizing is wrong rather than that the grid should cope with it.
    #[must_use]
    pub fn zeros(strike_count: usize, dip_count: usize) -> Self {
        assert!(
            strike_count > 0 && dip_count > 0,
            "spectrum extents must be non-zero, got {strike_count}x{dip_count}"
        );
        assert!(
            strike_count.is_multiple_of(2) && dip_count.is_multiple_of(2),
            "spectrum extents must be even, got {strike_count}x{dip_count}"
        );
        Self {
            strike_count,
            dip_count,
            values: vec![Complex32::default(); strike_count * dip_count],
        }
    }

    /// Number of samples along strike.
    #[must_use]
    pub const fn strike_count(&self) -> usize {
        self.strike_count
    }

    /// Number of samples down dip.
    #[must_use]
    pub const fn dip_count(&self) -> usize {
        self.dip_count
    }

    /// Flat index of `(strike, dip)`.
    #[must_use]
    pub const fn offset(&self, strike: usize, dip: usize) -> usize {
        strike + dip * self.strike_count
    }

    /// The grid as one flat slice, along-strike index fastest.
    #[must_use]
    pub fn as_slice(&self) -> &[Complex32] {
        &self.values
    }

    /// The grid as one mutable flat slice.
    pub fn as_mut_slice(&mut self) -> &mut [Complex32] {
        &mut self.values
    }
}

impl std::ops::Index<(usize, usize)> for Spectrum {
    type Output = Complex32;

    fn index(&self, (strike, dip): (usize, usize)) -> &Complex32 {
        &self.values[self.offset(strike, dip)]
    }
}

impl std::ops::IndexMut<(usize, usize)> for Spectrum {
    fn index_mut(&mut self, (strike, dip): (usize, usize)) -> &mut Complex32 {
        let offset = self.offset(strike, dip);
        &mut self.values[offset]
    }
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
        spectrum[(strike_count - strike, 0)] = spectrum[(strike, 0)].conj();
        spectrum[(strike_count - strike, dip_nyquist)] = spectrum[(strike, dip_nyquist)].conj();
    }

    // The zero-strike and Nyquist-strike columns.
    for dip in 1..dip_nyquist {
        spectrum[(0, dip_count - dip)] = spectrum[(0, dip)].conj();
        spectrum[(strike_nyquist, dip_count - dip)] = spectrum[(strike_nyquist, dip)].conj();
    }

    // The interior, both diagonals.
    for dip in 1..dip_nyquist {
        for strike in 1..strike_nyquist {
            spectrum[(strike_count - strike, dip_count - dip)] = spectrum[(strike, dip)].conj();
            spectrum[(strike, dip_count - dip)] = spectrum[(strike_count - strike, dip)].conj();
        }
    }
}
