"""Units and conversions, named once each.

genslip works in **CGS** -- centimetres, grams, seconds, and therefore dyne-cm for
moment and dyne/cm^2 for rigidity. Its inputs are in kilometres and km/s, because
that is how a fault and a velocity model are written down. Every conversion in the
package is a consequence of that one mismatch.

The Rust core has the same constants in `crates/genslip/src/units.rs`, with the
derivations. These are the ones the Python side needs; `tests/test_units.py` asserts
the two agree, so a change on either side that does not reach the other goes red.
"""

import numpy as np

CM_PER_KM = 1.0e5
"""Centimetres per kilometre. The root of every length conversion here."""

CM2_PER_KM2 = CM_PER_KM * CM_PER_KM
"""Square centimetres per square kilometre -- subfault areas, as an SRF stores them."""

SRF_FLOAT = np.float32
"""What an SRF file's numbers are.

The core computes in float64 and this is where that stops. An SRF writes `%13.5e`,
six significant figures, so float32 is the format's own resolution rather than a
shortcut -- see `_core.pyi`.
"""
