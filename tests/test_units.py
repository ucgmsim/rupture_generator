"""The two sides agree about what a kilometre is.

`rupture_generator/units.py` and `crates/genslip/src/units.rs` define the same
conversions, because both sides need them and neither can import the other's. That
duplication is the point of this file: a change to one that does not reach the other
goes red here rather than showing up as a rupture off by a factor of ten.

Ten is the realistic error. Getting `1e5` wrong by one decade is a factor of 100 in
an area, and an entire magnitude unit in the seismic moment that area feeds -- from a
literal nobody would look at twice, which is why neither side has one any more.
"""

import re
from pathlib import Path

import numpy as np
import pytest

from rupture_generator import units

RUST = (
    Path(__file__).resolve().parent.parent / "crates" / "genslip" / "src" / "units.rs"
)


def rust_constant(name: str) -> float:
    """The value of a `pub const` in the Rust units module.

    Read rather than linked, because the alternative is exposing arithmetic constants
    through PyO3 for a test's benefit, which would make the boundary wider to check
    that it is narrow.
    """
    source = RUST.read_text()
    match = re.search(rf"pub const {name}: f64 = ([^;]+);", source)
    assert match, f"{name} is not a `pub const` in {RUST.name}"

    # The derived ones are written as products of other constants, which is the whole
    # point of them. A product is all the grammar this needs, so it is a `split` rather
    # than an `eval` -- a test that runs arbitrary code from a source file to check a
    # constant has the wrong risk profile for what it buys.
    value = 1.0
    for factor in match.group(1).split("*"):
        factor = factor.strip()
        try:
            value *= float(factor)
        except ValueError:
            value *= rust_constant(factor)
    return value


@pytest.mark.parametrize("name", ["CM_PER_KM", "CM2_PER_KM2"])
def test_the_two_units_modules_agree(name: str) -> None:
    assert getattr(units, name) == rust_constant(name)


def test_a_square_kilometre_is_ten_billion_square_centimetres() -> None:
    assert units.CM2_PER_KM2 == pytest.approx(1.0e10)


def test_the_derived_constant_is_derived() -> None:
    """Not a literal. A reader can check the relationship rather than the decimal."""
    assert units.CM2_PER_KM2 == units.CM_PER_KM**2
    assert "CM_PER_KM * CM_PER_KM" in RUST.read_text()


def test_an_srf_stores_single_precision() -> None:
    """The core is float64 and the file is not, and that is the format's own limit.

    An SRF writes six significant figures, so the extra digits have nowhere to go.
    `assemble.py` is the one place the two meet.
    """
    assert units.SRF_FLOAT is np.float32
