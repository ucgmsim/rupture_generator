"""Tests for rupture_generator.utils option serialisation.

The contract these pin is genslip's, not ours: arguments are ``name=value`` pairs
consumed by ``getpar``, which only overwrites a variable when it finds the name.
So *omitting* an argument is how a built-in default is requested, and emitting a
name with an unparseable value is an error rather than a no-op.
"""

from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from rupture_generator.config import (
    KModel,
    RiseTimeNormalisation,
    SlipRateFunction,
    Stype,
)
from rupture_generator.utils import _serialise_value, serialise_options


class TestSerialiseValue:
    def test_bool_is_zero_or_one(self) -> None:
        # genslip has no boolean type; every flag is an int getpar.
        assert _serialise_value(True) == "1"
        assert _serialise_value(False) == "0"

    def test_enum_unwraps_to_its_value(self) -> None:
        assert _serialise_value(KModel.MAI) == "2"
        assert _serialise_value(RiseTimeNormalisation.SLIP_WEIGHTED) == "1"

    def test_the_two_stype_vocabularies_are_distinct(self) -> None:
        """`stype` means different things to genslip and to generic_slip2srf.

        Both binaries take an argument spelled `stype`; the sets overlap only at
        brune/urs/ucsb. genslip's default OliuP2 is not a generic_slip2srf value,
        and the configured point-source `cos` is not a genslip value.
        """
        assert _serialise_value(SlipRateFunction.oliu_p2) == "OliuP2"
        assert _serialise_value(Stype.cos) == "cos"

        genslip = {member.value for member in SlipRateFunction}
        point_source = {member.value for member in Stype}
        assert genslip & point_source == {"brune", "urs", "ucsb"}
        assert "OliuP2" not in point_source
        assert "cos" not in genslip

    def test_list_is_comma_joined(self) -> None:
        # `gwid` and `rvfac_seg` are getpar "vf" vectors, which are comma-separated.
        assert _serialise_value([1.0, 2.0, 3.0]) == "1.0,2.0,3.0"

    def test_path_becomes_its_string(self) -> None:
        assert _serialise_value(Path("/tmp/fault.gsf")) == "/tmp/fault.gsf"

    def test_unsupported_type_raises(self) -> None:
        with pytest.raises(TypeError, match="Unsupported type"):
            _serialise_value({"not": "serialisable"})


class TestSerialiseOptions:
    def test_none_is_omitted(self) -> None:
        assert serialise_options({"kx_corner": None}) == []

    def test_empty_list_is_omitted(self) -> None:
        # Regression: an empty list used to fall through to the `case _` arm and
        # raise TypeError. `gwid` and `rvfac_seg` both default to [], so a default
        # configuration could not be serialised at all.
        assert serialise_options({"gwid": [], "rvfac_seg": []}) == []

    def test_zero_and_false_are_not_omitted(self) -> None:
        # The falsey-but-meaningful case. `seg_delay=0` and `rupture_delay=0.0` are
        # real instructions, not absent values.
        assert serialise_options({"seg_delay": False, "rupture_delay": 0.0}) == [
            "seg_delay=0",
            "rupture_delay=0.0",
        ]

    def test_empty_string_is_not_omitted(self) -> None:
        assert serialise_options({"velfile": ""}) == ["velfile="]


NAMES = st.text(
    alphabet=st.characters(min_codepoint=97, max_codepoint=122), min_size=1, max_size=12
)
VALUES = st.one_of(
    st.booleans(),
    st.integers(min_value=-(2**31), max_value=2**31 - 1),
    st.floats(allow_nan=False, allow_infinity=False),
    st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1, max_size=4),
    st.sampled_from(list(KModel)),
    st.none(),
)


@given(options=st.dictionaries(NAMES, VALUES, max_size=20))
def test_every_argument_has_exactly_one_equals_prefix(
    options: dict[str, object],
) -> None:
    """Each rendered argument splits into a name genslip can look up.

    getpar splits on the *first* ``=``, so a value containing one is fine; a name
    containing one is not. Names here are alphabetic, so the invariant is that the
    part before the first ``=`` is exactly the key.
    """
    for key, rendered in zip(
        [k for k, v in options.items() if v is not None and v != []],
        serialise_options(options),
        strict=True,
    ):
        assert rendered.startswith(f"{key}=")


@given(options=st.dictionaries(NAMES, VALUES, max_size=20))
def test_serialisation_is_deterministic(options: dict[str, object]) -> None:
    assert serialise_options(options) == serialise_options(options)


@given(options=st.dictionaries(NAMES, VALUES, max_size=20))
def test_omitted_arguments_are_exactly_the_absent_ones(
    options: dict[str, object],
) -> None:
    """Nothing is dropped except None and []; nothing else is invented."""
    expected = {k for k, v in options.items() if v is not None and v != []}
    emitted = {argument.split("=", 1)[0] for argument in serialise_options(options)}
    assert emitted == expected


# Deliberately not asserted:
#
# - The *formatting* of a float. `str(0.1)` is "0.1" and genslip's `atof` accepts
#   whatever Python produces, but the exact repr is not a contract and pinning it
#   would break on a Python version bump. genslip's own `xshift`/`yshift` are read
#   as strings and `atof`'d precisely to dodge this question.
# - Ordering. `to_cmd()` is deterministic (see test_unroll) but getpar does not
#   care about argument order, so no test should depend on it.
