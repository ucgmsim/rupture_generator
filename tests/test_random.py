"""Properties of the substreams: what a field's noise is a function of, and what it is not.

Every assertion here is of one shape -- two streams that should agree do, or two that
should differ do. That is the whole contract, and it is worth stating as tests because
the alternative failure is silent: a rupture that is still perfectly plausible, still
reproducible within one process, and simply not the rupture the seed names.

The one asymmetry worth reading twice is
:func:`test_renaming_one_segment_leaves_its_siblings_alone`. Position-keyed streams did
not have that property, and nothing in the pipeline would have reported its absence.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from rupture_generator.random import Streams, _key

CALCULATIONS = ("propagation", "slip", "rise_time", "rake", "onset")


def _draw(streams: Streams, calculation: str, segment: str | None = None) -> np.ndarray:
    """A few numbers off one stream -- enough that two streams agreeing is not luck."""
    return streams.stream(calculation, segment).standard_normal(8)


# ============================================================================
# What the noise is a function of
# ============================================================================


def test_one_seed_gives_one_earthquake() -> None:
    """The same key, twice, gives the same numbers. The point of a seed."""
    first = Streams(seed=1234, realisation=0)
    second = Streams(seed=1234, realisation=0)

    for calculation in CALCULATIONS:
        assert np.array_equal(
            _draw(first, calculation, "hope"), _draw(second, calculation, "hope")
        )


def test_each_calculation_draws_from_its_own_stream() -> None:
    """Two calculations on one segment share a seed and nothing else.

    This is what lets a stage be added, reordered, or -- as the fields batch does --
    merged with its neighbours, without moving any other field's values.
    """
    streams = Streams(seed=1234)
    drawn = [_draw(streams, calculation, "hope") for calculation in CALCULATIONS]

    for first in range(len(drawn)):
        for second in range(first + 1, len(drawn)):
            assert not np.array_equal(drawn[first], drawn[second])


def test_each_segment_draws_from_its_own_stream() -> None:
    """One calculation on two faults draws twice, not once.

    Two faults handed one stream would carry the same slip pattern up to their
    different shapes -- which looks like a correlation nobody asked for.
    """
    streams = Streams(seed=1234)

    assert not np.array_equal(
        _draw(streams, "slip", "kaikoura:0"), _draw(streams, "slip", "kaikoura:1")
    )


def test_a_realisation_is_an_independent_earthquake() -> None:
    """The realisation index moves every stream. That is what a campaign varies."""
    first = Streams(seed=1234, realisation=0)
    second = Streams(seed=1234, realisation=1)

    assert not np.array_equal(_draw(first, "slip", "hope"), _draw(second, "slip", "hope"))


def test_an_event_level_draw_is_not_any_segments() -> None:
    """The causality tree belongs to the system, so it names no segment.

    A two-element spawn key and a three-element one are different streams, so omitting
    the segment is not the same as passing some particular one.
    """
    streams = Streams(seed=1234)
    event = _draw(streams, "propagation")

    for segment in ("hope", "kaikoura:0", ""):
        assert not np.array_equal(event, _draw(streams, "propagation", segment))


# ============================================================================
# What the noise is *not* a function of
# ============================================================================


def test_renaming_one_segment_leaves_its_siblings_alone() -> None:
    """A fault's noise depends on its own name and on no other fault's.

    The property position-keying did not have. Keyed by position, inserting a fault --
    or renaming one in a way that reorders the dict -- renumbers every fault after it,
    and silently redraws every field on all of them. Here, ``hope`` draws what ``hope``
    draws, whatever it is standing next to.
    """
    streams = Streams(seed=1234)
    alone = _draw(streams, "slip", "hope")

    for sibling in ("kaikoura", "alpine", "wairau", "aaaaa"):
        assert np.array_equal(alone, _draw(streams, "slip", "hope"))
        # And the sibling itself draws something else, so the comparison is not
        # vacuous -- both names are live.
        assert not np.array_equal(alone, _draw(streams, "slip", sibling))


def test_the_order_streams_are_asked_for_does_not_matter() -> None:
    """Streams are addressed, not consumed in sequence.

    `Streams` holds no generator state; every call builds its own from the key. So a
    caller that draws rake before slip gets exactly what one drawing slip first gets,
    which is what makes the pipeline's stage order a convention.
    """
    streams = Streams(seed=1234)

    forwards = [_draw(streams, name, "hope") for name in CALCULATIONS]
    backwards = [_draw(streams, name, "hope") for name in reversed(CALCULATIONS)]

    assert all(
        np.array_equal(first, second)
        for first, second in zip(forwards, reversed(backwards), strict=True)
    )


# ============================================================================
# The key itself
# ============================================================================


def test_a_name_keys_the_same_stream_in_every_process() -> None:
    """`_key` is a hash of the name, not Python's -- which is salted per process.

    Run in a subprocess with a hostile ``PYTHONHASHSEED``, because that is the failure:
    within one process the built-in ``hash`` is perfectly consistent, so a test in this
    interpreter would pass on an implementation that reproduces nothing.
    """
    program = (
        "from rupture_generator.random import _key; print(_key('hope'), _key('slip'))"
    )
    runs = {
        subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            text=True,
            check=True,
            env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": seed},
        ).stdout.strip()
        for seed in ("0", "1", "12345")
    }

    assert len(runs) == 1, f"the key moved between processes: {runs}"
    assert runs == {f"{_key('hope')} {_key('slip')}"}


@pytest.mark.parametrize("name", ["hope", "kaikoura:0", "", "a" * 500, "wairau  "])
def test_a_key_fits_a_spawn_key(name: str) -> None:
    """Eight bytes, unsigned: what `np.random.SeedSequence` takes as a spawn entry.

    A key wider than 64 bits raises there rather than being truncated, and a name of
    any length has to be usable -- segment names come from a config file.
    """
    key = _key(name)

    assert 0 <= key < 2**64
    np.random.SeedSequence(entropy=1234, spawn_key=(0, key))
