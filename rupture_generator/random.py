"""Which stream of numbers a calculation draws from, and where in it.

Noise is a pure function of ``(seed, realisation, calculation, segment)`` and of
nothing else. That is what makes the pipeline's order a convention rather than a
contract: reordering the calculations, re-batching them into one function, or changing
one's parameters cannot move another's numbers.

Keyed by **name**, not by position. Keying on a calculation's index in a fixed tuple
and a segment's index in a dict makes insertion order semantically significant --
inserting a fault renumbers every fault after it, and every field on them is redrawn.
That is exactly the accident naming the streams was meant to prevent, left in place
for the segments.
"""

from __future__ import annotations

import dataclasses
import hashlib

import numpy as np


def _key(name: str) -> int:
    """A name as a stable integer, for a spawn key.

    blake2b rather than the built-in :func:`hash`, which is randomised per process for
    strings unless ``PYTHONHASHSEED`` is set -- so the same seed would give two
    different earthquakes in two runs, which is the one thing a seed exists to prevent.

    Eight bytes is the whole of a name's identity here. A collision would hand two
    calculations one stream; over the handful of names in a rupture, at 2**64, it does
    not happen.
    """
    return int.from_bytes(hashlib.blake2b(name.encode(), digest_size=8).digest(), "big")


@dataclasses.dataclass(frozen=True)
class Streams:
    """The event's randomness, split by the name of whatever is drawing.

    Attributes
    ----------
    seed : int
        The event seed. Every stream in the run descends from it.
    realisation : int
        Which independent realisation of that seed -- what makes a campaign of
        ruptures on one fault reproducible one by one.
    """

    seed: int
    realisation: int = 0

    def stream(
        self, calculation: str, segment: str | None = None
    ) -> np.random.Generator:
        """The generator one calculation draws from, on one segment.

        Parameters
        ----------
        calculation : str
            ``"slip"``, ``"rise_time"``, ``"rake"``, ``"onset"``, ``"propagation"``.
            The *calculation's* name rather than the stage's, so batching several
            calculations into one pass over the segments moves no field's noise.
        segment : str, optional
            The segment's name, or omitted for a draw that belongs to no segment --
            the causality tree, which is one draw for the whole system. A two-element
            spawn key and a three-element one are different streams, so the omission
            collides with nothing.

        Returns
        -------
        np.random.Generator
        """
        spawn_key: tuple[int, ...] = (self.realisation, _key(calculation))
        if segment is not None:
            spawn_key += (_key(segment),)
        return np.random.default_rng(
            np.random.SeedSequence(entropy=self.seed, spawn_key=spawn_key)
        )


__all__ = ["Streams"]
