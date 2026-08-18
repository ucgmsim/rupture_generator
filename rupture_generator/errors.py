"""What this package raises, and what each type means.

One root, so a caller can tell *this package refused* from *this package broke*: the
CLI catches :class:`RuptureGeneratorError` and renders it, and lets everything else
traceback. Catching `Exception` there instead reported a bug in the generator as though
the user had mistyped their config.

The type carries the category and the message carries the specifics, so the classes
stay few and shallow. Where a builtin already means the right thing it is used
unwrapped: :class:`TypeError` for a dtype the kernels will not take, :class:`KeyError`
for a segment that is not in a realisation, :class:`FileNotFoundError` for a path.
"""

from __future__ import annotations


class RuptureGeneratorError(Exception):
    """Anything this package refuses to do, and the only thing the CLI catches."""


class ConfigError(RuptureGeneratorError):
    """The configuration does not describe an earthquake this can generate.

    Raised for what the config *says* rather than how it parses: a serialisation
    library's own exceptions are wrapped in this at the config boundary so they stop
    leaking into the CLI.
    """


class GeometryError(RuptureGeneratorError):
    """The fault geometry is not a chart.

    Planes that meet at a seam but are cut into different rows down dip, spacings too
    far apart to average, a surface that violates what the sampler assumes about its
    grid.
    """


class PropagationError(RuptureGeneratorError):
    """The rupture cannot propagate the way the configuration asks.

    Causality that is cyclic or disconnected, a jump with no target, a segment nothing
    triggers.
    """


class FormatError(RuptureGeneratorError):
    """A file cannot be read or written as asked.

    An extension that names no format, a schema version this does not know, a file
    that holds a fault surface where a rupture was wanted.
    """


class CapacityError(RuptureGeneratorError):
    """The job is too big for the machine, at this resolution.

    Not a bad value: the same configuration at coarser subfaults would be accepted.
    Its own type because it is the one refusal a batch driver can act on -- catch it,
    coarsen, retry -- and the message names both the size asked for and the limit.
    """


__all__ = [
    "CapacityError",
    "ConfigError",
    "FormatError",
    "GeometryError",
    "PropagationError",
    "RuptureGeneratorError",
]
