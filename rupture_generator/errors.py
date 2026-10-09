"""The one error type the package raises for input no rupture answers to."""


class RuptureGeneratorError(ValueError):
    """A fault, model or parameter set that does not describe a rupture.

    Everything else that escapes the package is a bug in it.
    """
