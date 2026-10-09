"""The one error type the package raises for input that describes no rupture."""


class RuptureGeneratorError(ValueError):
    """A fault, model or parameter set that describes no rupture.

    Any other exception that escapes the package is a bug in it.
    """
