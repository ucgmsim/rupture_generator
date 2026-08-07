from enum import Enum
from pathlib import Path
from typing import Any


def _serialise_value(value: Any) -> str:
    match value:
        case Enum(value=inner):
            return _serialise_value(inner)
        case str():
            return value
        case bool():
            return str(int(value))
        case int() | float() | Path():
            return str(value)
        case [_x, *_xs]:
            return ",".join(_serialise_value(x) for x in value)
        case _:
            e = TypeError(f"Unsupported type {type(value)!r}")
            e.add_note(f"{value=}")
            raise e


def serialise_options(options: dict[str, Any]) -> list[str]:
    """Render options as genslip ``name=value`` arguments.

    ``None`` and empty lists are omitted rather than rendered. Both mean "say
    nothing and let genslip apply its own default": ``getpar`` only overwrites a
    variable when it finds the name on the command line, so an absent argument
    *is* how a default is requested. ``gwid=`` with no value would be a parse
    error, and ``gwid`` defaults to an empty list.

    Parameters
    ----------
    options : dict[str, Any]
        Mapping of genslip parameter name to value.

    Returns
    -------
    list[str]
        One ``name=value`` string per option that has a value.
    """
    return [
        f"{key}={_serialise_value(value)}"
        for key, value in options.items()
        if value is not None and value != []
    ]
