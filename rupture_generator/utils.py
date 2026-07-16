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
    return [
        f"{key}={_serialise_value(value)}"
        for key, value in options.items()
        if value is not None
    ]
