import dataclasses
from collections.abc import Generator
from typing import Any


class _ValidateMixin:
    def __post_init__(self):
        for f in dataclasses.fields(self):
            if validator := f.metadata.get("validator"):
                value = getattr(self, f.name)
                if value is not None:
                    setattr(self, f.name, validator(value))


def _unroll_dataclass(obj: Any) -> Generator[tuple[str, Any], None, None]:
    for f in dataclasses.fields(obj):
        metadata = f.metadata
        value = getattr(obj, f.name)
        if metadata.get("skip"):
            continue
        elif dataclasses.is_dataclass(value):
            yield from _unroll_dataclass(value)
        else:
            name = metadata.get("alias") or f.name
            if serializer := metadata.get("serializer"):
                value = serializer(value)
            yield name, value
