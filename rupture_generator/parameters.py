import dataclasses
from collections.abc import Generator
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Nested:
    a: str
    b: str = field(metadata=dict(skip=True))
    c: str = field(metadata=dict(alias="hi"))


@dataclass
class Parameters:
    x: int
    y: int
    z: int
    n: Nested

    def to_cmd(self) -> dict[str, Any]:
        return dict(_unroll_dataclass(self))


def _unroll_dataclass(obj: Any) -> Generator[tuple[str, Any], None, None]:
    for field in dataclasses.fields(obj):
        metadata = field.metadata
        value = getattr(obj, field.name)
        if metadata.get("skip"):
            continue
        elif dataclasses.is_dataclass(value):
            yield from _unroll_dataclass(value)
        else:
            name = metadata.get("alias") or field.name
            if serializer := metadata.get("serializer"):
                value = serializer(value)

            yield name, value
