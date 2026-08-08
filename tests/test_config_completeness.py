"""The configuration *is* the compiled core's types, checked mechanically.

`README.md` and `tests/harness/README.md` both state the rule:

    The configuration **is** the compiled core's types. [...] the moment they appear in
    the library there are two descriptions of a rupture model and they start to drift.

`config/rupture.py` is written under it -- every field carries the same name and unit as
the ``_core`` constructor argument it feeds. That is a discipline, and a discipline is
worth exactly as much as the thing that checks it. This is the thing.

# How

The stub is parsed for each spec's ``__init__`` signature, the config class for its
dataclass fields, and the two are compared as sets. A field added to the core and
forgotten here goes red; so does a field here the core does not have, which is the case
that produces a config key that is *read and silently ignored*.

`README.md` records what that costs when nothing checks: genslip's ``getpar`` never asks
for names it does not recognise, so five parameters have been silently discarded in
production for as long as the workflow has pointed at a binary that does not know them.

The stub rather than the extension because the extension is compiled and PyO3 keyword
names are not introspectable from Python. `tests/test_boundary.py` separately asserts
the stub describes the extension member for member, so the chain closes:
extension -> stub -> config.
"""

import ast
import dataclasses
from pathlib import Path

import pytest

from rupture_generator import _core
from rupture_generator.config import rupture as config

STUB = ast.parse((Path(_core.__file__).parent / "_core.pyi").read_text())

TAG_FIELDS = frozenset({"type"})
"""The discriminator tag. It names *which* config class this is, not a core argument."""

# Every spec the core takes, against the config class that transliterates it. Kept as a
# table rather than derived, because being explicit about which pairs are claimed to
# correspond is the whole content of the claim.
PAIRS = [
    ("SourceSpec", config.FiniteSourceConfig),
    ("PointSourceSpec", config.PointSourceConfig),
    ("SlipSpec", config.SlipConfig),
    ("TimingSpec", config.TimingConfig),
    ("VelocityModel1D", config.VelocityModelConfig),
    ("Ramp", config.RampConfig),
]


def stub_arguments(class_name: str) -> set[str]:
    """The keyword names of a stub class's ``__init__``, excluding ``self``."""
    for node in STUB.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                arguments = item.args
                names = [
                    argument.arg
                    for argument in (
                        *arguments.posonlyargs,
                        *arguments.args,
                        *arguments.kwonlyargs,
                    )
                ]
                return {name for name in names if name != "self"}
        raise AssertionError(f"{class_name} has no __init__ in the stub")
    raise AssertionError(f"{class_name} is not in the stub")


def config_fields(config_class: type) -> set[str]:
    """The dataclass field names of a config class, excluding its discriminator tag."""
    return {
        field.name
        for field in dataclasses.fields(config_class)
        if field.name not in TAG_FIELDS
    }


BY_NAME = pytest.mark.parametrize(
    ("spec_name", "config_class"),
    PAIRS,
    ids=[name for name, _ in PAIRS],
)


@BY_NAME
def test_the_config_describes_every_argument_the_core_takes(
    spec_name: str, config_class: type
) -> None:
    """Nothing the core accepts is unreachable from a config file.

    A missing field is a parameter nobody can set: the generator has a knob and the
    format has no way to turn it, so whatever default the constructor carries is the
    only value anyone ever gets.
    """
    missing = stub_arguments(spec_name) - config_fields(config_class)
    assert not missing, (
        f"{config_class.__name__} cannot set {sorted(missing)}, which {spec_name} takes"
    )


@BY_NAME
def test_the_config_describes_nothing_the_core_does_not_take(
    spec_name: str, config_class: type
) -> None:
    """Nothing in a config file is read and then ignored.

    The failure mode `README.md` records from genslip: a name the reader does not
    recognise is silently discarded, and the run proceeds with a different earthquake
    than the file describes.
    """
    extra = config_fields(config_class) - stub_arguments(spec_name)
    assert not extra, (
        f"{config_class.__name__} has {sorted(extra)}, which {spec_name} does not take"
    )


def test_the_table_covers_every_spec_the_core_exposes() -> None:
    """So a new spec cannot be added to the core and skip the comparison.

    Without this, `PAIRS` is a list someone has to remember to extend, which is the
    same class of omission the whole file exists to catch.
    """
    specs = {
        node.name
        for node in STUB.body
        if isinstance(node, ast.ClassDef)
        and any(
            isinstance(item, ast.FunctionDef) and item.name == "__init__"
            for item in node.body
        )
    }
    # Geometry is described by `config/geometry.py` in its own vocabulary, because there
    # is no existing type for it to mirror -- that is why it is a separate file. The
    # generated rupture is an output. Neither is a spec a config sets.
    not_configured = {
        "Projected",
        "Plane",
        "Fault",
        "PointSource",
        "Cuts",
        "RefinedMesh",
        "FaultGrid",
        "GeneratedRupture",
    }
    claimed = {name for name, _ in PAIRS}
    assert specs - not_configured == claimed, (
        f"unclaimed specs: {sorted(specs - not_configured - claimed)}; "
        f"claimed but absent: {sorted(claimed - specs)}"
    )


@BY_NAME
def test_to_core_is_a_copy_rather_than_a_translation(
    spec_name: str, config_class: type
) -> None:
    """Every field is mentioned by name in the ``to_core`` that feeds it.

    Weaker than reading the arithmetic and stronger than nothing: it catches the field
    that is declared, validated, and then quietly not passed on -- which looks exactly
    like a working config and produces a rupture built on a default.

    `VelocityModel1D` and `Ramp` take their arguments positionally, so this reads the
    source rather than the call's keywords.
    """
    import inspect

    if not hasattr(config_class, "to_core"):
        pytest.skip(f"{config_class.__name__} has no to_core")

    source = inspect.getsource(config_class.to_core)
    unmentioned = {
        name for name in config_fields(config_class) if f"self.{name}" not in source
    }
    assert not unmentioned, (
        f"{config_class.__name__}.to_core never reads {sorted(unmentioned)}"
    )
