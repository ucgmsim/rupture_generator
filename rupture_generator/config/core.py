"""The base every config object is built on.

One class, and the thing it does that is not obvious: **validation rides on the type
annotation**. A field declared ``Annotated[float, in_range(0.0, 90.0)]`` is checked
after construction by walking the class's own hints, so the constraint lives beside the
type rather than in a separate schema or a hand-written ``__post_init__`` per class.

Three consequences worth knowing before writing a config class.

A validator may **coerce**. A non-``None`` return is written back to the field, which is
how a path gets resolved and a longitude gets folded. Returning nothing means "accepted
as is", which is what most of them do.

A failure becomes mashumaro's own ``InvalidFieldValue``, carrying the field name, its
declared type, the offending value and the class it belongs to. That matters because it
is the same exception mashumaro raises for a decode failure, so the CLI renders a bad
*value* and a bad *type* through one path -- see ``scripts/errors.py``.

``forbid_extra_keys`` is on. A misspelt key is an error rather than a silently ignored
line. `README.md` records the cost of the alternative: genslip's ``getpar`` never asks
for names it does not recognise, so five parameters have been *silently discarded in
production* for as long as the workflow has pointed at a binary that does not know them.

Copied, deliberately and almost verbatim, from ``nzcvm/config/core.py``. It is fifty
lines and it is the whole mechanism; a second version of it that drifted would be worse
than a shared one that did not.
"""

from typing import Annotated, Any, get_args, get_origin, get_type_hints

from mashumaro.config import BaseConfig
from mashumaro.exceptions import InvalidFieldValue
from mashumaro.mixins.dict import DataClassDictMixin
from mashumaro.mixins.json import DataClassJSONMixin
from mashumaro.mixins.toml import DataClassTOMLMixin
from mashumaro.mixins.yaml import DataClassYAMLMixin


class ConfigObject(
    DataClassJSONMixin, DataClassYAMLMixin, DataClassTOMLMixin, DataClassDictMixin
):
    """A dataclass that reads from TOML, YAML, JSON or a dict, and validates itself.

    Subclasses get ``from_dict``/``to_dict`` and the three format pairs from mashumaro,
    and the ``Annotated`` validator sweep from here.
    """

    def __post_init__(self) -> None:
        """Run every validator attached to an ``Annotated`` field.

        Raises
        ------
        InvalidFieldValue
            If a validator rejects a value. The message is the validator's own, so it
            says what was wanted rather than which check ran.
        """
        hints = get_type_hints(type(self), include_extras=True)

        for field_name, hint in hints.items():
            if get_origin(hint) is not Annotated:
                continue

            declared, *validators = get_args(hint)
            for validator in validators:
                if not callable(validator):
                    continue

                value = getattr(self, field_name)
                try:
                    coerced = validator(value)
                except (TypeError, ValueError) as error:
                    raise InvalidFieldValue(
                        field_name=field_name,
                        field_type=declared,
                        field_value=value,
                        holder_class=type(self),
                        msg=str(error),
                    ) from error
                if coerced is not None:
                    setattr(self, field_name, coerced)

    def refuse(self, field_name: str, message: str) -> None:
        """Reject a value for a reason no single field could have caught.

        For invariants *between* fields -- a bottom depth above a top one, a
        discretisation given twice. Raising ``InvalidFieldValue`` rather than a bare
        ``ValueError`` is what makes the CLI point at a key: it carries the field name,
        so the error panel names ``bottom_depth_km`` rather than the class.

        Pick the field the reader should *change*, which is not always the one that
        looks wrong.

        Raises
        ------
        InvalidFieldValue
            Always. This exists to raise.
        """
        raise InvalidFieldValue(
            field_name=field_name,
            field_type=type(getattr(self, field_name, None)),
            field_value=getattr(self, field_name, None),
            holder_class=type(self),
            msg=message,
        )

    class Config(BaseConfig):
        """mashumaro's settings for every config class.

        **The name matters.** mashumaro reads an inner class called ``Config``; one
        called ``Meta`` is ignored in silence, and every setting in it does nothing. The
        version of this pattern that was copied from spelled it ``Meta``, so its
        ``forbid_extra_keys`` had never once refused a misspelt key.

        A subclass that needs its own settings -- a `Discriminator`, say -- must inherit
        from *this* rather than from `BaseConfig`, or it replaces these instead of
        adding to them. `tests/test_config.py::TestMisspellingsAreErrors` is
        parametrised over the sections for exactly that reason.
        """

        serialize_by_alias = True
        omit_none = True
        forbid_extra_keys = True


def field_path(error: Exception) -> tuple[str, Exception]:
    """The dotted path to the field that failed, and the error that says why.

    mashumaro reports a nested failure as a chain: the outer object says "``planes`` has
    an invalid value", and its ``__context__`` says which field of which plane. Walking
    the chain collects the breadcrumbs and finds the *most specific* error, which is the
    one worth showing.

    Returns
    -------
    tuple of (str, Exception)
        The dotted path, and the innermost ``InvalidFieldValue`` or ``MissingField``.
        The path is ``"<unknown>"`` if the chain carried no field names at all.
    """
    from mashumaro.exceptions import MissingField

    names: list[str] = []
    innermost: Exception = error
    current: Any = error

    while current is not None:
        if isinstance(current, InvalidFieldValue | MissingField):
            innermost = current
            if getattr(current, "field_name", None):
                names.append(current.field_name)
        current = getattr(current, "__context__", None)

    return (".".join(names) if names else "<unknown>", innermost)
