"""The base every config object is built on."""

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
    """A dataclass that reads from TOML, YAML, JSON or a dict, and validates itself."""

    def __post_init__(self) -> None:
        """Run every validator attached to an ``Annotated`` field.

        Raises
        ------
        InvalidFieldValue
            If a validator rejects a value, with the validator's message.
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
        """Reject a value for a given reason.

        Raises
        ------
        InvalidFieldValue
            Always.
        """
        raise InvalidFieldValue(
            field_name=field_name,
            field_type=type(getattr(self, field_name, None)),
            field_value=getattr(self, field_name, None),
            holder_class=type(self),
            msg=message,
        )

    class Config(BaseConfig):
        """mashumaro's settings for every config class."""

        serialize_by_alias = True
        omit_none = True
        forbid_extra_keys = True


def field_path(error: Exception) -> tuple[str, Exception]:
    """Collect error field tracebak.

    Returns
    -------
    tuple of (str, Exception)
        The dotted path, ``"<unknown>"`` if the chain carried no field names, and the
        innermost ``InvalidFieldValue`` or ``MissingField``.
    """
    from mashumaro.exceptions import ExtraKeysError, MissingField

    names: list[str] = []
    innermost: Exception = error
    current: Any = error

    while current is not None:
        # `ExtraKeysError` is included because a misspelt key raises one *inside* an
        # `InvalidFieldValue` about the whole containing list.
        if isinstance(current, InvalidFieldValue | MissingField | ExtraKeysError):
            innermost = current
            if getattr(current, "field_name", None):
                names.append(current.field_name)
        current = getattr(current, "__context__", None)

    return (".".join(names) if names else "<unknown>", innermost)
