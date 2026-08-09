"""Turning a bad config file into something a person can act on.

Two failures, rendered differently because they want different things.

A **syntax** error wants the line. The reader has a stray bracket or an unclosed string
and needs to see where, so this prints a window of the file with the offending line
highlighted.

A **validation** error wants the *key*. The file parsed, so nothing is visibly wrong
with it; what is wrong is a value, and the reader needs the dotted path to it and the
constraint it broke.

# Why the path takes walking

mashumaro reports a nested failure as a chain. A bad dip inside a plane inside a fault
arrives as an ``InvalidFieldValue`` about ``surfaces``, whose ``__context__`` is about
``planes``, whose ``__context__`` is about ``dip_deg``. The outermost is true and
useless. `config.field_path` walks to the innermost and collects the breadcrumbs, and
this renders what it finds.

Everything here writes to **stderr**, so stdout stays pipeable.
"""

from __future__ import annotations

import dataclasses
import difflib
import json
import tomllib
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import typer
import yaml
from mashumaro.exceptions import ExtraKeysError, InvalidFieldValue, MissingField
from rich.console import Console, Group
from rich.panel import Panel
from rich.syntax import Syntax

from rupture_generator.config import field_path

console = Console(stderr=True)
"""One console, on stderr. A CLI whose diagnostics go to stdout cannot be piped."""

CONTEXT_LINES = 2
"""How much of the file to show either side of a syntax error."""

VALUE_WIDTH = 60
"""How much of an offending value to print.

A value can be a whole list of planes. Printing all of it buries the one word that is
wrong under the twenty that are right.
"""


def _brief(value: object) -> str:
    """A value, short enough to read."""
    text = repr(value)
    return text if len(text) <= VALUE_WIDTH else f"{text[:VALUE_WIDTH]}..."


def _closest(unknown: str, known: Iterable[str]) -> str | None:
    """The real field name a misspelling is nearest to, if it is near one.

    A near miss is the common case -- `dipp_deg` for `dip_deg`, `rake_deg` for
    `average_rake_deg` -- and naming the intended key turns a rejection into an
    instruction. Cut off at 0.6 so an unrelated word gets no suggestion rather than a
    confusing one.
    """
    matches = difflib.get_close_matches(unknown, list(known), n=1, cutoff=0.6)
    return matches[0] if matches else None


def _snippet(document: str, line_number: int) -> tuple[str, int]:
    """The lines around `line_number`, and what the first one is numbered."""
    lines = document.splitlines(keepends=True)
    start = max(line_number - 1 - CONTEXT_LINES, 0)
    end = min(line_number + CONTEXT_LINES, len(lines))
    return "".join(lines[start:end]), start + 1


def print_syntax_error(
    error: tomllib.TOMLDecodeError | json.JSONDecodeError,
    document: str,
    language: str = "toml",
) -> None:
    """Show where a file stopped parsing, with the line highlighted.

    Parameters
    ----------
    error : TOMLDecodeError or JSONDecodeError
        What the parser raised.
    document : str
        The file's text. Needed because `tomllib` does not keep it.
    language : str
        For the highlighter: ``"toml"``, ``"json"`` or ``"yaml"``.
    """
    line_number = getattr(error, "lineno", None)
    column = getattr(error, "colno", None)

    heading = str(error)
    if line_number is not None:
        heading = (
            f"  [bold]Line:[/bold] {line_number}"
            + (f", [bold]column:[/bold] {column}" if column else "")
            + f"\n  [bold]Problem:[/bold] {getattr(error, 'msg', error)}\n"
        )

    body: list[Any] = [heading]
    if line_number is not None:
        text, first = _snippet(document, line_number)
        if text:
            body.append(
                Syntax(
                    text,
                    language,
                    theme="ansi_dark",
                    line_numbers=True,
                    start_line=first,
                    highlight_lines={line_number},
                )
            )

    console.print()
    console.print(
        Panel(
            Group(*body),
            border_style="red",
            title=f"[bold red]{language.upper()} syntax error[/bold red]",
            title_align="left",
        )
    )
    console.print()


def print_config_error(error: Exception) -> None:
    """Show which key is wrong and what was wanted.

    Walks to the innermost failure, so the panel names ``dip_deg`` rather than the
    outermost container that happens to hold it.
    """
    path, innermost = field_path(error)

    if isinstance(innermost, MissingField):
        holder = getattr(innermost, "holder_class", None)
        body = (
            f"  [bold]Missing key:[/bold] [bold yellow]{path}[/bold yellow]\n"
            f"  [bold]Section:[/bold]     {getattr(holder, '__name__', holder)}\n\n"
            "  It has no default, so there is nothing sensible to run without it."
        )
        title = "[bold red]Configuration incomplete[/bold red]"
    elif isinstance(innermost, InvalidFieldValue):
        holder = getattr(innermost, "holder_class", None)
        declared = getattr(innermost, "field_type", None)
        reason = getattr(innermost, "msg", None) or str(innermost)
        body = (
            f"  [bold]Key:[/bold]     [bold blue]{path}[/bold blue]\n"
            f"  [bold]Section:[/bold] {getattr(holder, '__name__', holder)}\n"
            f"  [bold]Type:[/bold]    {getattr(declared, '__name__', declared)}\n"
            f"  [bold]Value:[/bold]   {_brief(getattr(innermost, 'field_value', '?'))}\n"
            f"  [bold]Problem:[/bold] [white]{reason}[/white]"
        )
        title = "[bold red]Configuration invalid[/bold red]"
    elif isinstance(innermost, ExtraKeysError):
        target = innermost.target_type
        known = (
            [field.name for field in dataclasses.fields(target)]
            if dataclasses.is_dataclass(target)
            else []
        )
        lines = []
        for key in sorted(innermost.extra_keys):
            suggestion = _closest(key, known)
            lines.append(
                f"  [bold]Unknown key:[/bold] [bold yellow]{key}[/bold yellow]"
                + (
                    f"  -- did you mean [green]{suggestion}[/green]?"
                    if suggestion
                    else ""
                )
            )
        body = (
            "\n".join(lines)
            + f"\n  [bold]Section:[/bold]     {innermost.target_class_name}\n\n"
            "  Unknown keys are refused rather than ignored, because a key that is read\n"
            "  and dropped is a different earthquake than the one written down."
        )
        title = "[bold red]Unknown configuration key[/bold red]"
    else:
        # An unknown discriminator tag, or anything else mashumaro raises. Rendered
        # plainly rather than guessed at -- a panel that invents a field name is worse
        # than one that quotes the library.
        body = f"  {error}"
        title = f"[bold red]{type(error).__name__}[/bold red]"

    console.print()
    console.print(Panel(body, border_style="red", title=title, title_align="left"))
    console.print()


def load_config[T](path: Path, read: Callable[[Path], T]) -> T:
    """Read a config file with `read`, rendering a failure rather than raising it.

    Only the decode is wrapped. Anything after it is a bug in the caller rather than a
    mistake in the file, and a traceback is the right answer for that.

    `read` is `config.read_config` or `config.read_geometry`; the two differ in nothing
    else, which is why this is one function.
    """
    try:
        return read(path)
    except (InvalidFieldValue, MissingField) as error:
        print_config_error(error)
        raise typer.Exit(1) from error
    except tomllib.TOMLDecodeError as error:
        print_syntax_error(error, path.read_text(), "toml")
        raise typer.Exit(1) from error
    except json.JSONDecodeError as error:
        print_syntax_error(error, path.read_text(), "json")
        raise typer.Exit(1) from error
    except yaml.YAMLError as error:
        console.print(f"[red]{path}: {error}[/red]")
        raise typer.Exit(1) from error
    except Exception as error:
        print_config_error(error)
        raise typer.Exit(1) from error


__all__ = [
    "CONTEXT_LINES",
    "console",
    "load_config",
    "print_config_error",
    "print_syntax_error",
]
