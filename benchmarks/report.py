"""Render pytest-benchmark JSON as a Markdown report, head against base when given one.

Usage::

    python benchmarks/report.py HEAD.json [--base BASE.json] [--head-ref REF]
        [--base-ref REF]

The first line is an HTML comment the CI workflow finds its own earlier comment by,
so each run rewrites the pull request's one report.
"""

import argparse
import json
import statistics
from pathlib import Path

MARKER = "<!-- rupture-generator benchmarks -->"

NOISE = 0.05
"""Changes smaller than this are left unmarked: a shared runner wanders about that much."""


def _load(path: Path | None) -> dict[str, dict]:
    if path is None or not path.exists():
        return {}
    return {
        bench["fullname"]: bench for bench in json.loads(path.read_text())["benchmarks"]
    }


def _seconds(value: float) -> str:
    for unit, scale in (("s", 1.0), ("ms", 1e-3), ("µs", 1e-6)):
        if value >= scale:
            return f"{value / scale:.3g} {unit}"
    return f"{value / 1e-9:.3g} ns"


def _bytes(value: float) -> str:
    return f"{value / 2**20:,.0f} MiB"


def _change(head: float, base: float | None) -> str:
    if base is None or base == 0.0:
        return "new"
    ratio = head / base - 1.0
    text = f"{ratio:+.1%}"
    return f"**{text}**" if abs(ratio) >= NOISE else text


def _name(bench: dict) -> str:
    return bench["name"].removeprefix("test_")


def _median(bench: dict | None) -> float | None:
    return None if bench is None else bench["stats"]["median"]


def _padded(bench: dict, before: dict | None) -> str:
    """The embedding's padded grid size, with its change, for the cases that have one."""
    cells = bench["extra_info"].get("padded_cells")
    if cells is None:
        return ""
    previous = None if before is None else before["extra_info"].get("padded_cells")
    return f"{cells:,} ({_change(cells, previous)})" if previous else f"{cells:,}"


def _kernels(head: dict[str, dict], base: dict[str, dict]) -> list[str]:
    lines = [
        "| kernel | base | head | change | head IQR | rounds | padded cells |",
        "|---|--:|--:|--:|--:|--:|--:|",
    ]
    for key, bench in head.items():
        if bench["group"] != "kernels":
            continue
        before = _median(base.get(key))
        lines.append(
            f"| `{_name(bench)}` | {'' if before is None else _seconds(before)} "
            f"| {_seconds(_median(bench) or 0.0)} | {_change(_median(bench) or 0.0, before)} "
            f"| {_seconds(bench['stats']['iqr'])} | {bench['stats']['rounds']} "
            f"| {_padded(bench, base.get(key))} |"
        )
    return lines


def _end_to_end(head: dict[str, dict], base: dict[str, dict]) -> list[str]:
    lines = [
        (
            "| example | base time | head time | change | base peak RSS "
            "| head peak RSS | change |"
        ),
        "|---|--:|--:|--:|--:|--:|--:|",
    ]
    for key, bench in head.items():
        if bench["group"] != "end-to-end":
            continue
        before = base.get(key)
        time_before = _median(before)
        rss = bench["extra_info"]["peak_rss_bytes"]
        rss_before = None if before is None else before["extra_info"]["peak_rss_bytes"]
        lines.append(
            f"| `{_name(bench).removeprefix('example[').removesuffix(']')}` "
            f"| {'' if time_before is None else _seconds(time_before)} "
            f"| {_seconds(_median(bench) or 0.0)} "
            f"| {_change(_median(bench) or 0.0, time_before)} "
            f"| {'' if rss_before is None else _bytes(rss_before)} "
            f"| {_bytes(rss)} | {_change(rss, rss_before)} |"
        )
    return lines


def render(
    head_path: Path,
    base_path: Path | None = None,
    head_ref: str = "head",
    base_ref: str = "base",
) -> str:
    """Render a report as Markdown.

    Parameters
    ----------
    head_path : Path
        The pull request's pytest-benchmark JSON.
    base_path : Path, optional
        The base branch's, when it has a suite to compare against.
    head_ref : str
        How to name the head in the report, such as a short commit hash.
    base_ref : str
        How to name the base.

    Returns
    -------
    str
        The report, starting with the marker comment.
    """
    head, base = _load(head_path), _load(base_path)
    machine = json.loads(head_path.read_text())["machine_info"]
    cpu = machine.get("cpu", {}).get("brand_raw", "an unknown CPU")
    if base:
        against = f"`{head_ref}` against base `{base_ref}`"
        changes = [
            abs((_median(bench) or 0.0) / (_median(base[key]) or 1.0) - 1.0)
            for key, bench in head.items()
            if key in base
        ]
        summary = (
            f" The median change is {statistics.median(changes):.1%}; changes past "
            f"{NOISE:.0%} are in bold."
            if changes
            else ""
        )
    else:
        against = f"`{head_ref}`; the base has no benchmark suite to compare against"
        summary = ""
    return "\n".join(
        [
            MARKER,
            "## Benchmarks",
            "",
            f"{against}, on {cpu}. Times are medians over the rounds.{summary}",
            "",
            "### Kernels",
            "",
            *_kernels(head, base),
            "",
            "### End to end",
            "",
            (
                "Each round is the command line in a fresh interpreter, start-up "
                "included. Peak RSS is the largest the child process reached over the "
                "rounds."
            ),
            "",
            *_end_to_end(head, base),
            "",
        ]
    )


def main() -> None:
    """Print the report for the files on the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("head", type=Path)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--head-ref", default="head")
    parser.add_argument("--base-ref", default="base")
    args = parser.parse_args()
    print(render(args.head, args.base, args.head_ref, args.base_ref))


if __name__ == "__main__":
    main()
