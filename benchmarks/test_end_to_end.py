"""Every example under examples/, from its TOML to its SRF, with its peak memory.

Each round runs the command line in a fresh interpreter, so the peak resident set
the child reports is that run's alone and no earlier round's allocations count
towards it. The time is the whole child process, interpreter start-up and imports
included, because a user waits for all of it.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).parents[1] / "examples"

ROUNDS = 3

CHILD = """
import json, resource, sys, time
from rupture_generator.cli import main

start = time.perf_counter()
status = main(sys.argv[1:])
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
# ru_maxrss is kibibytes on Linux and bytes on macOS.
peak_bytes = peak if sys.platform == "darwin" else peak * 1024
print(json.dumps({"pipeline_s": time.perf_counter() - start, "peak_bytes": peak_bytes}))
sys.exit(status)
"""

pytestmark = pytest.mark.benchmark(group="end-to-end")


@pytest.mark.parametrize(
    "example", sorted(path.stem for path in EXAMPLES.glob("*.toml"))
)
def test_example(benchmark, example, tmp_path):
    runs = []

    def run():
        output = tmp_path / f"{example}.srf"
        child = subprocess.run(
            [
                sys.executable,
                "-c",
                CHILD,
                str(EXAMPLES / f"{example}.toml"),
                str(output),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        runs.append(json.loads(child.stdout.strip().splitlines()[-1]))
        # The SRF is large and only its making is under test.
        output.unlink()

    benchmark.pedantic(run, rounds=ROUNDS, iterations=1)
    benchmark.extra_info["peak_rss_bytes"] = max(run["peak_bytes"] for run in runs)
    benchmark.extra_info["pipeline_s"] = sorted(run["pipeline_s"] for run in runs)[
        len(runs) // 2
    ]
