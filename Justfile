default: test

# The Python suite, then the doctests, then the Rust suite.
test: pytest cargo

pytest:
    uv run --extra test pytest tests
    uv run --extra test pytest --doctest-modules rupture_generator/ -q

cargo:
    cargo test --manifest-path crates/kernels/Cargo.toml

lint: ty ruff clippy numpydoc

# The kernels, and every example end to end with its peak RSS. Extra arguments go to
# pytest, e.g. `just bench --benchmark-json run.json`.
bench *args:
    uv run --extra bench pytest benchmarks {{args}}

ty:
    uv run ty check rupture_generator

ruff:
    uv run ruff format
    uv run ruff check --fix

clippy:
    cargo clippy --manifest-path crates/kernels/Cargo.toml -- -D warnings

numpydoc:
    uv run numpydoc lint rupture_generator/*.py

deptry:
    uv run deptry .

# Line counts of the package's modules, less the viewer, largest first.
budget:
    @find rupture_generator -name '*.py' ! -name view.py | xargs wc -l | sort -rn

# A rupture on the shipped Alpine-Hope example, end to end.
# Writes /tmp/alpine_hope.srf, with the config it ran in its header comments.
demo:
    uv run rupture-generator examples/alpine_hope.toml /tmp/alpine_hope.srf
