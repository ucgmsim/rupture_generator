default: test

# The Python suite, then the doctests, then both Rust suites.
test: pytest cargo

pytest:
    uv run --extra test pytest tests
    uv run --extra test pytest --doctest-modules rupture_generator/ -q

cargo:
    cargo test --manifest-path crates/kernels/Cargo.toml
    cargo test --manifest-path crates/srf/Cargo.toml

lint: ty ruff clippy numpydoc

ty:
    uv run ty check rupture_generator

ruff:
    uv run ruff format
    uv run ruff check --fix

clippy:
    cargo clippy --manifest-path crates/kernels/Cargo.toml -- -D warnings
    cargo clippy --manifest-path crates/srf/Cargo.toml -- -D warnings

numpydoc:
    uv run numpydoc lint rupture_generator/*.py

deptry:
    uv run deptry .

# What the 5,000-line budget is measured against: the package, less the viewer.
budget:
    @find rupture_generator -name '*.py' ! -name view.py | xargs wc -l | sort -rn

# A rupture on the shipped crustal example, end to end.
demo:
    uv run rupture-generator mesh examples/alpine_hope.geometry.toml /tmp/alpine_hope.h5
    uv run rupture-generator generate examples/crustal.toml /tmp/alpine_hope.h5 /tmp/rupture.srf
