#!/bin/sh
# Every gate, in the order that fails fastest. Errors and warnings both count --
# `cargo test` alone will happily pass while `--all-targets` does not compile.
#
# There is no --no-default-features stage any more, and no features to drop: FFTW and
# the Fortran eikonal solver were the two compatibility backends and both are gone.
# The default build IS the shipping build.
set -eu
cd "$(dirname "$0")"

echo "== clippy (workspace, all targets, -D warnings) =="
cargo clippy --workspace --all-targets -- -D warnings

echo "== fmt =="
cargo fmt --all --check

echo "== test, debug =="
cargo test --workspace

echo "== test, release =="
# Debug and release must agree with each other. A disagreement means the port
# depends on optimisation-level float behaviour, which is a real bug class.
cargo test --workspace --release

echo "== pytest =="
.venv/bin/python -m pytest tests/ -q

echo "== all gates green =="
