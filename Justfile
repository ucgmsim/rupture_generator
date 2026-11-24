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

# The README's picture: one draw of the Beavan example, its slip and rupture-time
# contours. `uv run --with` fetches pyvista and matplotlib for this alone.
hero:
    uv run rupture-generator examples/beavan.toml /tmp/beavan.h5
    uv run --with pyvista --with matplotlib python docs/hero.py /tmp/beavan.h5 docs/images/beavan_slip.png

# The README's turntable: one draw of the Maule example, circled, with Natural Earth's
# coastline, which is fetched once to /tmp/naturalearth.
orbit:
    uv run rupture-generator examples/maule.toml /tmp/maule.h5
    mkdir -p /tmp/naturalearth
    test -f /tmp/naturalearth/ne_10m_coastline.zip || curl -sSfL -o /tmp/naturalearth/ne_10m_coastline.zip https://naciscdn.org/naturalearth/10m/physical/ne_10m_coastline.zip
    uv run --with pyvista --with matplotlib --with imageio --with pyshp python docs/orbit.py /tmp/maule.h5 docs/images/maule_orbit.gif /tmp/naturalearth/ne_10m_coastline.zip

# The recording the turntable links to: the same draw of the Maule example in Rerun.
# The README opens it in Rerun's web viewer at the installed rerun-sdk's version,
# so bump the link's version whenever this recording is remade under a new one.
maule-rrd:
    uv run rupture-generator examples/maule.toml /tmp/maule.srf
    uv run --extra vis rupture-view /tmp/maule.srf --save docs/maule.rrd
    @uv run --extra vis python -c "import rerun; print('README link version:', rerun.__version__)"
