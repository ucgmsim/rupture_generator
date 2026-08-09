"""Kinematic rupture model generation.

A pipeline in Python over stateless Rust kernels. A rupture realisation is a
composition of stages -- mesh, fields, wavefront, pulses -- each a pure function of
``(mesh, params, rng)``; `pipeline.py` is the one place their order is written down,
and `PLAN.md` is the argument for the whole shape.

The interchange type is an `xarray.Dataset` per fault segment (node coordinates plus
whatever fields the pipeline has attached so far), which is also what the rupture
file stores: the pipeline's output *is* the file.

Units are MKS -- slip in metres, moment in newton-metres -- except geometry, which
is written and meshed in kilometres. `units.py` has the argument.
"""
