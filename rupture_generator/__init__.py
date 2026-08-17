"""Kinematic rupture model generation.

A pipeline in Python over stateless Rust kernels: each stage a pure function of
``(mesh, params, rng)``, and `pipeline.py` the one place their order is written down.
The interchange type is an `xarray.Dataset` per fault segment, which is also what the
rupture file stores.

Units are MKS -- slip in metres, moment in newton-metres -- except geometry, which is
written and meshed in kilometres.
"""
