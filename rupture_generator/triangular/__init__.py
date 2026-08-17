"""The triangular-mesh track: a fault as a triangulated parametric chart.

A parallel track beside the structured pipeline; the geometry is curved and the solvers
are flat. See MESH.md and HYBRID.md for the architecture.

- :mod:`~rupture_generator.triangular.mesh` -- the geometry container, a Monge patch
  ``X(u, v) = O + u e_u + v e_v + h(u, v) n``, supplying true areas, depths, positions
  and outline.
- :mod:`~rupture_generator.triangular.lattice` -- the regular grid over that parameter
  domain, and the two solvers that run on it.
- :mod:`~rupture_generator.triangular.pipeline` -- the stage order, on faces.
- :mod:`~rupture_generator.triangular.gocad` -- reading a modeller's TSurf.
"""
