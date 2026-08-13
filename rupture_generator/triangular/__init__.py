"""The triangular-mesh track: a fault as a triangulated parametric chart.

A parallel track beside the structured pipeline, built and validated before anything
switches over to it. `MESH.md` is the plan; the three modules here are its three
components:

- :mod:`~rupture_generator.triangular.mesh` -- the geometry container, a Monge patch
  ``X(u, v) = O + u e_u + v e_v + h(u, v) n`` triangulated in the parameter domain and
  lifted to R^3.
- :mod:`~rupture_generator.triangular.spde` -- the correlation sampler, the
  Whittle-Matern SPDE solved by P1 finite elements on the lifted triangles rather than
  circulant embedding on a lattice.
- :mod:`~rupture_generator.triangular.fim` -- the eikonal solver, meshFIM with an
  analytic geodesic-ball boundary condition rather than a Cartesian fast sweep.

Nothing here is wired into :mod:`rupture_generator.pipeline`. The structured path stays
green until the triangular one reproduces its answers.
"""
