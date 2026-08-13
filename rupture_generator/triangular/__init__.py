"""The triangular-mesh track: a fault as a triangulated parametric chart.

A parallel track beside the structured pipeline. Its four modules split the way the
architecture does -- the geometry is curved and the solvers are flat:

- :mod:`~rupture_generator.triangular.mesh` -- the geometry container, a Monge patch
  ``X(u, v) = O + u e_u + v e_v + h(u, v) n`` triangulated in the parameter domain and
  lifted to R^3. It supplies **true areas, true depths, true positions and the true
  outline**, which is the whole of its job and, on the curvature study's measurements,
  where nearly all of the value is.
- :mod:`~rupture_generator.triangular.lattice` -- the regular grid over that parameter
  domain, and the two solvers that run on it: the factored fast sweep with a slowness
  wall on the off-fault cells, and circulant embedding. Both are the structured track's
  own; both project their answers onto the mesh's faces.
- :mod:`~rupture_generator.triangular.pipeline` -- the stage order, and the three things
  a triangulation genuinely does differently: the taper, the depths the slowness is
  sampled at, and the seed seam.
- :mod:`~rupture_generator.triangular.gocad` -- reading a modeller's TSurf.

So the two tracks differ in their **mesh container** and in nothing else: one wavefront
solver, one field sampler, and one of every field model in the package.
"""
