"""Kinematic rupture model generation.

A port of genslip v5.6.2, in Rust with a Python front door.

The configuration **is** the compiled core's own types — there is no second
description of a rupture model in this package, and nothing here speaks genslip's
`getpar` vocabulary. That vocabulary exists only in `tests/harness`, which drives the
reference binary the port is compared against.

```python
import numpy as np
from rupture_generator import (
    FaultGrid, Ramp, SlipSpec, SourceSpec, SpectrumModel, TimingSpec,
    VelocityModel1D, generate_rupture,
)

grid = FaultGrid(
    24, 14, 28, 16, 1.0, 1.0,
    depth_km=depths, base_rake_deg=rakes, velocity_fraction=fractions,
)
rupture = generate_rupture(
    grid, velocity_model, source, slip, timing,
    seed=1234, hypocentre_strike=12, hypocentre_dip=8,
)
```
"""

from rupture_generator._core import (
    FaultGrid,
    GeneratedRupture,
    PointSourceSpec,
    Ramp,
    RiseTimeWeighting,
    SlipRateShape,
    SlipSpec,
    SourceSpec,
    SpectrumModel,
    TimingSpec,
    VelocityModel1D,
    generate_point_source,
    generate_rupture,
)

__all__ = [
    "FaultGrid",
    "GeneratedRupture",
    "PointSourceSpec",
    "Ramp",
    "RiseTimeWeighting",
    "SlipRateShape",
    "SlipSpec",
    "SourceSpec",
    "SpectrumModel",
    "TimingSpec",
    "VelocityModel1D",
    "generate_point_source",
    "generate_rupture",
]
