"""Units and conversions, named once each.

The pipeline works in **MKS**: slip in metres, slip rate in metres per second,
rigidity in pascals, moment in newton-metres. Geometry is the one deliberate
exception -- a fault is written down and meshed in kilometres, because that is the
scale a fault has and the scale the mesh file stores -- so the km-to-metre
conversions below exist for the single place each is undone.

The CGS the C worked in survives only inside the SRF writer. An SRF stores slip in
centimetres, slip rate in cm/s and area in cm^2; those constants live here so the
writer converts by name rather than by magic number, and nothing outside `srf.py`
has a reason to touch them.
"""

import numpy as np

M_PER_KM = 1.0e3
"""Metres per kilometre.

Two distinct jobs with one number: the projection reports metres where the mesh
speaks kilometres (`mesh.py` divides once on the way in, multiplies once on the way
out), and the moment closure needs subfault areas in square metres where the mesh
derives them in square kilometres.
"""

M2_PER_KM2 = M_PER_KM * M_PER_KM
"""Square metres per square kilometre -- the moment closure's one conversion:
moment = rigidity [Pa] * area [m^2] * slip [m]."""

CM_PER_M = 1.0e2
"""Centimetres per metre. SRF territory: slip and slip rate cross into the format's
own units at the writer and nowhere else."""

CM2_PER_M2 = CM_PER_M * CM_PER_M
"""Square centimetres per square metre -- an SRF stores subfault areas in cm^2."""

CM2_PER_KM2 = M2_PER_KM2 * CM2_PER_M2
"""Square centimetres per square kilometre: the mesh's area unit straight to the
SRF's, composed from the two conversions above rather than being a third fact."""

SRF_FLOAT = np.float32
"""What an SRF file's numbers are.

The pipeline computes in float64 and this is where that stops. An SRF writes
`%13.5e`, six significant figures, so float32 is the format's own resolution rather
than a shortcut.
"""
