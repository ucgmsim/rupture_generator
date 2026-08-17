"""Units and conversions, named once each.

The pipeline works in **MKS**: slip in metres, slip rate in metres per second,
rigidity in pascals, moment in newton-metres. Geometry is the one exception -- a fault
is written down and meshed in kilometres -- so the km-to-metre conversions below exist
for the single place each is undone.

CGS survives only inside the SRF writer, which stores slip in centimetres, slip rate
in cm/s and area in cm^2; nothing outside `srf.py` has a reason to touch them.
"""

import numpy as np

M_PER_KM = 1.0e3
"""Metres per kilometre. Two jobs with one number: the projection reports metres where
the mesh speaks kilometres, and the moment closure needs subfault areas in square
metres where the mesh derives them in square kilometres."""

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

CM_PER_KM = CM_PER_M * M_PER_KM
"""Centimetres per kilometre. A velocity model writes shear speed in km/s and an SRF
version 2.0 point stores it in cm/s -- a factor of 1e5 that crosses at the SRF
assembler and nowhere else."""

SRF_FLOAT = np.float32
"""What an SRF file's numbers are. The pipeline computes in float64 and this is where
that stops: an SRF writes `%13.5e`, six significant figures, so float32 is the
format's own resolution rather than a shortcut."""
