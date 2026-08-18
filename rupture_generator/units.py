"""Units and conversions, named once each.

The pipeline works in **MKS**: slip in metres, slip rate in metres per second, rigidity
in pascals, moment in newton-metres. Geometry is the one exception -- a fault is written
down and meshed in kilometres. CGS survives only inside the SRF writer, which stores
slip in centimetres, slip rate in cm/s and area in cm^2.
"""

import numpy as np

M_PER_KM = 1.0e3
"""Metres per kilometre."""

M2_PER_KM2 = M_PER_KM * M_PER_KM
"""Square metres per square kilometre, for moment = rigidity [Pa] * area * slip."""

CM_PER_M = 1.0e2
"""Centimetres per metre. Slip and slip rate cross into SRF units at the writer."""

CM2_PER_M2 = CM_PER_M * CM_PER_M
"""Square centimetres per square metre -- an SRF stores subfault areas in cm^2."""

CM2_PER_KM2 = M2_PER_KM2 * CM2_PER_M2
"""Square centimetres per square kilometre: the mesh's area unit to the SRF's."""

CM_PER_KM = CM_PER_M * M_PER_KM
"""Centimetres per kilometre: km/s shear speed to the cm/s an SRF 2.0 point stores."""

SRF_FLOAT = np.float32
"""What an SRF file's numbers are. An SRF writes `%13.5e`, six significant figures, so
float32 is the format's own resolution rather than a shortcut."""
