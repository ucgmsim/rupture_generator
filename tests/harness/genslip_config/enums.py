from enum import IntEnum, StrEnum


class Stype(StrEnum):
    """Slip-rate function shapes accepted by ``generic_slip2srf`` (point sources).

    This is **not** the same vocabulary as genslip's finite-fault ``stype`` -- see
    :class:`SlipRateFunction`. Both are spelled ``stype`` on their respective
    command lines, and the two sets overlap only at ``brune``, ``urs``, ``ucsb``.
    Passing one binary's value to the other silently selects a different function
    or falls through to that binary's default.

    Dispatch is at ``generic_slip2srf/generic_slip2srf.c:404-450``, using
    ``strncmp``, so ``ucsb-T`` accepts a numeric suffix giving ``tau1r``
    (e.g. ``ucsb-T0.2``).
    """

    brune = "brune"
    delta = "delta"
    esg2006 = "esg2006"
    urs = "urs"
    ucsb = "ucsb"
    ucsb2 = "ucsb2"
    ucsb_T = "ucsb-T"
    ucsb_varT1 = "ucsb-varT1"
    cos = "cos"
    seki = "seki"


class SlipRateFunction(StrEnum):
    """Slip-rate function shapes accepted by genslip v5.6.2 (finite faults).

    Dispatch is in ``load_slip_srf_dd5_vsden``
    (``gslip_srf_subs.c:1407-1627``), the only loader ``main`` calls
    (``genslip_v5.6.2.c:2964``). The other ``load_slip_srf_dd*`` variants accept
    subsets of these names but are unreachable.

    The default is ``OliuP2``: ``genslip_v5.6.2.c`` sets ``OliuP`` at line 699 and
    then overwrites it with ``OliuP2`` at line 787 in the v5.6.x defaults block, so
    the declared initialiser is not the effective default.

    ``OliuP2`` reads the slip-rate shape parameter beta from the per-subfault
    ``psrc`` array, which ``main`` fills from the ``beta_shal``/``beta_mid``/
    ``beta_deep`` depth ramps. ``OliuP`` and ``MliuP`` instead recompute beta
    in-loop from the *legacy* ``beta_depth``/``beta_depth_range`` pair that
    ``stfpar2`` still carries -- so those two see a different depth model than the
    one configured.
    """

    brune = "brune"
    urs = "urs"
    ucsb = "ucsb"
    mliu = "Mliu"
    mliu_p = "MliuP"
    oliu_p = "OliuP"
    oliu_p2 = "OliuP2"
    tri = "tri"


class RiseTimeNormalisation(IntEnum):
    """How the fault-wide rise-time normalisation ``rt_scalefac`` is averaged.

    genslip ``svr_wt``, read as an int (``genslip_v5.6.2.c:1061``) and applied at
    lines 2461-2466. Despite the name it is a mode selector, not a weight.

    ``rt_scalefac`` itself is a getpar float, but genslip overwrites it in place at
    line 2476 with the computed constant -- only its sign is an input, selecting
    whether rise time is scaled with slip at all.
    """

    UNWEIGHTED_MEAN = 0
    SLIP_WEIGHTED = 1
    SLIP_AND_RUPTURE_VELOCITY_WEIGHTED = 2


class KModel(IntEnum):
    """Wavenumber correlation-length model (genslip ``kmodel``).

    Selects both the magnitude-to-correlation-length relation
    (``genslip_v5.6.2.c:1292-1409``) and the spectral falloff shape in
    ``kfilt_gaus2`` (``slip.c:1636-1657``). Unrecognised values below 100 are
    silently coerced to ``SOMERVILLE``.

    ``MAI_SOMERVILLE`` looks incomplete in v5.6.2: it computes a Somerville
    correlation-length pair alongside the Mai one, but never consumes it.
    """

    SOMERVILLE = 1
    MAI = 2
    FRANKEL = 3
    MAI_SOMERVILLE = 4
    SUZUKI = 5
    INPUT_CORNERS = -1
