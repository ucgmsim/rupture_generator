"""The port against the stored reference, field by field.

This is what the mapping was built for, and it needs no genslip binary: the reference
is `tests/corpus/`, recorded from the real v5.6.2 and committed.

# Two kinds of claim, kept apart on purpose

**Asserted** are the things that must agree: slip, and the point ordering. Slip is the
whole stochastic pipeline -- draws, spectrum, taper, moment scaling -- so agreement to
the SRF's text precision is a strong statement about all of it.

**Recorded** are the things that do not yet agree. Each carries the measured number
and an envelope around it, so it is a regression pin on a known-wrong quantity rather
than a silence. `README.md` says to record the divergence rather than assert it away;
an envelope is how a recorded divergence still fails when it *changes*.

The envelopes are deliberately loose enough not to be flaky and tight enough that a
real move breaks them. They are not tolerances anyone argued for -- they are fences
around measurements.

# What this found

`rake_sigma` reaches nothing. `realisation.rs` passes the slip field's coefficient of
variation where the rake field's standard deviation in *degrees* belongs, so every
rake comes out with a spread of 0.75 degrees where genslip gives 15. Confirmed against
all five cases and written up as `DEFECTS.md` 14.

The per-function parity tests could not have caught it: `rake_field` is correct, and
is tested with whatever sigma the test hands it. The defect is in the **call**, which
is exactly the seam an end-to-end corpus closes and a per-function suite cannot.
"""

import numpy as np
import pytest

from tests.harness import corpus

CASES = [case.name for case in corpus.CASES]


@pytest.fixture(scope="module")
def compared() -> dict:
    """Every case, generated once and compared in the reference's point order.

    Module-scoped because generating five ruptures is seconds of FFT work and every
    test below wants the same five.

    Returns
    -------
    dict
        Case name to `(reference points, port arrays, base rake, geometry)`.
    """
    results = {}
    for case in corpus.CASES:
        reference = corpus.load_reference(case.name)
        geometry = corpus.load_geometry(case.name)
        order = corpus.segment_order(geometry)
        rupture = corpus.run_port(case)
        results[case.name] = dict(
            reference=reference,
            points=reference.points,
            geometry=geometry,
            order=order,
            slip_cm=rupture.slip_cm[order],
            rake_deg=rupture.rake_deg[order],
            onset_s=rupture.onset_s[order],
            rise_time_s=rupture.rise_time_s[order],
            base_rake_deg=geometry.rake_deg[order],
        )
    return results


def relative(mine: np.ndarray, theirs: np.ndarray) -> float:
    """Largest absolute difference, over the reference's own largest magnitude."""
    scale = max(float(np.abs(theirs).max()), 1e-30)
    return float(np.abs(mine - theirs).max()) / scale


class TestTheCorpusIsWellFormed:
    def test_every_case_is_present(self) -> None:
        for case in corpus.CASES:
            assert corpus.gsf_path(case.name).exists(), case.name
            assert corpus.args_path(case.name).exists(), case.name
            assert corpus.srf_path(case.name).exists(), case.name

    def test_every_case_says_what_it_is_for(self) -> None:
        # A fixture whose purpose is not recorded gets deleted by the next person to
        # look at the directory.
        for case in corpus.CASES:
            assert case.why.strip()

    @pytest.mark.parametrize("name", CASES)
    def test_the_reference_has_one_point_per_subfault(self, name: str) -> None:
        case = corpus.BY_NAME[name]
        reference = corpus.load_reference(name)
        assert len(reference.points) == case.strike_count * case.dip_count

    @pytest.mark.parametrize("name", CASES)
    def test_the_stored_arguments_describe_the_stored_case(self, name: str) -> None:
        case = corpus.BY_NAME[name]
        arguments = corpus.load_arguments(name)
        assert float(arguments["mag"]) == case.magnitude
        assert int(arguments["nstk"]) == case.strike_count
        assert int(arguments["ndip"]) == case.dip_count
        assert int(arguments["seed"]) == case.seed
        # ns and nh are the ones whose absence makes genslip write nothing at all.
        assert arguments["ns"] == "1"
        assert arguments["nh"] == "1"

    def test_the_spread_is_actually_spread(self) -> None:
        # Five cases that were secretly the same fault would pass everything below
        # and mean nothing.
        shapes = {(c.strike_count, c.dip_count) for c in corpus.CASES}
        assert len(shapes) == len(corpus.CASES)
        assert len({c.parameters().kmodel for c in corpus.CASES}) >= 3
        assert len({c.parameters().dt for c in corpus.CASES}) >= 2
        dips = {float(corpus.load_geometry(c.name).mean_dip_deg) for c in corpus.CASES}
        assert min(dips) < 20.0, "no shallow-dipping case"
        assert max(dips) > 75.0, "no steep case"


class TestThePointOrdering:
    """The GSF's order is not the SRF's, and only the multi-segment case shows it."""

    @pytest.mark.parametrize("name", CASES)
    def test_the_ordering_puts_the_subfaults_back_where_genslip_had_them(
        self, name: str, compared: dict
    ) -> None:
        # Positions agree to the SRF's text precision once regrouped. This is the
        # check that `segment_order` is right, and it is the only thing standing
        # between the comparisons below and comparing one subfault against another.
        result = compared[name]
        geometry, order, points = (
            result["geometry"],
            result["order"],
            result["points"],
        )
        assert np.abs(points.longitude_deg - geometry.longitude_deg[order]).max() < 2e-5
        assert np.abs(points.latitude_deg - geometry.latitude_deg[order]).max() < 2e-5
        assert np.array_equal(points.strike_deg, geometry.strike_deg[order])

    def test_it_is_the_identity_for_a_single_segment(self) -> None:
        geometry = corpus.load_geometry("crustal_small")
        assert np.array_equal(corpus.segment_order(geometry), np.arange(len(geometry)))

    def test_and_is_not_for_a_bent_fault(self) -> None:
        # If this ever becomes the identity, `on_two_planes` stopped interleaving and
        # every bent-fault comparison quietly started proving nothing.
        geometry = corpus.load_geometry("bent")
        order = corpus.segment_order(geometry)
        assert not np.array_equal(order, np.arange(len(geometry)))

        # And the cost of getting it wrong, measured: comparing in file order would
        # compare subfaults 180 metres apart or more.
        reference = corpus.load_reference("bent")
        wrong = np.hypot(
            reference.points.longitude_deg - geometry.longitude_deg,
            reference.points.latitude_deg - geometry.latitude_deg,
        ).max()
        assert wrong > 0.1, "the mis-ordering has to be visible or this proves nothing"


class TestSlipAgrees:
    """The whole stochastic pipeline, in one field."""

    @pytest.mark.parametrize("name", [n for n in CASES if n != "frankel_corners"])
    def test_slip_matches_to_the_format_s_precision(
        self, name: str, compared: dict
    ) -> None:
        # 2.6e-06 is the worst of the four, and the SRF writes slip with six
        # significant figures -- so this is agreement to the file's own resolution,
        # not a tolerance. Every draw, the spectrum, the taper and the moment
        # scaling are behind this one number.
        result = compared[name]
        assert relative(result["slip_cm"], result["points"].slip_cm) < 1e-5

    @pytest.mark.parametrize("name", [n for n in CASES if n != "frankel_corners"])
    def test_slip_is_the_same_field_and_not_merely_the_same_size(
        self, name: str, compared: dict
    ) -> None:
        # A scaled or shuffled field would pass a mean-and-max check.
        result = compared[name]
        correlation = np.corrcoef(result["slip_cm"], result["points"].slip_cm)[0, 1]
        assert correlation > 1.0 - 1e-9


class TestTheRecordedDivergences:
    """What does not agree yet, measured rather than tolerated.

    Each number below was recorded from the corpus as it stands. They are fences
    around measurements: a test here fails when the divergence *changes*, which is
    what makes it a record rather than a silence.
    """

    @pytest.mark.parametrize("name", CASES)
    def test_rake_carries_slip_sigma_where_rake_sigma_belongs(
        self, name: str, compared: dict
    ) -> None:
        """`DEFECTS.md` 14. The corpus found this; nothing else could have.

        genslip normalises the rake field to a standard deviation of `rake_sigma`
        **degrees** and adds it to the base rake (`genslip_v5.6.2.c:2068`,
        `sigfac = rake_sigma/rk_sig`). The port passes the slip field's coefficient
        of variation instead (`realisation.rs`, the `rake_field` call), so the spread
        is 0.75 where it should be 15 -- a factor of twenty, identical on every case,
        which is what makes it a wiring error rather than a numerical one.
        """
        result = compared[name]
        case = corpus.BY_NAME[name]
        reference_spread = float(
            np.std(result["points"].rake_deg - result["base_rake_deg"])
        )
        port_spread = float(np.std(result["rake_deg"] - result["base_rake_deg"]))

        # genslip delivers what was asked for, to a fraction of a degree.
        assert reference_spread == pytest.approx(case.parameters().rake_sigma, abs=0.1)
        # The port delivers slip_sigma, exactly.
        assert port_spread == pytest.approx(case.parameters().slip_sigma, abs=0.01)
        # Stated as the ratio, so the day someone fixes it this test says so loudly.
        assert reference_spread / port_spread == pytest.approx(20.0, rel=0.02), (
            "rake_sigma now reaches the port -- delete this test, move DEFECTS.md 14 "
            "to fixed, and turn the rake comparison into an assertion"
        )

    @pytest.mark.parametrize("name", CASES)
    def test_rise_time_is_within_two_percent_in_the_mean(
        self, name: str, compared: dict
    ) -> None:
        # Recorded: 0.989 to 1.018 across the corpus. The field is the right shape
        # (correlations 0.89 to 0.95) and slightly the wrong size, which points at
        # the rise-time perturbation's amplitude rather than at its draws -- a
        # desynchronised stream would not correlate at all.
        result = compared[name]
        ratio = result["rise_time_s"].mean() / result["points"].rise_time_s.mean()
        assert 0.97 < ratio < 1.04, f"{name}: rise-time mean ratio {ratio}"

    @pytest.mark.parametrize("name", CASES)
    def test_onset_is_the_right_shape_and_not_yet_the_right_times(
        self, name: str, compared: dict
    ) -> None:
        # Recorded: correlations 0.92 to 0.996, differences with standard deviation
        # 0.33 s to 1.05 s. Structure right, amplitude not -- the same signature the
        # rise time has, and probably the same cause.
        result = compared[name]
        correlation = np.corrcoef(result["onset_s"], result["points"].onset_s)[0, 1]
        assert correlation > 0.85, f"{name}: onset correlation {correlation}"

    def test_frankel_slip_does_not_agree_and_the_falloff_is_not_why(
        self, compared: dict
    ) -> None:
        # The one case where slip diverges: 0.39 relative, correlation 0.993. Not the
        # exponent -- `kfilt_gaus2` hardwires `beta2 = 2.0` at `slip.c:1610` and the
        # port hardwires the same. Not the corners either, now that `DEFECTS.md` 11
        # is fixed and Frankel takes Mai's relation with kx_corner/ky_corner at their
        # 2.50/1.50 defaults. Unexplained, and recorded so it stays visible.
        result = compared["frankel_corners"]
        divergence = relative(result["slip_cm"], result["points"].slip_cm)
        assert 0.2 < divergence < 0.6, f"frankel slip divergence {divergence}"
        correlation = np.corrcoef(result["slip_cm"], result["points"].slip_cm)[0, 1]
        assert correlation > 0.98, "mostly the same field, which is the puzzle"


class TestTheGeometryDivergence:
    """Measured, not asserted away -- and it is in the header, not the points."""

    @pytest.mark.parametrize("name", CASES)
    def test_the_point_positions_do_not_diverge_at_all(
        self, name: str, compared: dict
    ) -> None:
        # The README expected lon/lat/dep to disagree in the fourth decimal. They do
        # not, and the reason is worth writing down: genslip copies point positions
        # straight out of the GSF (`spar[ip].lon`), so there is nothing to
        # approximate. What it *derives* is the plane header -- see below.
        result = compared[name]
        points, geometry, order = (
            result["points"],
            result["geometry"],
            result["order"],
        )
        assert np.abs(points.depth_km - geometry.depth_km[order]).max() < 1e-3
        assert np.abs(points.latitude_deg - geometry.latitude_deg[order]).max() < 2e-5

    @pytest.mark.parametrize(
        ("name", "metres"),
        [
            ("crustal_small", 43.0),
            ("crustal_large", 210.0),
            ("subduction", 1899.0),
            ("bent", 287.0),
            ("frankel_corners", 353.0),
        ],
    )
    def test_the_plane_centre_diverges_and_by_how_much(
        self, name: str, metres: float, compared: dict
    ) -> None:
        """genslip's tangent-plane approximation, in metres, per case.

        It recomputes each segment's top-edge centre from a length, a width and a
        dip on a flat earth. The port never does this -- it uses the positions it was
        given -- so the difference is genslip's error, not a disagreement.

        It scales with the fault: 43 m across a 10 km crustal plane, **1.9 km** at
        subduction scale. That is the same effect as the 944 m at 100 km that
        `README.md` records for the flat-earth approximation, and it is the reason
        `Wgs84Geodesic` is a Stage 3 item.
        """
        result = compared[name]
        reference, geometry = result["reference"], result["geometry"]

        worst = 0.0
        for index, plane in enumerate(reference.planes):
            segment = geometry.segment == index
            shallowest = geometry.depth_km[segment].min()
            top = segment & (geometry.depth_km == shallowest)
            centre_longitude = geometry.longitude_deg[top].mean()
            centre_latitude = geometry.latitude_deg[top].mean()
            east = (
                (plane.centre_longitude_deg - centre_longitude)
                * 111.195
                * np.cos(np.radians(centre_latitude))
            )
            north = (plane.centre_latitude_deg - centre_latitude) * 111.195
            worst = max(worst, float(np.hypot(east, north) * 1000.0))

        assert worst == pytest.approx(metres, rel=0.1), (
            f"{name}: tangent-plane divergence is now {worst:.1f} m, recorded "
            f"as {metres} m"
        )

    def test_the_divergence_grows_with_the_fault(self, compared: dict) -> None:
        # The claim that makes it an approximation error rather than noise.
        def divergence(name: str) -> float:
            result = compared[name]
            plane = result["reference"].planes[0]
            geometry = result["geometry"]
            top = geometry.depth_km == geometry.depth_km.min()
            east = (
                plane.centre_longitude_deg - geometry.longitude_deg[top].mean()
            ) * 111.195
            north = (
                plane.centre_latitude_deg - geometry.latitude_deg[top].mean()
            ) * 111.195
            return float(np.hypot(east, north))

        assert divergence("subduction") > 10.0 * divergence("crustal_small")
