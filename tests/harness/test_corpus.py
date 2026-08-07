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

Three defects, all of the same kind: a correct function called wrongly, or not
guarded the way the original guards it. `DEFECTS.md` 14-16.

1. **`rake_sigma` reached nothing.** The slip field's coefficient of variation was
   handed to `rake_field` where the rake field's spread in *degrees* belongs.
2. **The shallow rise-time blend read the wrong slip.** The original blends against
   the *reloaded* slip spectrum brought back to space, not the tapered field the
   reload was built from.
3. **A subfault that does not slip was getting a pulse.** The `|slip| > MINSLIP`
   guard is in the SRF loader, outside the generator, so porting the generator
   faithfully did not reproduce it.

None could have been caught by the per-function parity tests. Each of `rake_field`,
the blend and `oliu_p` is correct and is tested against the C; the defects are in the
**calls**, and a suite that checks one function at a time cannot see a caller handing
the right function the wrong argument. That is the seam an end-to-end corpus closes.

What remains is onset -- see `TestTheRecordedDivergences`, which records what has been
ruled out as well as what is left.
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

        # The slip-rate rows. The reference stores them as a sparse matrix, one row
        # per subfault; the port as one concatenated array plus offsets. Compare the
        # lengths first -- `nt1` is stored, not derived -- and the samples only where
        # the lengths agree, since otherwise there is nothing to line up.
        sparse = reference.slip_rate
        reference_length = np.diff(sparse.indptr).astype(int)
        offsets = rupture.slip_rate_offsets
        pulse_length = np.diff(offsets)[order].astype(int)

        worst = 0.0
        for row, index in enumerate(order):
            if pulse_length[row] != reference_length[row] or reference_length[row] == 0:
                continue
            theirs = sparse.data[sparse.indptr[row] : sparse.indptr[row + 1]]
            mine = rupture.slip_rate[offsets[index] : offsets[index + 1]]
            worst = max(worst, relative(mine, theirs))

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
            pulse_length=pulse_length,
            reference_length=reference_length,
            slip_rate_relative=worst,
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


class TestRakeAgrees:
    """`DEFECTS.md` 14, fixed. The corpus found it; nothing else could have."""

    @pytest.mark.parametrize("name", CASES)
    def test_rake_matches_exactly_at_the_format_s_resolution(
        self, name: str, compared: dict
    ) -> None:
        """Every subfault, every case.

        The SRF stores rake as whole degrees, so the format itself is the floor here:
        the port's continuous value rounds to the reference's stored integer on
        100% of subfaults, with the largest deviation 0.4999 -- entirely
        quantisation, and provably so, because a real disagreement could not stay
        below half a degree on every subfault of five different faults.
        """
        result = compared[name]
        reference = result["points"].rake_deg
        # The reference really is quantised; if that ever stops being true this test
        # is measuring something else.
        assert np.array_equal(reference, np.round(reference))

        assert np.array_equal(np.round(result["rake_deg"]), reference)
        assert np.abs(result["rake_deg"] - reference).max() < 0.5

    @pytest.mark.parametrize("name", CASES)
    def test_the_spread_is_rake_sigma_and_not_slip_sigma(
        self, name: str, compared: dict
    ) -> None:
        # The shape of the defect that was: `rake_sigma` is 15 degrees and
        # `slip_sigma` is 0.75, so taking the wrong one was a factor of twenty. Both
        # values are in the fixture, so this cannot pass by coincidence.
        result = compared[name]
        parameters = corpus.BY_NAME[name].parameters()
        spread = float(np.std(result["rake_deg"] - result["base_rake_deg"]))
        assert spread == pytest.approx(parameters.rake_sigma, abs=0.05)
        assert spread != pytest.approx(parameters.slip_sigma, abs=1.0)


class TestTheSlipRatePulsesAgree:
    """The pulses, and the sample counts the format stores.

    `nt1` is what the slip-rate generator *returned*, not `rise_time / dt`. Comparing
    the port's rise time against `nt1 * dt` compares two different quantities and
    shows a bounded, meaningless offset -- which is what it did until the pulse
    lengths themselves were checked.
    """

    @pytest.mark.parametrize("name", [n for n in CASES if n != "frankel_corners"])
    def test_the_pulse_lengths_are_the_reference_s(
        self, name: str, compared: dict
    ) -> None:
        # 100% on three cases, 99.83% on `subduction` -- two subfaults of 1152.
        result = compared[name]
        exact = float(np.mean(result["pulse_length"] == result["reference_length"]))
        assert exact > 0.998, f"{name}: {exact:.4%} of pulse lengths exact"

    @pytest.mark.parametrize("name", [n for n in CASES if n != "frankel_corners"])
    def test_a_subfault_that_does_not_slip_emits_no_pulse(
        self, name: str, compared: dict
    ) -> None:
        # genslip guards the whole generator on `|slip| > MINSLIP`
        # (`gslip_srf_subs.c:1496`) and writes `nt1 = 0` otherwise. The guard is in
        # the loader, not in the generator, so a faithful port of the generator does
        # not reproduce it. On a tapered fault this is every edge subfault: 21 of 240
        # on the smallest case here.
        result = compared[name]
        silent = result["reference_length"] == 0
        assert silent.any(), f"{name} has no silent subfaults; this proves nothing"
        assert np.array_equal(
            result["pulse_length"][silent], result["reference_length"][silent]
        )

    @pytest.mark.parametrize("name", [n for n in CASES if n != "frankel_corners"])
    def test_the_samples_agree_where_the_lengths_do(
        self, name: str, compared: dict
    ) -> None:
        # 4.2e-05 relative at worst, which is the SRF's text precision for the
        # slip-rate rows. This is the pulse shape, its normalisation and the rise
        # time that set its duration, all at once.
        result = compared[name]
        assert result["slip_rate_relative"] < 1e-4


class TestTheRecordedDivergences:
    """What does not agree yet, measured rather than tolerated.

    Each number below was recorded from the corpus as it stands. They are fences
    around measurements: a test here fails when the divergence *changes*, which is
    what makes it a record rather than a silence.
    """

    @pytest.mark.parametrize("name", CASES)
    def test_onset_is_the_right_shape_and_not_yet_the_right_times(
        self, name: str, compared: dict
    ) -> None:
        """Correlations 0.92 to 0.997; differences of 0.33 s to 1.05 s.

        What has been ruled out, measured rather than argued:

        - **Not the perturbation's amplitude.** Generating with
          `rupture_time_scale = 0` and differencing gives a perturbation whose
          standard deviation is `|tsfac_main| * tsfac1_sigma` exactly.
        - **Not a desynchronised draw stream.** The rise-time field is drawn *after*
          this one and now agrees exactly, so the stream is right through both.
        - **Not wholly the perturbation field either.** The error correlates with the
          port's own perturbation at only -0.43 to +0.21, and on three cases its
          spread is *smaller* than the perturbation's -- so the two perturbations
          largely agree and this is a residual on top.

        What is left is the travel times: the eikonal solve, or the speed field it
        runs on. Not yet isolated.
        """
        result = compared[name]
        correlation = np.corrcoef(result["onset_s"], result["points"].onset_s)[0, 1]
        assert correlation > 0.85, f"{name}: onset correlation {correlation}"

    @pytest.mark.parametrize("name", CASES)
    def test_the_onset_error_stays_within_what_was_recorded(
        self, name: str, compared: dict
    ) -> None:
        result = compared[name]
        spread = float(np.std(result["onset_s"] - result["points"].onset_s))
        assert spread < 1.2, f"{name}: onset difference spread {spread} s"

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
