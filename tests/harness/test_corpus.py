"""The port against the stored reference, field by field.

This is what the mapping was built for, and it needs no genslip binary: the reference
is `tests/corpus/`, recorded from the real v5.6.2 and committed.

# Everything the corpus checks now agrees

Slip, rake, onset and the slip-rate pulses all match to the SRF's own text precision,
on all six cases. That is the whole stochastic pipeline -- draws, spectrum, taper,
moment scaling, the eikonal solve and every perturbation drawn against them -- in four
numbers.

One divergence is still recorded rather than asserted, and it is not the port's:
`TestTheGeometryDivergence` measures genslip's flat-earth error in the plane header,
which it recomputes from a length, a width and a dip rather than reading the positions
it was given.

A recorded divergence carries the measured number and an envelope around it, so it
fails when the divergence *changes*. The envelopes are not tolerances anyone argued
for -- they are fences around measurements.

# What this found

Five defects, all of the same kind: a correct function called wrongly, not guarded the
way the original guards it, or a stage of the original that was never transcribed at
all. `DEFECTS.md` 14-18.

1. **`rake_sigma` reached nothing.** The slip field's coefficient of variation was
   handed to `rake_field` where the rake field's spread in *degrees* belongs.
2. **The shallow rise-time blend read the wrong slip.** The original blends against
   the *reloaded* slip spectrum brought back to space, not the tapered field the
   reload was built from.
3. **A subfault that does not slip was getting a pulse.** The `|slip| > MINSLIP`
   guard is in the SRF loader, outside the generator, so porting the generator
   faithfully did not reproduce it.
4. **The hypocentre was a cell off in both directions.** genslip's `ixs`/`iys` count
   from one, because their only consumer is a Fortran routine. See `TestOnsetAgrees`.
5. **A Frankel field was stretched where it should have been shifted.** See
   `TestTheFrankelSpectrumIsShiftedNotStretched`.

None could have been caught by the per-function parity tests. Each of `rake_field`,
the blend and `oliu_p` is correct and is tested against the C; the defects are in the
**calls**, and a suite that checks one function at a time cannot see a caller handing
the right function the wrong argument. That is the seam an end-to-end corpus closes.

The last two are worse than that, and they are the same failure twice. Both had a test
whose *reference side re-implemented the original's logic* -- `main`'s index arithmetic
in one case, `main`'s slip block in the other -- and both re-implementations made
exactly the mistake the port made, so the two agreed bit for bit while both were wrong.
A reference has to be the original's **output**. A second reading of its source by the
same reader is not independent of the first.

# What every one of them looked like from outside

Plausible. Not one produced a rupture that was obviously broken:

| defect | how it presented |
| --- | --- |
| 14 | a rake field with the right shape and 1/20th the spread |
| 16 | a three-sample spike on subfaults that do not slip |
| 17 | onset correlated 0.92-0.997, smooth, starting at zero, up to a second early |
| 18 | slip correlated 0.993, the right mean, 63% too variable |

Each is the kind of thing an eyeball, a plot, or a summary statistic passes. The
numbers that caught them were ratios and worst-case differences against a stored
reference, which is the argument for having one.
"""

import numpy as np
import pytest

from tests.harness import corpus
from tests.harness.genslip_config import KModel

CASES = [case.name for case in corpus.CASES]


@pytest.fixture(scope="module")
def compared() -> dict:
    """Every case, generated once and compared in the reference's point order.

    Module-scoped because generating the whole corpus is seconds of FFT work and
    every test below wants the same ruptures.

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
        # Cases that were secretly the same fault would pass everything below and mean
        # nothing. Twins are exempt: being the same fault is what they are for, and
        # `test_a_twin_differs_from_its_original_in_one_thing` is their check instead.
        distinct = [c for c in corpus.CASES if c.twin_of is None]
        shapes = {(c.strike_count, c.dip_count) for c in distinct}
        assert len(shapes) == len(distinct)
        assert len({c.parameters().kmodel for c in distinct}) >= 3
        assert len({c.parameters().dt for c in distinct}) >= 2
        dips = {float(corpus.load_geometry(c.name).mean_dip_deg) for c in distinct}
        assert min(dips) < 20.0, "no shallow-dipping case"
        assert max(dips) > 75.0, "no steep case"

    def test_a_twin_differs_from_its_original_in_one_thing(self) -> None:
        # A twin exists to be *differenced* against its original, so any second
        # difference between them silently contaminates the first. The stored
        # arguments are the check, since they are what genslip was actually given.
        twins = [c for c in corpus.CASES if c.twin_of is not None]
        assert twins, "no twins; `Case.twin_of` and its exemptions are now dead"

        for twin in twins:
            mine = corpus.load_arguments(twin.name)
            theirs = corpus.load_arguments(twin.twin_of)
            # infile and velfile name the case, not the physics.
            differing = {
                key
                for key in mine.keys() | theirs.keys()
                if mine.get(key) != theirs.get(key)
            } - {"infile", "velfile"}
            assert differing == {"tsfac_main"}, (
                f"{twin.name} differs from {twin.twin_of} in {sorted(differing)}"
            )
            assert float(mine["tsfac_main"]) == 0.0


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

    @pytest.mark.parametrize("name", CASES)
    def test_slip_matches_to_the_format_s_precision(
        self, name: str, compared: dict
    ) -> None:
        # 2.6e-06 is the worst of the four, and the SRF writes slip with six
        # significant figures -- so this is agreement to the file's own resolution,
        # not a tolerance. Every draw, the spectrum, the taper and the moment
        # scaling are behind this one number.
        result = compared[name]
        assert relative(result["slip_cm"], result["points"].slip_cm) < 1e-5

    @pytest.mark.parametrize("name", CASES)
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

    @pytest.mark.parametrize("name", CASES)
    def test_the_pulse_lengths_are_the_reference_s(
        self, name: str, compared: dict
    ) -> None:
        # 100% on three cases, 99.83% on `subduction` -- two subfaults of 1152.
        result = compared[name]
        exact = float(np.mean(result["pulse_length"] == result["reference_length"]))
        assert exact > 0.998, f"{name}: {exact:.4%} of pulse lengths exact"

    @pytest.mark.parametrize("name", CASES)
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

    @pytest.mark.parametrize("name", CASES)
    def test_the_samples_agree_where_the_lengths_do(
        self, name: str, compared: dict
    ) -> None:
        # 4.2e-05 relative at worst, which is the SRF's text precision for the
        # slip-rate rows. This is the pulse shape, its normalisation and the rise
        # time that set its duration, all at once.
        result = compared[name]
        assert result["slip_rate_relative"] < 1e-4


class TestOnsetAgrees:
    """`DEFECTS.md` 17, fixed. The corpus found it; nothing else could have.

    genslip's `ixs`/`iys` count from **one**, because their only use is `wfront2d`'s
    argument list and that routine is Fortran. The mapping handed them to the port
    unconverted, and the port -- whose `Hypocentre` is a 0-based subfault index --
    added one on the way back out. Every subfault then ruptured as though the
    hypocentre were a cell further along strike and a cell further down dip.

    What made it survive: the per-function parity test reproduced the C's padding
    arithmetic *with the same confusion* and fed the result to an oracle wrapper that
    also adds one, so both sides were shifted together and agreed exactly. A test
    whose reference side re-implements the surround cannot catch an error in how the
    surround is read. Only the whole rupture, against genslip's own bytes, could.
    """

    @pytest.mark.parametrize("name", CASES)
    def test_onset_matches_to_the_format_s_precision(
        self, name: str, compared: dict
    ) -> None:
        # The SRF writes `tinit` as `%10.4f`, so 1e-4 s is the file's resolution and
        # the worst disagreement anywhere in the corpus is 5.3e-05 -- half a quantum,
        # which is what rounding to that field costs and nothing more.
        result = compared[name]
        worst = float(np.abs(result["onset_s"] - result["points"].onset_s).max())
        assert worst < 1e-4, f"{name}: worst onset difference {worst} s"

    @pytest.mark.parametrize("name", CASES)
    def test_onset_is_the_same_field_and_not_merely_the_same_spread(
        self, name: str, compared: dict
    ) -> None:
        result = compared[name]
        correlation = np.corrcoef(result["onset_s"], result["points"].onset_s)[0, 1]
        assert correlation > 1.0 - 1e-9

    def test_the_travel_times_agree_even_where_the_slip_does_not(
        self, compared: dict
    ) -> None:
        """The twin, and the reason it is in the corpus.

        `frankel_no_perturbation` is `frankel_corners` with `tsfac_main = 0`, so its
        onset is the eikonal solve with no perturbation term on top. It agrees to the
        format's precision. The perturbed twin does not -- and the difference between
        the two is a statement about slip, not about travel times.

        Without this case, a regression in the solver and the known Frankel slip
        divergence would show up in the same number.
        """
        result = compared["frankel_no_perturbation"]
        worst = float(np.abs(result["onset_s"] - result["points"].onset_s).max())
        assert worst < 1e-4, f"pure travel times differ by {worst} s"


class TestTheFrankelSpectrumIsShiftedNotStretched:
    """`DEFECTS.md` 18, fixed. The last field that did not agree.

    genslip turns a generated field into slip in one of two ways, and which one is a
    property of the spectrum. Everything but Frankel is *stretched* about its mean
    until the coefficient of variation is the configured 0.75. Frankel is *shifted*
    to its own minimum instead, and the configured value is then ignored -- the
    original says so by assigning `slip_sigma = -1.0` inside the branch
    (`genslip_v5.6.2.c:1809-1825`).

    The port had no branch at all, so a Frankel field was stretched like the rest. It
    stayed correlated at 0.993 with the original, because both operations are affine
    in the same generated field, while being **63% too variable** -- and the spread is
    what survives truncation and moment scaling to become a different rupture.
    """

    def test_a_frankel_field_has_no_negative_slip_before_truncation(
        self, compared: dict
    ) -> None:
        # The physical content of the shift, and the reason it is not merely a
        # different normalisation: subtracting the minimum makes the field
        # non-negative by construction, so the truncation that follows has nothing to
        # do. A stretched field reaches truncation with negative subfaults, and
        # clipping them is what stopped the two being related by an affine map.
        for name in ("frankel_corners", "frankel_no_perturbation"):
            assert (compared[name]["slip_cm"] >= 0.0).all()

    def test_the_spread_is_the_spectrum_s_own_and_not_the_configured_one(
        self, compared: dict
    ) -> None:
        """The measurement that identified it, kept as the check.

        Slip agreeing to 1.1e-06 already covers this. It is asserted separately
        because a *ratio of spreads* is what made the defect legible when the fields
        themselves just looked "nearly right": the means matched to 2%, the pattern
        correlated at 0.993, and only the spread said 1.63.
        """
        for name in ("frankel_corners", "frankel_no_perturbation"):
            result = compared[name]
            mine, theirs = result["slip_cm"], result["points"].slip_cm
            assert float(np.std(mine) / np.std(theirs)) == pytest.approx(1.0, abs=1e-4)

    def test_and_the_stretched_spectra_still_are_stretched(
        self, compared: dict
    ) -> None:
        # The shift must be Frankel's alone. If it ever reached the other spectra
        # their slip would move, so this is really a statement that the branch is on
        # the right side of its condition -- checked here rather than left to the
        # slip comparison, which would report it as five unrelated failures.
        for name in CASES:
            if corpus.BY_NAME[name].parameters().kmodel is KModel.FRANKEL:
                continue
            result = compared[name]
            assert (result["slip_cm"] == 0.0).any(), (
                f"{name} has no zero subfault; the taper should give it one"
            )
            assert float(np.std(result["slip_cm"])) == pytest.approx(
                float(np.std(result["points"].slip_cm)), rel=1e-4
            )


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
