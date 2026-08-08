"""Tests for the GSF reader and writer.

The interesting claims are not about the file format, which is eleven numbers a line.
They are about the four quantities genslip *derives* from a GSF and then uses
everywhere else, because those are what a caller has to pass back on the command line
and what ends up in the SRF header. `top_depth_km` is pinned against a header the real
binary wrote.
"""

from pathlib import Path

import numpy as np
import pytest

from tests.harness import gsf

STRIKE, DIP = 20, 12
SUBFAULT_KM = 0.5


def plane() -> gsf.GsfSubfaults:
    """The fixture geometry: a 10 x 6 km plane dipping 80 degrees, 0.5 km subfaults."""
    return gsf.on_a_plane(
        strike_count=STRIKE,
        dip_count=DIP,
        along_strike_km=SUBFAULT_KM,
        down_dip_km=SUBFAULT_KM,
        centre_longitude_deg=172.0,
        centre_latitude_deg=-43.5,
        strike_deg=45.0,
        dip_deg=80.0,
        top_depth_km=0.0,
        rake_deg=175.0,
    )


class TestTheFileRoundTrips:
    def test_every_column_survives(self, tmp_path: Path) -> None:
        original = plane()
        path = tmp_path / "geometry.gsf"
        gsf.write_gsf(original, path)
        reread = gsf.read_gsf(path)

        assert len(reread) == len(original)
        for name in (
            "longitude_deg",
            "latitude_deg",
            "depth_km",
            "along_strike_km",
            "down_dip_km",
            "strike_deg",
            "dip_deg",
            "rake_deg",
            "slip_cm",
            "onset_s",
        ):
            # The file stores six decimals, so this is equality to that and not to
            # float32.
            assert getattr(reread, name) == pytest.approx(
                getattr(original, name), abs=1e-6
            ), name
        assert np.array_equal(reread.segment, original.segment)

    def test_comment_lines_are_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "commented.gsf"
        gsf.write_gsf(plane(), path)
        path.write_text("# written by something\n#\n" + path.read_text())
        assert len(gsf.read_gsf(path)) == STRIKE * DIP

    def test_a_short_file_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "truncated.gsf"
        gsf.write_gsf(plane(), path)
        lines = path.read_text().splitlines()
        path.write_text("\n".join(lines[:-1]) + "\n")
        with pytest.raises(ValueError, match="declares 240 subfaults and supplies 239"):
            gsf.read_gsf(path)


class TestTheDerivedQuantities:
    def test_the_subfault_sizes_average_to_themselves(self) -> None:
        subfaults = plane()
        assert subfaults.mean_along_strike_km == pytest.approx(SUBFAULT_KM)
        assert subfaults.mean_down_dip_km == pytest.approx(SUBFAULT_KM)

    def test_the_dip_averages_to_itself(self) -> None:
        assert plane().mean_dip_deg == pytest.approx(80.0)

    def test_the_top_depth_lifts_the_shallowest_centre_to_the_edge(self) -> None:
        # The shallowest centre is half a subfault down dip from the top edge, so the
        # edge is that much shallower: 0.25 * sin(80 degrees) above 0.25.
        subfaults = plane()
        shallowest = float(subfaults.depth_km.min())
        assert shallowest == pytest.approx(0.25 * np.sin(np.radians(80.0)), abs=1e-4)
        assert subfaults.top_depth_km == pytest.approx(0.0, abs=1e-4)

    def test_the_top_depth_is_the_one_genslip_writes(self) -> None:
        """Pinned against a header the real binary wrote.

        genslip v5.6.2, handed a GSF whose shallowest subfault centre is at 0.25 km
        with 0.5 km subfaults dipping 80 degrees, put `dtop = 0.0038` in its SRF
        header. That is `0.25 - 0.25 * sin(80 deg)` with genslip's own truncated
        radians-per-degree constant, and it is the arithmetic `top_depth_km` does.
        """
        flat = gsf.GsfSubfaults(
            longitude_deg=np.full(4, 172.0, dtype=np.float32),
            latitude_deg=np.full(4, -43.5, dtype=np.float32),
            depth_km=np.array([0.25, 0.75, 1.25, 1.75], dtype=np.float32),
            along_strike_km=np.full(4, 0.5, dtype=np.float32),
            down_dip_km=np.full(4, 0.5, dtype=np.float32),
            strike_deg=np.full(4, 45.0, dtype=np.float32),
            dip_deg=np.full(4, 80.0, dtype=np.float32),
            rake_deg=np.full(4, 175.0, dtype=np.float32),
            slip_cm=np.full(4, -1.0, dtype=np.float32),
            onset_s=np.zeros(4, dtype=np.float32),
            segment=np.zeros(4, dtype=int),
        )
        assert flat.top_depth_km == pytest.approx(0.0038, abs=5e-5)

    def test_the_top_depth_is_floored_at_the_surface(self) -> None:
        # A fault whose top row is shallower than half a subfault would otherwise
        # produce a negative depth, and genslip clamps rather than allowing one.
        subfaults = plane()
        subfaults.depth_km = np.zeros_like(subfaults.depth_km)
        assert subfaults.top_depth_km == 0.0


class TestRefusesMismatchedColumns:
    def test_ragged_columns_are_refused(self) -> None:
        subfaults = plane()
        with pytest.raises(ValueError, match="disagree on length"):
            gsf.GsfSubfaults(
                **(
                    {
                        name: getattr(subfaults, name)
                        for name in gsf._COLUMNS
                        if name != "depth_km"
                    }
                    | {
                        "depth_km": subfaults.depth_km[:-1],
                        "segment": subfaults.segment,
                    }
                )
            )


# Deliberately not asserted:
#
# - That `on_a_plane` puts subfaults where a fault really is. It is a flat-earth
#   layout for producing fixture input; what matters is that it is fixed and that both
#   sides are handed the same one. The library does no geodesy at all -- see
#   assemble.py -- and genslip's own is a tangent-plane approximation that DEFECTS.md
#   measures at a kilometre out on subduction geometry.
