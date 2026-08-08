#!/bin/sh
# Does the suite still catch the defects it says it catches?
#
# `contracts.rs` claims, in its docstring, to catch four of the five defects the
# corpus found. A claim like that decays: a refactor moves an assertion, a fixture
# stops having silent subfaults, a tolerance widens by a factor nobody notices, and
# the file goes on saying it catches something it no longer does.
#
# So the claim is executable. Each defect below is reintroduced into the library, the
# contract that names it is run, and the run is expected to FAIL. A "MISSED" line is a
# hole in the suite, not a passing test.
#
# This is not in `gate.sh`: it edits the working tree, and the gate must never do
# that. Run it when changing `contracts.rs`, and after any change to the pipeline that
# might move where a guard lives.
#
#   crates/genslip/tests/teeth.sh
#
# Every mutation is reverted with `git checkout` whether it is caught or not, so the
# script is safe to interrupt -- but it refuses to start on a dirty tree, because on
# one it would revert *your* work rather than its own.
set -eu
cd "$(dirname "$0")/../../.."

if ! git diff --quiet -- crates/genslip/src/; then
    echo "crates/genslip/src/ has uncommitted changes; this script reverts that \
directory and would discard them." >&2
    exit 1
fi

status=0

check () {
    name="$1"
    filter="$2"
    if cargo test -p genslip --test contracts "$filter" 2>&1 |
        grep -q "test result: FAILED"; then
        printf '  caught   %s\n' "$name"
    else
        printf '  MISSED   %s -- %s did not fail\n' "$name" "$filter"
        status=1
    fi
    git checkout -- crates/genslip/src/
}

echo "Reintroducing each defect and expecting the suite to notice."

# DEFECTS.md 17. The solver is handed a source one cell along strike and one down
# dip, which is what reading genslip's 1-based `ixs` as 0-based did. The onset field
# stays smooth and correlated 0.92-0.997 with the truth; only the registration moves.
python3 - <<'MUTATION'
import pathlib
p = pathlib.Path("crates/genslip/src/rupture/wavefront.rs")
s = p.read_text()
s = s.replace("i32::try_from(strike.source + 1).expect", "i32::try_from(strike.source + 2).expect")
s = s.replace("i32::try_from(dip.source + 1).expect", "i32::try_from(dip.source + 2).expect")
p.write_text(s)
MUTATION
check "17 hypocentre a cell off" rupture_starts_at_the_hypocentre

# DEFECTS.md 16. The `|slip| > MINSLIP` guard lives in genslip's SRF loader rather
# than in its generator, so a faithful port of the generator does not have it.
python3 - <<'MUTATION'
import pathlib
p = pathlib.Path("crates/genslip/src/realisation.rs")
s = p.read_text()
s = s.replace("slip_rate.push(if cm.abs() > slip_rate::MIN_SLIP_CM {",
              "slip_rate.push(if true || cm.abs() > slip_rate::MIN_SLIP_CM {")
p.write_text(s)
MUTATION
check "16 pulse on a silent subfault" a_silent_subfault_emits_no_pulse

# DEFECTS.md 14. The slip field's dimensionless coefficient of variation handed to
# the rake field, where a spread in degrees belongs. A factor of twenty.
python3 - <<'MUTATION'
import pathlib
p = pathlib.Path("crates/genslip/src/realisation.rs")
s = p.read_text()
s = s.replace("slip_spec.rake_sigma_deg,", "slip_spec.spectrum.coefficient_of_variation,")
p.write_text(s)
MUTATION
check "14 rake spread in the wrong units" the_rake_spread_is_in_degrees

# DEFECTS.md 18. A Frankel field stretched about its mean where the original shifts
# it to its minimum. Slip stays correlated 0.993 with the truth and 63% too variable.
python3 - <<'MUTATION'
import pathlib
p = pathlib.Path("crates/genslip/src/slip.rs")
s = p.read_text()
s = s.replace("let from_minimum = spectrum_spec.shape.normalises_from_its_minimum();",
              "let from_minimum = false;")
p.write_text(s)
MUTATION
check "18 Frankel stretched not shifted" the_slip_field_has_the_spread_its_spectrum_implies

if [ "$status" -eq 0 ]; then
    echo "every defect is still caught"
else
    echo "a defect slipped through; the suite has a hole" >&2
fi
exit "$status"
