#!/bin/sh
# Does the suite still catch the defects it says it catches?
#
# `contracts.rs` and `test_corpus.py` both claim, in their docstrings, to catch
# specific historical defects. A claim like that decays silently: a refactor moves an
# assertion, a fixture stops having silent subfaults, a bound widens by a factor
# nobody notices, and the file goes on saying it catches something it no longer does.
#
# So the claim is executable. Each defect below is reintroduced into the library and
# both suites are run, each expected to FAIL. A "MISSED" line is a hole.
#
# This matters most for the corpus, whose bounds were deliberately loosened from the
# SRF's text precision to `ENGINEERING_RULES.md`'s physical ones -- four orders of
# magnitude. That loosening is what makes refactoring possible, and this is the
# evidence it did not cost the teeth. As of the last run every defect was caught by
# both suites, with the narrowest margin 14x.
#
# A mutation that no longer applies is the failure this script is worst at reporting,
# because it looks exactly like a hole in the suite: nothing changed, so nothing
# failed, so both columns read MISSED. It has happened twice -- once when the mutated
# file became dead code, and once when a refactor dropped a trailing comma the pattern
# matched on. So `mutate` now *asserts the file changed*, and says "did not apply"
# rather than letting it read as lost coverage.
#
#   ./teeth.sh              both suites
#   ./teeth.sh --rust-only  skip the corpus, which needs a rebuild per mutation
#
# Not in `gate.sh`: it edits the working tree, and the gate must never do that. It
# refuses to start on a dirty tree, because on one it would revert *your* work.
set -eu
cd "$(dirname "$0")"

python_too=1
[ "${1:-}" = "--rust-only" ] && python_too=0

if ! git diff --quiet -- crates/genslip/src/; then
    echo "crates/genslip/src/ has uncommitted changes; this script reverts that \
directory and would discard them." >&2
    exit 1
fi

status=0

# Apply a search-and-replace to a library file, and fail loudly if it matched nothing.
#
#   mutate <file> <needle> <replacement>
mutate () {
    if ! python3 - "$@" <<'APPLY'
import pathlib, sys
path, needle, replacement = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
source = path.read_text()
if needle not in source:
    sys.exit(1)
path.write_text(source.replace(needle, replacement))
APPLY
    then
        echo "  MUTATION DID NOT APPLY in $1 -- the pattern is stale, not the suite" >&2
        status=1
        return 1
    fi
    return 0
}

restore () {
    git checkout -- crates/genslip/src/
    [ "$python_too" -eq 1 ] && uv sync --extra test --group dev >/dev/null 2>&1
    return 0
}

check () {
    name="$1"
    contract="$2"
    corpus="$3"

    if [ "$python_too" -eq 1 ]; then
        # The corpus runs against the compiled extension, so a mutation only reaches
        # it through a rebuild. This is the slow part, and the reason for --rust-only.
        uv sync --extra test --group dev >/dev/null 2>&1
    fi

    if cargo test -p genslip --test contracts "$contract" 2>&1 |
        grep -q "test result: FAILED"; then
        rust="caught"
    else
        rust="MISSED"
        status=1
    fi

    if [ "$python_too" -eq 0 ]; then
        printf '  %-8s --       %s\n' "$rust" "$name"
        restore
        return 0
    fi

    if PYTHONPATH="$PWD" .venv/bin/python -m pytest tests/harness/test_corpus.py \
        -q -k "$corpus" 2>&1 | grep -qE "[0-9]+ failed"; then
        py="caught"
    else
        py="MISSED"
        status=1
    fi

    printf '  %-8s %-8s %s\n' "$rust" "$py" "$name"
    restore
}

printf 'Reintroducing each defect and expecting both suites to notice.\n\n'
printf '  %-8s %-8s %s\n' contracts corpus defect
printf '  %-8s %-8s %s\n' --------- ------ ------

# DEFECTS.md 17. The source is displaced one cell along strike and one down dip,
# which is what reading genslip's 1-based `ixs` as 0-based did. The onset field stays
# smooth and correlated 0.92-0.997 with the truth; only the registration moves.
#
# Mutated in the DEFAULT solver, which is the point. It used to be applied to
# `wavefront.rs`, and when `FactoredSweep` took over as the default this script went
# on reporting "caught" for the Rust contracts while the corpus silently stopped
# seeing anything -- the mutation was no longer in the path Python takes. A teeth
# check that mutates code nobody runs is the exact failure it exists to prevent.
mutate crates/genslip/src/rupture/sweeping.rs \
    "let source_slowness = slowness(hypocentre.strike, hypocentre.dip);" \
    "let hypocentre = Hypocentre {
            strike: (hypocentre.strike + 1).min(strike_count - 1),
            dip: (hypocentre.dip + 1).min(dip_count - 1),
        };
        let source_slowness = slowness(hypocentre.strike, hypocentre.dip);" &&
    check "17 hypocentre a cell off" sweeping::rupture_starts onset

# DEFECTS.md 16. The `|slip| > MINSLIP` guard lives in genslip's SRF loader rather
# than in its generator, so a faithful port of the generator does not have it.
mutate crates/genslip/src/realisation.rs \
    "if cm.abs() > slip_rate::MIN_SLIP_CM {" \
    "if true || cm.abs() > slip_rate::MIN_SLIP_CM {" &&
    check "16 pulse on a silent subfault" a_silent_subfault_emits_no_pulse emits_no_pulse

# DEFECTS.md 14. The slip field's dimensionless coefficient of variation handed to
# the rake field, where a spread in degrees belongs. A factor of twenty.
mutate crates/genslip/src/realisation.rs \
    "slip_spec.rake_sigma_deg" \
    "slip_spec.spectrum.coefficient_of_variation" &&
    check "14 rake spread in the wrong units" the_rake_spread_is_in_degrees rake

# DEFECTS.md 18. A Frankel field stretched about its mean where the original shifts
# it to its minimum. Slip stays correlated 0.993 with the truth and 63% too variable.
mutate crates/genslip/src/slip.rs \
    "let from_minimum = spectrum_spec.shape.normalises_from_its_minimum();" \
    "let from_minimum = false;" &&
    check "18 Frankel stretched not shifted" the_slip_field_has_the_spread_its_spectrum_implies slip

echo
if [ "$status" -eq 0 ]; then
    echo "every defect is still caught"
else
    echo "a defect slipped through; the suite has a hole" >&2
fi
exit "$status"
