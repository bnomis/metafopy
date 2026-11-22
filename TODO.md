# TODO

Open work only, in implementation order, nearest first. Finished items move to
`DONE.md` unchanged. See the shared `todo-add` and `todo-review` skills for
the shape of an item.

## Numerical cores have no property-based tests

<!-- doc-counts: accept-start found on 2026-11-21 - a dated record, not a claim about the present -->
Found on 2026-11-21 by the first auditme audit (`agents/audit-reviews/review-1.md`, question T2, at `31d809f`): 144 pure functions with
well-defined domains have no hypothesis test. 132 of them are under `src/metafopy`, and the rest under `src/audit` and `bin`. The only
property tests (`src/tests/correctness/test_properties.py`) work on whole fits. Each finding in `agents/audit-reviews/ledger/T2.jsonl` states
the property to test. Decision D5 of that review (answered 2026-11-21) files it here as one item.
<!-- doc-counts: accept-end -->

Cause: the functions were checked against R fixtures at fixed points, never over their domains.

Fix direction: a spec with `/specdev`, starting with the numerical cores, where a property catches a real bug. These are `core/bvn.py`,
`core/dbinom.py`, `core/farebrother.py`, `core/tukey.py`, `core/nchg.py`, the distribution functions in `methods/emm_adjust.py`, the weight
functions in `models/sel_spec.py`, and `models/sel_lik.py::map_delta`/`map_inverse`. Then the formatting helpers, then the harness.

Spec: `agents/specs/property-tests-for-metafopy-helpers/spec.md`; tasks: not yet written.

Note 2026-11-22: the T2 findings under `src/audit` and `bin` move to the item "Audit-harness and bin helpers with branches have no
direct test" (decision D2 of `agents/reviews/property-tests-for-metafopy-helpers-spec.md`); this item now covers `src/metafopy` only.

## Audit-harness and bin helpers with branches have no direct test

<!-- doc-counts: accept-start found on 2026-11-21 - a dated record, not a claim about the present -->
Found on 2026-11-21 by the first auditme audit (`agents/audit-reviews/review-1.md`, question T1, at `31d809f`): 108 functions under
`src/audit` (most in `rlib.py`, `transpile.py`, `pycall.py`, `generate.py` and `apidiff.py`), and 9 in `bin/logo.py`,
`bin/metafor-audit.py` and `bin/platform-probe.py`, branch and have no direct test. The list is in
`agents/audit-reviews/ledger/T1.jsonl`. Decision D4 of that review (answered 2026-11-21) files it here.
<!-- doc-counts: accept-end -->

Cause: the harness is exercised end to end by `bin/metafor-audit.sh --recall` and by a few step tests, not function by function.

Fix direction: a spec with `/specdev`. It decides which harness functions need direct tests (the R-library shims in `rlib.py` and the
transpiler first), and dismisses the rest in the ledger with the reason that the recall check covers them.

Note 2026-11-22: this item also takes the T2 (property test) findings under `src/audit` and `bin`, moved from the item "Numerical cores
have no property-based tests" (decision D2 of `agents/reviews/property-tests-for-metafopy-helpers-spec.md`). The list is in
`agents/audit-reviews/ledger/T2.jsonl`.
