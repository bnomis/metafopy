# TODO

Open work only, in implementation order, nearest first. Finished items move to
`DONE.md` unchanged. See the shared `todo-add` and `todo-review` skills for
the shape of an item.

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

Spec: `agents/specs/direct-tests-for-the-audit-harness/spec.md`; tasks: `agents/tasks/direct-tests-for-the-audit-harness/tasks-1.md`.

## R-library shims disagree with R

Found on 2026-11-22 by the direct tests of the audit harness (`src/tests/test_harness_rlib.py`, strict `xfail` cases in `DISAGREE`), against R
values recorded in `src/tests/reference/audit/snippets.json`:

- `audit.rlib.r_as_dataframe` of an unnamed matrix has no `V1`, `V2` column names (snippet `as_dataframe_matrix`);
- `audit.rlib.r_as_numeric` of a single string fails with "iteration over a 0-d array" (`as_numeric_text`);
- `audit.rlib.r_cbind` with named arguments gives no column names (`cbind_named`);
- `audit.rlib.r_class` gives `float` for a number where R gives `numeric`, and one name for a matrix where R gives `matrix` and `array`
  (`class_numeric`, `class_matrix`);
- `audit.rlib.r_inherits` of a number against `numeric` is false (`inherits_number`).

Cause: each shim was written for the calls the audited scripts make, not for R's full behaviour.

Fix direction: change the harness shim to R's behaviour (or document the narrower contract in its docstring and drop the case), then run
`bin/metafor-audit.sh --recall`, since a harness change can change what an audit reports. Remove the matching `DISAGREE` entry in the same change.
