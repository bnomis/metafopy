# TODO

Open work only, in implementation order, nearest first. Finished items move to
`DONE.md` unchanged. See the shared `todo-add` and `todo-review` skills for
the shape of an item.

## Public metafopy helpers with branches have no direct test

<!-- doc-counts: accept-start found on 2026-11-21 - a dated record, not a claim about the present -->
Found on 2026-11-21 by the first auditme audit (`agents/audit-reviews/review-1.md`, question T1, at `31d809f`): 162 functions under
`src/metafopy` that branch have no test calling them directly (under the T1 rulings that review records). Most are public helpers that the
tests reach only through a higher-level function. The per-function list is in `agents/audit-reviews/ledger/T1.jsonl`: run
`python3 ~/.claude/skills/auditme/checks/phaseb.py report . --question T1`. Decision D4 of that review (answered 2026-11-21) files the
catch-up here, one item per package.
<!-- doc-counts: accept-end -->

Cause: tests were written per feature against the public entry points (`rma_uni`, `escalc`, `selmodel`, ...), so a helper is tested only
through its caller, which T1 does not count for a public function.

Fix direction: a spec with `/specdev` first. It chooses which helpers deserve a direct test (the branchy numerics and formatters first), and
which should become private, or stay covered through their callers with a ledger dismissal giving the reason. Then a task list.

Spec: `agents/specs/direct-tests-for-metafopy-helpers/spec.md`; tasks: `agents/tasks/direct-tests-for-metafopy-helpers/tasks-1.md`.

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
