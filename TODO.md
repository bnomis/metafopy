# TODO

Open work only, in implementation order, nearest first. Finished items move to
`DONE.md` unchanged. See the shared `todo-add` and `todo-review` skills for
the shape of an item.

## `RRandom.walker` draws differently from R's `sample.int` with `prob` and more than 200 weights

Found on 2026-11-22 while writing the caller tests of Task 6 of `agents/tasks/direct-tests-for-metafopy-helpers/tasks-1.md`. Base R 4.6.2
(`Rscript`, no metafor needed) gives, after `set.seed(42)`:

- `sample.int(300, 5, TRUE, prob = rep(1, 300))` is `49 153 228 122 128`, and `metafopy.core.rng.set_seed(42).sample_int(300, 5, True, prob=numpy.ones(300))`
  (through `RRandom.walker`, `src/metafopy/core/rng.py`) gives `275 282 86 250 193`;
- `sample.int(300, 6, TRUE, prob = (1:300)/sum(1:300))` is `276 153 228 122 128 289`, and metafopy gives `275 282 258 250 193 156`.

The cases with at most 200 weights (`prob_sample_replace`, `prob_sample_no_replace`) and `sample_int` without `prob` agree with R, so the
difference is in the Walker alias branch.

Cause: not found. R's first two draws, 49 and 153, equal the first two of `set.seed(42); sample.int(300, 3, TRUE)`, but its third (228) does
not, so the alias branch is not a plain rejection-sampler draw either. metafopy's walker draws `unif_rand() * n`. Compare with R's
`src/main/random.c` for the reference version before changing it.

Fix direction: fix `RRandom.walker` to draw as R does, then add the two cases above as tests. `RRandom.sample_int` in `gosh`
(`src/metafopy/methods/gosh.py`) passes `prob` and `replace=True` and may reach the branch for large `k`; not checked.

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
