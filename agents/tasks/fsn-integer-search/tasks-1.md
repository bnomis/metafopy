# Tasks: fail-safe N integer search

**Source:** TODO.md, "The fail-safe N root search rounds up one too far on a step function"
**Created:** 2026-11-15
**Version:** 1

The user answered *follow the reference* on 2026-11-15 (recorded in the TODO.md item and in
`agents/specs/correctness-publication-bias/mismatches.md`), so this list changes metafopy's output without a new review. It is row 2 of
`agents/tasks/order-numerical-fixes.md` (planned in `agents/reviews/order-numerical-fixes-specs.md`) and waits for no other row. No task stops for
the operator.

Checked against the code on 2026-11-15 (metafopy at `89ec047`; `src/metafopy/bias/fsn.py` last changed in `72257ca`). `fsn` calls
`metafopy.core.zeroin.uniroot` in two places:

- the `Rosenthal` branch with `pool='fisher'` (`fsn_fisher`: the statistic adds `int(N)` studies of p = 0.5, but the degrees of freedom
  `2 (k + N)` take the real N);
- `root_of` in the `General` branch, used both without a target (p-value criterion) and with one (`fsn_gen` refits with `floor(N)` studies).

Both treat N as a real number over a step function of N, then round the root up with `rnd_fsn`, so the result can be one above the least integer
that meets the criterion (General with a target: 117 where the reference is 116) or one below it (Fisher: 47 where the reference is 48). The
reference is `formulas.fsn_of` in `src/tests/correctness/formulas.py`: the least integer N whose criterion holds, found by
`formulas.smallest_integer`. R metafor 5.2-1 gives metafopy's 47 and 117 (operator session, 2026-11-15), so every changed value is a new
difference from metafor, recorded in `docs/differences.md` under "Where metafor departs from its reference" and in `src/audit/data/known.toml`.

**The other types (checked 2026-11-15, by reading `fsn.py`):** Rosenthal with `pool='stouffer'`, Orwin and Rosenberg compute N in closed form
from a criterion that is continuous and monotone in N (`k (z / z_alpha)^2 - k`, `k (est - target) / target`, and Rosenberg's weighted
equivalent), so the ceiling of the closed form is already the least integer, and their correctness cases pass today. Binomial steps over the
integers in pairs (a sign test needs one positive and one non-positive study per step, as metafor's code does). None of them has the
step-function pattern, and this list does not change them. One near relative is left as it is: `rnd_fsn` rounds a closed form within
`sqrt(eps)` of an integer to that integer rather than up, which mirrors metafor and is a floating-point guard, not a step.

Decisions made while writing the list, by the project's conventions (each reversible):

- `fsn` keeps the `interval`, `tol` and `maxiter` arguments, since the API mirrors metafor's. `interval` gives the starting bracket of the integer
  search, which is then extended upwards as uniroot's `extend_int` did; `maxiter` bounds the number of bracket extensions and bisection steps;
  `tol` no longer changes the Rosenthal (Fisher) and General results, and the docstring says so.
- With `exact=True` the refits draw a different number of random deviates per call, so the global R-exact generator ends in a different state from
  R's after `fsn`. The refitted values do not depend on the draws (the file-drawer values are standardised to the mean `mumiss` and a fixed
  spread, and the refit sees them through their sum and sum of squares only), so `fsnum`, `est_fsn`, `tau2_fsn` and `pval_fsn` do not change
  with the seed. Task 1 checks that.
- The R parity fixtures stay as R wrote them (`src/tests/r/generate_bias.R`, `src/tests/r/generate_verbose.R`); a test of a value that now differs
  compares it with the reference criterion and asserts that R's value differs, as `agents/tasks/confint-ht-centre/tasks-1.md` Task 1 does for
  `ht_ci`.
- No version bump: no rule of the project asks for one (`pyproject.toml` holds `version = "0.0.1"`, and neither `AGENTS.md` nor
  `claude/rules/metafopy.md` names a bump).

Worktrees have no `.audit/Rlib`: every audit run and R call points at the main checkout's library,
`METAFOPY_AUDIT_RLIB=/Users/simonb/src/metafopy/.audit/Rlib` for the audit and `R_LIBS=/Users/simonb/src/metafopy/.audit/Rlib` for `Rscript`.
The main checkout is the first entry of `git worktree list --porcelain` (`/Users/simonb/src/metafopy`).

The project has no affected-tests tool, so each task names its test files, and the last task runs the full suite, as the run order asks.

---

## Task 1: Search the integers in the Rosenthal (Fisher) and General branches

**Status:** done
**Priority:** high
**Complexity:** medium
**Source Reference:** the item's fix direction (the integer search, the General step resolved before rounding, the two cases, the R parity fixtures)

**Technical Goal:** `fsn` returns the least integer N whose criterion holds for Rosenthal's type with Fisher pooling and for the General type,
with and without a target, and the tests say where that differs from R metafor.

**Subtasks:**

- [x] Re-check the starting point: `bias.fsn.egger2001.rosenthal.fisher` and `bias.fsn.egger2001.general.target` in
      `src/tests/correctness/cases/bias.toml` have `status = "mismatch"`, `grep -c 'metafopy.core.zeroin.uniroot' src/metafopy/bias/fsn.py`
      prints 2, and `grep -rnwE 'fsn_search' src/metafopy` finds nothing. Stop as `blocked`, naming what differs, if any is not so
- [x] Confirm that both cases fail today: `pytest src/tests/correctness/test_derived.py -k 'egger2001 and (rosenthal.fisher or general.target)' -rxX`
      reports each of the two as `XFAIL` (the strict expected failure of a case that fails). Stop as `blocked` if either is reported as passing
- [x] Add `fsn_search(satisfied, lower, upper, maxint, maxiter)` to `src/metafopy/bias/fsn.py`: the least integer `n` in `[ceil(lower), maxint]`
      with `satisfied(n)`, for a predicate that stays true once true (the search of `formulas.smallest_integer`, starting from the bracket
      `interval` gives). Extend `upper` by doubling until `satisfied(upper)` or `upper` reaches `maxint`, then bisect on the integers; return
      `maxint` when the criterion does not hold below it. Raise `metafopy.errors.InputError('Could not find fail-safe N ...')` (the message of
      the branch it serves) when the extensions and bisection steps together exceed `maxiter`
- [x] In the `Rosenthal` branch with Fisher pooling, replace the `uniroot` call by `fsn_search(lambda n: fsn_fisher(n, pi, alpha) >= 0, ...)`, so
      the number of added studies and the degrees of freedom are both the integer N
- [x] In the `General` branch, replace `root_of` by the integer search: without a target the criterion is `pval_fsn >= alpha`, with a target it is
      `abs(est_fsn) <= abs(tgt)` (as `formulas.fsn_of` states them), both from `fsn_gen(..., newest=True)` at the integer N, so the step is resolved
      before any rounding and `est_fsn`, `tau2_fsn` and `pval_fsn` are those of the returned N. Keep `ub_sign = '>'` when the result is `maxint`,
      and keep the `InputError('Could not find fail-safe N based on a random-effects model.')` for a refit that fails. Keep the `verbose` trace of
      `fsn_gen`: one line per N tried
- [x] Leave the Stouffer, Orwin, Rosenberg and Binomial branches unchanged (see the header)
- [x] Update `fsn`'s docstring: the fail-safe N of the Rosenthal (Fisher) and General types is the least integer whose criterion holds, found by a
      search over the integers, as in Becker (2005, doi:10.1002/0470870168.ch7) and Viechtbauer (2024); metafor finds a real root with `uniroot`
      and rounds it up, which can be one off on these step functions; `interval` is the starting bracket, `maxiter` bounds the search, `tol` has no
      effect on these types. Link the `docs/differences.md` subsection Task 2 adds (heading `### fsn(): Fisher pooling and the General type`, linked
      the way the `confint` docstring links its subsection)
- [x] In `src/tests/correctness/cases/bias.toml`, set `bias.fsn.egger2001.rosenthal.fisher` and `bias.fsn.egger2001.general.target` to
      `status = "pass"` and drop their `reason`. `bias.fsn.egger2001.general.target.tight` stays a pass case
- [x] Run `pytest src/tests/test_bias.py src/tests/test_verbose.py`. For each `FSN_CASES` key of `test_fsn` whose R fixture `fsnum` now differs,
      check that metafopy's N meets the criterion and N - 1 does not (with `fsn_fisher` or `fsn_gen` directly), then mark it as differing in
      `src/tests/test_bias.py`: a module-level `FSN_DIFFERS: dict[str, int]` mapping the key to metafopy's N, with a comment naming the
      correctness cases and the `docs/differences.md` subsection. For such a key `test_fsn` compares `type`, `est`, `tau2`, `pval`, `target` and
      `ub_sign` with the fixture, asserts that `out.fsnum` equals the mapped N and that the fixture's `fsnum` differs, and compares `est_fsn`,
      `tau2_fsn` and `pval_fsn` with `fsn_gen` at N instead of the fixture; it does not compare the printed text. A key whose N meets the criterion
      at N - 1 too is a bug in this change: fix it, never mark it
- [x] If the `fsn` key of `test_verbose_matches_r` in `src/tests/test_verbose.py` now fails (the search tries other values of N than uniroot),
      take it out of that parametrisation and add `test_verbose_fsn`, which checks, as `test_verbose_mv` does for its trace, that every stdout line
      has the form of R's lines (`fsnum = <N>  est = ...  diff = ...`), that the N values tried are integers, that stderr equals R's, and that the
      last line printed is for the returned N or a neighbour of it; with a comment naming the `docs/differences.md` subsection
- [x] Add regression tests to `src/tests/test_bias.py`: on `dat.egger2001` without rows 8 and 16 (0-based 7 and 15), log risk ratios with
      `add=0.5, to='all'` (the setup of the two cases), `fsn(yi, vi, pool='fisher')` gives 48 and
      `fsn(yi, vi, type='General', method='REML', exact=True, target=math.log(0.95))` gives 116; the General result is the same for `tol` 1e-10
      and the default; and, for the General type with `exact=True`, `fsnum`, `est_fsn`, `tau2_fsn` and `pval_fsn` are the same for two seeds

**Acceptance Criteria:**

- [x] `grep -c 'metafopy.core.zeroin.uniroot' src/metafopy/bias/fsn.py` prints 0 (and the import of `metafopy.core.zeroin` is gone if unused)
- [x] Both cases pass with `status = "pass"`, and every other `bias.fsn.*` case still passes
- [x] Every `FSN_DIFFERS` key meets its criterion at its N and not at N - 1, and no other `test_fsn` key changed

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy` passes
- [x] `pytest src/tests/test_bias.py src/tests/test_verbose.py src/tests/correctness/test_derived.py src/tests/correctness/test_coverage.py` passes

**Notes:**

- Key files: `src/metafopy/bias/fsn.py`, `src/tests/test_bias.py`, `src/tests/test_verbose.py`, `src/tests/correctness/cases/bias.toml`;
  reference search `smallest_integer` and `fsn_of` in `src/tests/correctness/formulas.py`
- `src/metafopy/bias/fsn.py` has no test file of its own; its tests are in `src/tests/test_bias.py` (R parity and errors), with further `fsn`
  tests in `src/tests/test_audit5_inputs.py`, `src/tests/test_audit8_crashes.py`, `src/tests/test_audit8_na_action.py`,
  `src/tests/test_audit8_refusals.py` and `src/tests/test_audit8_values_a.py`, which Task 3's full run covers
- The R fixtures most likely to move are the General keys at the default tolerance (`fsn_general*`) and `fsn_fisher`; the Stouffer, Orwin,
  Rosenberg and Binomial keys should not move, and a move in one of them is a bug in this change
- The monotonicity the search relies on: each added Fisher study adds `2 ln 2` to the statistic and 2 to its degrees of freedom, so the p-value
  rises with N; the General criteria are those of `formulas.fsn_of`, which makes the same assumption

---

## Task 2: Record the difference in the audit ledger and the documentation

**Status:** done
**Priority:** high
**Complexity:** small
**Source Reference:** the item's fix direction (`known.toml`, `docs/differences.md`)

**Technical Goal:** `bin/metafor-audit.sh` accepts the new fail-safe N values with their reason, `docs/differences.md` explains the difference,
and the printed examples in the manual show the new values.

**Subtasks:**

- [x] Run the audit slice for `fsn` from the worktree, against the main checkout's R library, in the background:
      `METAFOPY_AUDIT_RLIB=/Users/simonb/src/metafopy/.audit/Rlib bin/metafor-audit.sh --only fsn` (`--only` keeps the cases whose id contains
      the text: `matrix/fsn/*`, `standalone/fsn*` and their sweeps). If `check-r` reports that the R library is missing or differs from
      `src/audit/data/r-versions.json`, stop as `blocked` and say so; never pass `--allow-r-drift`
- [x] For each finding on `fsnum`, `est.fsn`, `tau2.fsn`, `pval.fsn` or the printed output (`Fail-safe N:` and the "with file drawer" values) of a
      Rosenthal (Fisher) or General call, check that metafopy's N meets the criterion and N - 1 does not, and add a `[[known]]` entry to
      `src/audit/data/known.toml` matching exactly those case ids, with `kind`, `path` (or `detail` for a print-diff), a `reason` (metafor finds a
      real root of a step function with `uniroot` and rounds it up; metafopy returns the least integer N whose criterion holds, as Becker 2005 and
      Viechtbauer 2024 define it) and `ref = "docs/differences.md"`. Add no entry for a finding the run does not report. Stop as `blocked`, naming
      it, for a finding on a Stouffer, Orwin, Rosenberg or Binomial call, or one whose N fails the criterion check
- [x] Rerun the same command and check that its report lists no finding left from this change
- [x] Add a subsection `### fsn(): Fisher pooling and the General type` to the section "Where metafor departs from its reference" of
      `docs/differences.md`, after the existing subsections: a short summary (the fail-safe N is the least integer whose criterion holds; metafor's
      root search over real N can give one more or one fewer on these step functions; the other types are unchanged), the Fisher example of
      `bias.fsn.egger2001.rosenthal.fisher` in R and in Python with the output of both, naming both correctness cases, and *Why:* Becker (2005)
      and Viechtbauer (2024) count added studies, so N is an integer and the criterion is a step function of it. Produce the R output with
      `R_LIBS=/Users/simonb/src/metafopy/.audit/Rlib LC_ALL=C.UTF-8 Rscript -e '<the R block>'` (R block:
      `library(metafor); dat <- escalc(measure="RR", ai=ai, n1i=n1i, ci=ci, n2i=n2i, data=dat.egger2001[-c(8,16),], add=1/2, to="all"); fsn(yi, vi, data=dat, pool="fisher")`),
      and the Python output with metafopy; give the date, R and metafor versions in the *Why* line as the other subsections do
- [x] Rerun the printed `fsn` examples of the manual and update each output block that changed: `docs/guide/bias.md` (section "Fail-safe N",
      `print(fsn(res))`) and `docs/analyses.md` (`print(fsn(res))`). Search `docs/` for `Fail-safe N:` for any other printed copy and update it in
      the same way

**Acceptance Criteria:**

- [x] The audit run with `--only fsn` reports no finding from this change, and every new `known.toml` entry matches at least one case of that run
- [x] `docs/differences.md` names the difference under "Where metafor departs from its reference"
- [x] Every printed `fsn` output in `docs/` is what metafopy prints now

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy` passes
- [x] `pytest src/tests/test_docs_examples.py src/tests/test_harness_accounts.py` passes

**Notes:**

- Depends on Task 1
- Prerequisite outside this repository: R metafor 5.2-1 and the pinned packages in `/Users/simonb/src/metafopy/.audit/Rlib` (the main
  checkout's library); the worktree has none of its own
- A full audit run is the user's call; a difference the slice does not reach surfaces in the next full run

---

## Task 3: Recall check, the full test run and closing the item

**Status:** done
**Priority:** high
**Complexity:** small
**Source Reference:** the item as a whole

**Technical Goal:** the harness still reproduces the earlier audits, the whole branch is checked once, and the item moves to `DONE.md`.

**Subtasks:**

- [x] Run the recall check in the background: `METAFOPY_AUDIT_RLIB=/Users/simonb/src/metafopy/.audit/Rlib bin/metafor-audit.sh --recall`.
      Compare its summary with the newest `recall.md` under `/Users/simonb/src/metafopy/.audit/runs/recall-*/`: every item reproduced there is
      reproduced again. Stop as `blocked`, naming the item, if one is newly missed
- [x] Move the item from `TODO.md` to `DONE.md`: append its heading and body unchanged at the end of `DONE.md`, then a line
      `Done: 2026-MM-DD, agents/tasks/fsn-integer-search/tasks-1.md` with the date of the commit
- [x] Run the full test suite from the worktree and save its whole output for the merge gate, in the main checkout (the first entry of
      `git worktree list --porcelain`, `/Users/simonb/src/metafopy`):
      `mkdir -p /Users/simonb/src/metafopy/ralpher-logs/fsn-integer-search/tasks-1 && source bin/activate.sh && pytest src/tests 2>&1 | tee /Users/simonb/src/metafopy/ralpher-logs/fsn-integer-search/tasks-1/final-tests.log`

**Acceptance Criteria:**

- [x] The recall check reproduces every item the newest recall report reproduced
- [x] The item is in `DONE.md` with its body unchanged and a `Done:` line naming this list, and no longer in `TODO.md`
- [x] `/Users/simonb/src/metafopy/ralpher-logs/fsn-integer-search/tasks-1/final-tests.log` holds the whole output of the full run

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `pre-commit run --all-files` passes
- [x] `pre-commit run auditme-all --hook-stage manual --all-files` passes
- [x] `mypy` passes
- [x] The full test run above passes, `slow` tests included
- [x] `git show $(git log --diff-filter=A --format=%H -- agents/tasks/fsn-integer-search/tasks-1.md):TODO.md > <scratch>/todo-before.md` and
      `python3 ~/.claude/skills/todo-review/verify_bodies.py <scratch>/todo-before.md TODO.md DONE.md` exits 0 (`<scratch>` is the session's
      scratch directory, outside the repository; the commit that adds this list is `HEAD` at the list's start, so this is `git show HEAD:TODO.md`
      taken then)

**Notes:**

- Depends on Task 2
- The recall check runs the harness on the commits the earlier audits used, so this list's change to `fsn.py` does not reach it; it checks that
  the `known.toml` entries of Task 2 hide no earlier item
- No version bump (see the header)

---
