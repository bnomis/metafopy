# Tasks: selmodel: a fixed first weight other than 1 on a step-function selection model

**Spec:** `agents/specs/selmodel-fixed-delta0/spec.md` (version 1.1)
**Created:** 2026-11-17
**Version:** 1

**Waits for:** row 1 of the run order `agents/tasks/order-todo-items-1-10.md`, the list `agents/tasks/docs-strict-links/tasks-1.md`, which rewrites the
docstring link `[docs/differences.md](../../../docs/differences.md#...)` of `selmodel` in `src/metafopy/models/selmodel.py`, the docstring this list
also edits. This list starts from main after row 1 merges; Task 1 stops it as `blocked` if row 1 is not merged or the link still has the
`../../../docs/differences.md` form.

The user answered the spec's decision on 2026-11-17 on the review page of `agents/reviews/order-todo-items-1-10-specs.md` (D1, what
`selmodel(type='stepfun', delta=[d0, ...])` does with `d0` neither `nan` nor 1): **(a) reject with `InputError`**. The spec's choice "Which
step-function forms the check covers" is **(i), every step form**: `type='stepfun'` with or without `decreasing` and `type='stepcon'` raise the same
error. This list builds option (a) with choice (i) only, as the spec's "Tests and records" section lists it, and changes metafopy's output without a
new review; options (b) and (c) are the spec's record of what was weighed, not tasks.

Checked against the code on 2026-11-17 (metafopy at `c75f3d0`, before row 1): in `selmodel()` of `src/metafopy/models/selmodel.py`,
`check_bounds('delta', delta_arg, spec)` is followed by the two lines that set `delta_arg[0] = 1.0` (for `'stepfun'` with `decreasing` or a `nan`
first value, and for `'stepcon'`); `metafopy.models.sel_spec.STEPWISE` is `('stepcon', 'stepfun')`; for `'stepcon'` and `decreasing=True` the
bounds of `delta[0]` are [0, 1]. No R parity fit and no correctness case has a first step weight other than 1, and the spec expects no `known.toml`
entry.

The project has no affected-tests tool, so each task names its test files, and the last task runs `pytest src/tests` as the run order requires.
Every R call and audit run sets `METAFOPY_AUDIT_RLIB=/Users/simonb/src/metafopy/.audit/Rlib` (`R_LIBS` to the same path for `Rscript`): a worktree
has no `.audit/Rlib` of its own. Scratch scripts stay outside the repository. No version bump (`pyproject.toml` version `0.0.1`).

**Operator stops:** none expected. Task 1 names one conditional stop, "stops for the operator: an existing test or case reaches the new error",
which fires only if a test, parity fit or correctness case the spec did not list starts raising the new `InputError`.

---

## Task 1: Re-check the starting point, and reject a first step weight other than 1

**Status:** done
**Priority:** high
**Complexity:** small
**Spec Reference:** API; Error handling, option (a); Implementation, option (a); Choices made, "Which step-function forms the check covers" (i);
Tests and records, items 1 and 4; Acceptance

**User Story:** As a user of `selmodel`, I want a fixed first weight other than 1 on a step-function selection model to raise an error that tells
me to give the weights relative to the first, so that I never get a fit of weights I did not write or an LRT that moves with a constant.

**Subtasks:**

- [x] Check that row 1 of the run order is merged on main: in the main checkout (the first entry of `git worktree list --porcelain`,
      `main=$(git worktree list --porcelain | sed -n '1s/^worktree //p')`), `git -C "$main" show main:DONE.md` holds the heading
      `## The Docs workflow fails on three docstring links to docs/differences.md`, `git -C "$main" show main:TODO.md` does not, and
      `git merge-base --is-ancestor main HEAD` exits 0. Stop as `blocked`, naming what is missing, if not
- [x] Check that row 1's edit is in this branch: `grep -nF '](../../../docs/differences.md' src/metafopy/models/selmodel.py` finds nothing. Stop as
      `blocked`, naming the line, if it finds one
- [x] Re-check the starting point, narrowly: `grep -nF "check_bounds('delta', delta_arg, spec)" src/metafopy/models/selmodel.py` finds the one call
      in `selmodel()`, followed by the `if type_ == 'stepfun' and (decreasing or math.isnan(delta_arg[0])):` and `if type_ == 'stepcon':` lines;
      `grep -rnF "first value of 'delta' must be 1" src/metafopy` finds nothing; and
      `grep -nwE 'test_stepfun_first_delta_must_be_one|test_stepfun_first_delta_one_or_nan_fits' src/tests/test_selmodel.py` finds nothing. Stop as
      `blocked`, naming what differs, if any is not so
- [x] In `src/tests/test_selmodel.py`, beside `bangertdrowns_ml()`, add `test_stepfun_first_delta_must_be_one`, parametrised over `(type, decreasing)`
      in `('stepfun', False)`, `('stepfun', True)`, `('stepcon', False)` and `d0` in `0.5` and `0.0`, plus `d0 = 2.0` for `('stepfun', False)` only
      (the other two forms bound `delta[0]` by 1, so `2.0` keeps its bounds error there): on `bangertdrowns_ml()` with `steps=[0.025, 0.5, 1]` and
      `delta=[d0, nan, nan]`, assert `metafopy.errors.InputError` with `match="first value of 'delta' must be 1"`
- [x] Add `test_stepfun_first_delta_one_or_nan_fits`: on `bangertdrowns_ml()` with `type='stepfun'` and `steps=[0.025, 0.5, 1]`, `delta=[1, nan, nan]`
      and `delta=[nan, nan, nan]` give the same fit, beta 0.10569 and ll -17.4217 (absolute tolerance 1e-4, the spec's values measured 2026-11-17),
      and `delta=[1, 0.5, 0.25]` fits with beta 0.09380 (the same tolerance)
- [x] Run the two new tests against the unchanged code and record in this task's Notes that `test_stepfun_first_delta_must_be_one` fails for every
      parameter (a fit is returned, or `EstimationError` for `d0 = 0.0` on `'stepfun'`) and `test_stepfun_first_delta_one_or_nan_fits` passes. A
      parameter that already passes, or a value of the second test other than the spec's, stops the task as `blocked`, naming it
- [x] In `selmodel()` of `src/metafopy/models/selmodel.py`, right after `check_bounds('delta', delta_arg, spec)` and before the two lines that set
      `delta_arg[0] = 1.0`, add
      `if type_ in metafopy.models.sel_spec.STEPWISE and not math.isnan(delta_arg[0]) and delta_arg[0] != 1.0:` raising
      `metafopy.errors.InputError("The first value of 'delta' must be 1 or nan for this type of selection model: the weight of the first p-value
      interval is fixed at 1, so give the other weights relative to it (see docs/differences.md).")`, before any fitting. Exact comparison with 1.0,
      no tolerance (settled in the spec's Implementation). Keep the two `delta_arg[0] = 1.0` lines; leave `ll_stepfun` and `w_step` in
      `src/metafopy/models/sel_lik.py` unchanged
- [x] Update the `delta` description of `selmodel`'s docstring to the spec's API text (for `'stepfun'` and `'stepcon'` the first value is the weight of
      the first p-value interval, fixed at 1 to identify the model (Vevea & Hedges, 1995): it must be 1 or ``nan``, and the other values are weights
      relative to it (``docs/differences.md``, "Where metafor departs from its reference")), and add "a first ``delta`` other than 1 or ``nan`` for
      ``type='stepfun'`` or ``'stepcon'``" to the `InputError` line of `Raises`. Write the reference as inline code, as the spec does, not as a link
- [x] Check that no other caller passes such a `delta`: `refit` in `src/metafopy/methods/selmodel.py` passes the fit's own `delta`, whose first value is
      1 for a step function (the spec's investigation). Run the Definition of Done's test files.
      **Stops for the operator: an existing test or case reaches the new error.** Only if an existing test, R parity fit or correctness case now
      raises the new `InputError` or changes value: do not edit it; stop the task as `blocked` and report each test id with the call it makes. The
      spec lists none, so a hit is a change to metafopy's output it did not approve. Otherwise there is no stop

**Acceptance Criteria:**

- [x] `selmodel` raises the spec's `InputError` for `'stepfun'` with and without `decreasing` and for `'stepcon'` with a first `delta` of 0.5 or 0.0,
      and for `'stepfun'` with 2.0, before any fitting; `'stepfun'` with `decreasing=True` and `'stepcon'` keep the bounds error "must be <= 1" for 2.0
- [x] A first value of 1 or `nan` fits as before; both new tests pass, and this task's Notes record the first failing on the code before the change
- [x] Every existing test in `src/tests/test_selmodel.py`, `src/tests/test_stepcon.py`, `src/tests/test_selmodel_parallel.py` and
      `src/tests/correctness` passes unchanged, with no fixture edited
- [x] The `delta` docstring and `Raises` say what the spec's API section gives

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `ruff format --check src/metafopy/models/selmodel.py src/tests/test_selmodel.py` and `ruff check` on the same files pass
- [x] `mypy` passes
- [x] `pytest src/tests/test_selmodel.py src/tests/test_stepcon.py src/tests/test_selmodel_parallel.py src/tests/correctness` passes

**Notes:**

- Run on the unchanged code on 2026-11-17: `test_stepfun_first_delta_must_be_one` failed for all seven parameters (six returned a fit, `stepfun`
  with `d0 = 0.0` raised `EstimationError`), and `test_stepfun_first_delta_one_or_nan_fits` passed with the spec's values. After the change both pass,
  and `src/tests/test_selmodel.py`, `test_stepcon.py`, `test_selmodel_parallel.py` and `src/tests/correctness` pass with no existing test edited
- Depends on `agents/tasks/docs-strict-links/tasks-1.md` (row 1 of `agents/tasks/order-todo-items-1-10.md`), merged on main
- Key files: `src/metafopy/models/selmodel.py` (`selmodel`, `check_bounds`), `src/metafopy/models/sel_spec.py` (`STEPWISE`, unchanged),
  `src/metafopy/models/sel_lik.py` (`ll_stepfun`, unchanged), `src/tests/test_selmodel.py`, `src/tests/reference/phase5/selmodel.json` (unchanged)
- The bounds error comes first, so `delta[0] = 2` with `decreasing=True` or `'stepcon'` still reads "must be <= 1"; `d0 = 0` on `'stepfun'` now
  raises this `InputError` instead of the optimiser's `EstimationError`
- `sel_spec.py` and `selmodel.py` have no test files of their own beyond `src/tests/test_selmodel.py`; `test_stepcon.py` and
  `test_selmodel_parallel.py` reach the same step-function branch

---

## Task 2: Record the difference in the user guide

**Status:** todo
**Priority:** high
**Complexity:** small
**Spec Reference:** Tests and records, item 2; Acceptance (the `docs/differences.md` entry and `test_docs_examples.py`); External dependencies

**Technical Goal:** `docs/differences.md` explains the rejection with R's output beside metafopy's error, so a user who ports an R call with a first
weight other than 1 learns why it fails and how to rewrite it.

**Subtasks:**

- [ ] Add a subsection `### selmodel(type="stepfun") with a fixed first weight` to the section `## Where metafor departs from its reference` of
      `docs/differences.md`, right after `### selmodel(type="stepfun") with a precision measure` and in its format: a short summary, the R call and the
      Python call side by side in a `compare` block, and `*Why:*` with the check date. The summary says that metafopy raises `InputError` for a
      first `delta` other than 1 or `nan` on `type="stepfun"`, with or without `decreasing=TRUE`, and on `type="stepcon"`; that metafor fits
      `"stepfun"` with the first weight as 1 in the normalising constant and `delta[1]` for the observed p-values, and silently replaces `delta[1]`
      by 1 for `decreasing=TRUE` and `"stepcon"`; and that dividing every weight by the first gives the model the user meant
- [ ] The R block: `library(metafor)`, `res <- rma(yi, vi, data=dat.bangertdrowns2004, method="ML")`,
      `selmodel(res, type="stepfun", steps=c(0.025, 0.5, 1), delta=c(2, NA, NA))`. Run it on the day the entry is written with
      `export METAFOPY_AUDIT_RLIB=/Users/simonb/src/metafopy/.audit/Rlib; LC_ALL=C.UTF-8 R_LIBS=$METAFOPY_AUDIT_RLIB Rscript <script>` (the script in
      the scratch directory) and paste the printed output as printed. Also print `logLik` and the LRT of the same call with `delta=c(1, NA, NA)`,
      and add below the block the line "With `delta=c(1, NA, NA)` the log-likelihood is -17.4217 and LRT 1.6810." The spec measured, on 2026-11-17,
      ll -7.7176 and LRT 21.0891 (p 2.6e-5) for `c(2, NA, NA)` and ll -17.4217, LRT 1.6810 for `c(1, NA, NA)`; a different value stops the task as
      `blocked`, naming both. Stop as `blocked` too if metafor does not load or its version is not 5.2.1
- [ ] The Python block starts `# docs: skip` and shows
      `selmodel(res, type='stepfun', steps=[0.025, 0.5, 1], delta=[2, math.nan, math.nan])` (with `import math` and the `res` of the
      `stepfun`-with-`prec` entry) raising `InputError`; run it once with metafopy (`source bin/activate.sh`) to paste the message as printed
- [ ] The `*Why:*`: with the other weights free, a fixed first weight `d0` only adds the constant `n1 * log(d0)` to the log-likelihood (`n1` the
      number of observed p-values in the first interval), so the estimates are those of `d0 = 1` while ll, LRT, AIC, BIC and AICc move; with every
      weight fixed, metafor fits `1, d1, d2` in place of the relative weights the user wrote (`c(0.5, 0.25, 0.125)` is fitted as `1 : 0.25 : 0.125`),
      so a sensitivity analysis with a priori weights (Vevea and Woods 2005) is given other weights; Vevea and Hedges (1995, Psychometrika
      60:419-435, pp. 424-425) and `?selmodel` fix the first weight at 1 to identify the model. Name the decision and its date (2026-11-17, D1 of
      `agents/reviews/order-todo-items-1-10-specs.md`; `agents/specs/selmodel-fixed-delta0/spec.md`) and the check date, R 4.6.2, metafor 5.2-1
- [ ] Check that the entry says the same as the `delta` docstring and `Raises` Task 1 wrote (the forms covered, 1 or `nan`, relative weights)
- [ ] Run `pytest src/tests/test_docs_examples.py`

**Acceptance Criteria:**

- [ ] `docs/differences.md` has the subsection under "Where metafor departs from its reference", after the `prec` entry, with R's output as printed
      on the day it was written and metafopy's error as printed
- [ ] The summary names every step form the check covers, and the entry and the `delta` docstring agree
- [ ] `pytest src/tests/test_docs_examples.py` passes

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `pytest src/tests/test_docs_examples.py src/tests/test_selmodel.py` passes

**Notes:**

- Depends on Task 1
- Prerequisite outside this repository: R 4.6.2 with metafor 5.2-1 and the pinned packages in `/Users/simonb/src/metafopy/.audit/Rlib`, the main
  checkout's R library
- `src/tests/test_docs_examples.py` runs every Python block of `docs/` not marked `# docs: skip`, so neither block of the new entry runs there; their
  output comes from the commands above
- Key file: `docs/differences.md`

---

## Task 3: Run the audit and its recall check

**Status:** todo
**Priority:** high
**Complexity:** small
**Spec Reference:** Tests and records, item 5; Acceptance (the audit with `--recall`)

**Technical Goal:** the whole audit and its recall check run on the change, adding a `known.toml` entry only if the audit flags `selmodel` with a
first `delta` other than 1.

**Subtasks:**

- [ ] Run the whole audit from the worktree, in the background:
      `METAFOPY_AUDIT_RLIB=/Users/simonb/src/metafopy/.audit/Rlib bin/metafor-audit.sh`. If `check-r` reports that the R library is missing or
      differs from `src/audit/data/r-versions.json`, stop as `blocked` and say so; never pass `--allow-r-drift`
- [ ] If its report flags `selmodel` with a first `delta` other than 1 (a step-function case whose `delta` starts with a value other than 1), add a
      `[[known]]` entry to `src/audit/data/known.toml` matching exactly those case ids, with `kind`, `path`, a `reason` citing the decision and its
      date (D1 of `agents/reviews/order-todo-items-1-10-specs.md`, 2026-11-17) and `ref = "docs/differences.md"`, and re-render the run's report with
      `bin/metafor-audit.sh --rerender .audit/runs/<run>` to see it accepted. Otherwise add no entry: the spec expects none, since the audit's only
      `delta` value is `[1, 0.5]` (`src/audit/data/domains.toml`)
- [ ] Any other finding that the newest earlier report under `/Users/simonb/src/metafopy/.audit/runs/` does not hold is new with this list: stop as
      `blocked`, naming it
- [ ] Run `METAFOPY_AUDIT_RLIB=/Users/simonb/src/metafopy/.audit/Rlib bin/metafor-audit.sh --recall`, in the background. A miss stops the task as
      `blocked`, naming the item (`AGENTS.md`: a miss is fixed in the harness, `src/audit/data/`, which is outside this list)
- [ ] Record in this task's Notes the run directories of both runs and whether a `known.toml` entry was added

**Acceptance Criteria:**

- [ ] The whole audit reports no finding new with this list, and every `known.toml` entry added matches at least one case of that run
- [ ] `bin/metafor-audit.sh --recall` reproduces every item
- [ ] This task's Notes name both run directories and whether a `known.toml` entry was added

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `pytest src/tests/test_harness_accounts.py` passes if `src/audit/data/known.toml` changed

**Notes:**

- Depends on Task 2
- Key files: `src/audit/data/known.toml` (only if the audit flags the change), `src/audit/README.md`
- If no file but this list changes, the commit records the Notes only

---

## Task 4: The full test run and closing the spec

**Status:** todo
**Priority:** high
**Complexity:** small
**Spec Reference:** Tests and records, item 3; Acceptance; the spec as a whole

**Technical Goal:** the whole branch is checked once, the spec is marked implemented and its item moves to `DONE.md` with the decision.

**Subtasks:**

- [ ] Set the spec's `**Status:**` in `agents/specs/selmodel-fixed-delta0/spec.md` to `implemented`
- [ ] Move the item "A fixed delta[0] other than 1 on a step-function selection model makes the likelihood a constant multiple" from `TODO.md` to
      `DONE.md`: append its heading and body unchanged at the end of `DONE.md`, then a line
      `Done: 2026-MM-DD, agents/tasks/selmodel-fixed-delta0/tasks-1.md; option (a), a first delta other than 1 or nan on every step form raises InputError (agents/specs/selmodel-fixed-delta0/spec.md, D1 answered 2026-11-17 in agents/reviews/order-todo-items-1-10-specs.md)`
      with the date of the commit
- [ ] Run the final tests from the worktree and save their whole output for the merge gate, in the main checkout (the first entry of
      `git worktree list --porcelain`, here `/Users/simonb/src/metafopy`):
      `mkdir -p /Users/simonb/src/metafopy/ralpher-logs/selmodel-fixed-delta0/tasks-1 && source bin/activate.sh && pytest src/tests 2>&1 | tee /Users/simonb/src/metafopy/ralpher-logs/selmodel-fixed-delta0/tasks-1/final-tests.log`

**Acceptance Criteria:**

- [ ] The spec's status is `implemented`
- [ ] The item is in `DONE.md` with its body unchanged and the `Done:` line, and no longer in `TODO.md`

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `pre-commit run --all-files` passes
- [ ] `pre-commit run auditme-all --hook-stage manual --all-files` passes
- [ ] `ruff format --check`, `ruff check` and `mypy` pass
- [ ] The final test run above passes, and its log is saved at
      `/Users/simonb/src/metafopy/ralpher-logs/selmodel-fixed-delta0/tasks-1/final-tests.log`
- [ ] `git show $(git log --diff-filter=A --format=%H -- agents/tasks/selmodel-fixed-delta0/tasks-1.md):TODO.md > <scratch>/todo-before.md` (the
      `TODO.md` at the list's start) and `python3 ~/.claude/skills/todo-review/verify_bodies.py <scratch>/todo-before.md TODO.md DONE.md` exits 0
      (`<scratch>` is the session's scratch directory, outside the repository)

**Notes:**

- Depends on Task 3
- The baseline of `verify_bodies.py` is the commit that adds this list, so the item row 1 moved to `DONE.md` before it is found in `DONE.md`
- The full `pytest src/tests` run is the run order's required final check

---
