# Tasks: the rma_mv funnel test skip

**Source:** TODO.md, "The rma_mv funnel test still skips when rstandard is missing"
**Created:** 2026-11-18
**Version:** 1

The item was filed on 2026-11-15 by a TODO review: `test_funnel_matches_metafor` in `src/tests/test_fits.py` skips its `funnel_mvm` case when the
test helper `has_mv_residuals` finds no `rstandard` for `rma_mv` fits, but `rstandard_rma_mv` now exists in `src/metafopy/methods/mv_diagnostics.py`.
This list is row 3 of `agents/tasks/order-todo-items-1-5.md`. The plan was approved on 2026-11-18 (item 3 of
`agents/reviews/order-todo-items-1-5-specs.md`): confirm the `mvm` case runs and matches R, then remove `has_mv_residuals` and the skip.

<!-- doc-counts: accept-start checked on 2026-11-18 - a dated record, not a claim about the present -->
Checked on 2026-11-18 at 11:52 UTC on `main` (`552f50e` plus one local commit), read only: `has_mv_residuals` is defined and used only in
`src/tests/test_fits.py` (`grep -rnw has_mv_residuals src bin docs` finds its definition and the one call in `test_funnel_matches_metafor`); the R
fixture exists, key `funnel_mvm` with `ret` and `rec` in `src/tests/reference/phase7/fits.json`, written by `src/tests/r/generate_fits.R` (the
`funnel_` loop over `mh`, `peto`, `mv`, `mvm`, `ls`, `sel`, `glmm`); and `pytest src/tests/test_fits.py -k funnel_mvm` already reports the case
`PASSED`, not skipped. `rtol_for` gives `funnel_mvm` the optimiser tolerance `RTOL_OPT` (its key contains `_mv`), so the case compares `x`, `y` and
`slab` with R and skips the drawn-geometry check, as for `funnel_mv`.
<!-- doc-counts: accept-end -->

No fixture is regenerated and no R call is needed: the fixture is committed. No task changes `src/metafopy/`. One stop for the operator, in Task 1:
**stops for the operator: the `funnel_mvm` case of `test_funnel_matches_metafor` fails once the skip is removed** (metafopy's funnel of an `rma_mv`
fit differs from R's). No version bump (the project is at 0.0.1). The project has no affected-tests tool: the task names the test file it changes, and
runs the full suite as the run order requires.

---

## Task 1: Remove the funnel skip and close the item

**Status:** todo
**Priority:** medium
**Complexity:** small
**Source Reference:** the item's "Fix direction"; the approved plan (item 3 of `agents/reviews/order-todo-items-1-5-specs.md`, 2026-11-18)

**Technical Goal:** `test_funnel_matches_metafor` runs its `funnel_mvm` case without a guard, the helper `has_mv_residuals` is gone, and the item
moves to `DONE.md`.

**Subtasks:**

- [ ] Re-check the starting point, and stop as `blocked`, naming what differs, if any of these is not so:
      - `grep -rnw has_mv_residuals src` finds only `src/tests/test_fits.py` (its definition and its call in `test_funnel_matches_metafor`)
      - `src/tests/reference/phase7/fits.json` has the key `funnel_mvm` with `ret` and `rec`
      - the item is in `TODO.md` with a `tasks:` line naming `agents/tasks/mv-funnel-skip/tasks-1.md`
- [ ] In `src/tests/test_fits.py`, delete the function `has_mv_residuals` and the two lines of `test_funnel_matches_metafor` that call it and skip
      (`if fit == 'mvm' and not has_mv_residuals(...)` and its `pytest.skip`). Leave the imports that other tests still use (`metafopy.errors` is used
      elsewhere in the file; `ruff check` reports any import left unused)
- [ ] Run `pytest src/tests/test_fits.py -k funnel -rs` and check that `test_funnel_matches_metafor[funnel_mvm]` passes and nothing in the file is
      skipped for `rstandard`. **Stops for the operator: the `funnel_mvm` case of `test_funnel_matches_metafor` fails once the skip is removed** -
      stop as `blocked`, naming the failing field (`x`, `y` or `slab`), R's and metafopy's values from the assertion message; do not change
      `src/metafopy/`, the tolerance or the fixture
- [ ] Move the item from `TODO.md` to `DONE.md`: append its heading and body unchanged at the end of `DONE.md`, then a line
      `Done: 2026-MM-DD, agents/tasks/mv-funnel-skip/tasks-1.md; <outcome>` with the date of the commit, where `<outcome>` says in one line what the
      item ended as (for example "skip and has_mv_residuals removed; funnel_mvm matches R")
- [ ] Run the final tests from the worktree and save their whole output for the merge gate (the main checkout is the first entry of
      `git worktree list --porcelain`, `/Users/simonb/src/metafopy`):
      `mkdir -p /Users/simonb/src/metafopy/ralpher-logs/mv-funnel-skip/tasks-1 && source bin/activate.sh && pytest src/tests 2>&1 | tee /Users/simonb/src/metafopy/ralpher-logs/mv-funnel-skip/tasks-1/final-tests.log`

**Acceptance Criteria:**

- [ ] `grep -rnw has_mv_residuals src` finds nothing, and `test_funnel_matches_metafor` has no `pytest.skip`
- [ ] `test_funnel_matches_metafor[funnel_mvm]` passes against the committed fixture
- [ ] The item is in `DONE.md` with its body unchanged and a `Done:` line naming this list and the outcome, and is no longer in `TODO.md`
- [ ] No file under `src/metafopy/` or `src/tests/reference/` changes

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `pre-commit run --all-files` passes
- [ ] `pre-commit run auditme-all --hook-stage manual --all-files` passes
- [ ] `mypy` passes
- [ ] `pytest src/tests/test_fits.py` passes
- [ ] The final test run above passes
- [ ] `git show $(git log --diff-filter=A --format=%H -- agents/tasks/mv-funnel-skip/tasks-1.md):TODO.md > <scratch>/todo-before.md` and
      `python3 ~/.claude/skills/todo-review/verify_bodies.py <scratch>/todo-before.md TODO.md DONE.md` exits 0 (`<scratch>` is the session's scratch
      directory, outside the repository)

**Notes:**

- Key files: `src/tests/test_fits.py` (`FUNNEL_CASES`, `has_mv_residuals`, `test_funnel_matches_metafor`, `rtol_for`),
  `src/tests/reference/phase7/fits.json` (read only), `src/tests/r/generate_fits.R` (read only), `src/metafopy/methods/mv_diagnostics.py`
  (`rstandard_rma_mv`, read only)
- No version bump: no task changes library output
- The full `pytest src/tests` run is the run order's required final check (its checkpoint command)

---
