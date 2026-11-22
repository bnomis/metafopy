# Tasks: weibull_shape_start reaches the shape R reaches

**Source:** `TODO.md`, "`weibull_shape_start` stops at its start value for large shapes"
**Created:** 2026-11-22
**Version:** 1

How to run things, in every task (from the worktree root):

- **Tests:** `export PATH=/Users/simonb/src/metafopy/.venv/bin:$PATH PYTHONPATH=$PWD/src && pytest <files>`
- **Type check:** `mypy <files>`, with the same `PATH`.
- **R:** `LC_ALL=C.UTF-8 R_LIBS=/Users/simonb/src/metafopy/.audit/Rlib Rscript -e '<code>'`. Worktrees have no `.audit/Rlib`, so use the main
  checkout's.

The fix and its output change were approved on 2026-11-22 (item 2 of `agents/reviews/order-r-parity-fixes-from-property-tests-specs.md`, Answers section). That approval covers only the change the
item describes. Any other change to metafopy's output stops the task as `blocked`, naming the function, the input and both values
(`claude/rules/metafopy.md`).

---

## Task 1: Find the Weibull start shape by a root find on the coefficient of variation

**Status:** done
**Priority:** high
**Complexity:** medium
**Source Reference:** `TODO.md`, "`weibull_shape_start` stops at its start value for large shapes"; review item 2

**Technical Goal:** `weibull_shape_start` returns the shape whose Weibull coefficient of variation equals `sd / mean`. That shape is where
R's `nlminb` minimum lies, instead of the start value L-BFGS-B stops at.

**Subtasks:**

- [x] Before changing anything, run `src/tests/test_conv_fivenum.py` and `src/tests/test_conv_fivenum_methods.py`, saving the outcome of every
      test. For mean 1, record `weibull_shape_start` for the coefficients of variation of shapes 0.5, 1, 2, 4, 8, 20 and 50. Get R's
      `nlminb` value for the same inputs, with R's start and objective as in the item
- [x] In `src/metafopy/effects/fivenum_estimators.py::weibull_shape_start`, replace the `lbfgsb` call with
      `scipy.optimize.brentq(lambda k: cv(k) - sd_hat / mean_hat, 1e-3, upper)`. `cv(k)` is the existing `sqrt(temp1 / temp2**2 - 1)`, and
      `upper` starts at `max(2 * start, 1.0)` and doubles until `cv(upper) < sd_hat / mean_hat`, up to 1e6. Keep the fallback to `start`
      where no bracket is found or the input is not finite. Update the docstring: cite R's `nlminb` call, and say a root find reaches its
      minimum because the objective is 0 there
- [x] Compare the seven shapes with R's: each within 1e-8 relative. Shapes 0.5 to 4 within 1e-5 of the old values
- [x] Remove the strict `xfail` marker of
      `src/tests/test_conv_fivenum_methods.py::test_weibull_shape_start_property_reproduces_the_coefficient_of_variation`
- [x] Rerun both fivenum test files. Every test that passed before passes. A parity fixture that now fails, or a final estimate that
      moves past its fixture's tolerance, stops the task as `blocked`, naming the test and both values

**Acceptance Criteria:**

- [x] The seven shapes match R's `nlminb` to 1e-8 relative, and the property test passes without `xfail`
- [x] No fivenum test changes outcome except the former `xfail`

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy src/metafopy/effects/fivenum_estimators.py src/tests/test_conv_fivenum_methods.py` passes
- [x] `pytest src/tests/test_conv_fivenum.py src/tests/test_conv_fivenum_methods.py` passes

**Notes:**

- The only caller is `src/metafopy/effects/fivenum_estimators.py:348` (at `ea53581`), which uses the shape as a start for a later fit
- `lbfgsb` stays: other fits in the module use it

---

## Task 2: Full check and closing

**Status:** done
**Priority:** high
**Complexity:** small
**Source Reference:** `TODO.md`, "`weibull_shape_start` stops at its start value for large shapes"

**Technical Goal:** the branch passes every check once, and the `TODO.md` item moves to `DONE.md`.

**Subtasks:**

- [x] Move the `TODO.md` item "`weibull_shape_start` stops at its start value for large shapes" to the end of `DONE.md`: append its heading and body unchanged, then the line
      `Done: <date> - agents/tasks/weibull-shape-start/tasks-1.md Task 1 (<commit>); approved in agents/reviews/order-r-parity-fixes-from-property-tests-specs.md item 2.`, and remove it from `TODO.md`
- [x] Run the final tests from the worktree and save their whole output for the merge gate. The main checkout is the first entry of
      `git worktree list --porcelain`, `/Users/simonb/src/metafopy`:
      `mkdir -p /Users/simonb/src/metafopy/ralpher-logs/weibull-shape-start/tasks-1 && export PATH=/Users/simonb/src/metafopy/.venv/bin:$PATH PYTHONPATH=$PWD/src && pytest src/tests 2>&1 | tee /Users/simonb/src/metafopy/ralpher-logs/weibull-shape-start/tasks-1/final-tests.log`

**Acceptance Criteria:**

- [x] The item is in `DONE.md` with its body unchanged, and the final run passes

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `pre-commit run --all-files` passes
- [x] `pre-commit run auditme-all --hook-stage manual --all-files` passes
- [x] `mypy` passes
- [x] The final test run above passes
- [x] `git show $(git log --diff-filter=A --format=%H -- agents/tasks/weibull-shape-start/tasks-1.md):TODO.md > <scratch>/todo-before.md` and
      `python3 ~/.claude/skills/todo-review/verify_bodies.py <scratch>/todo-before.md TODO.md DONE.md` exits 0

**Notes:**

- Depends on Task 1
- The full run can outlast a ten-minute command limit; run it in the background with the longest limit the session allows, and wait for its
  exit status before going on

---

## Task 3: Fix the failing `finish_annotext` property test

**Status:** done
**Priority:** high
**Complexity:** small
**Source Reference:** found by the full run of Task 2 (`/Users/simonb/src/metafopy/ralpher-logs/weibull-shape-start/tasks-1/final-tests.log`)

**Technical Goal:** `src/tests/test_plots.py::test_finish_annotext_blanks_missing_rows_and_swaps_symbols` passes. Hypothesis falsifies it with `raw=['NNAA']`, `pi_row=True`: the
last output row is `'NA'`, but the test asserts `'NA' not in` it. Either the code in `src/metafopy` that finishes annotation text leaves a missing-value marker
that R blanks, or the test's property is wrong for text that contains `NA` formed after a replacement. Decide which against R metafor, and fix that one. A change to
metafopy's output needs the operator's approval first (`claude/rules/metafopy.md`).

**Subtasks:**

- [x] Reproduce with `pytest src/tests/test_plots.py -k finish_annotext_blanks`, and run R's equivalent on `NNAA`
- [x] Fix the code or the test, as R decides

**Acceptance Criteria:**

- [x] The test passes and the Weibull branch's other tests are unchanged

**Notes:**

- Not caused by Task 1: the branch changes only `src/metafopy/effects/fivenum_estimators.py` and its tests
- Task 2 stays `in-progress` until this passes and `pytest src/tests` is rerun

- Resolved 2026-11-22: R's `gsub("NA", "", fixed = TRUE)` is a single pass, so `NNAA` leaves `NA` in R too; metafopy matches R and the test property was too strong. Test fixed, no output change. Full run: `pytest src/tests` passed.
