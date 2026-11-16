# Tasks: SJIT convergence on the scaled scale-invariance data

**Source:** TODO.md, "Whether R metafor's SJIT converges on the scaled scale-invariance data was never checked"
**Created:** 2026-11-15
**Version:** 1

The item comes from Task 13 of `agents/tasks/correctness-location-scale/tasks-1.md`, which found that `rma_uni(method='SJIT')` raises
`EstimationError: Iterative SJ estimator did not converge.` on the scaled copy (scale 2.0) of the data that made
`src/tests/correctness/test_properties.py::test_scale_invariance[SJIT]` fail, and could not run R metafor to compare. The property test now rejects
a scaled copy that does not converge (`fit_or_reject`).

Probed on 2026-11-15 (metafopy at `89ec047`, R metafor 5.2-1 from `/Users/simonb/src/metafopy/.audit/Rlib`), with the property test's `CONTROL`
(`maxiter` 1000, `tau2_max` 1e6, `threshold` 1e-13, `tol` 1e-13): on the unscaled data both give `tau2` 0.0055136721559805 and intercept
-0.31447636711455818; on the scaled copy both stop with "Iterative SJ estimator did not converge."; R with `maxiter = 100000` converges to `tau2`
0.022054688609927535 (four times the unscaled value). So the expected path is Task 1 recording that R agrees and Task 2 closing as not needed; Task 2
keeps the fix branch in case Task 1's run disagrees with the probe.

The SJIT loop is in `src/metafopy/models/tau2.py` (the `if method == 'SJIT':` branch, called from `src/metafopy/models/rma_uni.py`). The project has
no affected-tests tool, so each task names its test files.

---

## Task 1: Run the scaled data through R metafor and metafopy

**Status:** done
**Priority:** high
**Complexity:** small
**Source Reference:** the item's fix direction (run the scaled data through R metafor 5.2-1 with the same control)

**Technical Goal:** whether R metafor's `SJIT` converges on the scaled copy is known and recorded in this task's Notes, with the numbers both
programs give.

**Subtasks:**

- [x] Write an R script in the session's scratch directory (outside the repository) that loads metafor, prints `packageVersion("metafor")`, and for
      `yi = c(0, -0.5625, 0, 0.875, -0.7734375, 0)`, `vi = c(0.5, 0.625, 0.5, 0.5, 0.125, 1)` and scale `s` in `1` and `2` runs
      `rma(yi * s, vi * s^2, method = "SJIT", control = list(maxiter = 1000, tau2.max = 1e6, threshold = 1e-13, tol = 1e-13))` inside `tryCatch`,
      printing `tau2` and `beta` with `sprintf("%.17g", ...)` or the error message
- [x] Run it with the main checkout's R library (the main checkout is the first entry of `git worktree list --porcelain`):
      `export METAFOPY_AUDIT_RLIB=/Users/simonb/src/metafopy/.audit/Rlib; LC_ALL=C.UTF-8 R_LIBS=$METAFOPY_AUDIT_RLIB Rscript <script>`.
      Stop as `blocked` if metafor does not load or its version is not 5.2.1
- [x] Run the same fits through metafopy (`source bin/activate.sh`, then `metafopy.rma_uni(yi=yi * s, vi=vi * s**2, method='SJIT', control=CONTROL)`
      with `CONTROL` copied from `src/tests/correctness/test_properties.py`), printing `repr(tau2)` and `beta` or the exception
- [x] Record in this task's Notes, dated: both programs' results for each scale, and the verdict - "R does not converge either" or "R converges and
      metafopy does not". If the unscaled results differ beyond 1e-12 relative, stop as `blocked`: that is a different difference from this item's

**Acceptance Criteria:**

- [x] This task's Notes record R's and metafopy's result on the unscaled and the scaled data, and one of the two verdicts

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)

**Notes:**

- Only this task list file changes; no code or test changes
- The R and Python scripts stay in the scratch directory and are not committed
- Result, 2026-11-16 (R metafor 5.2-1 from `/Users/simonb/src/metafopy/.audit/Rlib`; metafopy at `0038b6e`; control `maxiter` 1000, `tau2_max`
  1e6, `threshold` 1e-13, `tol` 1e-13):
  - Unscaled (scale 1): R `tau2` 0.0055136721559805452, `beta` -0.31447636711455818; metafopy `tau2` 0.0055136721559805426, `beta`
    -0.3144763671145582 (relative difference about 5e-16, within 1e-12)
  - Scaled (scale 2): R stops with "Iterative SJ estimator did not converge."; metafopy raises `EstimationError: Iterative SJ estimator did not
    converge.`
  - Verdict: R does not converge either

---

## Task 2: Match R where metafopy alone fails to converge, or record that R fails too

**Status:** todo
**Priority:** high
**Complexity:** small
**Source Reference:** the item's fix direction (fix the iteration if R converges; otherwise close with the result)

**Technical Goal:** metafopy's `SJIT` behaves as R metafor 5.2-1 does on the scaled copy, and `test_scale_invariance` says why that copy is
rejected or no longer is.

**Subtasks:**

- [ ] Read Task 1's verdict. If it is missing, stop as `blocked`
- [ ] **If R does not converge either:** add a comment above the `fit_or_reject` call for the scaled copy in `test_scale_invariance`
      (`src/tests/correctness/test_properties.py`): R metafor 5.2-1 also stops with "Iterative SJ estimator did not converge." on the item's scaled
      data at `maxiter` 1000, checked on the date of this task, so rejecting such a copy matches R. Change no code, and mark the other branch's subtasks
      ("In that branch") not applicable
- [ ] **If R converges and metafopy does not:** this is a port difference, which metafopy fixes to match R. Compare the `SJIT` branch of
      `src/metafopy/models/tau2.py` line by line with `rma.uni`'s `SJIT` loop in metafor 5.2-1 (`R/rma.uni.r` in the installed source under
      `$METAFOPY_AUDIT_RLIB/metafor`, or the CRAN 5.2-1 source), find the line where they differ (starting value, update, stopping rule or the
      `maxiter` check), and change metafopy's to R's. Fix it in `src/metafopy/models/tau2.py` (called from `src/metafopy/models/rma_uni.py`)
- [ ] In that branch, before committing, check the change does not alter output beyond matching R: `pytest src/tests/test_rma_uni_extras.py
      src/tests/test_rma_uni_parity.py src/tests/correctness` must pass unchanged. If any existing R parity value or correctness case changes, or the
      fix would change metafopy's `SJIT` result on data where it already converged, stop as `blocked` and report what changed: that needs the user
- [ ] In that branch, add a regression test to `src/tests/test_rma_uni_extras.py`: the scaled copy with the property test's `CONTROL` gives R's `tau2`
      and `beta` (from Task 1's Notes, absolute tolerance 1e-10), with a comment naming this list and the R version
- [ ] In that branch, if the scaled copy now converges, leave `fit_or_reject` in place (other data can still fail to converge) and update the docstring
      of `test_scale_invariance` so it no longer implies metafopy fails where R converges

**Acceptance Criteria:**

- [ ] R does not converge: `test_scale_invariance` carries the comment recording R's result, and no file under `src/metafopy/` changes
- [ ] R converges: metafopy gives R's `tau2` and `beta` on the scaled copy, the regression test passes, and no existing parity value changes

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy` passes
- [ ] `pytest src/tests/correctness/test_properties.py src/tests/test_rma_uni_extras.py` passes

**Notes:**

- Depends on Task 1
- Key files: `src/tests/correctness/test_properties.py` (`test_scale_invariance`, `fit_or_reject`, `CONTROL`), `src/metafopy/models/tau2.py`,
  `src/tests/test_rma_uni_extras.py`
- The 2026-11-15 probe in the header expects the first branch

---

## Task 3: The full test run and closing the item

**Status:** todo
**Priority:** high
**Complexity:** small
**Source Reference:** the item as a whole

**Technical Goal:** the whole branch is checked once, and the item moves to `DONE.md` with R's result.

**Subtasks:**

- [ ] Move the item from `TODO.md` to `DONE.md`: append its heading and body unchanged at the end of `DONE.md`, then a line
      `Done: 2026-MM-DD, agents/tasks/sjit-scaled-convergence/tasks-1.md; <R result>` with the date of the commit, where `<R result>` is
      "R metafor 5.2-1 does not converge on the scaled data either (maxiter 1000); test_scale_invariance keeps rejecting it" or "R metafor 5.2-1
      converges (tau2 <value>); metafopy's SJIT loop fixed to match", as Task 1 found
- [ ] Run the final tests from the worktree and save their whole output for the merge gate (the main checkout is the first entry of
      `git worktree list --porcelain`, `/Users/simonb/src/metafopy`):
      `mkdir -p /Users/simonb/src/metafopy/ralpher-logs/sjit-scaled-convergence/tasks-1 && source bin/activate.sh && pytest src/tests 2>&1 | tee /Users/simonb/src/metafopy/ralpher-logs/sjit-scaled-convergence/tasks-1/final-tests.log`

**Acceptance Criteria:**

- [ ] The item is in `DONE.md` with its body unchanged and a `Done:` line naming this list and R's result, and no longer in `TODO.md`

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `pre-commit run --all-files` passes
- [ ] `pre-commit run auditme-all --hook-stage manual --all-files` passes
- [ ] `mypy` passes
- [ ] The final test run above passes
- [ ] `git show $(git log --diff-filter=A --format=%H -- agents/tasks/sjit-scaled-convergence/tasks-1.md):TODO.md > <scratch>/todo-before.md` and
      `python3 ~/.claude/skills/todo-review/verify_bodies.py <scratch>/todo-before.md TODO.md DONE.md` exits 0 (`<scratch>` is the session's scratch
      directory, outside the repository)

**Notes:**

- Depends on Task 2
- No version bump: the expected branch changes no library output; if Task 2's fix branch ran, a bump is still not needed unless the run order asks for
  one against main
- The full `pytest src/tests` run is the run order's required final check (its checkpoint command)

---
