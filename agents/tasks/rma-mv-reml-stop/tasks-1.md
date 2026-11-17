# Tasks: rma_mv REML stop

**Spec:** `agents/specs/rma-mv-reml-stop/spec.md` (version 1.1)
**Created:** 2026-11-17
**Version:** 1

Row 2 of the run order `agents/tasks/order-todo-items-1-10.md`; it waits for no other row. The spec was approved by the operator on 2026-11-17 on the
review page of `agents/reviews/order-todo-items-1-10-specs.md`, and that approval is the confirmation `claude/rules/metafopy.md` asks for of the output
changes the spec lists (its "Measured on 2026-11-17 while writing this spec"). So this list changes metafopy's output within those bounds without a
new review, and builds the spec's chosen option, *capped* (`POLISH_MAX_STEP` 1e-3), only.

Rules every task follows:

- **No output change beyond the spec's measured bounds.** Task 2 compares every fit of `CASES` in `src/tests/test_rma_mv.py` with and without the
  polish against the bounds the spec measured; a change beyond them, or a parity test that fails, stops for the operator. Never widen a tolerance or
  mark a fixture differing to make a test pass.
- **No push.** CI on Linux, macOS and Windows is checked by the operator after the merge (spec, Acceptance); no task pushes, opens a pull request or
  reads CI. Never force-push, never `--no-verify`.
- **R and the audit** use the main checkout's R library (a worktree has none):
  `export METAFOPY_AUDIT_RLIB=/Users/simonb/src/metafopy/.audit/Rlib`, then `LC_ALL=C.UTF-8 R_LIBS=$METAFOPY_AUDIT_RLIB Rscript <script>` or
  `bin/metafor-audit.sh`. Scratch scripts and their outputs stay in the session's scratch directory, outside the repository.
- **The Linux check** uses `bin/linux-tests.sh` (added by `agents/tasks/cross-platform-numerics/tasks-1.md`, Task 2). A task that needs it stops as
  `blocked` if `docker info` does not exit 0, saying "Docker is not running".
- The project has no affected-tests tool, so each task names its test files; the last task runs `pytest src/tests`.
- No version bump (the project is at 0.0.1 and no list bumps it).

**Operator stops:** Task 2, "stops for the operator: an output change beyond the spec's measured bounds, or a parity test that fails"; Task 5,
"stops for the operator: a new audit finding on the `rma_mv` fixtures". Both fire only on what they name; otherwise the list runs to the end.

---

## Task 1: The guarded Newton polish in `mv_opt`

**Status:** done
**Priority:** high
**Complexity:** medium
**Spec Reference:** API; Implementation; Error handling; Settings; Tests (`src/tests/test_mv_opt.py`); Records (the module docstring)

**Technical Goal:** `maximise` finishes every default-optimiser fit with Newton steps on the analytic score, capped and guarded as the spec says, so
the BFGS stop no longer decides how close to the optimum a fit ends.

**Subtasks:**

- [x] Starting point: `grep -rnwE 'polish_score|score_jacobian|POLISH_GRADIENT|POLISH_MAX_STEP|POLISH_STEPS' src/metafopy` finds nothing,
      `src/tests/test_mv_opt.py` does not exist, and `grep -n 'SLACK_GRADIENT = 1e-4' src/metafopy/models/mv_opt.py` finds the constant. Stop as
      `blocked`, naming what differs, if any is not so
- [x] Add to `src/metafopy/models/mv_opt.py` the constants `POLISH_GRADIENT = 1e-10`, `POLISH_MAX_STEP = 1e-3` and `POLISH_STEPS = 8`, each with the
      comment of the spec's API, beside `SLACK_GRADIENT`
- [x] Add `score_jacobian(problem, par, free, reml)` as the spec's API gives it: central differences of `metafopy.models.mv_lik.gradient` in each free
      entry with step `1e-5 * max(1, |par_i|)`, symmetrised (`(H + H.T) / 2`), shape `len(free) x len(free)`
- [x] Add `polish_score(problem, par, free, reml)` returning `(par, -ll)`, with the step of the spec's Implementation: stop when the largest
      `|gradient|` is at most `POLISH_GRADIENT`; Cholesky of the Hessian (catch `numpy.linalg.LinAlgError` here only, and stop); `step = -H^-1 g` by
      the factor; stop if `max|step| > POLISH_MAX_STEP`; evaluate `-ll` and the gradient at `par + step`; stop if `-ll` or the gradient is not finite,
      `-ll` rose by more than `metafopy.core.stationarity.REL_TOL * (1 + |-ll|)`, or the largest gradient entry did not fall; else accept. At most
      `POLISH_STEPS` steps; it raises nothing of its own and returns the best point reached, at least its input. Docstring as the spec's API, with the
      reason the objective test is loose (the spec's paragraph under Implementation)
- [x] In `maximise`, default optimiser only (after the `choice is not None` return), after the L-BFGS-B fallback and its error check, and only when
      `not nonsmooth` and `free.size`: `par, val = polish_score(problem, expand(x), free, reml)` and return `OptResult(par=par, value=-val, ...)` with
      `iterations` still `int(res.nit)`. The polish runs whether BFGS reported success or not; it does not call `log` or `count` (spec, "Choices made,
      not decisions")
- [x] Add `polish_score` and `score_jacobian` (and the constants, if the module lists constants elsewhere) to `__all__`
- [x] Rewrite the module docstring's optimiser paragraph: BFGS is followed by a Newton polish on the score, because the gradient slack accepts BFGS
      stops ("precision loss") that vary by platform and by run, up to 1.1e-06 of the score's scale on CI Windows (measured 2026-11-17); the other
      optimisers of `control['optimizer']` are not polished. Link the spec and Nocedal and Wright (2006), ch. 3 (Newton's method with a step test)
- [x] Write `src/tests/test_mv_opt.py` with the spec's tests: (a) the `kon` model of `property.mv.reml-score.kon` (`dat.konstantopoulos2011`,
      `random='~ 1 | district/school'`, `mods='~ year'`, REML) with `control={'gtol': 1e-4}` ends with the largest log-`sigma2` gradient
      (`mv_lik.gradient` at the fit's parameters) at most `POLISH_GRADIENT`, and `sigma2` within 1e-9 relative of the default fit; (b) `polish_score`
      at the polished optimum of (a) takes no step (returns its input unchanged); (c) on the boundary fit `sim_R_known` of `src/tests/test_rma_mv.py`
      (import its `CASES` builder rather than copy it), `polish_score` from the BFGS stop returns the input unchanged (the cap); (d) `score_jacobian`
      on the `kon` fit agrees with a central difference of `mv_lik.negative_loglik` (second differences) to 1e-5 relative. Reach the `Problem`, `par`
      and `free` the way `maximise`'s caller in `src/metafopy/models/rma_mv.py` builds them, or by wrapping `maximise` in the test
- [x] Confirm test (a) fails with the polish call removed (patch `metafopy.models.mv_opt.polish_score` to return its input, by hand, not committed):
      the spec measured a gradient of 2.5e-05. Record the figure seen and the date in this task's Notes
- [x] Run `pytest src/tests/test_mv_opt.py src/tests/test_rma_mv.py src/tests/correctness/test_properties.py -m "not slow"`. A failure in
      `test_rma_mv.py` is Task 2's stop, not a reason to change this task's tolerances: record it and carry on to Task 2 only if this task's own tests
      pass

**Acceptance Criteria:**

- [x] `polish_score`, `score_jacobian` and the three constants exist with the spec's signatures and values; `maximise`'s signature and `OptResult`
      are unchanged
- [x] The four tests of `src/tests/test_mv_opt.py` pass, and (a) is shown to fail without the polish
- [x] `git diff main -- src/metafopy` touches `src/metafopy/models/mv_opt.py` only

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy` passes
- [x] `pytest src/tests/test_mv_opt.py src/tests/test_rma_mv.py src/tests/correctness/test_properties.py` passes, or fails only in `test_rma_mv.py`
      with the failure recorded for Task 2

**Notes:**

- Key files: `src/metafopy/models/mv_opt.py`, `src/metafopy/models/mv_lik.py` (`Problem`, `gradient`, `negative_loglik`, `numeric_gradient`;
  unchanged), `src/metafopy/core/stationarity.py` (`REL_TOL`; unchanged), `src/tests/test_mv_opt.py` (new)
- The spec's prototype (2026-11-17, not kept) took one Newton step on every `kon` stop, from the default control to `gtol=1e-4` and `maxiter=4`
- `SLACK_GRADIENT` and the L-BFGS-B fallback are unchanged (spec, "Choices made, not decisions")
- 2026-11-17: test (a) without the polish call ends with a largest log-`sigma2` gradient of 2.49e-05 (spec: 2.5e-05), so it fails there; with the polish it is at most 1e-10. `pytest src/tests/test_mv_opt.py src/tests/test_rma_mv.py src/tests/correctness/test_properties.py -m "not slow"`: 381 passed, 12 skipped, none failed

---

## Task 2: R parity and the output changes against the spec's bounds

**Status:** in-progress
**Priority:** high
**Complexity:** medium
**Spec Reference:** Measured on 2026-11-17 ("Every fit of `CASES`"); Tests ("R parity"); Choices made ("When the polish runs, and how far it may move")

**Technical Goal:** every output the polish moves is shown to move within the bounds the operator approved, and every R parity test of `rma_mv` passes
unchanged.

**Subtasks:**

- [ ] Write a scratch script (outside the repository) that fits every entry of `CASES` in `src/tests/test_rma_mv.py` twice, with the polish and with
      `metafopy.models.mv_opt.polish_score` patched to return its input, and prints per fit the largest relative change of `sigma2`, `tau2`, `rho`,
      `gamma2`, `phi`, `beta` and `se` (relative to `max(|value|, 1e-8)`), the change of the log-likelihood, and the time of each fit
- [ ] Compare with the spec's bounds (measured 2026-11-17): largest relative change 7.1e-08 (`ass_mods`, `ass_mods_t`), every other fit at most
      5.7e-08; log-likelihood change at most 2e-14 except `kon_weights` (+7.4e-10); `kon_spexp`, `sim_R_known`, `sim_R_cov0`, `sim_R_none`,
      `sim_R_sparse`, `sim_gdiag`, `sim_s_and_g`, `sim_two_inner_outer` and the fits with nothing to estimate unchanged (identical values); the
      largest fit time 0.10 s (`sim_na_un`). A relative change above 1e-7 (the largest measured, rounded up), a log-likelihood change above the
      figures named, a change in a fit the spec lists as unchanged, or a fit time above 0.5 s goes beyond the bounds
- [ ] Run every test file that names `rma_mv`, `RMAMV` or `mv_opt`, without editing any of them:
      `pytest $(grep -rlE 'rma_mv|RMAMV|mv_opt' src/tests --include='*.py') -m "not slow"` (the spec's run on 2026-11-17 with the prototype: 4098
      passed, 13 skipped, 1 xfailed, none failed). No fixture is regenerated and none is marked differing
- [ ] **Stops for the operator: an output change beyond the spec's measured bounds, or a parity test that fails.** If either happens, stop the task as
      `blocked` and report each fit or test with its figures beside the spec's, without changing a tolerance, a fixture or a `DIFFERS`/`FLAT` entry.
      The list resumes once the operator has answered in an operator session and the answer is committed in this task's Notes
- [ ] Record in this task's Notes, dated: the largest change per fit group, the log-likelihood changes, the largest fit time, and the test counts

**Acceptance Criteria:**

- [ ] Every fit of `CASES` changes within the spec's bounds, and the fits the spec lists as unchanged are identical
- [ ] Every test file that names `rma_mv`, `RMAMV` or `mv_opt` passes, `slow` tests skipped, with no test file changed by this task

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `pytest $(grep -rlE 'rma_mv|RMAMV|mv_opt' src/tests --include='*.py') -m "not slow"` passes

**Notes:**

- Depends on Task 1
- Only this task list file changes (its Notes); the comparison script stays in the scratch directory
- The spec's run of these files took about 13 minutes on the development Mac (2026-11-17); run it in the background
- The uncapped alternative (moving `kon_spexp` `rho` and the `sim_R_*` boundary fits) was not chosen; a fit that moves like it is a change beyond the
  bounds
- 2026-11-17 (blocked, for the operator): per-fit comparison with and without the polish over every entry of `CASES`: largest relative change 7.12e-08
  (`ass_mods`, `ass_mods_t`; bound 1e-7), then `sim_gen` 5.69e-08; the fits the spec lists as unchanged are identical; the largest fit time is 0.15 s
  (`sim_gen`; bound 0.5 s); `kon_weights` log-likelihood +7.44e-10 as measured. **Beyond the named log-likelihood figure (2e-14):** `ass_cs` -3.34e-13
  (log-likelihood -50.5, so 6.6e-15 relative; the sign is a loss, so rounding in evaluating the likelihood, not a move) and `kon_fix_sigma2` -2.13e-14.
  Question for the operator: accept these as rounding of the likelihood evaluation, or treat them as beyond the bounds
- 2026-11-17: `pytest <the 47 files naming rma_mv, RMAMV or mv_opt> -m "not slow"`: 4107 passed, 13 skipped, 1 xfailed, 1 failed:
  `src/tests/test_profile_plots.py::test_pipeline_through_figure_without_axes` (`assert 3 == 2`, an open matplotlib figure left by an earlier file). It
  passes alone and fails the same way, with the same 1116 passed, on the main checkout's code without the polish, so the polish does not cause it;
  filed as Task 7

---

## Task 3: Narrow the REML score property to 1e-10 of its scale

**Status:** done
**Priority:** high
**Complexity:** small
**Spec Reference:** Tests (`src/tests/correctness/cases/properties.toml`)

**Technical Goal:** `property.mv.reml-score.kon` asserts the rounding-size score the polish reaches on every platform, so a stop accepted short of the
optimum fails the test again.

**Subtasks:**

- [x] Starting point: `property.mv.reml-score.kon` in `src/tests/correctness/cases/properties.toml` has `tolerance = 2e-6`. Stop as `blocked`,
      naming the value found, if not
- [x] Set its `tolerance` to `1e-10` and rewrite its comment: relative to the score's scale `tr(P Z Z')`; the Newton polish of `mv_opt` ends the fit at
      a relative score of at most 8.6e-12 on the development Mac from every BFGS stop measured (default, `gtol` 1e-12, 1e-5 and 1e-4, `maxiter` 4),
      times 10 and rounded up (the margin rule of `agents/specs/cross-platform-numerics/spec.md`); measured 2026-11-17,
      `agents/specs/rma-mv-reml-stop/spec.md`
- [x] Rewrite the comment in `mv_score_is_zero` (`src/tests/correctness/test_properties.py`): BFGS's stop is finished by Newton steps on the score, so
      the REML score is a rounding-size fraction of its scale on every platform; it was up to 1.1e-06 of it on CI Windows before (probe run
      44002318304, 2026-11-17); the ML case stays absolute. No code change in the function
- [x] Confirm the case fails at 1e-10 without the polish under an early stop (by hand, not committed): with `polish_score` patched to return its input
      and the case refitted with `control={'gtol': 1e-5}`, the relative score is about 1.3e-07 (spec). Record the figure and the date in this task's
      Notes
- [x] Leave `property.mv.ml-score.kon` at its absolute 1e-6

**Acceptance Criteria:**

- [x] `property.mv.reml-score.kon` has `tolerance = 1e-10` with a dated comment, and passes
- [x] The case is shown to fail without the polish under `gtol=1e-5`

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy` passes
- [x] `pytest src/tests/correctness/test_properties.py src/tests/correctness/test_coverage.py` passes

**Notes:**

- Depends on Task 1
- Key files: `src/tests/correctness/cases/properties.toml`, `src/tests/correctness/test_properties.py`
- 2026-11-17: with `polish_score` disabled (`POLISH_STEPS` patched to 0) and `gtol=1e-5`, `mv_score_is_zero` fails at 1e-10: the REML score of `sigma2[0]` is 1.34e-05 (absolute); with the polish the same refit passes. `pytest src/tests/correctness/test_properties.py src/tests/correctness/test_coverage.py`: 368 passed, 1 skipped

---

## Task 4: The platform probe records the polish

**Status:** done
**Priority:** medium
**Complexity:** small
**Spec Reference:** Tests (`bin/platform-probe.py`); Module location (Records)

**Technical Goal:** the CI probe's `mv-reml-score` group shows whether the polish ran and where it ended, so the operator's CI check after the merge can
confirm the fix on Linux and Windows.

**Subtasks:**

- [x] In `measure_mv_reml_score` of `bin/platform-probe.py`, record the polish beside the `scipy.optimize.minimize` calls: wrap
      `metafopy.models.mv_opt.polish_score` during the refit the way `captured_minimize` wraps `minimize` (restoring it afterwards), and put
      `polish.steps` (the Newton steps taken, counted as the calls of `metafopy.models.mv_opt.score_jacobian` whose step was accepted, or, if that
      cannot be told from outside, `polish.hessians`, the calls of `score_jacobian`, with a `record.say` line saying it is an upper bound),
      `polish.gradient_before` and `polish.gradient_after` (the largest `|mv_lik.gradient|` at the input and the output of `polish_score`)
- [x] Keep the existing keys unchanged, so the earlier probe outputs in `agents/specs/cross-platform-numerics/measurements.md` stay comparable
- [x] Run `python bin/platform-probe.py --group mv-reml-score --out <scratch>/probe.json` and check the new keys hold finite values with
      `polish.gradient_after` at most `POLISH_GRADIENT`
- [x] Extend `src/tests/test_platform_probe.py` so the `mv-reml-score` group's values include the new keys

**Acceptance Criteria:**

- [x] The probe's `mv-reml-score` group records the polish's steps (or Hessians) and its gradient before and after
- [x] `src/tests/test_platform_probe.py` passes and checks the new keys

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy` passes
- [x] `pytest src/tests/test_platform_probe.py` passes

**Notes:**

- Depends on Task 1
- Key files: `bin/platform-probe.py`, `src/tests/test_platform_probe.py`
- `polish_score` returns no step count (spec, API), so the count is taken from outside; do not change `polish_score`'s return type for the probe
- 2026-11-17: the probe records `polish.hessians` (calls of `score_jacobian`, an upper bound on the steps, since acceptance cannot be told from outside), `polish.gradient_before` and `polish.gradient_after`. On the development Mac: 1 Hessian, gradient 1.3e-10 before, 2.5e-11 after (at most `POLISH_GRADIENT`). `pytest src/tests/test_platform_probe.py`: 6 passed

---

## Task 5: The records, the Linux container and the audit

**Status:** todo
**Priority:** high
**Complexity:** medium
**Spec Reference:** Records; Acceptance (the Linux container check); Tests ("R parity", the audit)

**Technical Goal:** the user guide's note says how the `rma_mv` stop now ends, the arm64 Linux container passes the spec's test files, and the audit
of the `rma_mv` fixtures shows nothing new.

**Subtasks:**

- [ ] In `docs/differences-notes.md`, "Optimisers and flat likelihoods", rewrite the `rma_mv` BFGS line: BFGS's stop is now finished by Newton steps
      on the analytic score, capped at 1e-3 in the transformed parameters so that ridge and boundary fits stay where BFGS stopped; the test
      `property.mv.reml-score.kon` asserts 1e-10 of the scale `tr(P Z Z')`; R metafor 5.2-1's own estimates (`nlminb`) have a relative score of
      8e-09 of it (measured 2026-11-17). Search `docs/` for any other copy of the "2e-6 of the scale" or "1.1e-6" figures and update each
- [ ] Run `pytest src/tests/test_docs_examples.py`
- [ ] Check `docker info` exits 0; stop as `blocked` with "Docker is not running" if not. Run
      `bin/linux-tests.sh -- src/tests/correctness/test_properties.py src/tests/test_rma_mv.py src/tests/test_mv_opt.py` (arm64) in the background.
      A failure that the Mac run of the same files does not show stops for the operator (the same stop as Task 2), with the failing test and figures
- [ ] Run the audit slice in the background:
      `METAFOPY_AUDIT_RLIB=/Users/simonb/src/metafopy/.audit/Rlib bin/metafor-audit.sh --only '*/mv_*'` (the `rma_mv` fixtures `mv_gen`, `mv_ks`,
      `mv_ks_mods`, `mv_slab`, `mv_sp`, `mv_wide`). If `check-r` reports that the R library is missing or differs from
      `src/audit/data/r-versions.json`, stop as `blocked` and say so; never pass `--allow-r-drift`
- [ ] Compare its findings with the same case ids in the newest whole-audit run under `/Users/simonb/src/metafopy/.audit/runs/` (not a `recall-*`
      directory). **Stops for the operator: a new audit finding on the `rma_mv` fixtures.** A finding that run does not hold stops the task as
      `blocked`, naming it with its detail; do not add a `known.toml` entry for it
- [ ] Record in this task's Notes, dated: the container's test counts, the audit run directory and the baseline run compared with

**Acceptance Criteria:**

- [ ] `docs/differences-notes.md` states the polish, the 1e-10 test and R's own relative score, with the date
- [ ] The arm64 container passes the three test files
- [ ] The `mv_*` audit slice reports no finding the newest whole-audit run does not hold

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `pytest src/tests/test_docs_examples.py src/tests/test_mv_opt.py` passes
- [ ] `bin/linux-tests.sh -- src/tests/correctness/test_properties.py src/tests/test_rma_mv.py src/tests/test_mv_opt.py` exits 0

**Notes:**

- Depends on Tasks 1 to 4
- The containers' unpolished stop was already at 1e-11 of the scale (`agents/specs/cross-platform-numerics/measurements.md`, "The rma_mv REML score"),
  so the container run shows no regression on Linux, not the CI fix; CI is the operator's check after the merge
- Key files: `docs/differences-notes.md`, `bin/linux-tests.sh` (unchanged), `bin/metafor-audit.sh` (unchanged)

---

## Task 6: The full test run, the records and closing

**Status:** todo
**Priority:** high
**Complexity:** small
**Spec Reference:** Acceptance; Records (`TODO.md`); the spec as a whole

**Technical Goal:** the branch is checked once locally, the spec is marked implemented and the item is closed.

**Subtasks:**

- [ ] Set the spec's `**Status:**` in `agents/specs/rma-mv-reml-stop/spec.md` to `implemented`, and its row in `agents/specs/index.md` from
      "Approved (2026-11-17)" to "Implemented (2026-MM-DD)" with the date of the commit
- [ ] Move the item "The rma_mv REML fit stops short of the optimum on Linux and Windows" from `TODO.md` to `DONE.md`: append its heading and body
      unchanged at the end of `DONE.md`, then a line
      `Done: 2026-MM-DD, agents/tasks/rma-mv-reml-stop/tasks-1.md; a capped Newton polish on the score follows the BFGS stop of rma_mv, property.mv.reml-score.kon asserts 1e-10 of its scale; CI on Linux, macOS and Windows is checked by the operator after the merge`
      with the date of the commit. (The spec's Records move it once CI is green after the merge; the run order closes it here, and a CI failure is
      filed anew with the probe's figures, as the spec's Acceptance says)
- [ ] Run the final tests from the worktree and save their whole output for the merge gate (the main checkout is the first entry of
      `git worktree list --porcelain`, `/Users/simonb/src/metafopy`):
      `mkdir -p /Users/simonb/src/metafopy/ralpher-logs/rma-mv-reml-stop/tasks-1 && source bin/activate.sh && pytest src/tests 2>&1 | tee /Users/simonb/src/metafopy/ralpher-logs/rma-mv-reml-stop/tasks-1/final-tests.log`

**Acceptance Criteria:**

- [ ] The spec is `implemented`; the item is in `DONE.md` with its body unchanged and the `Done:` line, and no longer in `TODO.md`
- [ ] `git diff main -- src/metafopy` changes `src/metafopy/models/mv_opt.py` only

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `pre-commit run --all-files` passes
- [ ] `pre-commit run auditme-all --hook-stage manual --all-files` passes
- [ ] `mypy` passes
- [ ] The final test run above passes, `slow` tests included, and its log is saved at
      `/Users/simonb/src/metafopy/ralpher-logs/rma-mv-reml-stop/tasks-1/final-tests.log`
- [ ] `git show $(git log --diff-filter=A --format=%H -- agents/tasks/rma-mv-reml-stop/tasks-1.md):TODO.md > <scratch>/todo-before.md` (the `TODO.md`
      at the list's start) and `python3 ~/.claude/skills/todo-review/verify_bodies.py <scratch>/todo-before.md TODO.md DONE.md` exits 0
      (`<scratch>` is the session's scratch directory, outside the repository)

**Notes:**

- Depends on Task 5
- The full `pytest src/tests` run is the run order's required final check
- No version bump

---

---

## Task 7: Close the figure that leaks into `test_pipeline_through_figure_without_axes`

**Status:** todo
**Priority:** low
**Complexity:** small
**Spec Reference:** none (found by Task 2)

**Technical Goal:** `src/tests/test_profile_plots.py::test_pipeline_through_figure_without_axes` passes whatever test files ran before it.

**Subtasks:**

- [ ] Find the earlier test file that leaves a matplotlib figure open (run the files named in Task 2's Notes in order up to `test_profile_plots.py`; the
      failure shows with `src/tests/test_audit*.py`, `test_fits.py`, `test_plots.py` and `test_mv_diagnostics.py` before it) and close its figures, or
      make the failing test count the axes of its own figure
- [ ] Run the same selection and see it pass

**Acceptance Criteria:**

- [ ] The test passes in the combined run and alone

**Notes:**

- Fails identically without the polish, so it is not caused by this spec
