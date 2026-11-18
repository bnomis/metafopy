# Tasks: Profile starts of location-scale models that reach the highest branch

**Spec:** `agents/specs/ls-profile-starts/spec.md` (version 1.1), with `agents/specs/ls-profile-starts/measurements.md`
**Created:** 2026-11-18
**Version:** 1

The spec adds two starts beyond the far end of the line of starts of `profile_starts` in `src/metafopy/methods/ls_profile.py`
(`LINE_WEIGHTS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)`), so that the refits of `profile_fit` reach the branch of highest log-likelihood of the profile of a
scale coefficient of a location-scale model. It closes the `TODO.md` item "metafopy's profile of the ls scale coefficient leaves the highest branch
before the crossing". The operator approved the spec on 2026-11-18 on the review page of `agents/reviews/order-todo-items-1-5-specs.md`, with every
output change its "Every output that moves" lists (the spec's Implementation section and the table of measurements.md of that name). That approval is
the confirmation `claude/rules/metafopy.md` asks for, so this list changes metafopy's `confint` and `profile` output by exactly those moves and no more.

**Operator stops:** none expected. Tasks 1 and 3 name one conditional stop, "stops for the operator: an output moves beyond the spec's list", which fires
only if a test or case the spec did not list as moving changes outcome or value with the new starts, or a listed output moves to a value other than the
spec's.

<!-- doc-counts: accept-start measured on 2026-11-18 - a dated record, not a claim about the present -->
Checked against the code on 2026-11-18 (metafopy at `552f50e`): `LINE_WEIGHTS = (0.25, 0.5, 0.75, 1.0)` in `src/metafopy/methods/ls_profile.py`;
the case `ls.confint.scale.ML.bcg` in `src/tests/correctness/cases/location_scale.toml` has `status = "mismatch"` and a `reason`; the spec measured,
at `dcfb11c`, that the tests of measurements.md's "Tests and run time" give one failure with the new starts, `test_derived[ls.confint.scale.ML.bcg]`
as `XPASS(strict)`, which removing the case's `status` and `reason` resolves.
<!-- doc-counts: accept-end -->

The project has no affected-tests tool, so each task names its test files, and the last task runs `pytest src/tests` as the run order requires.
Every R call sets `METAFOPY_AUDIT_RLIB=/Users/simonb/src/metafopy/.audit/Rlib` (`R_LIBS` to the same path for `Rscript`): a worktree has no
`.audit/Rlib` of its own. Scratch scripts stay outside the repository. No version bump (`pyproject.toml` version `0.0.1`).

---

## Task 1: Regression tests first, then the two new starts

**Status:** done
**Priority:** high
**Complexity:** small
**Spec Reference:** Module location; API; Error handling; Implementation, items 1 to 3; Choices made ("Which starts to add", "The identity-link moves
are part of this change"); Acceptance (the case passes, the two tests fail against the old weights)

**User Story:** As a user of `confint` and `profile` on a location-scale fit, I want the profile of a scale coefficient to stay on the branch of
highest log-likelihood, so that a profile-likelihood bound is the crossing of the cut-off, not a jump of the refit onto a `tau2 = 0` plateau or a
`<` bound at a point the profile does not reach.

**Subtasks:**

- [x] Re-check the starting point, narrowly: `grep -nxF 'LINE_WEIGHTS = (0.25, 0.5, 0.75, 1.0)' src/metafopy/methods/ls_profile.py` finds one line;
      `grep -nwE 'test_profile_keeps_the_continuous_branch_of_alpha2_on_bcg|test_confint_identity_lower_bound_is_a_crossing' src/tests/test_rma_ls.py`
      finds nothing; and the `[[case]]` with `id = "ls.confint.scale.ML.bcg"` in `src/tests/correctness/cases/location_scale.toml` has a
      `status = "mismatch"` line. Stop as `blocked`, naming what differs, if any is not so
- [x] In `src/tests/test_rma_ls.py`, after `test_profile_refit_keeps_the_highest_likelihood_on_bcg`, add
      `test_profile_keeps_the_continuous_branch_of_alpha2_on_bcg`: on `res = ls(bcg(), '~ ablat', '~ ablat', method='ML')`,
      `2 * (res.logLik() - metafopy.methods.ls_profile.profile_fit(res, 1, a)[0])` is 3.789459, 3.813933 and 3.838488 at a = 1.36, 1.37 and 1.38
      (`abs=1e-5`; assert each fit is not `None`), and `metafopy.confint(res, alpha=1, tol=1e-12)` gives `ci.ub` 1.3812075 (`abs=1e-6`) of its `random`
      table with `ub_sign == ''`. Its docstring says what it guards: every start of the line ends on the `tau2 = 0` plateau (3.868641) from a = 1.36,
      while the continuous branch, whose optimum has `alpha.1` near -74, stays below the cut-off until 1.3812
- [x] Add `test_confint_identity_lower_bound_is_a_crossing`: on `res = result('bcg_year_identity')` (`FITS['bcg_year_identity']`),
      `metafopy.confint(res, alpha=1, tol=1e-12)` gives `ci.lb` -0.1356308 (`abs=1e-6`) of its `random` table with `lb_sign == ''`. Its docstring
      says the brute-force profile (a one-dimensional search over the feasible interval) crosses the cut-off there, and that before the new starts
      every line start left a `tau2_i` negative and was dropped, so the bound was a `<` at -0.0820267
- [x] Run the two new tests against the unchanged `LINE_WEIGHTS` (`pytest src/tests/test_rma_ls.py -k "continuous_branch_of_alpha2 or
      identity_lower_bound_is_a_crossing"`) and record in this task's Notes that each fails, with the values it got (the spec expects the plateau
      3.868641 at 1.37 and 1.38 and `ci.ub` 1.3598266; `ci.lb` -0.0820267 with `lb_sign == '<'`). A test that already passes stops the task as
      `blocked`, naming it
- [x] In `src/metafopy/methods/ls_profile.py`, set `LINE_WEIGHTS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)` and replace the docstring of `profile_starts`
      with the spec's API text word for word. Leave the loop `starts.extend(own + weight * (line - own) for weight in LINE_WEIGHTS)`, the
      identity-link filter and `profile_fit` unchanged: the new starts come last, so a tie keeps the fit found now
- [x] In `src/tests/correctness/cases/location_scale.toml`, remove the `status` and `reason` lines of `ls.confint.scale.ML.bcg`, so it runs as a
      passing case
- [x] Run the Definition of Done's test files. Both new tests pass, and so does `ls.confint.scale.ML.bcg`.
      **Stops for the operator: an output moves beyond the spec's list.** Only if a test or correctness case other than the two new tests and
      `ls.confint.scale.ML.bcg` changes outcome, or a new test fails with a value other than the spec's: do not edit the test, case or fixture; stop
      the task as `blocked` and report each test id with its old and new value. The spec lists none, so a hit is a change to metafopy's output it did
      not approve. Otherwise there is no stop

**Acceptance Criteria:**

- [x] `LINE_WEIGHTS` is `(0.25, 0.5, 0.75, 1.0, 1.5, 2.0)` and the docstring of `profile_starts` is the spec's API text; no signature changes
- [x] Both new tests pass, and this task's Notes record each failing against `LINE_WEIGHTS = (0.25, 0.5, 0.75, 1.0)` with the values it got
- [x] `ls.confint.scale.ML.bcg` has no `status` and passes: metafopy's upper bound of `alpha.2` is within `LS_PROFILE`
      (`src/tests/correctness/test_derived.py`) of the reference's 1.38120753377
- [x] Every other test of the Definition of Done's files passes unchanged, with no fixture edited

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `ruff format --check src/metafopy/methods/ls_profile.py src/tests/test_rma_ls.py` and `ruff check` on the same files pass
- [x] `mypy` passes
- [x] `pytest src/tests/test_rma_ls.py src/tests/correctness/test_derived.py src/tests/correctness/test_formulas_ls.py src/tests/correctness/test_properties.py`
      passes

**Notes:**

- Run against the unchanged `LINE_WEIGHTS = (0.25, 0.5, 0.75, 1.0)` (2026-11-18): `test_profile_keeps_the_continuous_branch_of_alpha2_on_bcg`
  failed with 3.868641238 (the plateau) where 3.789459 was expected at a = 1.36; `test_confint_identity_lower_bound_is_a_crossing` failed with
  `ci.lb` -0.0820267 where -0.1356308 was expected. With the new starts both pass, and so do the Definition of Done's test files (820 passed, 1 skipped,
  no XPASS)
- The worktree has no `.venv`, so tests ran with the main checkout's `.venv/bin` and `PYTHONPATH` set to the worktree's `src`

- Key files: `src/metafopy/methods/ls_profile.py` (`LINE_WEIGHTS`, `profile_starts`, `profile_fit`), `src/metafopy/methods/ls_confint.py`
  (`lb_sign`, `ub_sign`; unchanged), `src/tests/test_rma_ls.py` (`ls`, `bcg`, `result`, `FITS`), `src/tests/correctness/cases/location_scale.toml`,
  `src/tests/correctness/test_derived.py` (`ls_confint_scale`, `LS_PROFILE`)
- `src/metafopy/methods/ls_profile.py` has no test file of its own; the tests that import it are `src/tests/test_rma_ls.py`,
  `src/tests/correctness/test_derived.py` and `src/tests/correctness/test_properties.py`, and `test_formulas_ls.py` runs the reference beside it
- The other outputs the spec lists as moving (the identity-link bounds of `ls.fit.ML.identity.bcg`, `ls.fit.REML.identity.bcg` and `bd_identity_ni`,
  the two `published.vl2022.*` profile points, the optimiser noise on `bd_3_unweighted` and `bd_3_weights`) are pinned by no test, so no test is
  expected to change for them (measurements.md, "Tests and run time")
- The spec measured `pytest src/tests/test_rma_ls.py` at 34.5 s before and about 38.8 s after (2026-11-18)

---

## Task 2: Record the change in the user guide and the changelog

**Status:** done
**Priority:** high
**Complexity:** small
**Spec Reference:** Implementation, item 4; "Every output that moves"; Motivation, "R"

**Technical Goal:** `docs/differences.md` says what metafopy's upper bound of `alpha.2` is on the ML `dat.bcg` fit beside R's, and `CHANGELOG.md`
names the new starts and every output that moves, so a user comparing with R or with an earlier metafopy knows why the numbers differ.

**Subtasks:**

- [x] Check R's bound on the day: write an R script in the scratch directory that loads metafor, prints `packageVersion("metafor")`, fits
      `rma(yi, vi, mods = ~ ablat, scale = ~ ablat, data = escalc(measure = "RR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg),
      method = "ML")` and prints `confint(res, alpha = 2)`. Run it with
      `export METAFOPY_AUDIT_RLIB=/Users/simonb/src/metafopy/.audit/Rlib; LC_ALL=C.UTF-8 R_LIBS=$METAFOPY_AUDIT_RLIB Rscript <script>`. The spec
      expects the upper bound `>0.4245`; a different value stops the task as `blocked`, naming both. Stop as `blocked` too if metafor does not load
      or its version is not 5.2.1
- [x] In `docs/differences.md`, section "Profile-likelihood bounds of the scale coefficients", add after the first difference (the refit from
      metafor's default start) a sentence that on the same fit metafopy's upper bound of `alpha.2` is 1.3812, where R prints `>0.4245`: metafopy also
      tries starts beyond the line from the estimate, because the profile optimum there has `alpha.1` near -74, past every start of the line, while R's
      refits from its default start sit on the `tau2 = 0` plateau at every value. Keep the section's other text and its list of correctness cases
- [x] In `CHANGELOG.md` under "Unreleased (1.0 candidate)", add a behaviour-change entry at the top, in the style of the others: the profile refits of
      a location-scale fit (`confint(res, alpha=j)` and `profile()`) also start one and a half and twice the way along the line start, so they keep
      the branch of highest log-likelihood. Name the outputs that move, from the spec's "Every output that moves": the ML `dat.bcg` log-link upper
      bound of `alpha.2`, 1.3598 (a jump) to 1.3812; the identity-link lower bounds that were a `<` or NaN and are now crossings (ML `dat.bcg`
      `alpha.2` -0.0082 to -0.0136, REML `dat.bcg` `alpha.1` 0 to -0.2686 and `alpha.2` NaN to -0.0288, `dat.bangertdrowns2004` with
      `scale = ~ meta + imag` `alpha.2` 0 to -0.0049), with the profile values that were NaN now fitted; and two log-link profile points of the
      Viechtbauer and Lopez-Lopez (2022) fits that rise to a higher refit, their bounds unchanged
- [x] Run `pytest src/tests/test_docs_examples.py`

**Acceptance Criteria:**

- [x] `docs/differences.md` gives metafopy's 1.3812 beside R's `>0.4245` (checked in R on the day) in the section named above
- [x] `CHANGELOG.md` has the behaviour-change entry naming the starts and every output of the spec's list, with no output the spec does not list
- [x] `pytest src/tests/test_docs_examples.py` passes

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `pytest src/tests/test_docs_examples.py` passes

**Notes:**

- Depends on Task 1
- Prerequisite outside this repository: R 4.6.2 with metafor 5.2-1 in `/Users/simonb/src/metafopy/.audit/Rlib`, the main checkout's R library
- Key files: `docs/differences.md`, `CHANGELOG.md`
- The spec's figures (measurements.md, "Every output that moves", 2026-11-18 at `dcfb11c`) are the ones to quote; round as the table does

---

## Task 3: The measured tests, the full test run and closing the spec

**Status:** done
**Priority:** high
**Complexity:** small
**Spec Reference:** Implementation, item 5; Not in scope; Acceptance; the spec as a whole

**Technical Goal:** the tests the spec measured pass with the change, the reference's identity-link gap the spec left out of scope is filed, the
whole branch is checked once, the spec is marked implemented and its item moves to `DONE.md`.

**Subtasks:**

- [x] Run the tests of measurements.md's "Tests and run time":
      `pytest src/tests/test_rma_ls.py src/tests/correctness/test_derived.py src/tests/correctness/test_formulas_ls.py src/tests/correctness/test_properties.py src/tests/correctness/test_published.py src/tests/test_audit5_numerics.py src/tests/test_audit8_args.py src/tests/test_profile_parallel.py src/tests/test_profile_plots.py src/tests/test_audit13_fixes.py src/tests/test_audit3_misc.py src/tests/test_audit5_plots.py src/tests/test_audit5_printing.py src/tests/test_audit8_crashes.py src/tests/test_audit8_fields.py src/tests/test_audit8_values_a.py src/tests/test_emmeans.py`
      (the spec measured about 839 s on 2026-11-18). The expected result is every test passing, `ls.confint.scale.ML.bcg` included, and no R parity
      fixture marked as differing.
      **Stops for the operator: an output moves beyond the spec's list.** Only if a test fails or changes outcome there: do not edit it; stop the task
      as `blocked` and report each test id with its old and new value. Otherwise there is no stop
- [x] File the spec's "Not in scope" as a new `TODO.md` item at the end of `TODO.md`, in the shape of the other items (a dated finding inside a
      `<!-- doc-counts: accept-start found on 2026-11-18 - a dated record, not a claim about the present -->` ... `<!-- doc-counts: accept-end -->`
      block, then `Cause:` and `Fix direction:` paragraphs; no `Spec:` or `tasks:` line), titled "The reference ls_profile_ll cannot fit the identity
      link at some feasible values". Its finding: found on 2026-11-18 by `agents/specs/ls-profile-starts/spec.md` ("Not in scope") and this list,
      `agents/tasks/ls-profile-starts/tasks-1.md`; `formulas.ls_profile_ll` in `src/tests/correctness/formulas.py` gives -1e10 at `alpha.2 = -0.00615`
      of the ML identity-link `dat.bcg` fit (`ls.fit.ML.identity.bcg` in `src/tests/correctness/cases/location_scale.toml`), where the brute-force
      profile is -8.045852; no `ls.confint.scale.*` case uses the identity link, so no case is affected. Its Cause: the SLSQP starts of
      `ls_profile_ll` (`start[free]`, the adjusted start and the line between them) are infeasible there (`Z alpha < 0`), so no refit runs. Its Fix
      direction: give `ls_profile_ll` a feasible start for the identity link (for example the starts of `profile_starts`, filtered as it filters them),
      then add an identity-link `ls.confint.scale.*` case checked against the brute-force profile
- [x] Set the spec's `**Status:**` in `agents/specs/ls-profile-starts/spec.md` to `implemented`
- [x] Move the item "metafopy's profile of the ls scale coefficient leaves the highest branch before the crossing" from `TODO.md` to `DONE.md`: append
      its heading and body unchanged at the end of `DONE.md`, then a line
      `Done: 2026-MM-DD, agents/tasks/ls-profile-starts/tasks-1.md; LINE_WEIGHTS gains 1.5 and 2.0, the upper bound of alpha.2 is the crossing 1.3812 and ls.confint.scale.ML.bcg passes (agents/specs/ls-profile-starts/spec.md)`
      with the date of the commit
- [x] Run the final tests from the worktree and save their whole output for the merge gate, in the main checkout (the first entry of
      `git worktree list --porcelain`, here `/Users/simonb/src/metafopy`):
      `mkdir -p /Users/simonb/src/metafopy/ralpher-logs/ls-profile-starts/tasks-1 && source bin/activate.sh && pytest src/tests 2>&1 | tee /Users/simonb/src/metafopy/ralpher-logs/ls-profile-starts/tasks-1/final-tests.log`

**Acceptance Criteria:**

- [x] The tests of measurements.md's "Tests and run time" pass, `ls.confint.scale.ML.bcg` included
- [x] `TODO.md` ends with the new item on the reference's identity-link starts; it is the only item added
- [x] The spec's status is `implemented`
- [x] The item is in `DONE.md` with its body unchanged and the `Done:` line, and no longer in `TODO.md`

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `pre-commit run --all-files` passes
- [x] `pre-commit run auditme-all --hook-stage manual --all-files` passes
- [x] `ruff format --check`, `ruff check` and `mypy` pass
- [x] The final test run above passes, judged by its exit status, and its log is saved at
      `/Users/simonb/src/metafopy/ralpher-logs/ls-profile-starts/tasks-1/final-tests.log`
- [x] `git show $(git log --diff-filter=A --format=%H -- agents/tasks/ls-profile-starts/tasks-1.md):TODO.md > <scratch>/todo-before.md` (the
      `TODO.md` at the list's start) and `python3 ~/.claude/skills/todo-review/verify_bodies.py <scratch>/todo-before.md TODO.md DONE.md` exits 0
      (`<scratch>` is the session's scratch directory, outside the repository)

**Notes:**

- Depends on Task 2
- `verify_bodies.py` accepts an added item, so the new `TODO.md` item does not fail it
- The full `pytest src/tests` run is the run order's required final check

---
