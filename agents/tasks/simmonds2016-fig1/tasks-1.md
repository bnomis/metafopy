# Tasks: Simmonds and Higgins (2016) Fig 1 one-stage random effects

**Source:** TODO.md, "Simmonds and Higgins (2016) Fig 1 one-stage random effects differs from metafopy and R metafor"
**Created:** 2026-11-17
**Version:** 1

**Run order:** row 4 of `agents/tasks/order-todo-items-1-10.md`; waits for no other list. Row 5 (`agents/tasks/stijnen2010-pn/tasks-1.md`) waits for
this one, because both edit `src/tests/correctness/cases/glmm_published.toml`.

The user answered *investigate* for the case `glmm.published.simmonds2016.hrt.random.mismatch` on 2026-11-14 (`agents/specs/correctness-glmm/mismatches.md`,
"Answers"), and approved this plan on 2026-11-17: compare the marginal likelihood of the one-stage random-effects model (`rma_glmm`, `measure='OR'`,
`model='UM.FS'`, `method='ML'`) at `tau2 = 0` and at the fitted 0.213 with the formulas of `src/tests/correctness/glmm_formulas.py`. If it is higher
at 0.213, the printed value is recorded as a source difference in the case's `reason`, as the other accepted published mismatches are; if it is not,
metafopy is wrong and the list stops for the operator with the figures. No file under `src/metafopy/` changes in either branch.

The paper is in the papers directory as `/Users/simonb/src/metafopy papers/A general framework for the use of logistic regression models in
meta-analysis.pdf`. Its appendix gives the R code of the Fig 1 one-stage odds-ratio model:
`glmer(cbind(event,n-event) ~ factor(study) + factor(treat) + (treat-1|study), family=binomial(link="logit"))`, with `treat` 1 for the treated arm and
0 for the control arm, and `glmer`'s default `nAGQ` of 1. That puts the random effect on the treated arm only, which is `rma_glmm`'s `coding=1`;
`rma_glmm` defaults to `coding=0.5` (plus and minus one half).

<!-- doc-counts: accept-start probed on 2026-11-17 - a dated record, not a claim about the present -->
Probed on 2026-11-17 (metafopy at `c75f3d0`; R metafor 5.2-1 and lme4 2.1.0 from `/Users/simonb/src/metafopy/.audit/Rlib`), on the data of the case:

- Formulas (`glmm_formulas.rma_glmm_fit`, adaptive quadrature, 7 points): `ML` gives log OR 0.58260, `tau2` 0.213079, marginal log-likelihood
  -15.847071; `FE` (the same model at `tau2 = 0`, log OR 0.55461) gives -15.871109. `profile_loglik_at` at the `ML` log OR gives -15.872378 at
  `tau2 = 0` and -15.847071 at 0.21307. So the likelihood is higher at 0.213, by about 0.024, and the expected branch is the source difference.
- metafopy's `fit_stats_ml.ll` is not the marginal log-likelihood for `UM.FS` with `ML` (it is metafor's saturated-model log-likelihood minus half
  the conditional deviance; "`ll` of the `lme4` mixed fits" in `agents/specs/correctness-glmm/mismatches.md`): -15.061842 for `ML` against
  -15.871109 for `FE`, the same as R's `logLik` (-15.06185973 and -15.87110947). Only the formulas' values compare the two `tau2`.
- With `coding=1` both metafopy and R metafor 5.2-1 give `tau2` 0 and OR 1.74126 (0.58695, 5.16564), at `nAGQ` 1 and 7, which rounds to the printed
  1.74 (0.59, 5.17); the appendix's `glmer` call gives the same (singular fit, `tau2` 0, `logLik` -15.87110947). With `coding=0.5`, `nAGQ` 1 gives
  OR 1.79826, `tau2` 0.24699 (R) and 0.24701 (metafopy); `nAGQ` 7 gives the case's 1.79068 (0.54155, 5.92107).
<!-- doc-counts: accept-end -->

So the paper's random-effects result equals its fixed-effect result because its model (random effect on the treated arm only) has its maximum at
`tau2 = 0` on these data, not because `glmer` stopped early; `rma_glmm`'s default model is a different parametrisation with its maximum at 0.213.

The project has no affected-tests tool, so each task names its test files, and the last task runs `pytest src/tests` as the run order requires.
Every R call sets `METAFOPY_AUDIT_RLIB=/Users/simonb/src/metafopy/.audit/Rlib` (a worktree has no `.audit/Rlib`). Scratch scripts stay outside the
repository. No version bump.

**Operator stop (conditional):** Task 1, "stops for the operator: the marginal likelihood is not higher at tau2 0.213 than at 0". It stops only if the
formulas' likelihood is not higher at 0.213; the 2026-11-17 probe above expects it not to stop.

---

## Task 1: Compare the marginal likelihood at tau2 0 and 0.213

**Status:** todo
**Priority:** high
**Complexity:** small
**Source Reference:** the item's fix direction (compare the marginal likelihood at `tau2 = 0` and at 0.213)

**Technical Goal:** whether the one-stage random-effects model's marginal likelihood is higher at the fitted `tau2` than at zero is known and recorded
in this task's Notes, with the figures of the formulas, metafopy and R metafor 5.2-1, and the fit under the paper's own coding.

**Subtasks:**

- [ ] Starting-point check: `src/tests/correctness/cases/glmm_published.toml` holds the case `glmm.published.simmonds2016.hrt.random.mismatch` with
      `status = "mismatch"` and a `reason` without the words "accepted as a source difference"; if not, stop as `blocked`, naming what differs
- [ ] Write a Python script in the session's scratch directory (outside the repository) that, on the case's data (`ai`, `n1i`, `ci`, `n2i` copied
      from the case), prints with `repr`: `glmm_formulas.rma_glmm_fit(measure='OR', model='UM.FS', method=m, ...)` for `m` in `'ML'` and `'FE'`
      (`beta[0]`, `tau2`, `loglik`, `gradient_norm`); `glmm_formulas.profile_loglik_at(problem, 'ML', beta, tau2, rule=rule, points=points)` with the
      problem, rule and points of `glmm_formulas.rma_glmm_problem` for the `ML` call, at the `ML` and the `FE` `beta`, each at `tau2` 0 and at the `ML`
      `tau2`; and `metafopy.rma_glmm` for `method` `'ML'` and `'FE'`, and for `coding` in `0.5` and `1.0` with `nAGQ` in `1` and `7` (`exp` of
      `beta[0]`, `ci_lb[0]`, `ci_ub[0]`, `tau2`, `fit_stats_ml.ll`). Run it with `source bin/activate.sh` and `src/tests` on `sys.path`
- [ ] Write an R script in the scratch directory that prints `packageVersion` of metafor and lme4, the same `rma.glmm` fits (`coding` 1/2 and 1,
      `nAGQ` 1 and 7: OR, CI, `tau2`, `logLik`), and the paper's appendix call `glmer(cbind(event, n-event) ~ factor(study) + factor(treat) +
      (treat-1|study), family=binomial(link="logit"))` on the data in long form (`treat` 1 for the treated arm), printing the OR, its Wald interval
      (1.959964 standard errors), the variance and `logLik`. Run it with
      `export METAFOPY_AUDIT_RLIB=/Users/simonb/src/metafopy/.audit/Rlib; LC_ALL=C.UTF-8 R_LIBS=$METAFOPY_AUDIT_RLIB Rscript <script>`. Stop as
      `blocked` if metafor does not load or its version is not 5.2.1
- [ ] Record in this task's Notes, dated, with the metafopy commit: every figure above, and the verdict "the likelihood is higher at 0.213" (the
      formulas' `ML` `loglik` exceeds both `tau2 = 0` values by more than 1e-6) or "the likelihood is not higher at 0.213"
- [ ] If the verdict is "not higher at 0.213": stops for the operator: the marginal likelihood is not higher at tau2 0.213 than at 0, so metafopy's
      `ML` fit (and R metafor's, which agrees) is not the maximum. Change no file but this task list, record the figures, and stop as `blocked`
      naming this stop. Do not touch `src/metafopy/`
- [ ] If the formulas, metafopy or R disagree with each other on the `coding=0.5`, `nAGQ` 7 fit beyond the case's last printed digit (OR 1.79068,
      CI 0.54155 to 5.92107, `tau2` 0.21307), stop as `blocked`: that is a different difference from this item's

**Acceptance Criteria:**

- [ ] This task's Notes record the formulas' likelihood at `tau2` 0 and 0.213, metafopy's and R's fits at both codings, the appendix `glmer` fit, and
      one of the two verdicts

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)

**Notes:**

- Only this task list file changes; no code or test changes
- Key files: `src/tests/correctness/glmm_formulas.py` (`rma_glmm_fit`, `rma_glmm_problem`, `profile_loglik_at`),
  `src/tests/correctness/cases/glmm_published.toml`
- The header's 2026-11-17 probe expects "the likelihood is higher at 0.213"

---

## Task 2: Record the printed value as a source difference, and the paper's model as a passing case

**Status:** todo
**Priority:** high
**Complexity:** small
**Source Reference:** the item's fix direction (record the printed value as a source difference)

**Technical Goal:** the mismatch case says why the paper prints the fixed-effect result for its random-effects model, in the form of the other
accepted published mismatches, and a passing case checks that `rma_glmm` with the paper's coding reproduces the printed figure.

**Subtasks:**

- [ ] Read Task 1's verdict. If it is missing, or it is "not higher at 0.213", stop as `blocked`
- [ ] In `src/tests/correctness/cases/glmm_published.toml`, append to the `reason` of `glmm.published.simmonds2016.hrt.random.mismatch`, after its
      values and in the form of the Jackson et al. cases: `; accepted as a source difference on <date> (the paper's glmer code puts the random
      effect on the treated arm only, coding = 1, where tau2 is 0 and the fit is the fixed-effect 1.74 (0.59, 5.17); with coding = 0.5 the marginal
      likelihood is higher at tau2 0.21307 than at 0, <the two formulas' values from Task 1>; metafopy and R metafor 5.2-1 agree)`. Keep `status =
      "mismatch"`, `printed`, `data` and `args` unchanged
- [ ] Add a case `glmm.published.simmonds2016.hrt.random.coding1` after the mismatch case, with the same `data`, `args` plus `coding = 1.0`,
      `check = "test_published.glmm_printed"`, `printed = { "exp_beta[0]" = "1.74", "exp_ci_lb[0]" = "0.59", "exp_ci_ub[0]" = "5.17", "tau2" = "0" }`
      if the checker accepts a printed `"0"` (otherwise leave `tau2` out), `covers = ["rma_glmm.fit.OR.UM.FS.ML", "rma_glmm.coding.1"]`, and a
      `[case.source]` with the mismatch case's citation and doi, `locator = "Fig 1 (read from the page image), one-stage model, random effects; data
      of Table 1; model of the appendix R code (treat-1|study)"` and `read = <date>`. Update the comment above the Simmonds cases to say the paper's
      model is `coding = 1`
- [ ] Append to the paragraph after the published-mismatches table in `agents/specs/correctness-glmm/mismatches.md` (the one ending "but not for
      `ML` (`tau2` 0.21307).") one dated sentence: investigated on <date> by `agents/tasks/simmonds2016-fig1/tasks-1.md`; the paper's model is
      `coding = 1`, whose maximum is at `tau2 = 0`, and the `coding = 0.5` likelihood is higher at 0.21307, so the row is a source difference. Change
      nothing else in that file (its Answers section is a dated record)
- [ ] Run `pytest src/tests/correctness/test_published.py src/tests/correctness/test_coverage.py`; the new case passes and the mismatch case stays a
      strict expected failure. If the new case fails, stop as `blocked` with the values it got

**Acceptance Criteria:**

- [ ] The mismatch case's `reason` names the source difference with its date and both likelihoods; its status is unchanged
- [ ] `glmm.published.simmonds2016.hrt.random.coding1` passes
- [ ] No file under `src/metafopy/` changes

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `pytest src/tests/correctness/test_published.py src/tests/correctness/test_coverage.py` passes

**Notes:**

- Depends on Task 1
- Key files: `src/tests/correctness/cases/glmm_published.toml`, `src/tests/correctness/test_published.py` (`glmm_printed`),
  `src/tests/correctness/cases.py`, `agents/specs/correctness-glmm/mismatches.md`
- The passing `coding = 1` case adds a test only; it changes no metafopy output

---

## Task 3: The full test run and closing the item

**Status:** todo
**Priority:** high
**Complexity:** small
**Source Reference:** the item as a whole

**Technical Goal:** the whole branch is checked once, and the item moves to `DONE.md` with its outcome.

**Subtasks:**

- [ ] Move the item from `TODO.md` to `DONE.md`: append its heading and body unchanged at the end of `DONE.md`, then a line
      `Done: 2026-MM-DD, agents/tasks/simmonds2016-fig1/tasks-1.md; source difference: the paper's glmer model is coding = 1 (tau2 0, OR 1.74), and
      the coding = 0.5 likelihood is higher at tau2 0.213; case reason updated and a passing coding = 1 case added` with the date of the commit
- [ ] Run the final tests from the worktree and save their whole output for the merge gate (the main checkout is the first entry of
      `git worktree list --porcelain`, `/Users/simonb/src/metafopy`):
      `mkdir -p /Users/simonb/src/metafopy/ralpher-logs/simmonds2016-fig1/tasks-1 && source bin/activate.sh && pytest src/tests 2>&1 | tee /Users/simonb/src/metafopy/ralpher-logs/simmonds2016-fig1/tasks-1/final-tests.log`

**Acceptance Criteria:**

- [ ] The item is in `DONE.md` with its body unchanged and a `Done:` line naming this list and the outcome, and no longer in `TODO.md`

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `pre-commit run --all-files` passes
- [ ] `pre-commit run auditme-all --hook-stage manual --all-files` passes
- [ ] `mypy` passes
- [ ] The final test run above passes
- [ ] `git show $(git log --diff-filter=A --format=%H -- agents/tasks/simmonds2016-fig1/tasks-1.md):TODO.md > <scratch>/todo-before.md` and
      `python3 ~/.claude/skills/todo-review/verify_bodies.py <scratch>/todo-before.md TODO.md DONE.md` exits 0 (`<scratch>` is the session's scratch
      directory, outside the repository)

**Notes:**

- Depends on Task 2
- No version bump: the list changes no library output
- The full `pytest src/tests` run is the run order's required final check

---
