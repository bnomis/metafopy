# Tasks: QE without an intercept

**Source:** TODO.md, "QE and the moment tau2 estimators centre yi in models without an intercept"
**Created:** 2026-11-14
**Version:** 1

The user answered *file it and fix it in this order* on 2026-11-13 (`agents/reviews/order-correctness-follow-ups-specs.md`, D1, "the QE and tau2
defect in models without an intercept"), so this list changes metafopy's output without a new review, within what that decision names, widened by the user on
2026-11-14 to ML, REML and EB (see the Notes of Task 2).

Checked against the code on 2026-11-14 (metafopy at `1bd2712`). `yi` is centred (`ymc = yi - yi.mean()`) in three places:

- `estimate_tau2` in `src/metafopy/models/tau2.py`: every `ymc @ P @ ymc` (HS, HSk, HE, DL, DLIT, SJ, SJIT, the generalised Q estimators), the
  PM/MP/PMM root finding through `qe(t, ymc, ...)`, the Fisher scoring steps of ML, REML and EB (`ymc @ PP @ ymc`, `ymc @ P @ ymc`) and the
  `ll0check` likelihoods (`log_likelihood(method, ymc, ...)`);
- the heterogeneity block of `rma_uni` in `src/metafopy/models/rma_uni.py` (QE);
- the heterogeneity block of `src/metafopy/models/rma_ls_fit.py` (QE).

`y' P y` is unchanged by the centring when the column space of X holds the constant (an intercept, or a full set of dummy columns such as
`~ factor(alloc) - 1`), so only fits whose design lacks the constant change. `confint(type='qp')` already uses the uncentred `x.yi`
(`src/metafopy/methods/confint.py`), so for such fits it is today inconsistent with the PM estimate; the fix makes them agree.

**Measured on 2026-11-14** on `dat.bangertdrowns2004` (rows with `ni` and `yi`), `mods = ni/100`, on a scratch copy with `ymc = yi` in all three
places. With an intercept, QE and every tau2 are unchanged to ten digits except SJ and SJIT (see Task 2). Without an intercept (`intercept=False`):

| Method | QE now -> fixed | tau2 now -> fixed |
|--------|-----------------|-------------------|
| EE, FE, CE | 95.742 -> 135.998 | 0 -> 0 (I2 50.91 -> 65.44) |
| DL | as above | 0.03838 -> 0.07008 |
| DLIT | | 0.06133 -> 0.11864 |
| HE, CO, VC | | 0.08218 -> 0.13999 |
| HS | | 0.03463 -> 0.06383 |
| HSk | | 0.03611 -> 0.06592 |
| SJ | | 0.09340 -> 0.14710 |
| SJIT | | 0.06134 -> 0.11864 |
| ML | | 0.03458 -> 0.08738 |
| REML | | 0.03876 -> 0.09303 |
| EB | | 0.06134 -> 0.11864 |
| PM, MP | | 0.06133 -> 0.11864 |
| PMM | | 0.06331 -> 0.12153 |

`GENQ` and `GENQM` need `weights` and were not measured; Task 1 measures them. R metafor 5.2-1 gives today's values (QE 95.742 was checked against it
on 2026-11-12), so every fixed value is a new difference from metafor, recorded in `docs/differences.md` under "Where metafor departs from its
reference" (created by `agents/tasks/confint-ht-centre/tasks-1.md`) and in `src/audit/data/known.toml`.

The project has no affected-tests tool, so each task names its test files, and the last task runs, for every module the list changed, its own tests
and the test files that import it, and `src/tests/correctness`.

---

## Task 1: Derived and property cases, written before the fix

**Status:** done
**Priority:** high
**Complexity:** small
**Source Reference:** the item's fix direction (the derived and property cases)

**Technical Goal:** the correctness harness checks QE and every tau2 estimator on a model without an intercept against the defining formulas, so
the fix in Task 2 is seen to turn them from mismatches into passes.

**Subtasks:**

- [x] Re-check the starting point: `grep -n 'ymc = yi - yi.mean()' src/metafopy/models/tau2.py src/metafopy/models/rma_uni.py
      src/metafopy/models/rma_ls_fit.py` finds one line in each, and the `## Where metafor departs from its reference` heading exists in
      `docs/differences.md`. Stop as `blocked`, naming what differs, if not
- [x] Give `model_matrix` and `rma_tau2` in `src/tests/correctness/formulas.py` an `intercept: bool = True` argument (no column of ones when
      false), and pass it on wherever those functions build X for a tau2 estimator or Cochran's Q
- [x] Add `derived` cases on `dat.bangertdrowns2004` with `mods = '~ ni'` and `intercept = false` (`ni` rather than `ni/100`: scaling a column does
      not change the projection, and the formula parser takes column names): `QE` against Cochran's `y' P y`, and `tau2` for every estimator in
      `metafopy.models.tau2.ESTIMATORS`, with `weights = ni` for `GENQ` and `GENQM`. Put them in `src/tests/correctness/cases/heterogeneity.toml`
      and `tau2.toml`, beside the cases with an intercept
- [x] Add a `property` case: QE and tau2 of a model with an intercept are unchanged when `yi` is shifted by a constant, and QE of a no-intercept
      model whose columns span the constant (`~ factor(alloc) - 1` on `dat.bcg`) equals QE of the same model with an intercept
- [x] Commit the new no-intercept derived cases with `status = "mismatch"` and a `reason` naming both values, as the harness requires for a case that
      fails today; Task 2 flips them. Record the GENQ and GENQM values measured here in this task's Notes

**Acceptance Criteria:**

- [x] `pytest src/tests/correctness` passes, with the new no-intercept cases as strict expected failures and the property case passing

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy` passes
- [x] `pytest src/tests/correctness/test_derived.py src/tests/correctness/test_properties.py src/tests/correctness/test_coverage.py` passes

**Notes:**

- Measured on 2026-11-14 with `mods = '~ ni'`, `intercept = false`, `weights = ni` (the model of the new cases, so its figures differ from the header's `ni/100` model): QE
  95.13523 -> 135.99842; GENQ tau2 0.04351 -> 0.07667; GENQM tau2 0.04539 -> 0.07976. EE, FE and CE give tau2 0 before and after, so their cases pass today
- Depends on `agents/tasks/confint-ht-centre/tasks-1.md` (it creates the `docs/differences.md` section Task 3 adds to)
- These cases record a mismatch the user has already decided (D1), so they do not go to a review; they exist so that the fix is checked
- Key files: `src/tests/correctness/formulas.py`, `src/tests/correctness/cases/heterogeneity.toml`, `src/tests/correctness/cases/tau2.toml`,
  `src/tests/correctness/cases/properties.toml`

---

## Task 2: Use yi, not the centred yi, in each quadratic form and likelihood

**Status:** in-progress
**Priority:** high
**Complexity:** small
**Source Reference:** the item's fix direction (the code change, the R parity fixtures)

**Technical Goal:** QE is Cochran's `y' P y` and every tau2 estimator uses it, for models with and without an intercept.

**Subtasks:**

- [x] In `estimate_tau2` (`src/metafopy/models/tau2.py`), use `yi` in every quadratic form, in the `qe(...)` calls of PM, MP and PMM, in the Fisher
      scoring steps and in the `ll0check` likelihoods. Keep the SJ and SJIT starting value as it is, `numpy.mean((yi - yi.mean()) ** 2)`: it is
      Sidik and Jonkman's initial estimate, a variance about the mean and not a quadratic form in P, and changing it changes SJ with an intercept
      (0.09323 -> 0.10501 on the model in the header, measured 2026-11-14). Keep that one centred expression under a name that says what it is
      (`sj_start`), and drop `ymc`
- [x] In the heterogeneity blocks of `src/metafopy/models/rma_uni.py` and `src/metafopy/models/rma_ls_fit.py`, compute QE from `yi`
- [x] Update the docstrings that describe QE or the estimators to say that QE is `y' P y` (Cochran's Q) for every design, and that metafor centres
      `yi` (`ymci` in `rma.uni`), which differs for designs without the constant; link the `docs/differences.md` section
- [x] Flip the no-intercept cases of Task 1 to `status = "pass"` and drop their `reason`
- [x] Run the R parity tests (`pytest src/tests/test_rma_uni_parity.py src/tests/test_rma_uni_help.py src/tests/test_rma_uni_extras.py
      src/tests/test_rma_ls.py src/tests/test_fits.py`). For each failure, check that the fit has no intercept and that its columns do not span the
      constant; mark that fixture value as differing as `agents/tasks/confint-ht-centre/tasks-1.md` Task 1 marks `ht_ci` (the JSON stays as R wrote
      it; the test compares the value with the derived reference and asserts that R's differs, with a comment naming the section). A failure on a
      fit with an intercept, or with columns that span the constant, is a bug in this change: fix it, never mark it
- [x] Measure `rma_ls` without an intercept in the location part (`intercept=False`) before and after, and record both QE values in this task's
      Notes

**Acceptance Criteria:**

- [x] `grep -n 'ymc' src/metafopy/models/tau2.py src/metafopy/models/rma_uni.py src/metafopy/models/rma_ls_fit.py` finds nothing
- [x] Every fit with an intercept gives the same QE and tau2 as before (the R parity tests of such fits pass unchanged)
- [x] The no-intercept correctness cases pass

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy` passes
- [x] `pytest src/tests/test_rma_uni_parity.py src/tests/test_rma_uni_help.py src/tests/test_rma_uni_extras.py src/tests/test_rma_ls.py src/tests/correctness/test_derived.py src/tests/correctness/test_properties.py` passes

**Notes:**

- Depends on Task 1
- Scope confirmed by the user on 2026-11-14 (recorded in `agents/reviews/order-correctness-follow-ups-specs.md`, Answers): the fix covers ML, REML
  and EB as well as QE and the moment estimators, since their Fisher scoring steps use the centred `yi` (REML 0.0388 -> 0.0930 on the model in the
  header, where the uncentred REML likelihood peaks at 0.0930)
- `src/metafopy/models/tau2.py` has no test file of its own; `rma_uni`'s tests reach it
- Measured on 2026-11-14, `rma_uni(..., mods='~ ni - 1', scale='~ 1')` on `dat.bangertdrowns2004`: QE 95.74239 before, 135.99842 after (no-intercept location part)
- Task 1's cases passed `mods = '~ ni', intercept = false`, but `rma_uni` ignores `intercept` when `mods` is a formula (as metafor does), so metafopy fitted an intercept model
  and the cases' reasons (tau2 0.03981 for DL) were of that model. The cases now use `mods = '~ ni - 1'`, and `formulas.model_matrix` reads the `- 1`
- The R parity tests of Task 2's subtask pass unchanged: none of their fixtures fits a design without the constant, so no fixture value is marked as differing

---

## Task 3: Record the difference, the full test run and closing the item

**Status:** done
**Priority:** high
**Complexity:** small
**Source Reference:** the item's fix direction (`known.toml`, `docs/differences.md`) and the item as a whole

**Technical Goal:** `bin/metafor-audit.sh` accepts the difference with its reason, `docs/differences.md` explains it, the whole branch is checked
once, and the item moves to `DONE.md`.

**Subtasks:**

- [ ] Run the audit on the no-intercept constructions from the worktree, against the main checkout's R oracle, in the background:
      `main=$(git worktree list --porcelain | sed -n '1s/^worktree //p'); METAFOPY_AUDIT_WORK="$main/.audit" bin/metafor-audit.sh --only 'construct/*/intercept=false'`.
      If `check-r` reports that the R library is missing or differs from `src/audit/data/r-versions.json`, stop as `blocked` and say so; never
      pass `--allow-r-drift`
- [ ] For each finding it reports on `QE`, `QEp`, `I2`, `H2`, `tau2` or a value that follows from tau2 (`se`, `beta`, the fit statistics, and so on),
      check that the fixture's design lacks the constant, and add a `[[known]]` entry to `src/audit/data/known.toml` matching exactly those case
      ids, with `kind`, `path`, a `reason` (metafor centres `yi` once, `ymci` in `rma.uni`, so its QE is not Cochran's Q when the design does not
      span the constant; metafopy uses `y' P y`) and `ref = "docs/differences.md"`. Add no entry for a finding the run does not report. Stop as
      `blocked`, naming it, for a finding on a fixture whose design spans the constant
- [ ] Rerun the same command and check that its report lists no finding left from this change
- [ ] Add a subsection `### QE and tau2 of a model without an intercept` to the section "Where metafor departs from its reference" of
      `docs/differences.md`: a short summary, the same call in R and in Python with the output of both (the model in the header), and *Why:*
      Cochran's Q is `sum w (y - X b)^2`, which centring changes only when X does not span the constant. Produce the R output with
      `R_LIBS="$main/.audit/Rlib" LC_ALL=C.UTF-8 Rscript -e '<the R block>'`
- [ ] Move the item from `TODO.md` to `DONE.md`: append its heading and body unchanged at the end of `DONE.md`, then a line
      `Done: 2026-MM-DD, agents/tasks/qe-no-intercept/tasks-1.md` with the date of the commit
- [ ] Run the final tests from the worktree and save their output for the merge gate:
      `main=$(git worktree list --porcelain | sed -n '1s/^worktree //p'); mkdir -p "$main/ralpher-logs/qe-no-intercept/tasks-1" && pytest src/tests/test_rma_uni_parity.py src/tests/test_rma_uni_help.py src/tests/test_rma_uni_extras.py src/tests/test_rma_ls.py src/tests/test_fits.py src/tests/test_audit13_fixes.py src/tests/test_audit4_predict.py src/tests/test_audit8_values_a.py src/tests/test_audit8_refusals.py src/tests/test_audit9_fixes.py src/tests/test_hettest.py src/tests/test_docs_examples.py src/tests/correctness 2>&1 | tee "$main/ralpher-logs/qe-no-intercept/tasks-1/final-tests.log"`

**Acceptance Criteria:**

- [ ] The audit run on `construct/*/intercept=false` reports no finding from this change, and every new `known.toml` entry matches at least one case
      of that run
- [ ] `docs/differences.md` names the difference under "Where metafor departs from its reference"
- [ ] The item is in `DONE.md` with its body unchanged, and no longer in `TODO.md`

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `pre-commit run --all-files` passes
- [ ] `pre-commit run auditme-all --hook-stage manual --all-files` passes
- [ ] `mypy` passes
- [ ] The final test run above passes, `slow` tests included
- [ ] `git show $(git log --diff-filter=A --format=%H -- agents/tasks/qe-no-intercept/tasks-1.md):TODO.md > <scratch>/todo-before.md` and
      `python3 ~/.claude/skills/todo-review/verify_bodies.py <scratch>/todo-before.md TODO.md DONE.md` exits 0 (`<scratch>` is the session's scratch
      directory, outside the repository)

**Notes:**

- Depends on Task 2
- Prerequisite outside this repository: R metafor 5.2-1 and the pinned packages in the main checkout's `.audit/Rlib` (present on 2026-11-14)
- The audit's no-intercept constructions are `construct/<fixture>/intercept=false` (seen in `.audit/runs/20261102-223017/findings.json`:
  `uni_*`, `ls_*`, `mv_*` and `glmm_*` fixtures). `rma_mv` and `rma_glmm` do not centre `yi` (`grep -rn 'yi.mean()' src/metafopy` on 2026-11-14), so
  a finding on an `mv_*` or `glmm_*` fixture is not from this change. A full audit run is the user's call; a difference the slice does not reach
  (a help example or `testthat` unit with such a model) surfaces in the next full run
- The test files in the final run are, for `rma_uni`, `rma_ls_fit` and `tau2`, their own tests and the test files that import
  `metafopy.models.rma_uni` (`grep -l 'metafopy\.models\.rma_uni\b' src/tests/*.py` on 2026-11-14), the docs examples, and `src/tests/correctness`.
  If a file has been added since, run the grep again and extend the command

---
