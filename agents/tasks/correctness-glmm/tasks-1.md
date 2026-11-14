# Tasks: Correctness tests for rma_glmm

**Spec:** `agents/specs/correctness-glmm/spec.md`
**Created:** 2026-11-14
**Version:** 1

The spec (version 1.1) was approved on 2026-11-13 in `agents/reviews/order-correctness-follow-ups-specs.md`; its Decisions section records the
answers that apply here: the Jackson et al. (2018) values count as `published`, with a `derived` case for every coverage item as well (D3 of that
review), and R `glmmML` 1.1.7 is installed in `/Users/simonb/.local/share/metafopy/r-correctness-lib`, the library `METAFOPY_CORRECTNESS_RLIB`
names, outside the audit's pinned library (D4). The papers were supplied before the order started (D5).

This list is one row of the run order `agents/tasks/order-correctness-follow-ups.md`, run by `/rro` one list at a time in a worktree, each merged to
`main` before the next. It waits for two earlier lists of that order: `correctness-harness-checks` (the `check` functions for back-transformed
estimates and `predict` intervals in `src/tests/correctness/test_derived.py`, used by the published cases of Task 9) and `correctness-mv-robust` (its
changes to `src/tests/correctness/cases.py` and `test_coverage.py`, which Task 1 builds on).

The project has no affected-tests tool. This list changes no module under `src/metafopy/`: it adds tests, formulas and cases under
`src/tests/correctness/` and writes `agents/specs/correctness-glmm/mismatches.md`. So each task names its own test files, and the last task runs the
whole correctness suite, `slow` cases included (`slow` tests run by default; `pyproject.toml` has no `-m` in `addopts`).

**Recording a mismatch (every task that writes cases):** when a case's reference disagrees with metafopy beyond the case's stated tolerance, commit it
with `status = "mismatch"` (a strict xfail) and a `reason` naming both values, and append an item to `agents/specs/correctness-glmm/mismatches.md`,
creating the file from the review-document layout in Task 13 if it does not exist yet. The item gives the source, locator, reference value,
metafopy's value, and metafor's value: from the R parity fixture in `src/tests/reference/` where one holds it, otherwise computed with the audit's
pinned library (`main=$(git worktree list --porcelain | sed -n '1s/^worktree //p'); R_LIBS="$main/.audit/Rlib" LC_ALL=C.UTF-8 Rscript -e '...'`).
A difference that comes from a documented integration approximation (adaptive quadrature with `nAGQ = 7`, the Laplace approximation, `CM.EL`'s
`int_rel_tol`) is a convention, not a mismatch: it goes in `mismatches.md` under "Conventions" only when it changes a printed digit of a published
value. Never change metafopy's output to match a reference in this list. A reference that cannot be read is listed under "Sources needed" in the same
file, with the table or example wanted, and its cases are not written.

**Sources:** PDFs in `/Users/simonb/src/metafopy papers`, outside the repository (`download_report.md` there lists the files). Read them with
`pdftotext` (`/opt/homebrew/bin/pdftotext`); never copy them into the repository; copy only the numbers used, quoted with their citation. If that
directory cannot be read from the queue's session, the task stops as `blocked` and says so.

**R:** nothing is installed by this list, nothing is installed into the audit's pinned library, and `src/audit/data/r-versions.json` must not change.

---

## Task 1: Harness - formula modules, rma_glmm coverage items, optimiser exemption test

**Status:** todo
**Priority:** high
**Complexity:** medium
**Spec Reference:** Module location; Coverage list; Error handling; Implementation step 1

**Technical Goal:** case files can name formulas in `glmm_formulas.py`, and the coverage gate lists every `rma_glmm` item as missing.

**Subtasks:**

- [ ] Starting-point check: `ls src/tests/correctness/glmm_formulas.py src/tests/correctness/cases/glmm*.toml` finds nothing, and
      `grep -rnw 'rma_glmm' src/tests/correctness/test_coverage.py` finds no coverage item. Stop as `blocked`, naming what exists, if either fires.
      `FORMULA_MODULES` already present in `src/tests/correctness/cases.py` (made by a sibling list) is not a clash: then only add `'glmm_formulas'`
      to it
- [ ] `src/tests/correctness/cases.py`: a `FORMULA_MODULES = ('formulas', 'glmm_formulas')` tuple; a formula is named `"<module>.<name>"` and resolved
      in its module; the loader refuses a module not in the tuple with `ValueError` naming file and case. Update the module docstring and the error
      messages that say `"formulas.<name>"`
- [ ] Create `src/tests/correctness/glmm_formulas.py` holding only its module docstring (no `metafopy` import; each function cites its equation), and
      `cases/glmm.toml`, `cases/glmm_published.toml`, each empty except a header comment
- [ ] `src/tests/correctness/test_coverage.py`:
  - the `ast` import check runs over every module of `FORMULA_MODULES`;
  - `coverage_items()` adds the `rma_glmm` items of the spec's Coverage list, read from `metafopy.models.rma_glmm.SUPPORTED`, `EXPERIMENTAL`,
    `MODELS`, `UNCONDITIONAL_ONLY`, `METHODS`, `metafopy.models.glmm_fit.ONE_GROUP`, `PACKAGES`, `OPTIMIZERS` and
    `metafopy.core.glm.LINKS` (the tuple of the `Link` alias's values), with the test, coding, `i2def` and heterogeneity values listed literally as
    the spec says;
  - a test that each `[[exempt]]` entry for an `rma_glmm.optimizer.<o>` item whose reason names `NotSupportedError` still raises it on a `CM.EL` `OR`
    fit of `dat.nielweise2007`, failing with the item's name when it does not;
  - mark `test_every_core_item_has_a_case` `xfail(strict=True, reason='rma_glmm coverage incomplete; see agents/tasks/correctness-glmm/tasks-1.md')`
    until Task 7 removes the mark
- [ ] Inline-TOML unit tests in `test_coverage.py`: a `glmm_formulas.<name>` formula resolves; a formula in an unknown module is refused, naming file
      and case

**Acceptance Criteria:**

- [ ] The gate's missing list names every `rma_glmm` item and no other new item
- [ ] A `glmm_formulas.py` that imports `metafopy` fails `test_coverage.py`

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy` passes
- [ ] `pytest src/tests/correctness/test_coverage.py src/tests/correctness/test_derived.py src/tests/correctness/test_properties.py src/tests/correctness/test_published.py src/tests/correctness/test_other_software.py` passes

**Notes:**

- Depends on the `correctness-mv-robust` list of the order (merged to `main` before this one), which also edits `cases.py` and `test_coverage.py`
- `Link` is a PEP 695 alias (`src/metafopy/core/glm.py`); `LINKS` holds the same values (checked 2026-11-14)
- `NotSupportedError` is raised from `src/metafopy/models/glmm_fit.py` for an optimiser with no counterpart; which ones raise is found by running
  them in Task 6, not assumed here

---

## Task 2: glmm_formulas.py - likelihoods and the three integration rules

**Status:** todo
**Priority:** high
**Complexity:** large
**Spec Reference:** Kinds of evidence (derived, integration as a convention); API; Error handling; Implementation step 2

**Technical Goal:** independent conditional likelihoods and their normal mixtures, integrated exactly, by adaptive Gauss-Hermite quadrature and by the
Laplace approximation, each coded from its paper.

**Subtasks:**

- [ ] `binomial_loglik` and a Poisson counterpart, with each link of `LINKS` and the family constants that `logLik()` reports, from Stijnen et al.
      (2010) and Bagos and Nikolopoulos (2009)
- [ ] `nchg_loglik`: Fisher's non-central hypergeometric log-probability by direct summation over the support (van Houwelingen et al. 1993; Stijnen et
      al. 2010), and the approximate (binomial) conditional likelihood of `CM.AL`
- [ ] `marginal_loglik` with `rule` `'exact'` (`scipy.integrate.quad`; `dblquad` for the bivariate `UM.RS` effects), `'agq'` (`points`, centred at
      the conditional mode, scaled by its curvature) and `'laplace'`, from Pinheiro and Bates (1995)
- [ ] Turn `scipy.integrate.IntegrationWarning` and a non-converged `scipy.optimize` result into `ArithmeticError` naming the model
- [ ] `src/tests/correctness/test_glmm_formulas.py`: the three rules agree to `1e-6` on a dataset where they should (a random-intercept logistic model
      with large counts); `agq` with one point equals `laplace`; a binomial log-likelihood against `scipy.stats.binom.logpmf`; `nchg_loglik` against
      `scipy.stats.nchypergeom_fisher.logpmf`; a forced integration failure raises `ArithmeticError`

**Acceptance Criteria:**

- [ ] Every function's docstring names its source and equation
- [ ] `glmm_formulas.py` imports `numpy` and `scipy` only

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy` passes
- [ ] `pytest src/tests/correctness/test_glmm_formulas.py src/tests/correctness/test_coverage.py` passes

**Notes:**

- Depends on Task 1
- The spec names no test file for the formulas' own unit checks; `test_glmm_formulas.py` is this list's choice (see the closing report)
- Read for option names only, never for formulas: `src/metafopy/models/glmm_fit.py`, `src/metafopy/models/rma_glmm.py`

---

## Task 3: glmm_formulas.py - the formulas' own fits

**Status:** todo
**Priority:** high
**Complexity:** large
**Spec Reference:** Kinds of evidence (derived, fixed-effects fits, optimiser dependence); API; Implementation step 2

**Technical Goal:** each model fitted by the formulas' own `scipy.optimize` search from their own start, returning a frozen `GLMMReference`.

**Subtasks:**

- [ ] `GLMMReference` (frozen dataclass): estimates, standard errors from the observed information, `tau2` (and `sigma2`, `rho` for `UM.RS`),
      log-likelihood, gradient norm at the optimum
- [ ] `glmm_fit` for the two-group binomial models (`UM.FS`, `UM.RS`, `CM.AL`, `CM.EL`) with `method` `FE` and `ML`, `rule` and `points`; and the
      `poisson_`, `one_group_` and `conditional_` entry points with their measure's arguments (person-times for `IRR`, `IRLN`; one group for `PLO`,
      `PR`, `PLN`)
- [ ] Fixed-effects fits: the closed forms `logit(sum xi / sum ni)` (`PLO`) and `log(sum xi / sum ti)` (`IRLN`) without moderators; otherwise the
      score equations solved with `scipy.optimize.root`. The `CM.EL` `FE` standard error from the analytic conditional information (the sum of the
      conditional variances)
- [ ] A function giving the formulas' log-likelihood and its gradient at given estimates, for Task 4's check
- [ ] Unit checks in `test_glmm_formulas.py`: the closed forms equal the `root` solution; the gradient at each fit's optimum is below `1e-6`

**Acceptance Criteria:**

- [ ] No fit starts from, or calls, metafopy

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy` passes
- [ ] `pytest src/tests/correctness/test_glmm_formulas.py src/tests/correctness/test_coverage.py` passes

**Notes:**

- Depends on Task 2
- Sources: Stijnen et al. (2010), Simmonds and Higgins (2016) for `UM.FS`, Turner et al. (2000) for `UM.RS`, van Houwelingen et al. (1993) and
  Breslow (1981) for the conditional models, Bagos and Nikolopoulos (2009) for Poisson, Hamza et al. (2008) and Zhou et al. (1999) for `PLO`

---

## Task 4: Derived fit cases for OR

**Status:** todo
**Priority:** high
**Complexity:** large
**Spec Reference:** Coverage list (`rma_glmm.fit`); Kinds of evidence (optimiser dependence); Implementation step 3

**Technical Goal:** each `rma_glmm.fit.OR.<model>.<FE|ML>` item has a `derived` case, checked by the likelihood at metafopy's estimate and by the
estimates themselves.

**Subtasks:**

- [ ] `src/tests/correctness/test_derived.py`: `glmm_likelihood_at_estimate` as in the spec's API - the formulas' log-likelihood at metafopy's
      reported estimates equals `result.fit_stats_ml.ll` and the formulas' maximum (`rtol=1e-8`), and its gradient there is below `1e-4`; and a check
      that the estimates (`beta`, `se`, `tau2`, `sigma2` for `UM.RS`) agree with the formulas' optimum at the case's stated tolerance (`1e-5` for
      `nlminb` and the mixed models; `1e-8` relative on `FE` `beta`; `1e-6` on the `CM.EL` `FE` standard error)
- [ ] `src/tests/correctness/cases.py`: `formula_value` passes `rma_glmm`'s own arguments by name, as it does for `escalc`, if Task 1 left it unable to
- [ ] Cases in `cases/glmm.toml` for `OR` with each model of `MODELS` and `FE`, `ML` on `dat.nielweise2007` and `dat.collins1985a`, and on the
      Jackson et al. (2018) measles data inline in `[case.data]` with its locator (Table 3); each states its tolerance
- [ ] Record any mismatch as the header says

**Acceptance Criteria:**

- [ ] The gate's missing list names no `rma_glmm.fit.OR.*` item
- [ ] Each `ML` case runs both checks

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy` passes
- [ ] `pytest src/tests/correctness/test_derived.py src/tests/correctness/test_coverage.py src/tests/correctness/test_glmm_formulas.py` passes

**Notes:**

- Depends on Task 3
- The likelihood check is the stronger one: it fails for a wrong likelihood even when both searches stop in the same wrong place

---

## Task 5: Derived fit cases for IRR, the one-group and the experimental measures; methods and links

**Status:** todo
**Priority:** high
**Complexity:** large
**Spec Reference:** Coverage list (`rma_glmm.fit`, `method`, `link`); Implementation step 3

**Technical Goal:** every other `rma_glmm.fit` item, and every `method` and `link` item, has a `derived` case.

**Subtasks:**

- [ ] `IRR` with each model on `dat.nielweise2008`; the `IRR` `CM.AL` case shows it is fitted as `CM.EL` (same `beta`, `se`, `ll`)
- [ ] The one-group measures in `ONE_GROUP` with model `none`, `FE` and `ML`: `PLO`, `PR`, `PLN` on `dat.pritz1997`, `IRLN` on `dat.nielweise2008`'s
      one arm (or inline data with its locator)
- [ ] The experimental two-group measures `RR`, `RD` with `UM.FS`, `UM.RS` only (`UNCONDITIONAL_ONLY`), on `dat.nielweise2007`
- [ ] `rma_glmm.method.<m>` for each of `METHODS`: `EE` and `CE` give the `FE` fit
- [ ] `rma_glmm.link.<link>` for each value of `LINKS`, on the unconditional and one-group models that accept it
- [ ] Record any mismatch as the header says

**Acceptance Criteria:**

- [ ] The gate's missing list names no `rma_glmm.fit`, `rma_glmm.method` or `rma_glmm.link` item

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy` passes
- [ ] `pytest src/tests/correctness/test_derived.py src/tests/correctness/test_coverage.py` passes

**Notes:**

- Depends on Task 4
- A link a model refuses (a `ValueError` from `metafopy.core.glm.check_link` or the model) gets an `[[exempt]]` entry naming the refusal, not a case

---

## Task 6: Derived package and optimiser cases

**Status:** todo
**Priority:** high
**Complexity:** medium
**Spec Reference:** Coverage list (`package`, `optimizer`); Kinds of evidence (integration as a convention, optimiser dependence)

**Technical Goal:** each package's integration rule and each optimiser has a case, and the approximations are compared with the exact likelihood.

**Subtasks:**

- [ ] `rma_glmm.package.<p>` for each of `PACKAGES`: the log-likelihood at the estimates equals the formulas' approximation of the same kind -
      adaptive quadrature with `nAGQ` points for `lme4` and `GLMMadaptive`, Laplace for `glmmTMB` and for `lme4` with `UM.RS`
- [ ] Accurate-integral cases: `GLMMadaptive` with a large `nAGQ`, and `CM.EL` with `control` `int_rel_tol = 1e-10`, against the `'exact'` rule
- [ ] `rma_glmm.optimizer.<o>` for each of `OPTIMIZERS`, on `CM.EL` `OR`, estimates at `1e-4`; an optimiser that raises `NotSupportedError` gets an
      `[[exempt]]` entry saying so (checked by Task 1's exemption test)
- [ ] A difference between an approximation and the exact fit that changes a printed digit of a published value goes under "Conventions" in
      `mismatches.md`, as the header says
- [ ] Record any mismatch as the header says

**Acceptance Criteria:**

- [ ] The gate's missing list names no `package` or `optimizer` item
- [ ] Task 1's exemption test passes on every optimiser exemption

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy` passes
- [ ] `pytest src/tests/correctness/test_derived.py src/tests/correctness/test_coverage.py` passes

**Notes:**

- Depends on Task 4
- Package conventions: Bates et al. (2015) and Brooks et al. (2017) are open access, not in the papers folder; fetch them with the PubMed, Paperclip
  or alphaXiv tools if a convention needs checking. If neither can be fetched, list it under "Sources needed"; the cases still rest on Pinheiro and
  Bates (1995), which is in the folder

---

## Task 7: Derived inference and heterogeneity cases, then close the gate

**Status:** todo
**Priority:** high
**Complexity:** large
**Spec Reference:** Coverage list (inference); Implementation step 2 (heterogeneity statistics) and step 3

**Technical Goal:** the test, coding, correlation, heterogeneity, `i2def`, `tau2` interval and prediction-interval items have `derived` cases; with
them the gate passes.

**Subtasks:**

- [ ] `glmm_formulas.py`: the saturated-model log-likelihoods for `QE_LRT`, the Wald statistic for `QE_Wld`, `QM`, the typical within-study variance
      for `I2`/`H2` (`i2def='1'`, Higgins and Thompson 2002), `1 / mean(1 / vi)` (`i2def='2'`), the Wald interval on `log tau2` of `CM.EL` with `ML`,
      and the prediction interval
- [ ] Unit checks of these in `test_glmm_formulas.py`
- [ ] Cases: `rma_glmm.test.z`, `rma_glmm.test.t`; `rma_glmm.coding.<c>` for `0.5`, `1`, `0`; `rma_glmm.cor` (`UM.RS`, `cor=True`, `rho`);
      `rma_glmm.heterogeneity.<s>` for `QE_Wld`, `QE_LRT`, `QM`, `I2`, `H2`; `rma_glmm.i2def.<d>`; `rma_glmm.tau2_ci` (`se_tau2`, `ci_lb_tau2`,
      `ci_ub_tau2`); `rma_glmm.predict.pi` with the existing `predict_pi` check or the `predict` check the `correctness-harness-checks` list added
- [ ] `i2def='2'` is metafor's definition: look for a published source; if none is found, exempt `rma_glmm.i2def.2` with that reason
- [ ] Record any mismatch as the header says
- [ ] Remove the `xfail` mark from `test_every_core_item_has_a_case`

**Acceptance Criteria:**

- [ ] The gate passes with no xfail mark

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy` passes
- [ ] `pytest src/tests/correctness/test_derived.py src/tests/correctness/test_coverage.py src/tests/correctness/test_glmm_formulas.py` passes

**Notes:**

- Depends on Tasks 5 and 6
- `confint`, `blup` and `ranef` refuse `rma_glmm` fits, as in metafor; they have no items

---

## Task 8: Property cases

**Status:** todo
**Priority:** medium
**Complexity:** medium
**Spec Reference:** Kinds of evidence (property); Implementation step 4

**Technical Goal:** the invariances and limiting cases the spec names are checked, beside the derived cases.

**Subtasks:**

- [ ] `check` functions in `src/tests/correctness/test_properties.py` and `property` cases in `cases/glmm.toml`:
  - swapping the groups negates `beta` and keeps `tau2` (`UM.FS` with coding `0.5`, `CM.AL`, `CM.EL`);
  - swapping events and non-events negates the log odds ratio;
  - multiplying both person-times by one constant leaves `IRR` unchanged;
  - `IRR` `CM.AL` equals `CM.EL`;
  - the codings give one `FE` fit;
  - `ML` with `tau2` below `tau2tol` gives the `FE` estimates;
  - `I2 = 100 (H2 - 1) / H2`.
- [ ] Record any failing property as a mismatch, as the header says

**Acceptance Criteria:**

- [ ] Each property names the models or measures it runs over, read from the code's tuples where they exist

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy` passes
- [ ] `pytest src/tests/correctness/test_properties.py src/tests/correctness/test_coverage.py` passes

**Notes:**

- Depends on Task 4
- No `simulation` cases: the spec finds no exact finite-sample property to assert

---

## Task 9: Published cases

**Status:** todo
**Priority:** high
**Complexity:** large
**Spec Reference:** Seed sources; Decisions 1; Implementation step 5

**Technical Goal:** every seed source gives `published` cases, Jackson et al. (2018) Table 4 first, with the value as printed, locator and date read.

**Subtasks:**

- [ ] Check that `/Users/simonb/src/metafopy papers` can be read; stop as `blocked` if not
- [ ] Prerequisite check: `grep -n '^def ' src/tests/correctness/test_derived.py` lists a back-transform check beside `predict_pi`. If it does not,
      write no case that needs `exp(beta)`, and list each such value in `mismatches.md` under "Harness limits" with its locator
- [ ] Jackson et al. (2018) Table 4: log odds ratio, its SE and `tau2` per model for outcomes 1, 2, 4 and 4b (4b drops the double-zero study, a
      `drop00` check), on the Table 3 data inline
- [ ] Then each other seed source the spec's table lists (Stijnen et al. 2010; Simmonds and Higgins 2016; Turner et al. 2000; van Houwelingen et al.
      1993; Bagos and Nikolopoulos 2009; Hamza et al. 2008 and Zhou et al. 1999; Pinheiro and Bates 1995; Breslow 1981; Higgins and Thompson 2002): find
      its worked example and the metafopy dataset holding the same data (inline data with its locator otherwise), and add `published` cases in
      `cases/glmm_published.toml`, values as printed strings. A source with formulas only gets none, and that is said in the case file's header
- [ ] A source that cannot be read, or whose example needs a source not in the folder, goes under "Sources needed" in `mismatches.md` with its
      citation and the table or example wanted
- [ ] Record any mismatch as the header says

**Acceptance Criteria:**

- [ ] Every seed source has `published` cases, a "formulas only" note, or a "Sources needed" entry
- [ ] Every `published` case has `citation`, `locator` and `read`

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy` passes
- [ ] `pytest src/tests/correctness/test_published.py src/tests/correctness/test_coverage.py` passes

**Notes:**

- Depends on Task 7 (the derived cases every item also needs, Decision 1)
- Depends on the `correctness-harness-checks` list of the order for the back-transform check
- Every glmm source was in the folder on 2026-11-14 (`download_report.md` rows 1 and 27-35; Higgins and Thompson as `Quantifying heterogeneity in a
  meta-analysis.pdf`). Pinheiro and Bates (1995) is the authors' manuscript, not the typeset version: a value read from it says so in `locator`
- Jackson et al. computed Table 4 with metafor 1.9-9 and `glmer`; the derived cases carry the independent check

---

## Task 10: Stop for sources needed

**Status:** todo
**Priority:** high
**Complexity:** small
**Spec Reference:** Implementation step 5 (stop for the operator)

**Technical Goal:** the operator supplies each source the published cases could not be written without, or drops it. The list stops here only when
such a source exists.

**Subtasks:**

- [ ] If "Sources needed" in `agents/specs/correctness-glmm/mismatches.md` is absent or empty, write "No source needed" in that section, mark this
      task done and go on: there is nothing to stop for
- [ ] Otherwise: each entry names the citation, the table or example wanted, and the file name to save it under in
      `/Users/simonb/src/metafopy papers`; stop the list for the operator
- [ ] Once every entry has an answer, an operator session saves each supplied PDF in `/Users/simonb/src/metafopy papers` (never in the repository),
      writes `supplied <date>` or `dropped <date>` beside each entry in `mismatches.md`, commits that, and only then restarts the list

**Acceptance Criteria:**

- [ ] Every "Sources needed" entry names the table or example wanted

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy` passes
- [ ] `pytest src/tests/correctness -m 'not slow'` passes

**Notes:**

- Depends on Task 9
- Every seed source was in the folder on 2026-11-14, so this stop is expected to pass through

---

## Task 11: Published cases from supplied sources

**Status:** todo
**Priority:** medium
**Complexity:** medium
**Spec Reference:** Implementation step 5

**Technical Goal:** each source supplied at Task 10 gives its `published` cases.

**Subtasks:**

- [ ] Read "Sources needed" in `agents/specs/correctness-glmm/mismatches.md`. If it says "No source needed", mark this task done. If any entry has
      neither `supplied` nor `dropped` beside it, stop as `blocked`, naming the entries
- [ ] For each supplied source, write its `published` cases as in Task 9; a dropped source keeps its entry and gets no case
- [ ] Record any mismatch as the header says

**Acceptance Criteria:**

- [ ] Every supplied source has `published` cases or a note in its entry saying why none could be written

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy` passes
- [ ] `pytest src/tests/correctness/test_published.py src/tests/correctness/test_coverage.py` passes

**Notes:**

- Depends on Task 10

---

## Task 12: Other-software cases from base R and glmmML

**Status:** todo
**Priority:** low
**Complexity:** medium
**Spec Reference:** Other software; Error handling (generator); Decisions 2; Implementation step 6

**Technical Goal:** values from `stats::mantelhaen.test(exact = TRUE)` and `glmmML`, generated by a pinned script and committed.

**Subtasks:**

- [ ] If `/Users/simonb/.local/share/metafopy/r-correctness-lib` has no `glmmML` 1.1.7 (`grep -m1 '^Version' .../glmmML/DESCRIPTION`), stop as
      `blocked`, saying that the operator installs it there outside this repository
- [ ] Write `src/tests/correctness/r/other_software_glmm.R`, modelled on `r/other_software.R`:
  - it pins `glmmML` 1.1.7 at its top, reads `METAFOPY_CORRECTNESS_RLIB` and runs `.libPaths(c(lib, .libPaths()))`, and stops when the variable is
    unset or the library holds another `glmmML` version;
  - `mantelhaen.test(exact = TRUE)`: `CM.EL` with `FE` on `dat.nielweise2007`, `dat.collins1985a` and the Jackson data (estimate and exact interval);
  - `glmmML`: the one-group models and `CM.AL`;
  - it writes `cases/glmm_other.toml` with `kind = "other"`, and `glmmML`'s and R's versions in each case's `source`.
- [ ] Run it as `METAFOPY_CORRECTNESS_RLIB=/Users/simonb/.local/share/metafopy/r-correctness-lib R_LIBS="$main/.audit/Rlib" LC_ALL=C.UTF-8 Rscript
      src/tests/correctness/r/other_software_glmm.R` (`main` as in the header) and commit the output
- [ ] Record any mismatch as the header says

**Acceptance Criteria:**

- [ ] Every `glmmML` case names `glmmML`'s version; tests never run R
- [ ] `git diff --quiet main -- src/audit/data/r-versions.json` exits 0

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy` passes
- [ ] `pytest src/tests/correctness/test_other_software.py src/tests/correctness/test_coverage.py` passes

**Notes:**

- Depends on Task 7
- Prerequisite outside this repository: `glmmML` 1.1.7 in `/Users/simonb/.local/share/metafopy/r-correctness-lib`, installed 2026-11-13 (D4 of
  `agents/reviews/order-correctness-follow-ups-specs.md`); the audit's library at `$main/.audit/Rlib` supplies the datasets' `metafor` and base R
- Not used, because not independent: `lme4`, `GLMMadaptive`, `glmmTMB`, `survival::clogit`, `Epi::clogistic`, R `meta`'s GLMM methods
- `glmmML` uses plain Gauss-Hermite quadrature: compare with the formulas' rule of the same kind and points, or state the looser tolerance

---

## Task 13: Review the mismatches, conventions and sources needed

**Status:** todo
**Priority:** high
**Complexity:** small
**Spec Reference:** Implementation step 7

**Technical Goal:** the user decides each mismatch and sees each convention difference. The list stops here until every item has an answer.

**Subtasks:**

- [ ] Write `agents/specs/correctness-glmm/mismatches.md` in the review-document layout of the shared `reviews.md` rule:
  - what is reviewed and how;
  - automatic checks: the gate passes, every `mismatch` case in `cases/glmm*.toml` has an item, `src/audit/data/r-versions.json` is unchanged;
  - each mismatch, with source, locator and the three values;
  - "Conventions", "Sources needed" (with their Task 10 answers) and "Harness limits";
  - decisions, with options *follow the reference*, *keep metafor* and *investigate*, and a recommendation and its reason for each;
  - an empty Answers section.
- [ ] Publish its interactive page, with answers saved to the artifact's database, and put `**Review page:** <URL>` in the document's header
- [ ] If there are no mismatches and no convention differences, write that in the document, skip the page, and go on without stopping
- [ ] Otherwise stop the list for the operator. Once every item has an answer, an operator session reads the page's answers, copies each into the
      Answers section with the date, commits that, and only then restarts the list

**Acceptance Criteria:**

- [ ] Every `mismatch` case in `cases/glmm*.toml` has an item on the page

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy` passes
- [ ] `pytest src/tests/correctness -m 'not slow'` passes

**Notes:**

- Depends on Tasks 8, 11 and 12
- If the queue session has no Artifact tool, write the document and stop as `blocked`, saying that an operator session publishes the page

---

## Task 14: File the answers and close the spec

**Status:** todo
**Priority:** high
**Complexity:** small
**Spec Reference:** Implementation step 8

**Technical Goal:** each answer becomes recorded work, and the spec is marked implemented. Fixing metafopy is a follow-up change, not part of this
list.

**Subtasks:**

- [ ] Read the answers from the Answers section of `agents/specs/correctness-glmm/mismatches.md`, never from the page. Stop as `blocked`, naming the
      missing items, if the section is empty or incomplete. If Task 13 recorded that there was nothing to review, go on
- [ ] *Keep metafor*: correct the case to the convention metafor documents, cite where it documents it, and set `status = "pass"`
- [ ] *Follow the reference*: add a `TODO.md` item for each with the `todo-add` skill. The item names the case, the fix, the R parity fixtures and
      `known.toml` entries to mark, and the `docs/differences.md` section "Where metafor departs from its reference" to add
- [ ] *Investigate*: a `TODO.md` item naming what is in doubt, with the `todo-add` skill
- [ ] Set the spec's `**Status:**` to `implemented`, and its row in `agents/specs/index.md` to say so
- [ ] Run the final test run from the worktree:
      `main=$(git worktree list --porcelain | sed -n '1s/^worktree //p'); mkdir -p "$main/ralpher-logs/correctness-glmm/tasks-1" && pytest src/tests/correctness 2>&1 | tee "$main/ralpher-logs/correctness-glmm/tasks-1/final-tests.log"`

**Acceptance Criteria:**

- [ ] Every answered item has a case change or a `TODO.md` item
- [ ] `git diff --quiet main -- src/audit/data/r-versions.json` exits 0

**Definition of Done:**

- [ ] `pre-commit run --all-files` passes
- [ ] `pre-commit run auditme-all --hook-stage manual --all-files` passes
- [ ] `mypy` passes
- [ ] `pytest src/tests/correctness` passes, `slow` tests included (the last subtask's run, logged to
      `$main/ralpher-logs/correctness-glmm/tasks-1/final-tests.log`)

**Notes:**

- Depends on Task 13
- The list changes no module under `src/metafopy/`, so the full run is `src/tests/correctness` alone. If a task did change one, add its own tests and
  every test file importing it (`grep -rlw '<module>' src/tests`) to the selection
- This spec has no `TODO.md` item of its own: it is part of "Extend the correctness tests beyond the core models", which the
  `correctness-publication-bias` list, last in the order, moves to `DONE.md`
