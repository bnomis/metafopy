# Tasks: Correctness tests: location-scale models

**Spec:** `agents/specs/correctness-location-scale/spec.md`
**Created:** 2026-11-14
**Version:** 1

The spec (version 1.1) was approved on 2026-11-13 in `agents/reviews/order-correctness-follow-ups-specs.md`, whose Answers say: the paper's printed
values count as `published`, and every location-scale item needs a `derived` case as well (its D3); the QE defect of models without an intercept is
fixed earlier in the same order (its D1); the papers are supplied before the order starts (its D5). This list is a row of the run order
`agents/tasks/order-correctness-follow-ups.md` and runs after the lists `correctness-harness-checks` (the `check` functions for back-transformed
estimates, `predict` intervals and `confint` I2/H2 bounds) and `qe-no-intercept` (QE is Cochran's `y' P y` for models without an intercept), each
merged to `main` first.

The project has no affected-tests tool. The only module under `src/metafopy/` this list changes is `src/metafopy/models/rma_ls_fit.py` (Task 1, a
constant with no change of behaviour); everything else is under `src/tests/correctness/` and `agents/`. So each task names its own test files, and the
last task runs the whole correctness suite, `slow` cases included, with the tests that reach `rma_ls_fit.py`.

The test-only reference fit is `formulas.ls_reference_fit`, never `ls_fit`, which clashes with a name an audit check reserves.

**Recording a mismatch (every task that writes cases):** when a case's reference disagrees with metafopy, commit it with `status = "mismatch"` and a
`reason` naming both values, and append an item to `agents/specs/correctness-location-scale/mismatches.md`, creating the file from the
review-document layout of Task 11 if it does not exist yet. The item gives the source, locator, reference value, metafopy's value, and metafor's value
from the R parity fixture (`src/tests/reference/`) where one exists. Never change metafopy's output to match a reference in this list. A reference that
cannot be read is listed under "Sources needed" in the same file, and its case is not written.

**Sources:** PDFs in `/Users/simonb/src/metafopy papers`, outside the repository; read them with `pdftotext` (`/opt/homebrew/bin/pdftotext`). Never
copy them, or any text or table from them, into the repository; copy only the numbers a case uses. If that directory cannot be read from the queue
session, the task that needs it stops as `blocked` and says so.

**R:** only Task 10 runs R. Never list environment variables (`env`, `printenv`, `set`) or print an env file. To test that a variable is set, use
`[ -n "${VAR:-}" ]` and print only yes or no. To read one from an env file, search for its exact name (`grep -n '^export NAME='`), or list names without
values (`sed -n 's/^export \([A-Z_]*\)=.*/\1/p'`). Never match a generic token (`export`, `=`, `KEY`) in a file that may hold secrets.

**Caveat:** spec unclear in three places, decided here (see the Notes of the tasks named): the "per-file gate setting that requires a `derived` case"
(Decisions) does not exist in the harness yet (Task 1); `ls.optbeta` under REML has no reference objective in the spec's API (Task 4); the deviance of
`ls.fitstats.*` has no stated definition (Task 4).

---

## Task 1: Starting checks, `LS_METHODS`, the coverage items and the derived-case gate

**Status:** todo
**Priority:** high
**Complexity:** medium
**Spec Reference:** Module location; API (`LS_METHODS`); Coverage list; Decisions (derived case for every item); Implementation step 1

**Technical Goal:** the gate lists every location-scale item as missing, reads the option lists from the code, and requires a `derived` case for each
item, before any case is written.

**Subtasks:**

- [ ] Starting checks; stop as `blocked`, naming what fired, if any of them fails:
  - the TODO item "QE and the moment tau2 estimators centre yi in models without an intercept" is a heading in `DONE.md`
    (`grep -n '^## QE and the moment tau2 estimators centre yi' DONE.md`), so the `qe-no-intercept` list has merged;
  - the new names do not exist in the package or the correctness suite:
    `grep -rnwE 'LS_METHODS' src/metafopy` and `grep -rnwE 'ls_tau2|ls_loglik|ls_reference_fit|ls_wald|ls_knha_s2w|ls_profile_bounds|ls_lrt|ls_predict|ls_exact_permutation_p' src/tests/correctness/formulas.py`
    print nothing (`metafopy.methods.ls_predict` is a module, not a clash: the second search reads `formulas.py` only);
  - `src/tests/correctness/cases/location_scale.toml` does not exist.
- [ ] `src/metafopy/models/rma_ls_fit.py`: add `LS_METHODS: tuple[str, ...] = ('ML', 'REML')` and make `fit_rma_ls`'s check
      `if method not in LS_METHODS:`; the message is unchanged
- [ ] The per-file derived requirement: if `src/tests/correctness/cases.py` already has a per-file setting that makes the gate require a `derived` case
      (an earlier list of the order may add one; `grep -n 'TOP_KEYS' src/tests/correctness/cases.py`), use it unchanged. Otherwise add a top-level key
      `derived_required` to `TOP_KEYS`: a list of coverage-item prefixes; an item starting with one of them is missing until a `derived` case covers it
      or an `[[exempt]]` entry names it, a `published` case alone not being enough. Validate it as the loader validates other keys (a list of non-empty
      strings, else `ValueError` naming the file), and return it from a `load_derived_required()` beside `load_exemptions()`
- [ ] Create `src/tests/correctness/cases/location_scale.toml` with a header comment (what the file covers, the sources, the two decisions) and
      `derived_required = ["ls."]`
- [ ] `src/tests/correctness/test_coverage.py`: extend `coverage_items()` exactly as the spec's Coverage list says:
  - `ls.fit.<method>.<link>` over `metafopy.models.rma_ls_fit.LS_METHODS` and
    `typing.get_args(inspect.signature(metafopy.rma_uni).parameters['link'].annotation)`;
  - `ls.test.<test>` over `typing.get_args(metafopy.models.rma_uni.Test.__value__)` (`Test` is a PEP 695 alias);
  - `ls.optbeta`, `ls.scale.vcov`, `ls.scale.QS`, `ls.fitstats.ML`, `ls.fitstats.REML`;
  - `ls.heterogeneity.<statistic>` over a new constant `LS_HETEROGENEITY = ('H2', 'I2', 'QE')`;
  - `ls.confint.scale`, `ls.confint.location`;
  - `ls.predict.<situation>` over a constant `LS_PREDICT = ('both', 'fitted', 'newmods', 'newscale')`, the situations the
    `metafopy.methods.ls_predict` docstring names;
  - `ls.anova.<form>` over a constant `LS_ANOVA = ('X', 'Z', 'att', 'btt', 'lrt.ML', 'lrt.REML')`;
  - `ls.permutest.location.exact`, `ls.permutest.scale.exact`.
- [ ] Add a gate test that lists every item under a `derived_required` prefix lacking a `derived` case or an exemption. Mark it and the existing
      `test_every_core_item_has_a_case` `xfail(strict=True, reason='location-scale coverage incomplete; see agents/tasks/correctness-location-scale/tasks-1.md')`
      until Task 7 removes both marks
- [ ] Unit tests on inline TOML strings: a `derived_required` file whose item has only a `published` case is listed; a `derived` case clears it; an
      `[[exempt]]` entry clears it; an invalid `derived_required` value raises naming the file (skip those the setting's earlier list already tests)

**Acceptance Criteria:**

- [ ] Both gate tests are expected failures whose message names every `ls.` item and no other item
- [ ] `fit_rma_ls(method='EB', ...)` still raises `InputError` with the same message (`src/tests/test_rma_ls.py` passes unchanged)

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy` passes
- [ ] `pytest src/tests/correctness/test_coverage.py src/tests/test_rma_ls.py` passes

**Notes:**

- Key files: `src/metafopy/models/rma_ls_fit.py` (the check at `if method not in ('ML', 'REML'):`), `src/tests/correctness/cases.py`,
  `src/tests/correctness/test_coverage.py`, `src/metafopy/methods/ls_predict.py` (its module docstring lists the four situations)
- **Caveat:** spec unclear: the Decisions section names "the per-file gate setting that requires a `derived` case" as if it existed; on 2026-11-14 no
  such setting is in `cases.py`. Decided: reuse one if an earlier list added it, else add `derived_required` as above
- `rma_uni`'s `link` annotation is `typing.Literal['log', 'identity']` (checked 2026-11-14)

---

## Task 2: Sources for this list

**Status:** todo
**Priority:** high
**Complexity:** small
**Spec Reference:** Evidence; External dependencies; Implementation step 6

**Technical Goal:** every source the spec's Evidence table cites can be read before any formula is coded from it. The paper's equations are needed by
Tasks 3 to 6, so this check comes before them, not at the spec's step 6. When every source is present the task passes straight through.

**Subtasks:**

- [ ] Check that `/Users/simonb/src/metafopy papers` can be read (`ls "/Users/simonb/src/metafopy papers"`); stop as `blocked` saying so if not
- [ ] Check that each of these PDFs is there and gives text with `pdftotext` (first page, `-l 1`):
  - `Viechtbauer_2022_Locationscale_models_for_metaanalysis.pdf` (Viechtbauer and Lopez-Lopez 2022);
  - `Modelling Variance Heterogeneity- Residual Maximum Likelihood and Diagnostics.pdf` (Verbyla 1993);
  - `Generalized Linear Models with Varying Dispersion.pdf` (Smyth 1989);
  - `Harvey_1976_Estimating_regression_models_with_multiplicative.pdf` (Harvey 1976);
  - `Examining Correlates of Diversity.pdf` (Raudenbush and Bryk 1987);
  - `Improved tests for a random effects meta-regression with a single covariate.pdf` (Knapp and Hartung 2003);
  - `A LIKELIHOOD APPROACH TO META-ANALYSIS WITH RANDOM EFFECTS.pdf` (Hardy and Thompson 1996);
  - `Quantifying heterogeneity in a meta-analysis.pdf` (Higgins and Thompson 2002);
  - `Controlling the risk of spurious findings from meta-regression.pdf` (Higgins and Thompson 2004);
  - `Valid Inference in Random Effects Meta- Analysis.pdf` (Follmann and Proschan 1999).
- [ ] Check that the Viechtbauer and Lopez-Lopez PDF holds Tables 1, 2 and 3 with their numbers (`pdftotext -layout`, search for `Table 1`) and that its
      authors' analysis code link is readable (fetch it; it gives the recoding of `subject` and the rescaling of `ni`)
- [ ] If all are present, mark this task done and go on. Otherwise:
  - write the missing ones under "Sources needed" in `agents/specs/correctness-location-scale/mismatches.md` (citation with DOI, what is wanted from
    it, and which tasks wait for it), in the layout of Task 11;
  - stop the list for the operator. Once each source is in the folder, or the operator has written in the document's Answers section that it is
    dropped, an operator session commits that and only then restarts the list.
- [ ] On a restart, read the Answers section of `mismatches.md`: a source marked dropped is not used, and every case that needed it is left out with
      an `[[exempt]]` entry only where no `derived` case can cover its item; a source neither present nor answered stops the task as `blocked`, naming it

**Acceptance Criteria:**

- [ ] Every source in the spec's Evidence table except the open glmmTMB paper (Brooks et al. 2017) is present and readable, or answered as dropped

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`), if it commits anything
- [ ] `mypy` passes
- [ ] `pytest src/tests/correctness -m 'not slow'` passes (a task that may stop for the operator)

**Notes:**

- Depends on Task 1
- On 2026-11-14 every PDF above was in the folder (`download_report.md` there lists the Viechtbauer 2022 file as downloaded from PMC), so the task
  is expected to pass through
- The authors' analysis code is read for the recoding only; it is metafor code, so nothing else in it is a reference

---

## Task 3: The location-scale likelihood and its reference fit

**Status:** todo
**Priority:** high
**Complexity:** large
**Spec Reference:** API (`ls_tau2`, `ls_loglik`, `ls_reference_fit`); Error handling; Implementation step 2

**Technical Goal:** `formulas.py` codes the model of Viechtbauer and Lopez-Lopez (2022) from its equations, with a hand-coded gradient and observed
information, and maximises it independently of metafopy.

**Subtasks:**

- [ ] `formulas.ls_tau2(Z, alpha, link)`: `exp(Z alpha)` (log link) or `Z alpha` (identity link), citing the paper's equation
- [ ] `formulas.ls_loglik(yi, vi, X, Z, alpha, link, method)`: the profile log-likelihood (eq. 7, beta at its WLS value, eq. 6) for `ML`, the
      restricted log-likelihood (eq. 9) for `REML`, citing Verbyla (1993) for REML and Smyth (1989) / Harvey (1976) for the log-linear dispersion
- [ ] The analytic score and observed information of `ls_loglik` in alpha, coded by hand from the equations for both methods and links (helper names
      without a leading underscore, e.g. `ls_score`, `ls_information`)
- [ ] `formulas.ls_reference_fit(yi, vi, X, Z, link, method)`: BFGS (with the analytic score) from several starts, then Nelder-Mead from the best;
      SLSQP with `Z alpha >= 0` for the identity link. Returns `alpha`, `beta`, `vb` (eq. 10), `tau2`, `ll` and `va` (inverse of the analytic observed
      information). Raises `RuntimeError` with the optimiser's message when no start converges, never returning a non-optimum
- [ ] Harness tests in `src/tests/correctness/test_formulas_ls.py`: the analytic score and information agree with `scipy.optimize.approx_fprime` at
      several alpha on `dat.bcg` and `dat.bangertdrowns2004`, both methods, both links; `ls_reference_fit` raises `RuntimeError` on a case built not to
      converge (for example `maxiter=1`); the test module imports nothing from metafopy except the datasets it loads
- [ ] `test_coverage.py`'s import ban on `formulas.py` still passes

**Acceptance Criteria:**

- [ ] Each new `formulas.py` function names its source and equation in its docstring
- [ ] The score test fails if the information is replaced by a numerical Hessian of the wrong sign (checked once by hand, noted here)

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy` passes
- [ ] `pytest src/tests/correctness/test_formulas_ls.py src/tests/correctness/test_coverage.py` passes

**Notes:**

- Depends on Task 2
- Key files: `src/tests/correctness/formulas.py` (numpy and scipy only, no metafopy import)
- Read metafopy's `rma_ls_fit.py` for option names only, never for the formulas

---

## Task 4: Derived cases for the fits, fit statistics and heterogeneity

**Status:** todo
**Priority:** high
**Complexity:** large
**Spec Reference:** Coverage list (`ls.fit.*`, `ls.optbeta`, `ls.fitstats.*`, `ls.heterogeneity.*`); Implementation step 3

**Technical Goal:** each fit item has `derived` cases on `dat.bangertdrowns2004` and `dat.bcg` against `ls_reference_fit`.

**Subtasks:**

- [ ] Check functions in `src/tests/correctness/test_derived.py`, in the pattern of the existing `confint_*` checks: one fitting the reference with the
      case's data, method and link and comparing `alpha`, `beta`, `vb`, `tau2` and the log-likelihood; one for the fit statistics; one for `QE`, `I2`,
      `H2`
- [ ] `ls.fit.<method>.<link>` for each method and link: one moderator in both parts, on `dat.bangertdrowns2004` and on `dat.bcg` (log risk ratio,
      `ablat`). Tolerances: `1e-6` relative for alpha, beta and their variances, `1e-9` for the log-likelihood
- [ ] A `property` case per fit item: metafopy's log-likelihood is not below `ls_reference_fit`'s by more than `1e-9`
- [ ] `ls.optbeta`: `optbeta=True` under `ML`, against `ls_reference_fit` (the joint and the profile ML maximum coincide)
- [ ] `ls.fitstats.ML`, `ls.fitstats.REML`: log-likelihood, AIC, BIC and AICc from `ls_loglik` at the reference optimum and the parameter count (AICc
      with `k - p` observations under REML), each formula citing its source; the deviance as below
- [ ] `ls.heterogeneity.QE` as Cochran's `y' P y` with `P` at the fitted `tau2_i`, with and without an intercept; the no-intercept case is
      `dat.bangertdrowns2004` with `mods = ni/100`, `intercept=False` (Cochran's value 135.998 on 2026-11-12). `I2` and `H2` per study, from Higgins and
      Thompson (2002) with the typical within-study variance
- [ ] Rescaled and recoded columns (`ni/100`, the `subject` indicators) go in the case as inline `[case.data]` columns, with a comment naming the
      authors' code they come from; add a harness test that the inline columns equal the recoding of `dat.bangertdrowns2004`
- [ ] Record any mismatch as the header says

**Acceptance Criteria:**

- [ ] The derived gate lists none of these items
- [ ] The no-intercept QE case expects Cochran's value; if metafopy gives another, the case is a mismatch, not a changed expectation

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy` passes
- [ ] `pytest src/tests/correctness/test_derived.py src/tests/correctness/test_properties.py src/tests/correctness/test_coverage.py` passes

**Notes:**

- Depends on Task 3; depends on the `qe-no-intercept` list of the order (checked in Task 1)
- **Caveat:** spec unclear: `ls_reference_fit` has no `optbeta` objective, and under REML a joint maximisation over beta and alpha is a different
  objective from eq. 9. Decided: `ls.optbeta` is derived under ML only; REML with `optbeta=True` is not covered by this list and is named in the review
  document (Task 11) as an item for a later spec
- **Caveat:** spec unclear: the deviance's definition is not given. Decided: take it from the paper if it defines one; otherwise leave the deviance
  out of the derived case, say so in the case's comment, and compare log-likelihood, AIC, BIC and AICc only
- `QE` of the no-intercept model is computed in `rma_ls_fit.py` (the heterogeneity block); the `qe-no-intercept` list changes it there

---

## Task 5: Derived cases for tests, the scale covariance and confidence intervals

**Status:** todo
**Priority:** high
**Complexity:** large
**Spec Reference:** API (`ls_wald`, `ls_knha_s2w`, `ls_profile_bounds`, `ls_confint_scale`); Coverage list (`ls.test.*`, `ls.scale.*`, `ls.confint.*`);
Error handling (`nan` bounds)

**Technical Goal:** each `test` option, the scale covariance and omnibus test, and both kinds of interval have `derived` cases.

**Subtasks:**

- [ ] `formulas.ls_wald` (eq. 12/14: chi-square, or F with `(m, ddf)` after division by `m` for the t-type tests) and `formulas.ls_knha_s2w`
      (section 2.5.4: `RSS / (k - p)`, at least 1 with `adhoc`, citing Knapp and Hartung 2003)
- [ ] `formulas.ls_profile_bounds` (section 2.5.3, Hardy and Thompson 1996): `brentq` on `2 (ll_max - ll_j(a)) = chi2_{1, level}`, refitting the other
      alphas each time; `nan` for a bound not crossed within `+-` 20 standard errors
- [ ] Check functions `ls_vcov_alpha` (metafopy's `va` against the analytic observed information), `ls_qs`, `ls_confint_scale`
      (`confint(result, tol=1e-12)` for each alpha) and one for `confint(result, fixed=True)` (Wald location intervals) in `test_derived.py`
- [ ] Cases: `ls.test.<test>` for each `Test` value (location Knapp-Hartung factor and the t/F reference for both parts); `ls.scale.vcov`, `ls.scale.QS`;
      `ls.confint.scale`, `ls.confint.location`; on both datasets, `REML` and `ML`
- [ ] A case expects a `nan` bound only where the source says the profile is flat
- [ ] Record any mismatch as the header says

**Acceptance Criteria:**

- [ ] The derived gate lists none of these items
- [ ] `ls.scale.vcov` compares with the analytic information, not a numerical Hessian

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy` passes
- [ ] `pytest src/tests/correctness/test_derived.py src/tests/correctness/test_formulas_ls.py src/tests/correctness/test_coverage.py` passes

**Notes:**

- Depends on Task 3
- Key files: `src/metafopy/methods/ls_confint.py` (`confint_rma_ls`, `fixed`), `src/metafopy/methods/ls_anova.py` (`QS`)

---

## Task 6: Derived cases for predict and anova

**Status:** todo
**Priority:** high
**Complexity:** medium
**Spec Reference:** API (`ls_lrt`, `ls_predict`); Coverage list (`ls.predict.*`, `ls.anova.*`); Evidence (the REML LRT needs the same location part)

**Technical Goal:** each `predict` situation and each `anova` form has a `derived` case.

**Subtasks:**

- [ ] `formulas.ls_predict` (section 2.7: pred, se, CI, PI at `x0` and `z0`; the predicted `tau2` with its CI from `z0' va z0`, section 2.5.2) and
      `formulas.ls_lrt` (eq. 15)
- [ ] Check functions `ls_predict_fitted`, `ls_predict_newmods`, `ls_predict_newscale`, `ls_predict_both`, and `ls_anova_btt`, `ls_anova_att`,
      `ls_anova_X`, `ls_anova_Z`, `ls_anova_lrt` in `test_derived.py`; where the value is a back-transformed `tau2` or an interval bound, use the
      `check` functions the `correctness-harness-checks` list added rather than a new copy
- [ ] Cases for each `ls.predict.<situation>` and each `ls.anova.<form>`; `lrt.REML` only between models with the same location part, as the paper
      states
- [ ] Record any mismatch as the header says

**Acceptance Criteria:**

- [ ] The derived gate lists none of these items

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy` passes
- [ ] `pytest src/tests/correctness/test_derived.py src/tests/correctness/test_coverage.py` passes

**Notes:**

- Depends on Tasks 3 and 5; depends on the `correctness-harness-checks` list of the order (`grep -n '^def ' src/tests/correctness/test_derived.py`
  lists its back-transform and interval checks)
- Key files: `src/metafopy/methods/ls_predict.py`, `src/metafopy/methods/ls_anova.py`

---

## Task 7: Exact and random permutation tests, then close the gate

**Status:** todo
**Priority:** medium
**Complexity:** medium
**Spec Reference:** API (`ls_exact_permutation_p`); Coverage list (`ls.permutest.*`, random permutations); Implementation step 5

**Technical Goal:** the exact permutation p-values of both parts are derived by enumeration, the random ones are checked against them, and with them
every location-scale item has a `derived` case.

**Subtasks:**

- [ ] `formulas.ls_exact_permutation_p`: every distinct row permutation of X (`'location'`) or Z (`'scale'`), refitting with `ls_reference_fit`; cite
      Higgins and Thompson (2004) and Follmann and Proschan (1999)
- [ ] Check function `ls_permutest_exact` in `test_derived.py`; cases `ls.permutest.location.exact` and `ls.permutest.scale.exact` on a subset of
      `dat.bcg` small enough to enumerate, marked `slow`
- [ ] A `simulation` case in `test_simulation.py`: the random permutation p-value (metafopy's R-exact generator, fixed seed and stated `iter`) lies
      within 4 binomial standard errors of the exact one; marked `slow`
- [ ] Record any mismatch as the header says
- [ ] Remove the two `xfail` marks Task 1 added to `test_coverage.py`

**Acceptance Criteria:**

- [ ] Both gate tests pass with no xfail mark
- [ ] `pytest -m slow src/tests/correctness/test_derived.py src/tests/correctness/test_simulation.py` passes

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy` passes
- [ ] `pytest src/tests/correctness/test_derived.py src/tests/correctness/test_simulation.py src/tests/correctness/test_coverage.py` passes, `slow`
      tests included

**Notes:**

- Depends on Tasks 4, 5 and 6
- Key files: `src/metafopy/methods/ls_permutest.py` (`exact`, `iter`, `seed`)

---

## Task 8: Property cases

**Status:** todo
**Priority:** medium
**Complexity:** medium
**Spec Reference:** Evidence ("Properties the paper states"); Implementation step 4

**Technical Goal:** the reductions and invariances the spec lists hold for metafopy's fits.

**Subtasks:**

- [ ] Check functions in `test_properties.py`:
  - `ls_reduces_to_rma_uni`: an intercept-only scale model gives metafopy's `rma_uni` fit with the same method (`beta`, `se`,
    `tau2 = exp(alpha_0)`, log-likelihood; not the `se` of `tau2`);
  - `ls_subgroups_are_separate_fits`: one categorical moderator in both parts gives the separate `rma_uni` fits per category (`beta`, `tau2`), ML and
    REML;
  - `ls_score_is_zero`: the analytic score of `ls_loglik` is zero at metafopy's estimates;
  - `ls_invariance`: centring or rescaling a scale moderator changes alpha linearly and leaves `tau2_i` and the log-likelihood unchanged;
  - `ls_profile_peaks_at_estimate`: each alpha's profile log-likelihood peaks at the estimate;
  - `ls_optimisers_agree`: every optimiser in `metafopy.models.rma_ls_fit.OPTIMIZERS` that metafopy implements reaches the same maximum, read at test
    time; one that raises `InputError` as unavailable is skipped with its name in the message;
  - the log and identity links agree when the scale model is a set of group indicators.
- [ ] Their cases in `location_scale.toml` with `kind = "property"`
- [ ] Record any mismatch as the header says

**Acceptance Criteria:**

- [ ] Each property above has at least one case

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy` passes
- [ ] `pytest src/tests/correctness/test_properties.py src/tests/correctness/test_coverage.py` passes

**Notes:**

- Depends on Task 3
- The reductions call `metafopy.rma_uni`, so they are `property`, not `derived`

---

## Task 9: Published cases from Viechtbauer and Lopez-Lopez (2022)

**Status:** todo
**Priority:** medium
**Complexity:** medium
**Spec Reference:** Evidence ("Published values"); Decisions (printed values are `published`); Implementation step 6

**Technical Goal:** the paper's printed values on `dat.bangertdrowns2004` are `published` cases, a cross-check beside the derived ones.

**Subtasks:**

- [ ] Read the paper's PDF with `pdftotext -layout`: Tables 1 and 2 (coefficients, CIs, profile CIs), Table 3 (log-likelihood, AIC, BIC, AICc of five
      models under ML and REML), the predicted `tau2` of math and social science studies at 100 participants with CIs (0.006 to 0.121 and 0.004 to
      0.437), and the ML LRT on 3 df
- [ ] Write them as `published` cases (values as printed, strings; `citation`, `doi = "10.1002/jrsm.1562"`, `locator`, `read`), with the model as the
      paper fits it (REML, log link, KH for location, t/F for scale) and the recoded columns of Task 4
- [ ] The predicted `tau2` and its CI on the `tau2` scale use the back-transform and interval checks of the `correctness-harness-checks` list
- [ ] Record any mismatch as the header says

**Acceptance Criteria:**

- [ ] Every printed value named above is a case, or has a line in `mismatches.md` saying why not (for example a misprint, with both values)
- [ ] Every `published` case has `citation`, `locator` and `read`

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy` passes
- [ ] `pytest src/tests/correctness/test_published.py src/tests/correctness/test_coverage.py` passes

**Notes:**

- Depends on Tasks 2 and 4; depends on the `correctness-harness-checks` list of the order
- The paper's numbers were computed with metafor: agreement shows metafopy matches metafor as the paper ran it, not that the method is right

---

## Task 10: Other-software cases from R glmmTMB, and deduplication

**Status:** todo
**Priority:** low
**Complexity:** medium
**Spec Reference:** Evidence ("Other software"); Error handling (generator); Implementation step 7

**Technical Goal:** values from R `glmmTMB` 1.1.15.2 for the log-link fits, generated by a pinned script and committed; and the location-scale
formulas share, not copy, what `formulas.py` already has.

**Subtasks:**

- [ ] If `METAFOPY_CORRECTNESS_RLIB` is unset (`[ -n "${METAFOPY_CORRECTNESS_RLIB:-}" ] && echo yes || echo no`), stop as `blocked` naming the variable
- [ ] Check that `/Users/simonb/src/metafopy/.audit/Rlib/glmmTMB/DESCRIPTION` says `Version: 1.1.15.2`; stop as `blocked` if not. Install nothing
      into that library; `src/audit/data/r-versions.json` must not change
- [ ] Write `src/tests/correctness/r/other_software_ls.R` in the layout of `r/other_software.R`:
  - `lib <- Sys.getenv("METAFOPY_CORRECTNESS_RLIB")`, `.libPaths(c(lib, .libPaths()))`, glmmTMB found through `R_LIBS`;
  - it pins `GLMMTMB_VERSION <- "1.1.15.2"` and stops on any other version;
  - the known `vi` as an `equalto` covariance, the heterogeneity as the residual dispersion with `dispformula = ~ z`, ML and `REML = TRUE`;
  - whether its dispersion is `sigma` or `sigma^2` is read from glmmTMB's documentation and stated in a comment; `alpha` is compared after that
    factor;
  - it writes `cases/location_scale_other.toml` with `kind = "other"`, comparing `tau2_i`, `beta`, the log-likelihood and alpha, and puts glmmTMB's and
    R's versions in each case's `source`.
- [ ] Run it: `METAFOPY_CORRECTNESS_RLIB=$METAFOPY_CORRECTNESS_RLIB R_LIBS=/Users/simonb/src/metafopy/.audit/Rlib LC_ALL=C.UTF-8 Rscript src/tests/correctness/r/other_software_ls.R`,
      and commit its output
- [ ] Record any mismatch as the header says
- [ ] Hoist anything the location-scale formulas and checks duplicate from the core correctness functions to a single owner: grep
      `src/tests/correctness/formulas.py` and `test_derived.py` for each literal collection, constant and one-line helper the new code defines (the
      Wald and t-type p-values, the Knapp-Hartung factor, the QE quadratic form, `numbers_agree`-style comparisons), and call the existing one rather
      than a copy. Owners stay in `formulas.py` and `test_derived.py`

**Acceptance Criteria:**

- [ ] Every `other` case names glmmTMB's version
- [ ] Tests never run R
- [ ] `git diff main -- src/audit/data/r-versions.json` is empty

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy` passes
- [ ] `pytest src/tests/correctness/test_other_software.py src/tests/correctness/test_derived.py src/tests/correctness/test_formulas_ls.py src/tests/correctness/test_coverage.py`
      passes

**Notes:**

- Depends on Tasks 4 and 5
- Prerequisite outside this repository: `METAFOPY_CORRECTNESS_RLIB` names `/Users/simonb/.local/share/metafopy/r-correctness-lib` in the queue session.
  glmmTMB 1.1.15.2 is in the audit's pinned library `/Users/simonb/src/metafopy/.audit/Rlib` (checked 2026-11-14)
- glmmTMB has no identity link, so only `ls.fit.*.log` gets `other` cases
- The spec names `METAFOPY_AUDIT_RLIB`; this list uses `METAFOPY_CORRECTNESS_RLIB` with the audit library on `R_LIBS`, as `r/other_software.R` does, so
  both generators run the same way

---

## Task 11: Review the mismatches and the sources needed

**Status:** todo
**Priority:** high
**Complexity:** small
**Spec Reference:** Implementation step 8

**Technical Goal:** the user decides each mismatch. The list stops here until every item has an answer.

**Subtasks:**

- [ ] Write `agents/specs/correctness-location-scale/mismatches.md` in the review-document layout of the shared `reviews.md` rule (re-read the rule
      from disk first):
  - what is reviewed and how;
  - automatic checks: both gate tests pass, every `mismatch` case has an item, `r-versions.json` is unchanged;
  - each mismatch, with source, locator and the three values;
  - "Sources needed", and items this list leaves for a later spec (REML with `optbeta=True`, the deviance if Task 4 left it out);
  - decisions, with options *follow the reference*, *keep metafor* and *investigate*, and a recommendation for each;
  - an empty Answers section.
- [ ] Publish its interactive page, with answers saved to the artifact's database, and put `**Review page:** <URL>` in the document's header
- [ ] If there are no mismatches and no sources needed, write that in the document, skip the page, and mark this task done without stopping
- [ ] Otherwise stop the list for the operator. Once every item has an answer, an operator session reads the page's answers, copies each into the
      Answers section with the date, commits that, and only then restarts the list

**Acceptance Criteria:**

- [ ] Every `mismatch` case in `cases/location_scale*.toml` has an item on the page

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy` passes
- [ ] `pytest src/tests/correctness -m 'not slow'` passes

**Notes:**

- Depends on Tasks 4 to 10
- Model: `agents/specs/correctness-tests/mismatches.md` and Task 13 of `agents/tasks/correctness-tests/tasks-1.md`

---

## Task 12: File the answers and close the spec

**Status:** todo
**Priority:** high
**Complexity:** small
**Spec Reference:** Implementation step 8 (follow-up)

**Technical Goal:** each answer becomes recorded work; fixing metafopy is a follow-up, not part of this list. The spec is marked implemented.

**Subtasks:**

- [ ] Read the answers from the Answers section of `agents/specs/correctness-location-scale/mismatches.md`, never from the page. Stop as `blocked`,
      naming the missing items, if the section is empty or incomplete (skip this when Task 11 found nothing to review)
- [ ] *Keep metafor*: correct the case to the convention metafor documents, cite where it documents it, and set `status = "pass"`
- [ ] *Follow the reference*: add a `TODO.md` item for each with the `todo-add` skill. The item names the case, the fix, the R parity fixtures and
      `src/audit/data/known.toml` entries to mark, and the `docs/differences.md` section "Where metafor departs from its reference" to add to
- [ ] *Investigate*: a `TODO.md` item naming what is in doubt, with the `todo-add` skill
- [ ] Set the spec's `**Status:**` to `implemented`, and its row in `agents/specs/index.md` to say so
- [ ] Run the final test run from the worktree:
      `main=$(git worktree list --porcelain | sed -n '1s/^worktree //p'); mkdir -p "$main/ralpher-logs/correctness-location-scale/tasks-1" && pytest src/tests/correctness src/tests/test_rma_ls.py src/tests/test_ls_vif.py src/tests/test_ls_vif_sim.py src/tests/test_permutest_ls_plots.py src/tests/test_audit4_plots.py 2>&1 | tee "$main/ralpher-logs/correctness-location-scale/tasks-1/final-tests.log"`

**Acceptance Criteria:**

- [ ] Every answered item has a case change or a `TODO.md` item
- [ ] The spec says `**Status:** implemented`

**Definition of Done:**

- [ ] `pre-commit run --all-files` passes
- [ ] `pre-commit run auditme-all --hook-stage manual --all-files` passes
- [ ] `mypy` passes
- [ ] The final test run above passes: `src/tests/correctness` with `slow` cases (the project's `addopts` do not deselect them), the own tests of
      `src/metafopy/models/rma_ls_fit.py` (`src/tests/test_rma_ls.py`) and the test files that import the location-scale modules that import it
      (`src/tests/test_ls_vif.py`, `src/tests/test_ls_vif_sim.py`, `src/tests/test_permutest_ls_plots.py`, `src/tests/test_audit4_plots.py`)

**Notes:**

- Depends on Task 11
- This spec has no `TODO.md` item of its own to move: it is part of the umbrella item "Extend the correctness tests beyond the core models", which the
  `correctness-publication-bias` list, last in the order, moves to `DONE.md`
- Before running, re-check the importer list: `grep -rlE 'metafopy\.models\.rma_ls|methods\.ls_|methods\.location_scale' src/tests` and add any new
  file to the selection
