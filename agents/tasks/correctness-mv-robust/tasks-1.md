# Tasks: Correctness tests: rma_mv and robust

**Spec:** `agents/specs/correctness-mv-robust/spec.md`
**Created:** 2026-11-14
**Version:** 1

The spec (version 1.1) was approved on 2026-11-13 in `agents/reviews/order-correctness-follow-ups-specs.md`; its "Decisions" section records the answer
to D2 of that review (options that only software documentation defines are derived from that documentation; options defined only in metafor's code
get an `[[exempt]]` entry). This list is part of the run order `agents/tasks/order-correctness-follow-ups.md`, after `correctness-harness-checks` and
`qe-no-intercept`. It extends the harness built by `agents/tasks/correctness-tests/tasks-1.md` (all done) and follows that list's conventions.

The project has no affected-tests tool: run `source bin/activate.sh`, then `pytest` directly. The only modules under `src/metafopy/` this list
changes are `src/metafopy/models/rma_mv.py` and `src/metafopy/methods/robust.py` (Task 1, constants only, no change of behaviour). Their own tests are
`src/tests/test_rma_mv.py`, `src/tests/test_rma_mv_extras.py`, `src/tests/test_robust.py` and `src/tests/test_robust_extras.py`; the other test file
importing either module directly is `src/tests/test_audit8_crashes.py` (checked 2026-11-14 with
`grep -rnE 'metafopy\.(models\.rma_mv|methods\.robust)\b' src/tests`).

**Recording a mismatch (every task that writes cases):** when a case's reference disagrees with metafopy, commit it with `status = "mismatch"` and a
`reason` naming both values, and append an item to `agents/specs/correctness-mv-robust/mismatches.md`, creating the file from the review-document
layout in Task 12 if it does not exist yet. The item gives the source, locator, reference value, metafopy's value, and metafor's value from the R
parity fixture where one exists. Never change metafopy's output to match a reference in this list. A source that cannot be read is listed under
"Sources needed" in the same file, and its case is not written.

**Sources:** the PDFs are in `/Users/simonb/src/metafopy papers`, outside the repository (`download_report.md` there names each file). Read them with
`/opt/homebrew/bin/pdftotext`; never copy them, or any text or table from them, into the repository; copy only the numbers used. If that directory
cannot be read from the queue's session, the task stops as `blocked` and says so.

**R:** only Task 9 runs R. It reads the audit's pinned library `/Users/simonb/src/metafopy/.audit/Rlib` read-only, through `R_LIBS`, with
`METAFOPY_CORRECTNESS_RLIB=/Users/simonb/.local/share/metafopy/r-correctness-lib` prepended by `.libPaths(c(lib, .libPaths()))` as in
`src/tests/correctness/r/other_software.R`. Nothing is installed into the audit library, and `src/audit/data/r-versions.json` must not change.

**Secrets:** never list environment variables (`env`, `printenv`, `set`) or print an env file. To test that a variable is set, use
`[ -n "${VAR:-}" ]` and print only yes or no. To read one from an env file, search for its exact name (`grep -n '^export NAME='`), or list names
without values (`sed -n 's/^export \([A-Z_]*\)=.*/\1/p'`). Never match a generic token (`export`, `=`, `KEY`) in a file that may hold secrets.

**Caveat:** the spec's External dependencies say the generator reads the audit library "through `METAFOPY_AUDIT_RLIB`"; the existing generator and
the order use `METAFOPY_CORRECTNESS_RLIB` plus `R_LIBS`. This list uses the latter (Task 9); both reach the same pinned library read-only.

---

## Task 1: Option constants and the new coverage items

**Status:** todo
**Priority:** high
**Complexity:** small
**Spec Reference:** API (constants); Coverage list; Implementation steps 1 and 10

**Technical Goal:** the gate reads every `rma_mv` and `robust` option list from the code, and names each new item still lacking a case, without
hiding the core items that already pass.

**Subtasks:**

- [ ] Starting-point check: stop as `blocked` if any of these already exists, naming it:
      `grep -nE '^(METHODS|TESTS|DFS) = ' src/metafopy/models/rma_mv.py`, `grep -nE '^ADJUST = ' src/metafopy/methods/robust.py`,
      `grep -nwE 'build_v|vblocks|vconst' src/tests/correctness/cases.py`, `ls src/tests/correctness/cases/rma_mv.toml src/tests/correctness/cases/robust.toml`.
      (A `METHODS` in another module, such as `metafopy.methods.emm_contrasts`, is not a clash)
- [ ] In `src/metafopy/models/rma_mv.py` add `METHODS = (*FIXED_METHODS, 'ML', 'REML')`, `TESTS = ('z', 't', 'knha', 'adhoc')` and
      `DFS = ('residual', 'contain')`, and use them in place of the literals in the `method` and `dfs` validation and the `test` check after the
      `'hksj'` alias is mapped to `'knha'`. Error messages stay word for word
- [ ] In `src/metafopy/methods/robust.py` add `ADJUST = (False, True, 'Stata1', 'Stata2')` and use it in `check_adjust` and `default_inference`;
      `'Stata'` stays an accepted alias of `'Stata1'`, not an item. Add the new names to each module's `__all__` if it has one
- [ ] In `src/tests/correctness/test_coverage.py`, `coverage_items()` gains every item of the spec's Coverage list, each list read at test time:
      `rma_mv.METHODS`, `metafopy.models.mv_structs.STRUCTS`, `rma_mv.RSCALES`, `rma_mv.TESTS`, `rma_mv.DFS`, the named `rma_mv` items,
      `robust.rma_uni`, `robust.rma_mv`, `robust.adjust.<str(a)>` for `robust.ADJUST`, `metafopy.core.clubsandwich.CR_TYPES`, `robust.COEF_TESTS`,
      `robust.CONF_TESTS` and `metafopy.core.clubsandwich.WALD_TESTS`
- [ ] Keep the existing gate passing for the core items: it skips items starting `rma_mv.` or `robust.` while a module constant
      `PENDING_PREFIXES = ('rma_mv.', 'robust.')` is set, and a second test `test_gate_mv_robust` lists the missing items under those prefixes,
      marked `xfail(strict=True, reason='coverage incomplete; see agents/tasks/correctness-mv-robust/tasks-1.md')` until Task 7 removes both
- [ ] Create `src/tests/correctness/cases/rma_mv.toml` and `cases/robust.toml`, empty except a header comment

**Acceptance Criteria:**

- [ ] The existing `rma_mv` and `robust` tests pass unchanged
- [ ] `test_gate_mv_robust` is an expected failure that lists the new items; the core gate passes

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy` passes
- [ ] `pytest src/tests/correctness/test_coverage.py src/tests/test_rma_mv.py src/tests/test_rma_mv_extras.py src/tests/test_robust.py src/tests/test_robust_extras.py src/tests/test_audit8_crashes.py` passes

**Notes:**

- Depends on the `correctness-harness-checks` and `qe-no-intercept` lists of the order being merged to `main` (they edit `src/tests/correctness/`
  and `rma_uni`); this list starts from their merged state
- Key files: `src/metafopy/models/rma_mv.py` (the literals near `if method not in (*FIXED_METHODS, 'ML', 'REML')` and
  `dfs not in ('residual', 'contain')`), `src/metafopy/methods/robust.py` (`check_adjust`, `default_inference`)

---

## Task 2: Harness - the `then` follow-on call and the `vblocks` and `vconst` setups

**Status:** todo
**Priority:** high
**Complexity:** medium
**Spec Reference:** API (case-file additions); Error handling; Implementation step 2

**Technical Goal:** a case can call `robust` (or any follow-on function) on its fit, and can build `V` from block columns or a constant
within-cluster correlation, computed with numpy and pandas, never metafopy.

**Subtasks:**

- [ ] In `src/tests/correctness/cases.py`: the frozen dataclass `Step(function, args)`; `Case` gains `then: Step | None = None`; `then` joins
      `CASE_KEYS`; `run_case` applies it as `metafopy.<function>(result, **args)` and returns the follow-on result, which `expect` and `check` see
- [ ] `setup` accepts `escalc`, `vblocks` and `vconst` (at most one of the last two). `build_v(frame, setup, where)` returns the `V` matrix in data
      order: `vblocks = { cluster, columns }` is block-diagonal with block j the named columns of cluster j's rows; `vconst = { cluster, vi, rho }`
      has `vi` on the diagonal and `rho * sqrt(vi[i] * vi[l])` within a cluster. It is passed to the call as `V`, after any `escalc`
- [ ] Errors as in the spec's Error handling: `then` with keys other than `function` and `args`, both `vblocks` and `vconst`, either without
      `cluster`, raise `ValueError` naming the file and case id; an unknown follow-on function raises `LookupError` naming it; `build_v` raises
      `ValueError` naming the case when a `vblocks` cluster's row count differs from the number of `columns` or a block is not symmetric
- [ ] Inline-TOML tests in `src/tests/correctness/test_coverage.py`: a valid case of each new form (checked against a `V` written out by hand), and
      each malformed form above

**Acceptance Criteria:**

- [ ] A `then = { function = "robust", args = { cluster = "district" } }` case on an inline `rma_uni` fit returns the `robust` result
- [ ] Every error above is raised with the file and case id in its message

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy` passes
- [ ] `pytest src/tests/correctness/test_coverage.py` passes

**Notes:**

- Depends on Task 1
- `cases.py` keeps its rule: no `metafopy` call in building `V`; only the case's own call and its follow-on use metafopy

---

## Task 3: rma_mv - likelihood, GLS and the basic structures

**Status:** todo
**Priority:** high
**Complexity:** large
**Spec Reference:** API (`formulas.py`: `struct_matrix`, `mv_loglik`, `mv_fit`); Implementation steps 3 and 4; Coverage list (methods, random forms,
weights)

**Technical Goal:** the ML and REML likelihood, the GLS estimate and the structures `ID`, `CS`, `HCS`, `UN` and `DIAG` are coded from their sources,
and each `rma_mv.method.*`, `rma_mv.random.sigma2`, `rma_mv.random.nested`, `rma_mv.random.two_terms` and `rma_mv.weights` has a `derived` case.

**Subtasks:**

- [ ] `src/tests/correctness/formulas.py` (numpy and scipy only): the frozen dataclasses `RandomTerm` and `MVFit`; `struct_matrix` for `ID`, `CS`,
      `HCS`, `UN`, `DIAG` (Wolfinger 1993); `mv_loglik` (Harville 1977, the restricted likelihood); the GLS `b` and `vb = (X' M^-1 X)^-1`, with the
      weighted form for `W`; `mv_fit` (BFGS on log variances and atanh correlations, with the convergence check of the spec's Error handling: a fit
      with `success` false or gradient norm above `1e-6` fails the case naming the structure and the optimiser message)
- [ ] In `src/tests/correctness/test_derived.py`, a check function that reads the case's `random`, `struct`, `mods` and data, builds `X`, the grouping
      codes and the inner levels with `pandas.factorize` and a regular expression for `~ 1 | a`, `~ 1 | a/b`, `~ inner | outer`, `~ inner || outer`,
      and makes the spec's three comparisons: log-likelihood at metafopy's components (`rtol=1e-10`), `beta` and `se` from GLS at metafopy's
      components (`1e-10`), and `mv_fit`'s optimum (components `rtol=1e-6`, log-likelihood no more than `1e-8` above metafopy's)
- [ ] Derived cases in `cases/rma_mv.toml`: `EE`, `FE`, `CE`, `ML`, `REML`; `~ 1 | f` and `~ 1 | a/b` on `dat.konstantopoulos2011`; `UN` and `CS`
      and `HCS` and `DIAG` and `ID` on `dat.berkey1998` with `setup.vblocks`; a G and an H term together; a `W` case
- [ ] Record any mismatch as the header says

**Acceptance Criteria:**

- [ ] `test_gate_mv_robust`'s missing list no longer names `rma_mv.method.*`, these five structures, or the four named items above
- [ ] Each new `formulas.py` function names its source and equation in its docstring

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy` passes
- [ ] `pytest src/tests/correctness/test_derived.py src/tests/correctness/test_coverage.py` passes

**Notes:**

- Depends on Task 2
- Sources: Wolfinger 1993 (`Covariance structure selection in general mixed models.pdf`), Harville 1977 (`Maximum Likelihood Approaches to
  Variance Component Estimation and to Related Problems.pdf`), in the papers folder
- Read `src/metafopy/models/mv_structs.py` and `mv_lik.py` for option names and parametrisation only, never for formulas

---

## Task 4: rma_mv - serial, spatial and general structures, and `R`

**Status:** todo
**Priority:** high
**Complexity:** large
**Spec Reference:** API (`struct_matrix`); Implementation steps 3 and 4; Decisions (D2, `Rscale`); Coverage list (structures, `Rscale`,
`rma_mv.random.R`)

**Technical Goal:** every remaining structure in `STRUCTS`, the known correlation matrix `R` and each `Rscale` value has a `derived` case or, per D2,
an `[[exempt]]` entry.

**Subtasks:**

- [ ] `struct_matrix` gains `AR`, `HAR` (Wolfinger 1993), `CAR` (`rho ** |t_i - t_j|`), `SPEXP`, `SPGAU`, `SPLIN`, `SPRAT`, `SPSPH` from distances
      (Cressie 1993, section 2.3.1), `GEN` and `GDIAG` (Laird and Ware 1982). Spatial structures use metafor's documented parametrisation by range
      where it differs from Cressie's; the case's `source` says which, as a convention, not a mismatch
- [ ] Derived cases: `AR`, `HAR` and `CAR` on long-format inline data from `dat.ishak2007`; each spatial structure on a small inline dataset with
      coordinates; `GEN` and `GDIAG` as random coefficients
- [ ] `R` and `Rscale` on `dat.moura2021` with its tree (`dat.moura2021.tree`): the correlation matrix built in the test with numpy from the tree's
      branch lengths (Hadfield and Nakagawa 2010), each `RSCALES` value rescaled by the formula in metafor's `rma.mv` help page, the `source` naming
      the help page and saying the convention is metafor's (D2). Read the help page from the audit library, for example
      `R_LIBS=/Users/simonb/src/metafopy/.audit/Rlib Rscript -e 'tools::Rd2txt(tools::Rd_db("metafor")[["rma.mv.Rd"]])'`
- [ ] A `RSCALES` value whose rescaling the help page does not write down gets an `[[exempt]]` entry in `cases/rma_mv.toml` with the reason
      "defined only in metafor's code; R parity covers it". Apply D2 only to the `Rscale` values here; any other option met without a written formula
      stops the task as `blocked`, naming it
- [ ] Record any mismatch as the header says

**Acceptance Criteria:**

- [ ] `test_gate_mv_robust`'s missing list names no `rma_mv.struct.*`, no `rma_mv.Rscale.*` and not `rma_mv.random.R`

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy` passes
- [ ] `pytest src/tests/correctness/test_derived.py src/tests/correctness/test_coverage.py` passes

**Notes:**

- Depends on Task 3
- Sources in the papers folder: `Ishak_2007_Metaanalysis_of_longitudinal_studies.pdf`, `Statistics for Spatial Data.pdf`,
  `Laird_1982_Randomeffects_models_for_longitudinal_data.pdf` (a scan without text layer: read it as images), the Hadfield and Nakagawa (2010) and
  Nakagawa and Santos (2012) PDFs

---

## Task 5: rma_mv - inference, QE, QM, profile intervals and prediction intervals

**Status:** todo
**Priority:** high
**Complexity:** large
**Spec Reference:** Implementation steps 3 and 4; Decisions (D2, `knha`, `adhoc`, `contain`); Prerequisite; Coverage list (tests, dfs, `QE`, `QM`,
`confint.pl`, `confint.fixed`, `predict.pi`)

**Technical Goal:** each `rma_mv` `test` and `dfs` value, `QE`, `QM`, profile intervals of each kind of component, the fixed-effects intervals and the
prediction interval have `derived` cases.

**Subtasks:**

- [ ] `formulas.py`: the Wald `z` and `t` tests; the Knapp-Hartung factor `r' M^-1 r / (k - p)` and its ad hoc floor at 1, coded from metafor's
      `rma.mv` help page (the `rma_mv` form is metafor's generalisation of Knapp and Hartung 2003); containment degrees of freedom from the SAS/STAT
      PROC MIXED documentation of `DDFM=CONTAIN`; `QE` and `QM`; profile bounds (Hardy and Thompson 1996: the component where the likelihood ratio
      statistic equals the chi-square(1) quantile, solved with `scipy.optimize.brentq` over `mv_loglik` maximised in the other components)
- [ ] Per D2, each `knha`, `adhoc` and `contain` case's `source` names the documentation and says the convention is that software's
- [ ] Derived cases: each `rma_mv.TESTS` value and each `rma_mv.DFS` value; `QE` and `QM` on `dat.berkey1998`; `confint` profile intervals of a
      `sigma2`, a `tau2` and a `rho`; `confint` of the fixed effects; a `derived` `predict` interval case using the existing
      `test_derived.predict_pi` pattern
- [ ] Prerequisite check: if the `TODO.md` heading "The correctness harness cannot check exp(beta), prediction intervals or test-based tau2 and I2
      intervals" is in `DONE.md` and its `predict` check function exists in `src/tests/correctness/`, note it for Task 11's `published` prediction
      interval cases; if not, Task 13 files those cases as a `TODO.md` item
- [ ] Record any mismatch as the header says

**Acceptance Criteria:**

- [ ] `test_gate_mv_robust`'s missing list names no `rma_mv.test.*`, no `rma_mv.dfs.*` and none of `rma_mv.QE`, `rma_mv.QM`, `rma_mv.confint.pl`,
      `rma_mv.confint.fixed`, `rma_mv.predict.pi`

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy` passes
- [ ] `pytest src/tests/correctness/test_derived.py src/tests/correctness/test_coverage.py` passes

**Notes:**

- Depends on Task 3; the `predict` check function comes from the `correctness-harness-checks` list of the order
- Sources: `A LIKELIHOOD APPROACH TO META-ANALYSIS WITH RANDOM EFFECTS.pdf` and `Improved tests for a random effects meta-regression with a single
  covariate.pdf` (Knapp and Hartung 2003) in the papers folder; the SAS documentation is fetched in Task 10 if it is not read here

---

## Task 6: robust - the plain sandwich and its adjustments

**Status:** todo
**Priority:** high
**Complexity:** medium
**Spec Reference:** API (`cluster_robust_vcov`); Implementation steps 3 and 4; Decisions (D2, Stata adjustments); Coverage list (`robust.rma_uni`,
`robust.rma_mv`, `robust.adjust.*`)

**Technical Goal:** metafor's default `robust` on `rma_uni` and on `rma_mv`, and each `adjust` value, have `derived` cases.

**Subtasks:**

- [ ] `formulas.py`: `cluster_robust_vcov` for `CR0` and `CR1` (Hedges, Tipton and Johnson 2010, equations 6 and 8) and the Stata factors
      (Cameron and Miller 2015, section VI.B)
- [ ] A check function in `test_derived.py` that rebuilds `X`, the weights, residuals and cluster codes from the case and compares `vb`, `se`, the
      test statistics and `ddf` of the `robust` result reached through `then`
- [ ] Derived cases in `cases/robust.toml`: `robust` on an `rma_uni` fit (Sidik and Jonkman 2005) and on an `rma_mv` fit of
      `dat.tannersmith2016` with `setup.vconst`; each `robust.ADJUST` value. The Stata cases' `source` names the Stata manual and says the
      convention is Stata's (D2)
- [ ] Record any mismatch as the header says

**Acceptance Criteria:**

- [ ] `test_gate_mv_robust`'s missing list names neither `robust.rma_uni` nor `robust.rma_mv` and no `robust.adjust.*`

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy` passes
- [ ] `pytest src/tests/correctness/test_derived.py src/tests/correctness/test_coverage.py` passes

**Notes:**

- Depends on Task 2 (`then`, `vconst`) and Task 3 (the `rma_mv` design matrix helpers)
- Sources in the papers folder: `A Note on Variance Estimation in Random Effects Meta-Regression.pdf` (not the folder's other Sidik and Jonkman 2005
  paper), `Robust variance estimation in meta-regression with dependent effect size estimates.pdf`, `Cameron_2015_A_practitioners_guide_to_clusterrobust.pdf`

---

## Task 7: robust - club_sandwich estimators and tests, then close the gate

**Status:** todo
**Priority:** high
**Complexity:** large
**Spec Reference:** API (`cluster_robust_vcov`, `satterthwaite_dfs`, `hotelling_wald`, `saddlepoint_pvalue`); Implementation steps 3, 4 and 10;
Coverage list (`robust.vcov.*`, `robust.coef_test.*`, `robust.conf_test.*`, `robust.wald_test.*`)

**Technical Goal:** each `CR_TYPES`, `COEF_TESTS`, `CONF_TESTS` and `WALD_TESTS` value of `robust(club_sandwich=True)` has a `derived` case, and with
them the gate passes for every item.

**Subtasks:**

- [ ] `formulas.py`: `cluster_robust_vcov` gains `CR1p`, `CR1S`, `CR2` (Bell and McCaffrey 2002; Pustejovsky and Tipton 2018, equation 6) and `CR3`
      (MacKinnon and White 1985, the jackknife form); `satterthwaite_dfs` (Tipton 2015, equation 11; Pustejovsky and Tipton 2018, section 3);
      `hotelling_wald` (HTA, HTB, HTZ, EDF, EDT of Tipton and Pustejovsky 2015, section 3), plus the `chi-sq`, `Naive-F` and `Naive-Fp` forms;
      `saddlepoint_pvalue` (Kuonen 1999, equations 2 to 4). Never import `metafopy.core.clubsandwich`
- [ ] Derived cases in `cases/robust.toml` for each value, on `dat.tannersmith2016` (`vconst`) and one `rma_uni` fit
- [ ] Record any mismatch as the header says
- [ ] Remove `PENDING_PREFIXES` and `test_gate_mv_robust` from `test_coverage.py`, so the one gate covers every item. If any item is still missing,
      stop as `blocked`, naming it

**Acceptance Criteria:**

- [ ] The gate passes with no xfail mark and every new item in its list

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy` passes
- [ ] `pytest src/tests/correctness/test_derived.py src/tests/correctness/test_coverage.py` passes

**Notes:**

- Depends on Tasks 3 to 6
- Sources in the papers folder: `Small Sample Adjustments for Robust Variance Estimation With Meta-Regression.pdf`, `Small-sample adjustments for
  tests of moderators and model fit using robust variance estimation in meta-regression.pdf` (the JEBS version; not the similarly named file
  without "using"), `Pustejovsky_2018_Smallsample_methods_for_clusterrobust_variance_preprint.pdf`,
  `MacKinnon_1985_Some_heteroskedasticityconsistent_covariance_matrix_estimators_preprint.pdf`, `Saddlepoint approximations for distributions of
  quadratic forms in normal variables.pdf`. Bell and McCaffrey (2002) is open at Statistics Canada; fetch it with the connected tools or list it in
  Task 10
- `clubSandwich` agreement is parity, not evidence; do not use it

---

## Task 8: Property and simulation cases

**Status:** todo
**Priority:** medium
**Complexity:** medium
**Spec Reference:** Implementation steps 5 and 6

**Technical Goal:** the invariances, limiting cases and exact results the spec names are checked, the simulations behind the `slow` marker.

**Subtasks:**

- [ ] Property cases in `src/tests/correctness/test_properties.py` and `cases/properties.toml`: `rma_mv` with `~ 1 | id` (one row per id) equals
      `rma_uni` (beta, se, tau2, log-likelihood); a three-level fit with one level's sigma2 fixed at 0 equals the two-level fit; with two inner levels
      `UN` and `HCS` give the same fit; `AR` equals `CAR` at equally spaced times; location and scale invariance; the REML and ML score is zero at
      metafopy's optimum; `optbeta=True` reaches the same optimum; `CR0` with one row per cluster equals the HC0 sandwich; `CR2` is exactly unbiased
      when the working model is the true one (Pustejovsky and Tipton 2018, theorem 1), checked in closed form from `(I - H) M (I - H)'`
- [ ] Simulation cases in `src/tests/correctness/test_simulation.py`, each marked `slow`, using `numpy.random.default_rng(seed)` with a fixed seed and
      stated replicates, asserting at 4 standard errors: exact coverage of the `method="FE"` Wald interval with known `V`; unbiasedness of the GLS
      estimate
- [ ] Check once by hand that each simulation fails for a biased stand-in (for example `V` scaled by 2 in the interval), and note the figures here

**Acceptance Criteria:**

- [ ] `pytest -m slow src/tests/correctness/test_simulation.py` passes

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy` passes
- [ ] `pytest src/tests/correctness/test_properties.py src/tests/correctness/test_simulation.py` passes, `slow` cases included

**Notes:**

- Depends on Tasks 3 and 7

---

## Task 9: Other-software cases from mvmeta, nlme, robumeta and sandwich

**Status:** todo
**Priority:** low
**Complexity:** medium
**Spec Reference:** Implementation step 9; External dependencies; Error handling (generator)

**Technical Goal:** values from R packages that do not use metafor, generated by a pinned script and committed; tests never run R.

**Subtasks:**

- [ ] Check that `/Users/simonb/src/metafopy/.audit/Rlib` holds `mvmeta` 1.0.3, `nlme` 3.1-171, `robumeta` 2.1 and `sandwich` 3.1-3 (the versions in
      `src/audit/data/r-versions.json`); stop as `blocked`, naming what is missing, if not. Install nothing
- [ ] Write `src/tests/correctness/r/other_software_mv.R`, modelled on `r/other_software.R`: it reads `METAFOPY_CORRECTNESS_RLIB` and prepends it with
      `.libPaths(c(lib, .libPaths()))`; it pins each package version at its top and stops on any other; it stops unless
      `tools::package_dependencies(c("mvmeta", "nlme", "robumeta", "sandwich"), db = installed.packages(), recursive = TRUE)` shows none depends on
      metafor; it writes `cases/other_mv.toml` with `kind = "other"`, each case's `source` naming the package's and R's versions
- [ ] Cases: `mvmeta` (`UN` on `dat.berkey1998`); `nlme` (`lme` with `varFixed(~ vi)` and `sigma` fixed at 1, the three-level fit); `robumeta`
      (correlated-effects RVE with Tipton's correction on `dat.tannersmith2016`, compared with `robust(club_sandwich=True)` given robumeta's weights as
      `W`); `sandwich::vcovCL` (CR0, CR1 and the Stata factors on a weighted `lm`)
- [ ] Run it as `METAFOPY_CORRECTNESS_RLIB=/Users/simonb/.local/share/metafopy/r-correctness-lib R_LIBS=/Users/simonb/src/metafopy/.audit/Rlib LC_ALL=C.UTF-8 Rscript src/tests/correctness/r/other_software_mv.R`
      and commit the output
- [ ] Check `git diff --exit-code src/audit/data/r-versions.json`
- [ ] Record any mismatch as the header says; a difference that is a documented convention is pinned in the generator with a comment, as Task 12 of
      the correctness-tests list did

**Acceptance Criteria:**

- [ ] Every case in `cases/other_mv.toml` names its package's version
- [ ] `src/tests/correctness/test_other_software.py` runs the new cases (extend its parametrisation if it reads only `other.toml`)

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy` passes
- [ ] `pytest src/tests/correctness/test_other_software.py` passes

**Notes:**

- Depends on Tasks 2 and 7
- Prerequisite outside this repository: R (`/opt/homebrew/bin/Rscript`) and the audit's pinned library; checked present on 2026-11-14
- `clubSandwich` 0.7.0 is pinned too but not used: `metafopy.core.clubsandwich` ports it, so agreement is parity

---

## Task 10: Sources - fetch the open ones and stop for any still needed

**Status:** todo
**Priority:** high
**Complexity:** small
**Spec Reference:** Seed sources; Implementation step 7

**Technical Goal:** every seed source is read or listed as needed. The list stops here for the operator only if some source is still needed.

**Subtasks:**

- [ ] Check that `/Users/simonb/src/metafopy papers` can be read; stop as `blocked` if not
- [ ] For each row of the spec's Seed sources, find its file through `download_report.md` in that directory. Fetch the open-access rows not in it
      (Assink and Wibbelink 2016, Jackson, Riley and White 2011 at PMC3470931, the SAS/STAT PROC MIXED `DDFM=CONTAIN` page, Bell and McCaffrey 2002)
      with the connected PubMed, Paperclip or alphaXiv tools or the web, reading them without saving them in the repository
- [ ] If every source is present or fetched, write that under "Sources needed" in `agents/specs/correctness-mv-robust/mismatches.md` ("none, checked
      <date>"), commit, and continue to Task 11 without stopping
- [ ] Otherwise list each missing source under "Sources needed" with its citation and the table or example wanted, and options *supplied* and *drop*;
      publish the page for those items per the shared `reviews.md` rule (answers saved to the artifact's database) with `**Review page:** <URL>` in
      the document's header. If this session has no tool to publish the page, say so in the stop message, for the operator session to publish it
- [ ] Stop the list for the operator. Once every source item has an answer, an operator session reads the page's answers, copies each into the
      Answers section of `mismatches.md` under "Sources", with the date, commits that, and only then restarts the list

**Acceptance Criteria:**

- [ ] Every seed source is read, or listed under "Sources needed" with what is wanted

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy` passes
- [ ] `pytest src/tests/correctness -m 'not slow'` passes

**Notes:**

- Depends on Tasks 3 to 7 (they may already have read some sources)
- The user supplied the paywalled papers before the order started (review D5, 2026-11-13; `download_report.md` lists them as supplied
  2026-11-14), so this task is expected to pass through

---

## Task 11: Published cases

**Status:** todo
**Priority:** high
**Complexity:** large
**Spec Reference:** Seed sources; Implementation step 8; Kinds of evidence (`published`)

**Technical Goal:** each source with a worked example gives `published` cases, with value, locator and date read.

**Subtasks:**

- [ ] If "Sources needed" in `agents/specs/correctness-mv-robust/mismatches.md` lists any source, read its answers from the Answers section of that
      document, never from the page; stop as `blocked`, naming the missing answers, if that section has no answer for each listed source
- [ ] One source at a time (Berkey et al. 1998, Konstantopoulos 2011, Assink and Wibbelink 2016, Kalaian and Raudenbush 1996, Ishak et al. 2007,
      van Houwelingen et al. 2002, Gleser and Olkin 2009 in the Handbook PDF, Moura et al. via Nakagawa and Santos 2012 where its example applies,
      Sidik and Jonkman 2005, Hedges, Tipton and Johnson 2010, Tipton 2015, Tipton and Pustejovsky 2015, Tanner-Smith et al. 2016): find its worked
      example, the dataset that holds the same data (or inline data with its locator), and add `published` cases with values as printed, as strings,
      each with `citation`, `locator` and `read`
- [ ] `published` prediction-interval cases only if Task 5 found the harness `predict` check function; otherwise leave them for Task 13
- [ ] A source with no usable worked example: note it in the case file's header comment, with the reason
- [ ] Record any mismatch as the header says

**Acceptance Criteria:**

- [ ] Every source with a worked example has `published` cases; every `published` case has `citation`, `locator` and `read`

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy` passes
- [ ] `pytest src/tests/correctness/test_published.py` passes

**Notes:**

- Depends on Task 10
- Prefer the table with the most digits; a value printed with few digits gives a loose tolerance
- `Konstantopoulos_2011_...preprint.pdf` and `Pustejovsky_2018_..._preprint.pdf` are preprints: say so in the `locator`

---

## Task 12: Review the mismatches

**Status:** todo
**Priority:** high
**Complexity:** small
**Spec Reference:** Implementation step 11

**Technical Goal:** the user decides each mismatch. The list stops here until every item has an answer.

**Subtasks:**

- [ ] Complete `agents/specs/correctness-mv-robust/mismatches.md` in the review-document layout of the shared `reviews.md` rule (read the rule from
      disk first): what is reviewed and how; automatic checks (the gate passes, every `mismatch` case has an item); each mismatch with source,
      locator, reference, metafopy and metafor values; decisions with options *follow the reference*, *keep metafor* and *investigate*, each with a
      recommendation; an empty Answers section under "Mismatches"
- [ ] Publish its interactive page, with answers saved to the artifact's database and a button accepting every recommendation, and put
      `**Review page:** <URL>` in the document's header. If this session has no tool to publish the page, say so in the stop message, for the
      operator session to publish it
- [ ] If there are no mismatches, write that in the document, skip the page, and continue to Task 13 without stopping
- [ ] Otherwise stop the list for the operator. Once every item has an answer, an operator session reads the page's answers, copies each into the
      Answers section of `mismatches.md` under "Mismatches", with the date, commits that, and only then restarts the list

**Acceptance Criteria:**

- [ ] Every `mismatch` case in `src/tests/correctness/cases/*.toml` from this list has an item in the document and on the page

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy` passes
- [ ] `pytest src/tests/correctness -m 'not slow'` passes

**Notes:**

- Depends on Tasks 3 to 11

---

## Task 13: File the answers and close the spec

**Status:** todo
**Priority:** high
**Complexity:** small
**Spec Reference:** Implementation step 12

**Technical Goal:** each answer becomes recorded work, and the spec is marked implemented. Fixing metafopy is a follow-up change, not part of this
list.

**Subtasks:**

- [ ] Read the answers from the Answers section of `agents/specs/correctness-mv-robust/mismatches.md`, never from the page. If Task 12 recorded
      mismatches, stop as `blocked`, naming the missing items, if that section is empty or incomplete
- [ ] *Keep metafor*: correct the case to the convention metafor documents, cite where it documents it, and set `status = "pass"`
- [ ] *Follow the reference*: add a `TODO.md` item for each with the `todo-add` skill, naming the case, the fix, the case to flip, the R parity
      fixtures and `src/audit/data/known.toml` entries to mark, and the `docs/differences.md` section "Where metafor departs from its reference" to add
- [ ] *Investigate*: a `TODO.md` item with the `todo-add` skill naming what is in doubt
- [ ] If Task 5 found no harness `predict` check function, a `TODO.md` item with the `todo-add` skill for the `published` cases of
      `rma_mv.predict.pi`
- [ ] Set the spec's `**Status:**` to `implemented`, and its row in `agents/specs/index.md` to say so (leave the umbrella `TODO.md` item "Extend the
      correctness tests beyond the core models" where it is; the publication-bias list moves it)
- [ ] Run the final test selection from the worktree:
      `main=$(git worktree list --porcelain | sed -n '1s/^worktree //p'); mkdir -p "$main/ralpher-logs/correctness-mv-robust/tasks-1" && pytest src/tests/correctness src/tests/test_rma_mv.py src/tests/test_rma_mv_extras.py src/tests/test_robust.py src/tests/test_robust_extras.py src/tests/test_audit8_crashes.py 2>&1 | tee "$main/ralpher-logs/correctness-mv-robust/tasks-1/final-tests.log"`
      (`slow` cases included), judged by its exit status

**Acceptance Criteria:**

- [ ] Every answered item has a case change or a `TODO.md` item
- [ ] `src/audit/data/r-versions.json` is unchanged from `main` (`git diff --exit-code main -- src/audit/data/r-versions.json`)

**Definition of Done:**

- [ ] `pre-commit run --all-files` passes
- [ ] `pre-commit run auditme-all --hook-stage manual --all-files` passes
- [ ] `mypy` passes
- [ ] `pytest src/tests/correctness src/tests/test_rma_mv.py src/tests/test_rma_mv_extras.py src/tests/test_robust.py src/tests/test_robust_extras.py src/tests/test_audit8_crashes.py` passes, `slow` cases included (the last subtask's run)

**Notes:**

- Depends on Task 12
- The modules this list changes under `src/metafopy/` are `models/rma_mv.py` and `methods/robust.py` (Task 1); the selection is their own tests,
  the test file importing them directly, and the whole correctness suite
