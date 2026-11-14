# Tasks: Correctness tests: publication bias

**Spec:** `agents/specs/correctness-publication-bias/spec.md`
**Created:** 2026-11-14
**Version:** 1

The spec (version 1.1) was approved on 2026-11-13 in `agents/reviews/order-correctness-follow-ups-specs.md`, whose Answers settle the two decisions
recorded under the spec's "Decisions": options that metafor's help documents get `derived` cases, code-only ones get `[[exempt]]` entries (D2 in that
review), and R `weightr` 2.0.2 is installed beside R `meta` 8.5-0 in `/Users/simonb/.local/share/metafopy/r-correctness-lib` (D4). This list is the
last row of the run order `agents/tasks/order-correctness-follow-ups.md`, run by `/rro` one list at a time in a worktree, each merged to `main` before
the next. It builds on the lists merged before it: `correctness-harness-checks` (the `check` functions for a back-transformed estimate, `predict`
intervals and `confint` I2 and H2 bounds in `src/tests/correctness/test_derived.py`) and `correctness-mv-robust`, `correctness-glmm` and
`correctness-location-scale` (which extend `src/tests/correctness/cases.py`, `test_coverage.py` and `formulas.py`; build on them, never revert them).

This list changes no module under `src/metafopy/`: it adds tests, case files, formulas and a generator under `src/tests/correctness/`, writes
`agents/specs/correctness-publication-bias/mismatches.md`, and closes the spec. The project has no affected-tests tool, so each task names its own test
files, and the last task runs the whole correctness suite, `slow` cases included.

**Recording a mismatch (every task that writes cases):** when a case's reference disagrees with metafopy, commit it with `status = "mismatch"` and a
`reason` naming both values, and append an item to `agents/specs/correctness-publication-bias/mismatches.md`, creating the file from the
review-document layout in Task 13 if it does not exist yet. The item gives the source, locator, reference value, metafopy's value, and metafor's value
from the R parity fixture under `src/tests/reference/` where one exists. Never change metafopy's output to match a reference in this list. A
reference that cannot be read is listed under "Sources needed" in the same file, and its case is not written.

**Sources:** PDFs in `/Users/simonb/src/metafopy papers`, outside the repository (`download_report.md` there lists the files and which seed source each
one is). Read them with `pdftotext` (`/opt/homebrew/bin/pdftotext`); copy only the numbers used, never the files or their text, into the repository.
If that directory cannot be read from the queue's session, the task stops as `blocked` and says so. Before writing a case, read the value and
locator from the source and confirm that its data match the metafopy dataset; a source that does not reproduce its data gives no case, and says so
in its "Sources needed" row. As of 2026-11-14 every seed source is in the folder or open, except Citkowicz and Vevea (2017), doi:10.1037/met0000119:
the folder holds only Citkowicz's 2012 dissertation (`Citkowicz_dissertation_A_parsimonious_weight_function_for.pdf`).

**Secrets:** Never list environment variables (`env`, `printenv`, `set`) or print an env file. To test that a variable is set, use
`[ -n "${VAR:-}" ]` and print only yes or no. To read one from an env file, search for its exact name (`grep -n '^export NAME='`), or list names
without values (`sed -n 's/^export \([A-Z_]*\)=.*/\1/p'`). Never match a generic token (`export`, `=`, `KEY`) in a file that may hold secrets.

**Caveat:** spec unclear in two places, decided here and marked on the tasks they affect:

- `metafopy.models.sel_spec.TYPE_OPTIONS` also holds `negexppow` and `stepcon`, which the spec's Seed sources and Decisions do not name. The decision
  on options defined only by metafor (D2 in the review) names its options, so it is not extended to these two: Task 6 gives each a temporary
  `[[exempt]]` entry reading "pending the user's answer in mismatches.md", lists both as decisions in `mismatches.md`, and Task 14 applies the answer.
- The spec's Error handling lists the `setup` keys as `where`, `drop`, `escalc`, `rma_uni`; `correctness-mv-robust`, merged earlier, adds its own
  `setup` keys (`vblocks`, `vconst`). Task 1 adds this spec's keys to the accepted set and keeps every key already accepted.

---

## Task 1: Harness - `setup` filters and fitted models, `literal_options` and the bias coverage items

**Status:** todo
**Priority:** high
**Complexity:** medium
**Spec Reference:** Module location; API; Coverage list; Error handling; Implementation step 1

**Technical Goal:** cases can filter a dataset, drop rows and pass a fitted `rma_uni` model to the function under test, and the coverage gate lists
every publication-bias item still lacking a `published` or `derived` case, read from the option lists in the code.

**Subtasks:**

- [ ] Starting-point check: `grep -rnwE 'literal_options|kendall_z|trimfill_filled|selmodel_fit|hc_interval|selection_loglik' src/tests/correctness`
      prints nothing, and `ls src/tests/correctness/cases/bias.toml src/tests/correctness/cases/selmodel.toml` finds neither file. Stop as `blocked`,
      naming what was found, otherwise
- [ ] `cases.py`: `setup` accepts `where` (a table of column = value; keep the rows equal in every column), `drop` (a list of 0-based row indices)
      and `rma_uni` (arguments of `metafopy.rma_uni(yi='yi', vi='vi', data=frame, **rma_uni)`), beside `escalc` and every key the earlier lists
      added. `case_frame` applies `where`, then `drop`, then `escalc`; `run_case` with `rma_uni` fits the model and calls
      `metafopy.<function>(fit, **args)` without `data`, as in the spec's API
- [ ] Validation as in Error handling: an unknown `setup` key, a `where` column the dataset lacks, or a `drop` index out of range raises
      `ValueError` naming file and case
- [ ] `test_coverage.py`: `literal_options(func, parameter)` returns the string values of the `Literal` in the parameter's annotation, through PEP 695
      `type` aliases (`Predictor`, `FsnType` are `type` aliases, so their values are `typing.get_args(alias.__value__)`) and `| None`; any other
      annotation raises `TypeError` naming function and parameter
- [ ] Add the spec's Coverage list items to `coverage_items()`: `regtest.<model>.<predictor>`, `regtest.limit`, `ranktest.*`, `trimfill.*`,
      `fsn.<type>`, `fsn.pool.<pool>`, `tes.*`, `hc.estimate`, `hc.ci`, `selmodel.<type>` over `metafopy.models.sel_spec.TYPE_OPTIONS`,
      `selmodel.alternative.<a>` over `metafopy.models.selmodel.ALTERNATIVES`, `selmodel.prec.<p>` over `metafopy.models.selmodel.PRECISIONS`,
      `selmodel.fixed_delta`, `selmodel.decreasing`, `selmodel.LRT`. Read module constants with `importlib.import_module` (the package re-exports
      `regtest` the function over its submodule)
- [ ] The gate: the existing gate test keeps checking every item outside this spec. A new test `test_every_bias_item_has_a_case` checks the new
      items, marked `xfail(strict=True, reason='coverage incomplete; see agents/tasks/correctness-publication-bias/tasks-1.md')` until Task 6
      removes the mark
- [ ] Create `cases/bias.toml` and `cases/selmodel.toml`, empty except a header comment
- [ ] Unit tests of the harness on inline TOML strings and inline frames: each new `ValueError`; `where` and `drop` keep the right rows;
      `literal_options` through a `type` alias, through `| None`, and the `TypeError` on a non-`Literal` annotation; `run_case` with
      `setup.rma_uni` passes the fitted model as the first positional argument and no `data` (checked on `dat.hackshaw1998` with `trimfill`)

**Acceptance Criteria:**

- [ ] `test_every_bias_item_has_a_case` is an expected failure listing every new item
- [ ] Every other test in `src/tests/correctness` passes, the existing gate included
- [ ] A `setup` with an unknown key, an unknown `where` column or an out-of-range `drop` index fails naming file and case

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy` passes
- [ ] `pytest src/tests/correctness/test_coverage.py src/tests/correctness/test_derived.py src/tests/correctness/test_properties.py` passes

**Notes:**

- Depends on the `correctness-mv-robust` list in the run order (its `setup` keys and `then` key in `cases.py`), and on `correctness-glmm` and
  `correctness-location-scale` (their coverage items in `test_coverage.py`)
- **Caveat:** spec unclear on the accepted `setup` keys (see the header); keep every key already accepted
- Option lists checked 2026-11-14: `regtest(model: Literal['lm', 'rma'], predictor: Predictor)`, `trimfill(estimator: Literal['L0', 'R0', 'Q0'])`,
  `fsn(type: FsnType | None, pool: Literal['stouffer', 'fisher'])`, `tes(test: Literal['chi2', 'binom', 'exact'] | None)`

---

## Task 2: Derived cases - `regtest` and `ranktest`

**Status:** todo
**Priority:** high
**Complexity:** medium
**Spec Reference:** API (`kendall_z`); Coverage list (regtest, ranktest); Implementation step 2

**Technical Goal:** every `regtest` model and predictor, the limit estimate, and the rank test's `tau` and both p-values have `derived` cases from
formulas coded from the publications.

**Subtasks:**

- [ ] `formulas.py`, each function citing its source and equation:
  - Egger's regression in its original form (standard normal deviate on precision, ordinary least squares; Egger et al. 1997), so it checks
    metafopy's weighted form independently;
  - the weighted regression with each predictor as published (`sei`, `vi`: Moreno et al. 2009; `ni`, `ninv`: Macaskill et al. 2001, Peters et al.
    2006; `sqrtninv`: Pustejovsky and Rodgers 2019), for `model='lm'` and the random-effects `model='rma'` (Sterne and Egger 2005);
  - the limit estimate and its interval (Stanley and Doucouliagos 2014; Moreno et al. 2009);
  - Begg and Mazumdar's standardised deviates, Kendall's tau, its exact and normal p-values (Begg and Mazumdar 1994).
- [ ] `test_derived.py`: `kendall_z`, the normal deviate from `tau` and k, naming case, quantity and both values
- [ ] `derived` cases in `cases/bias.toml`, one or more per item, on `dat.egger2001` (`setup.drop` leaves out ISIS-4 as Egger's example does) and
      `dat.hackshaw1998`. Tolerances: closed forms `1e-10`, iterative (`model='rma'`) `1e-8`
- [ ] Record any mismatch as the header says

**Acceptance Criteria:**

- [ ] The bias gate's missing list names no `regtest.*` or `ranktest.*` item
- [ ] Each new `formulas.py` function names its source and equation

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy` passes
- [ ] `pytest src/tests/correctness/test_derived.py src/tests/correctness/test_coverage.py` passes

**Notes:**

- Depends on Task 1
- `sqrtni` and the other predictors are covariates in the same published regression test, not methods of their own, so their cases derive from
  that regression with the covariate metafor's `regtest` help defines; the case's `source` names the regression's paper and the help page
- Key files (read for option names only, not for formulas): `src/metafopy/bias/regtest.py`, `src/metafopy/bias/ranktest.py`

---

## Task 3: Derived cases - `trimfill`

**Status:** todo
**Priority:** high
**Complexity:** medium
**Spec Reference:** API (`trimfill_filled`); Coverage list (trimfill); Implementation steps 2 and 3 (`p_k0`)

**Technical Goal:** each estimator's `k0` and `se_k0`, the R0 test, the side chosen and the filled studies with the refitted estimate have
`derived` cases.

**Subtasks:**

- [ ] `formulas.py`: the L0 and R0 estimators and their standard errors (Duval and Tweedie 2000, Biometrics and JASA), Q0 (Duval 2005), the R0 null
      distribution (Duval and Tweedie 2000, JASA), the side rule, and the filling of mirrored studies
- [ ] `test_derived.py`: `trimfill_filled`, comparing the filled effects and variances order-free
- [ ] `derived` cases with `setup.rma_uni` on `dat.hackshaw1998` and one dataset where the missing studies are on the other side; items
      `trimfill.k0.<estimator>`, `trimfill.se_k0.<estimator>`, `trimfill.p_k0`, `trimfill.side`, `trimfill.fill`
- [ ] Check `p_k0` against the R0 null distribution read from the JASA paper; the code (as read 2026-11-14,
      `src/metafopy/bias/trimfill.py`) computes it as `1 - sum(comb(m + 1, m + 1) * 0.5 ** (m + 2) for m in range(-1, k0))`, which is
      `0.5 ** (k0 + 1)`. Any disagreement is a mismatch
- [ ] Record any mismatch as the header says

**Acceptance Criteria:**

- [ ] The bias gate's missing list names no `trimfill.*` item

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy` passes
- [ ] `pytest src/tests/correctness/test_derived.py src/tests/correctness/test_coverage.py` passes

**Notes:**

- Depends on Task 1

---

## Task 4: Derived cases - `fsn`, `tes` and `hc`

**Status:** todo
**Priority:** high
**Complexity:** large
**Spec Reference:** API (`hc_interval`); Coverage list (fsn, tes, hc); Decisions; Implementation step 2

**Technical Goal:** every fail-safe N type and pool, the excess-significance tests, powers and limit, and Henmi and Copas's estimate and interval have
`derived` cases or the exemptions the spec's Decisions give.

**Subtasks:**

- [ ] `formulas.py`: Rosenthal (Stouffer, Rosenthal 1979), Fisher pooling (Becker 2005), Orwin (1983), Rosenberg (2005), the General fail-safe N
      (Viechtbauer 2024 talk); TES powers, the chi-square, binomial and exact (Poisson-binomial) tests (Ioannidis and Trikalinos 2007); `theta_lim`
      from metafor's `?tes` help (5.2-1); Henmi and Copas's interval by their integral (Henmi and Copas 2010)
- [ ] `test_derived.py`: `hc_interval`
- [ ] `derived` cases for `fsn.<type>` (not `Binomial`), `fsn.pool.<pool>`, `tes.test.<test>`, `tes.E`, `tes.tau2`, `tes.theta_lim` (its `source`
      names the help page) and `hc.estimate`, `hc.ci` (`setup.rma_uni`, on `dat.lee2004`)
- [ ] `[[exempt]]` entry for `fsn.Binomial`: "defined only in metafor's code; R parity covers it"
- [ ] Record any mismatch as the header says

**Acceptance Criteria:**

- [ ] The bias gate's missing list names no `fsn.*`, `tes.*` or `hc.*` item

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy` passes
- [ ] `pytest src/tests/correctness/test_derived.py src/tests/correctness/test_coverage.py` passes

**Notes:**

- Depends on Task 1
- The General fail-safe N talk is open at https://www.wvbauer.com/lib/exe/fetch.php/talks:2024_viechtbauer_srsm_fail_safe_n.pdf; fetch it with
  WebFetch, or stop as `blocked` naming it if the session cannot

---

## Task 5: Derived cases - `selmodel` step functions and the selection likelihood

**Status:** todo
**Priority:** high
**Complexity:** large
**Spec Reference:** API (`selmodel_fit`); Coverage list (selmodel); Implementation step 2

**Technical Goal:** one general selection-model log-likelihood, maximised independently of metafopy, checks the step-function models and the options
common to every `selmodel` type.

**Subtasks:**

- [ ] `formulas.py`: `selection_loglik`, one log-likelihood taking a weight function (Iyengar and Greenhouse 1988; Vevea and Hedges 1995), and the
      step weight function (Hedges 1992; Hedges and Vevea 2005)
- [ ] `test_derived.py`: `selmodel_fit`, maximising `formulas.selection_loglik` with `scipy.optimize` from metafopy's start values and comparing
      `beta`, `tau2`, `delta` and `ll`; it fails the case, naming the case and the optimiser's message, if the optimiser does not converge
- [ ] `derived` cases in `cases/selmodel.toml` (`setup.rma_uni`) for `selmodel.stepfun`, `selmodel.fixed_delta` (a priori weights, Vevea and Woods
      2005, on `dat.cohen1981`), `selmodel.decreasing`, `selmodel.LRT`, `selmodel.alternative.<a>` and `selmodel.prec.<p>`. Each case states its own
      tolerance for `beta`, `tau2` and `delta`; `ll` at `1e-8`
- [ ] Record any mismatch as the header says

**Acceptance Criteria:**

- [ ] The bias gate's missing list names none of the items above
- [ ] A non-converging optimiser fails its case with the message (a unit test with `maxiter=1`)

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy` passes
- [ ] `pytest src/tests/correctness/test_derived.py src/tests/correctness/test_coverage.py` passes

**Notes:**

- Depends on Task 1
- Key files (option names only): `src/metafopy/models/selmodel.py`, `src/metafopy/models/sel_spec.py`

---

## Task 6: Derived cases - parametric and truncation `selmodel` types, then close the gate

**Status:** todo
**Priority:** high
**Complexity:** large
**Spec Reference:** Coverage list (selmodel types); Decisions; Implementation step 2

**Technical Goal:** each parametric and truncation selection model has a `derived` case or an exemption, and with them the bias gate passes.

**Subtasks:**

- [ ] `formulas.py`: the weight functions `beta` (Citkowicz and Vevea 2017, as the dissertation and Pustejovsky's 2024 post state it), `halfnorm`,
      `negexp`, `logistic`, `power` (Preston et al. 2004), and `trunc`, `truncest` from metafor's `?selmodel` help (5.2-1)
- [ ] `derived` cases through `selmodel_fit` for those types; `beta` on a dataset with a well-defined optimum (`docs/differences.md` records a flat
      likelihood on `dat.baskerville2012`)
- [ ] `[[exempt]]` entries for `halfnorm1`, `negexp1`, `logistic1`, `power1`, `halfnorm2`, `negexp2`, `logistic2`, `power2`: "defined only in
      metafor's code; R parity covers it"
- [ ] `negexppow` and `stepcon`: a temporary `[[exempt]]` entry for each reading "pending the user's answer in mismatches.md", and a decision item for
      each in `mismatches.md` (options: derive from metafor's help, exempt as code-only, or name a source)
- [ ] Record any mismatch as the header says
- [ ] Remove the `xfail` mark from `test_every_bias_item_has_a_case`

**Acceptance Criteria:**

- [ ] The bias gate passes with no xfail mark

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy` passes
- [ ] `pytest src/tests/correctness/test_derived.py src/tests/correctness/test_coverage.py` passes

**Notes:**

- Depends on Task 5
- **Caveat:** spec unclear on `negexppow` and `stepcon` (see the header); the spec's decisions name neither, so the user decides them in Task 13

---

## Task 7: Published cases - tests, trim and fill, fail-safe N, TES and Henmi-Copas

**Status:** todo
**Priority:** high
**Complexity:** large
**Spec Reference:** Seed sources; Implementation step 3

**Technical Goal:** each seed source for `regtest`, `ranktest`, `trimfill`, `fsn`, `tes` and `hc` gives `published` cases, value, locator and date
read.

**Subtasks:**

- [ ] Check the prerequisite: `grep -n 'def ' src/tests/correctness/test_derived.py` lists the back-transform check that the
      `correctness-harness-checks` list adds. Stop as `blocked`, naming that list, if it does not
- [ ] Borenstein et al. (2009) ch. 30 on `dat.hackshaw1998` (Rosenthal 398, Orwin 103, Begg, Egger, trim and fill; pooled ratios through the
      back-transform check) and Sutton (2009) ch. 23 on `dat.raudenbush1985`, from the folder
- [ ] Egger et al. (1997, PMC2127453) on `dat.egger2001`, Moreno et al. (2009, PMC2649158), and then every other seed source of these functions in the
      folder, one source at a time
- [ ] A source using another convention (a two-tailed alpha for Rosenthal; Peters' weights) sets the case's `args` to it where metafopy offers it and
      its `source` says so; where metafopy cannot express it, it is a mismatch item
- [ ] Values written as printed, as strings, in `cases/published.toml` or `cases/bias.toml`, each with `citation`, `locator` and `read`
- [ ] A source that cannot be read or does not reproduce its data goes under "Sources needed" in `mismatches.md`
- [ ] Record any mismatch as the header says

**Acceptance Criteria:**

- [ ] Every seed source of these functions has `published` cases or a "Sources needed" row
- [ ] Every `published` case has `citation`, `locator` and `read`

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy` passes
- [ ] `pytest src/tests/correctness/test_published.py src/tests/correctness/test_coverage.py` passes

**Notes:**

- Depends on Tasks 2 to 4, and on the `correctness-harness-checks` list in the run order (the back-transform check)
- A value printed with few digits gives a loose tolerance; prefer the table with the most digits

---

## Task 8: Stop for the operator - sources still missing

**Status:** todo
**Priority:** high
**Complexity:** small
**Spec Reference:** Seed sources; Implementation step 7 (first stop)

**Technical Goal:** the `selmodel` published cases have every source they need, or the operator has dropped it. The task passes through when
nothing is missing.

**Subtasks:**

- [ ] For each `selmodel` seed source (Iyengar and Greenhouse 1988, Hedges 1992, Vevea and Hedges 1995, Vevea and Woods 2005, Hedges and Vevea 2005,
      Citkowicz and Vevea 2017, Preston et al. 2004, Rust et al. 1990, Formann 2008, Lehmann et al. 2018), check from
      `/Users/simonb/src/metafopy papers/download_report.md` and the folder that the file is present, or that it is open (PMC). A file holding a
      different work (the 2012 dissertation for Citkowicz and Vevea 2017) does not count
- [ ] Add to "Sources needed" in `mismatches.md` every source still missing (and every row Task 7 added), with its citation, DOI and the table or
      example wanted, and a `Status` column (`needed`, or `dropped <date>` once the operator drops it)
- [ ] If no `selmodel` source is missing, or each missing one is marked `dropped`, mark this task done and go on
- [ ] Otherwise stop the list for the operator, naming each missing source. The operator saves each file in `/Users/simonb/src/metafopy papers`
      (outside the repository) or marks its row `dropped <date>` in `mismatches.md`, commits that, and only then restarts the list, which re-runs
      this task

**Acceptance Criteria:**

- [ ] Every `selmodel` seed source is present, open or marked `dropped`

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy` passes
- [ ] `pytest src/tests/correctness -m 'not slow'` passes

**Notes:**

- Depends on Task 7
- As of 2026-11-14 only Citkowicz and Vevea (2017), doi:10.1037/met0000119, is missing (it needs APA access); the stop fires only for what is still
  missing when it runs

---

## Task 9: Published cases - selection models

**Status:** todo
**Priority:** high
**Complexity:** large
**Spec Reference:** Seed sources (stepfun, fixed delta, beta, parametric, trunc, data only); Implementation step 3

**Technical Goal:** each available `selmodel` seed source gives `published` cases.

**Subtasks:**

- [ ] One source at a time, as in Task 7: Vevea and Woods (2005) on `dat.cohen1981`; Vevea and Hedges (1995), Hedges and Vevea (2005), Hedges
      (1992), Iyengar and Greenhouse (1988); Citkowicz and Vevea (2017) on `dat.baskerville2012` unless dropped; Preston et al. (2004) on
      `dat.hahn2001` (pooled ratios through the back-transform check); Rust et al. (1990), Formann (2008); Lehmann et al. (2018, PMC10480976) on
      `dat.lehmann2018`
- [ ] Each case's tolerance follows the printed digits; a source whose values metafopy cannot reproduce because the likelihood is flat says so in a
      mismatch item, not in a loosened tolerance
- [ ] A source that cannot be read or does not reproduce its data goes under "Sources needed"
- [ ] Record any mismatch as the header says

**Acceptance Criteria:**

- [ ] Every `selmodel` seed source not marked `dropped` has `published` cases or a "Sources needed" row
- [ ] Every `published` case has `citation`, `locator` and `read`

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy` passes
- [ ] `pytest src/tests/correctness/test_published.py src/tests/correctness/test_coverage.py` passes

**Notes:**

- Depends on Tasks 5, 6 and 8

---

## Task 10: Property and simulation cases

**Status:** todo
**Priority:** medium
**Complexity:** medium
**Spec Reference:** Implementation steps 4 and 5

**Technical Goal:** the invariances and limiting cases of the bias methods, and the exact sizes of two tests, are checked.

**Subtasks:**

- [ ] `test_properties.py` and `cases/properties.toml`: `selmodel` with every `delta` fixed at no selection equals the ML `rma_uni` fit; reflecting
      `yi` to `-yi` negates `regtest`'s `zval` and the limit estimate, keeps `ranktest`'s p-value, and moves `trimfill`'s missing studies to the
      other side with the same `k0`; a location shift leaves `ranktest` unchanged; `trimfill` with `k0 = 0` returns the input model; `hc` with a
      DL `tau2` of 0 gives the equal-effects interval
- [ ] `test_simulation.py`, each test marked `slow`, `numpy.random.default_rng(seed)` with a fixed seed and a stated number of replicates, asserting at
      4 standard errors of the Monte Carlo proportion: the size of `regtest(model='lm')` under the weighted normal linear model it assumes; the size
      of the exact TES test, at most `alpha`, with `theta` known
- [ ] Check each simulation once by hand against a stand-in that should fail it (a test at the wrong alpha), and note the result in this task's Notes
- [ ] Record any mismatch as the header says

**Acceptance Criteria:**

- [ ] `pytest -m slow src/tests/correctness/test_simulation.py` passes
- [ ] Each property names the functions and options it runs over

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy` passes
- [ ] `pytest src/tests/correctness/test_properties.py src/tests/correctness/test_simulation.py` passes, `slow` tests included

**Notes:**

- Depends on Tasks 2 to 6

---

## Task 11: Stop for the operator - R packages for the other-software cases

**Status:** todo
**Priority:** medium
**Complexity:** small
**Spec Reference:** External dependencies; Decisions (weightr installed); Implementation step 6 (its stop)

**Technical Goal:** R `weightr` 2.0.2 and R `meta` 8.5-0 load from the correctness library. The task passes through when they do.

**Subtasks:**

- [ ] Run
      `R_LIBS=/Users/simonb/src/metafopy/.audit/Rlib LC_ALL=C.UTF-8 Rscript -e 'lib <- "/Users/simonb/.local/share/metafopy/r-correctness-lib"; .libPaths(c(lib, .libPaths())); cat(as.character(packageVersion("weightr")), as.character(packageVersion("meta")), "\n"); invisible(loadNamespace("weightr"))'`
- [ ] If it prints `2.0.2 8.5-0` and loads `weightr`, mark this task done and go on
- [ ] Otherwise stop the list for the operator, naming the missing or wrong package and its version. The operator installs it into
      `/Users/simonb/.local/share/metafopy/r-correctness-lib` outside the repository, never into the audit's pinned library
      (`src/audit/data/r-versions.json` must not change), and then restarts the list

**Acceptance Criteria:**

- [ ] Both packages load at the pinned versions from that library

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy` passes
- [ ] `pytest src/tests/correctness -m 'not slow'` passes

**Notes:**

- Depends on Task 10
- Checked 2026-11-14: `weightr` 2.0.2, `meta` 8.5-0 and `glmmML` are in that library, so the task is expected to pass through

---

## Task 12: Other-software cases from R `weightr` and R `meta`

**Status:** todo
**Priority:** low
**Complexity:** medium
**Spec Reference:** Implementation step 6; External dependencies; Error handling (generator)

**Technical Goal:** values from `weightr` (Vevea-Hedges step functions, estimated and a priori weights) and from `meta` (`metabias` Egger, Begg,
Peters, Harbord; `trimfill` L0 and R0), generated by a pinned script and committed.

**Subtasks:**

- [ ] Write `src/tests/correctness/r/other_software_bias.R`, separate from `other_software.R`:
  - it reads the library from `METAFOPY_CORRECTNESS_RLIB` and puts it first with `.libPaths(c(lib, .libPaths()))`, as `other_software.R` does;
  - it pins `weightr` 2.0.2 and `meta` 8.5-0 at its top and stops on any other version;
  - it writes `cases/other_bias.toml` with `kind = "other"`, recording the package and R versions in each case's `source`.
- [ ] Run it as
      `METAFOPY_CORRECTNESS_RLIB=/Users/simonb/.local/share/metafopy/r-correctness-lib R_LIBS=/Users/simonb/src/metafopy/.audit/Rlib LC_ALL=C.UTF-8 Rscript src/tests/correctness/r/other_software_bias.R`
      and commit the output
- [ ] The `meta` random-effects refits take tau2 from metafor's `rma.uni`, so those cases check the pipeline around tau2, not the estimator; say so in
      their `source`
- [ ] `test_other_software.py` already runs every `other` case; confirm it picks up the new file
- [ ] Record any mismatch as the header says
- [ ] `git diff --exit-code src/audit/data/r-versions.json` exits 0

**Acceptance Criteria:**

- [ ] Every new `other` case names its package's version and R's
- [ ] Tests never run R

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy` passes
- [ ] `pytest src/tests/correctness/test_other_software.py src/tests/correctness/test_coverage.py` passes

**Notes:**

- Depends on Task 11
- Nothing is installed into `/Users/simonb/src/metafopy/.audit/Rlib`; it only supplies dependencies through `R_LIBS`

---

## Task 13: Stop for the operator - review the mismatches, decisions and sources

**Status:** todo
**Priority:** high
**Complexity:** small
**Spec Reference:** Implementation step 7 (second stop)

**Technical Goal:** the user decides each mismatch, the two `selmodel` types no decision names, and each source still needed. The list stops here
until every item has an answer.

**Subtasks:**

- [ ] Write `agents/specs/correctness-publication-bias/mismatches.md` in the review-document layout of the shared `reviews.md` rule (re-read it from
      disk first):
  - what is reviewed and how;
  - automatic checks: the gates pass, every `mismatch` case has an item, `src/audit/data/r-versions.json` is unchanged;
  - each mismatch, with source, locator and the reference's, metafopy's and metafor's values;
  - "Sources needed";
  - decisions: each mismatch with options *follow the reference*, *keep metafor* and *investigate*; `negexppow` and `stepcon` with the options of
    Task 6; each with a recommendation and its reason;
  - an empty Answers section.
- [ ] Publish its interactive page, with answers saved to the artifact's database and an accept-all button, and put `**Review page:** <URL>` in the
      document's header
- [ ] If there are no items, write that in the document and skip the page
- [ ] Stop the list for the operator. Once every item has an answer, an operator session reads the page's answers, copies each into the Answers
      section of `mismatches.md` with the date, commits that, and only then restarts the list

**Acceptance Criteria:**

- [ ] Every `mismatch` case in `cases/*.toml`, every pending exemption and every "Sources needed" row has an item on the page

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy` passes
- [ ] `pytest src/tests/correctness -m 'not slow'` passes

**Notes:**

- Depends on Tasks 1 to 12

---

## Task 14: File the answers and close the spec

**Status:** todo
**Priority:** high
**Complexity:** medium
**Spec Reference:** Implementation step 8

**Technical Goal:** each answer becomes a case change or a `TODO.md` item, the spec is marked implemented, and the umbrella item for the correctness
follow-ups moves to `DONE.md`. Fixing metafopy is a follow-up change, not part of this list.

**Subtasks:**

- [ ] Read the answers from the Answers section of `agents/specs/correctness-publication-bias/mismatches.md`, never from the page. Stop as `blocked`,
      naming the missing items, if the section is empty or incomplete
- [ ] *Keep metafor*: correct the case to the convention metafor documents, cite where it documents it, and set `status = "pass"`
- [ ] *Follow the reference*: add a `TODO.md` item for each with the `todo-add` skill, naming the case, the fix, the R parity fixtures and
      `known.toml` entries to mark, and the `docs/differences.md` section "Where metafor departs from its reference" to add
- [ ] *Investigate*: a `TODO.md` item with the `todo-add` skill naming what is in doubt
- [ ] `negexppow` and `stepcon`: replace each pending exemption as answered (a `derived` case, an exemption with its reason, or a published case)
- [ ] A source the operator supplied: write its `published` cases as in Task 9
- [ ] Set the spec's `**Status:**` to `implemented`, and its row in `agents/specs/index.md` to say so
- [ ] Move the `TODO.md` item "Extend the correctness tests beyond the core models" to `DONE.md`: append its heading and body unchanged, then a
      `Done:` line with the date naming the four specs `agents/specs/correctness-mv-robust/spec.md`, `agents/specs/correctness-glmm/spec.md`,
      `agents/specs/correctness-location-scale/spec.md` and `agents/specs/correctness-publication-bias/spec.md`
- [ ] Run the final tests from the worktree:
      `main=$(git worktree list --porcelain | sed -n '1s/^worktree //p'); mkdir -p "$main/ralpher-logs/correctness-publication-bias/tasks-1" && pytest src/tests/correctness 2>&1 | tee "$main/ralpher-logs/correctness-publication-bias/tasks-1/final-tests.log"`

**Acceptance Criteria:**

- [ ] Every answered item has a case change or a `TODO.md` item
- [ ] No `[[exempt]]` entry reads "pending the user's answer"
- [ ] `git diff --exit-code main -- src/audit/data/r-versions.json` exits 0

**Definition of Done:**

- [ ] `pre-commit run --all-files` passes
- [ ] `pre-commit run auditme-all --hook-stage manual --all-files` passes
- [ ] `mypy` passes
- [ ] `base=$(mktemp) && git show $(git log --diff-filter=A --format=%H -- agents/tasks/correctness-publication-bias/tasks-1.md):TODO.md > "$base" && python3 ~/.claude/skills/todo-review/verify_bodies.py "$base" TODO.md DONE.md`
      exits 0
- [ ] `pytest src/tests/correctness` passes, `slow` tests included (the list changes no module under `src/metafopy/`, so this is the full run), as
      logged in `ralpher-logs/correctness-publication-bias/tasks-1/final-tests.log`

**Notes:**

- Depends on Task 13
- This list is the last in the run order, so it alone moves the umbrella item; the sibling lists leave it in `TODO.md`
