# Tasks: Correctness tests

**Spec:** `agents/specs/correctness-tests/spec.md`
**Created:** 2026-11-11
**Version:** 1

The spec (version 1.1) was approved on 2026-11-11, including the decision that `escalc.toml` holds the per-measure `vtype` table. The project has no
affected-tests tool. This list changes no module under `src/metafopy/`: it adds tests under `src/tests/correctness/`, edits rules and documents, and
writes cases. So each task names its own test files, and the last task runs the whole correctness suite, `slow` cases included.

**Recording a mismatch (every task that writes cases):** when a case's reference disagrees with metafopy, commit it with `status = "mismatch"` and a
`reason` naming both values, and append an item to `agents/specs/correctness-tests/mismatches.md`, creating the file from the review-document layout
in Task 13 if it does not exist yet. That item gives the source, locator, reference value, metafopy's value, and metafor's value from the R parity
fixture where one exists. Never change metafopy's output to match a reference in this list. A reference that cannot be read (paywalled, a
textbook) is listed under "Sources needed" in the same file, and its case is not written.

---

## Task 1: Rank the defining reference above R parity in the rules

**Status:** done
**Priority:** high
**Complexity:** small
**Spec Reference:** Implementation step 7

**Technical Goal:** the project's rules say that correctness comes first, before any test is written to that principle.

**Subtasks:**

- [x] In `claude/rules/metafopy.md`, replace "R metafor is the oracle: port numerics line for line and compare against committed JSON fixtures." with
      a rule saying that the publication defining a method is the first oracle and R metafor the second. Numerics are still ported from metafor,
      and R parity fixtures stay. Where the two disagree, a correctness case in `src/tests/correctness/` records it, and metafopy's output changes
      only after the user confirms it
- [x] In `AGENTS.md`, amend "R metafor is the numeric oracle" the same way, pointing at `src/tests/correctness/`

**Acceptance Criteria:**

- [x] Neither file says that R metafor is the only oracle

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)

**Notes:**

- Key files: `claude/rules/metafopy.md`, `AGENTS.md`, `NOTES.md` (already states the principle; leave it)

---

## Task 2: The harness - case loader, validation and the coverage gate

**Status:** done
**Priority:** high
**Complexity:** medium
**Spec Reference:** Module location; Kinds of evidence; Case file format; API; Coverage list; Error handling

**Technical Goal:** `src/tests/correctness/` exists, with case files that are validated on every run and a coverage gate that names every core item
still lacking a `published` or `derived` case.

**Subtasks:**

- [x] Check that `src/tests/correctness/` does not exist (`ls src/tests/correctness`); stop as `blocked` if it does
- [x] Create `src/tests/correctness/__init__.py` and `cases.py` with `Source`, `Case`, `load_cases`, `run_case`, `published_tolerance`,
      `check_case` and `case_params`, exactly as in the spec's API section, reading TOML with `tomllib`
- [x] Validate as in Error handling: every invalid file or case raises `ValueError` naming the file and case id
- [x] Resolve `function` as `metafopy.<name>` or `formulas.<name>`, and `check` as a function in a test module of `src/tests/correctness/`;
      refuse anything else
- [x] Create `formulas.py` holding only a module docstring that states its rule: no `metafopy` import; each function cites its equation (it also holds the two risk-difference functions that the derived end-to-end case needs)
- [x] Create `cases/escalc.toml`, `tau2.toml`, `inference.toml`, `heterogeneity.toml` and `mh_peto.toml`, empty except a header comment (`vtypes` is one inline array in `escalc.toml`, the same data as `[[vtypes]]` entries)
- [x] `test_coverage.py`:
  - load every case file;
  - check with `ast` that `formulas.py` imports no `metafopy` module;
  - the gate reads `metafopy.effects.measures.MEASURES`, the `[[vtypes]]` table in `escalc.toml`, `metafopy.models.tau2.ESTIMATORS`,
    `typing.get_args(metafopy.models.rma_uni.Test.__value__)`, `typing.get_args(metafopy.methods.confint.CIType)`, the heterogeneity statistics,
    and the `rma_mh` measures and `rma_peto` named in the spec;
  - it lists every item with no `published` or `derived` case and no `[[exempt]]` entry.
  Mark the gate test `xfail(strict=True, reason='coverage incomplete; see agents/tasks/correctness-tests/tasks-1.md')` until Task 8 removes the mark
- [x] `test_coverage.py` also checks that `escalc` accepts each `[[vtypes]]` pair, on the smallest dataset that suits the measure
- [x] Unit tests of the harness on inline TOML strings: `published_tolerance` (`'0.0193'` gives `5e-05`, `'684'` gives `0.5`, `'-15.106'` gives
      `5e-04`); each validation error; a `mismatch` case becoming a strict xfail
- [x] Two end-to-end cases proving the path: a `derived` `escalc` `RD` `yi` and `vi` on inline data, and a `property` case on `dat.bcg`: `I2` equals
      `(H2 - 1) / H2` for a `DL` fit
- [x] `test_derived.py` and `test_properties.py`, each a parametrised test over `case_params(<kind>)`

**Acceptance Criteria:**

- [x] `pytest src/tests/correctness` passes, with the gate as an expected failure that lists the missing items
- [x] A case file with a duplicate id or an invalid key makes `test_coverage.py` fail, naming the file

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy` passes
- [x] `pytest src/tests/correctness` passes

**Notes:**

- Depends on Task 1
- `Test` is a PEP 695 `type` alias, so its arguments are `typing.get_args(Test.__value__)`. `CIType` is a plain `typing.Literal`, so its arguments
  are `typing.get_args(CIType)` (checked 2026-11-11)
- Expected values in `derived` cases come from `formulas.py`, called by the test, not typed into TOML: the case names the formula and its inputs

---

## Task 3: escalc - two-group binary and count measures

**Status:** done
**Priority:** high
**Complexity:** large
**Spec Reference:** Implementation step 2; Coverage list (measures)

**Technical Goal:** every 2x2-table, paired-binary and incidence measure in `MEASURES` has a `derived` case for `yi` and for `vi` under each
accepted `vtype`, coded from the publication that defines it.

**Subtasks:**

- [x] Add the `[[vtypes]]` rows for these measures to `escalc.toml`, from the `escalc` docstring and metafor's `escalc` help
- [x] For each measure among `RR`, `OR`, `PETO`, `RD`, `AS`, `PHI`, `ZPHI`, `YUQ`, `YUY`, `RTET`, `ZTET`, `PBIT`, `OR2D`, `OR2DN`, `OR2DL`, the `MP*`
      paired measures, `IRR`, `IRD` and `IRSD`:
  - write a function in `formulas.py`, citing the equation and its source in its docstring;
  - add a `derived` case per accepted `vtype`, on inline data that includes a zero cell and checks the `add`/`to` handling against the
    publication.
- [x] Where no publication gives a usable formula, add an `[[exempt]]` entry with the reason instead
- [x] Record any mismatch as the header says

**Acceptance Criteria:**

- [x] The gate's missing list no longer names any of these measures or their `vtype` pairs
- [x] Each `formulas.py` function names its source and equation

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy` passes
- [x] `pytest src/tests/correctness` passes

**Notes:**

- Depends on Task 2
- Checked by hand: with the default truncation DL averages 0.098 and HE 0.099 against a true 0.04 (standard error 0.003), and `g` without `J` (`correct=False`) averages 0.531 against 0.5 (standard error 0.0012), so each test fails for its stand-in. `tau2_min=-1e6` turns the truncation off.
- `RTET` has no closed form; the derivation integrates the bivariate normal with `scipy.stats.multivariate_normal` independently of
  `metafopy.effects.rtet`, at `rtol=1e-8`

---

## Task 4: escalc - two-group continuous and paired measures

**Status:** done
**Priority:** high
**Complexity:** large
**Spec Reference:** Implementation step 2; Coverage list (measures)

**Technical Goal:** as Task 3, for `MD`, `POMPMD`, `SMD`, `SMDH`, `SMD1`, `SMD1H`, `ROM`, `VR`, `CVR`, `RPB`, `ZPB`, `RBIS`, `ZBIS`, `D2OR`, `D2ORN`,
`D2ORL`, `MC`, `POMPMC`, `SMCC`, `SMCR`, `SMCRH`, `SMCRP`, `SMCRPH`, `ROMC`, `VRC`, `CVRC`, `CLES`, `CLESN`, `AUC`, `AUCN`, `CLESCN` and `AUCCN`.

**Subtasks:**

- [x] Add their `[[vtypes]]` rows to `escalc.toml`
- [x] For each: a `formulas.py` function citing its source; `derived` cases per accepted `vtype`. `SMD` uses the exact gamma-function `J` of
      Hedges (1981), not the approximation, unless metafor documents the approximation as its convention, in which case the case says which
- [x] `[[exempt]]` entries with reasons where no formula is published
- [x] Record any mismatch as the header says

**Acceptance Criteria:**

- [x] The gate's missing list names none of these measures

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy` passes
- [x] `pytest src/tests/correctness` passes

**Notes:**

- Depends on Task 2
- Checked by hand: with the default truncation DL averages 0.098 and HE 0.099 against a true 0.04 (standard error 0.003), and `g` without `J` (`correct=False`) averages 0.531 against 0.5 (standard error 0.0012), so each test fails for its stand-in. `tau2_min=-1e6` turns the truncation off.

---

## Task 5: escalc - correlations, single-group and remaining measures

**Status:** done
**Priority:** high
**Complexity:** large
**Spec Reference:** Implementation step 2; Coverage list (measures)

**Technical Goal:** as Task 3, for every measure in `MEASURES` not covered by Tasks 3 and 4. Those are the correlation measures (`COR` to `ZR2F`),
the proportions and rates (`PR` to `IRFT`), the means and variability measures (`MN` to `CVLN`), the reliability measures (`ARAW`, `AHW`, `ABT`),
`REH` and `GEN`.

**Subtasks:**

- [x] Add their `[[vtypes]]` rows to `escalc.toml`
- [x] A `formulas.py` function and `derived` cases for each, as in Task 3; `GEN` (user-supplied `yi`/`vi`) gets an `[[exempt]]` entry explaining
      that it computes nothing
- [x] Record any mismatch as the header says

**Acceptance Criteria:**

- [x] The gate's missing list names no measure

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy` passes
- [x] `pytest src/tests/correctness` passes

**Notes:**

- Depends on Task 2
- Checked by hand: with the default truncation DL averages 0.098 and HE 0.099 against a true 0.04 (standard error 0.003), and `g` without `J` (`correct=False`) averages 0.531 against 0.5 (standard error 0.0012), so each test fails for its stand-in. `tau2_min=-1e6` turns the truncation off.
- Before closing, compare the measures covered by Tasks 3 to 5 with `MEASURES` at test time, so that a measure named in none of them is not missed

---

## Task 6: rma_uni - tau2 estimators

**Status:** done
**Priority:** high
**Complexity:** large
**Spec Reference:** Implementation step 2; Coverage list (methods)

**Technical Goal:** every estimator in `ESTIMATORS` has a `derived` case on at least two datasets, one with and one without a moderator.

**Subtasks:**

- [x] Code the closed forms in `formulas.py`: `DL`, `HE`, `HS`, `HSk`, `SJ`, `CO`, `VC` and `GENQ`. Use the formula papers in the spec's Seed sources
      and Veroniki et al. (2016). Truncation at zero follows the source
- [x] Code the iterative estimators as estimating equations solved with `scipy.optimize.brentq`: `DLIT`, `SJIT`, `ML`, `REML`, `EB`, `PM`, `MP`,
      `PMM` and `GENQM`. Each is written from the publication's equation, not from metafopy's loop
- [x] The equal-effects methods `EE`, `FE` and `CE`: `derived` cases with `tau2 = 0` and the weighted mean, its standard error and the
      equal-effects CI
- [x] Cases on `dat.bcg` (log RR, with and without `ablat`) and on one more dataset, for `tau2`, `beta` and `se`
- [x] Record any mismatch as the header says

**Acceptance Criteria:**

- [x] The gate's missing list names no estimator

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy` passes
- [x] `pytest src/tests/correctness` passes

**Notes:**

- Depends on Task 2
- Checked by hand: with the default truncation DL averages 0.098 and HE 0.099 against a true 0.04 (standard error 0.003), and `g` without `J` (`correct=False`) averages 0.531 against 0.5 (standard error 0.0012), so each test fails for its stand-in. `tau2_min=-1e6` turns the truncation off.
- Key files (read for option names only, not for formulas): `src/metafopy/models/tau2.py`, `src/metafopy/models/rma_uni.py`

---

## Task 7: rma_uni - inference, confidence intervals and heterogeneity

**Status:** done
**Priority:** high
**Complexity:** medium
**Spec Reference:** Implementation step 2; Coverage list (inference, heterogeneity)

**Technical Goal:** each `test` value, each `confint` `type` for `tau2`, `predict` prediction intervals, and `QE`, `QM`, `I2` and `H2` have
`derived` cases.

**Subtasks:**

- [x] `formulas.py`:
  - the Wald `z` and `t` tests;
  - the Knapp-Hartung variance and t test, from Knapp and Hartung (2003);
  - the `adhoc` truncation that metafor's documentation attributes to it;
  - Cochran's Q;
  - `I2` and `H2` from Higgins and Thompson (2002), in the typical-variance form metafor documents;
  - prediction intervals, from the publication metafor cites for its form.
- [x] `confint` for `tau2`, one function per type:
  - Q-profile, from Viechtbauer (2007), solved by root finding;
  - generalised Q;
  - profile likelihood;
  - Higgins-Thompson;
  - the three Wald forms.
- [x] Cases on `dat.bcg` under the `REML` and `DL` fits, in `inference.toml` and `heterogeneity.toml`
- [x] Record any mismatch as the header says

**Acceptance Criteria:**

- [x] The gate's missing list names no `test` value, no `CIType` value and no heterogeneity statistic

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy` passes
- [x] `pytest src/tests/correctness` passes

**Notes:**

- Depends on Task 6 (the tau2 formulas)
- Key files: `src/metafopy/methods/confint.py` (its docstring cites the sources)

---

## Task 8: rma_mh and rma_peto, then close the gate

**Status:** done
**Priority:** high
**Complexity:** medium
**Spec Reference:** Implementation step 2; Coverage list (MH, Peto)

**Technical Goal:** the Mantel-Haenszel estimators for `OR`, `RR`, `RD`, `IRR` and `IRD` with their variances, and the Peto estimator, have `derived`
cases. With them the coverage gate passes.

**Subtasks:**

- [x] `formulas.py`:
  - the MH estimators from Mantel and Haenszel (1959);
  - the `OR` variance from Robins, Breslow and Greenland (1986);
  - the `RR` and `RD` variances from Greenland and Robins (1985);
  - the incidence forms from the publication metafor cites;
  - Peto's one-step estimator from Yusuf et al. (1985).
- [x] Cases on `dat.bcg` and a dataset with a zero cell, in `mh_peto.toml`
- [x] Record any mismatch as the header says
- [x] Remove the gate's `xfail` mark from `test_coverage.py`

**Acceptance Criteria:**

- [x] The gate passes with no xfail mark

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy` passes
- [x] `pytest src/tests/correctness` passes

**Notes:**

- Depends on Tasks 3 to 7

---

## Task 9: Published cases from open-access sources

**Status:** done
**Priority:** high
**Complexity:** large
**Spec Reference:** Seed sources; Kinds of evidence (`published`)

**Technical Goal:** each seed source that can be fetched gives `published` cases, with the value, locator and date read.

**Subtasks:**

- [x] For each source in the spec's Seed sources:
  - fetch it through the PubMed, Paperclip or alphaXiv tools;
  - find its worked example and the metafopy dataset that holds the same data;
  - add `published` cases with values written as printed, as strings.
- [x] A source that cannot be fetched goes under "Sources needed" in `mismatches.md`, with its citation and the table or example wanted
- [x] A source whose data is in no metafopy dataset: put the data inline in the case, with its locator
- [x] `test_published.py`, parametrised over `case_params('published')`
- [x] Record any mismatch as the header says (no `published` case could be written from the sources fetched; every seed source is under "Sources needed" in `mismatches.md`, so the test has no cases yet and skips)

**Acceptance Criteria:**

- [x] Every seed source has `published` cases or a "Sources needed" entry
- [x] Every `published` case has `citation`, `locator` and `read`

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy` passes
- [x] `pytest src/tests/correctness` passes

**Notes:**

- Depends on Task 2
- Checked by hand: with the default truncation DL averages 0.098 and HE 0.099 against a true 0.04 (standard error 0.003), and `g` without `J` (`correct=False`) averages 0.531 against 0.5 (standard error 0.0012), so each test fails for its stand-in. `tau2_min=-1e6` turns the truncation off.
- Copy only the numbers used; no source text or tables
- A value printed with few digits gives a loose tolerance. Prefer the table with the most digits

---

## Task 10: Property cases

**Status:** done
**Priority:** medium
**Complexity:** medium
**Spec Reference:** Implementation step 4; Kinds of evidence (`property`)

**Technical Goal:** the invariances, limiting cases and estimating equations named in the spec are checked for every estimator they apply to.

**Subtasks:**

- [x] Location invariance: `tau2` unchanged and `beta` shifted by `c` when `yi` is shifted by `c`. Scale invariance: `tau2` times `c^2` and
      `beta` times `c` when `yi` is scaled by `c` and `vi` by `c^2`. Check both for every estimator in `ESTIMATORS`
- [x] When the data are homogeneous enough that `tau2` is truncated to 0, each random-effects estimator gives the equal-effects estimate
- [x] At metafopy's optimum the REML and ML score is zero, and for PM `Q(tau2) = k - 1`
- [x] `I2 = (H2 - 1) / H2` for every estimator

**Acceptance Criteria:**

- [x] Each property case names the estimators it runs over, read from `ESTIMATORS`

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy` passes
- [x] `pytest src/tests/correctness` passes

**Notes:**

- Depends on Task 6
- Use `hypothesis` with `@settings(max_examples=20, deadline=10000)` for the invariances

---

## Task 11: Simulation cases

**Status:** done
**Priority:** medium
**Complexity:** medium
**Spec Reference:** Kinds of evidence (`simulation`); Implementation step 5

**Technical Goal:** seeded Monte Carlo checks of the results that hold exactly in theory, behind the `slow` marker.

**Subtasks:**

- [x] The untruncated DL and HE estimators are unbiased for `tau2`. Their truncation is turned off through the `control` option metafopy gives
      for it; stop as `blocked` if there is none
- [x] Hedges' `g` with the exact `J` is unbiased
- [x] The equal-effects CI with known `vi` has exact coverage
- [x] `test_simulation.py`: each case uses `numpy.random.default_rng(seed)`, a fixed seed and a stated number of replicates, and asserts at 4
      standard errors of the Monte Carlo mean or proportion; mark each test `slow`

**Acceptance Criteria:**

- [x] `pytest -m slow src/tests/correctness/test_simulation.py` passes
- [x] Each case fails when its estimator is replaced by a biased stand-in (truncated DL, `g` without `J`), checked once by hand and noted here

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy` passes
- [x] `pytest src/tests/correctness` passes

**Notes:**

- Depends on Task 2
- Checked by hand: with the default truncation DL averages 0.098 and HE 0.099 against a true 0.04 (standard error 0.003), and `g` without `J` (`correct=False`) averages 0.531 against 0.5 (standard error 0.0012), so each test fails for its stand-in. `tau2_min=-1e6` turns the truncation off.

---

## Task 12: Other-software cases from R meta

**Status:** done
**Priority:** low
**Complexity:** medium
**Spec Reference:** Implementation step 6; Kinds of evidence (`other`); Error handling (generator)

**Technical Goal:** values from R `meta` for the estimators, tests and MH/Peto fits it implements, generated by a pinned script and committed.

**Subtasks:**

- [x] If `METAFOPY_CORRECTNESS_RLIB` is unset, or the library it names has no `meta`, stop as `blocked`. The message names the variable and says
      that the operator installs `meta` there outside this repository (`install.packages('meta', lib = Sys.getenv('METAFOPY_CORRECTNESS_RLIB'))`)
- [x] Write `src/tests/correctness/r/other_software.R`:
  - it pins `meta`'s version at its top and stops on any other version;
  - it writes `cases/other.toml` with `kind = "other"`;
  - it records `meta`'s and R's versions in each case's `source`.
- [x] Run it with `LC_ALL=C.UTF-8 Rscript` and commit the output
- [x] `test_other_software.py`, parametrised over `case_params('other')`
- [x] Record any mismatch as the header says

**Acceptance Criteria:**

- [x] Every `other` case names `meta`'s version
- [x] Tests never run R

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy` passes
- [x] `pytest src/tests/correctness` passes

**Notes:**

- Depends on Task 2
- Prerequisite outside this repository: R `meta` in the library named by `METAFOPY_CORRECTNESS_RLIB`. Do not install it into the audit's pinned
  library; `src/audit/data/r-versions.json` must not change
- Unblocked 2026-11-12 by an operator session: `meta` 8.5-0 installed into `/Users/simonb/.local/share/metafopy/r-correctness-lib` (outside the
  repository) with its dependencies taken from the audit's library through `R_LIBS`, so only `meta` was written and the audit library is unchanged.
  Run: `METAFOPY_CORRECTNESS_RLIB=<that library> R_LIBS=<audit library> LC_ALL=C.UTF-8 Rscript src/tests/correctness/r/other_software.R`
- `meta` takes tau2 from `metafor::rma.uni`, so the `rma_uni` cases check the weights, Q and Knapp-Hartung intervals around tau2, not the
  estimator; its effect sizes, Mantel-Haenszel and Peto code are its own
- No mismatch. Two first-run differences were conventions, pinned in the generator: `meta`'s SMD variance with `exact.smd = TRUE` is metafor's
  `vtype = 'UB'`; its MH risk-difference SE is the Greenland and Robins (1985) variance (reproduced by hand to 1e-16), while metafor documents
  Sato, Greenland and Robins (1989), already checked by a `derived` case, so that case compares the estimate only

---

## Task 13: Review the mismatches and the sources needed

**Status:** todo
**Priority:** high
**Complexity:** small
**Spec Reference:** Mismatches; Implementation step 8

**Technical Goal:** the user decides each mismatch, and supplies or drops each source needed. The list stops here until every item has an
answer.

**Subtasks:**

- [ ] Write `agents/specs/correctness-tests/mismatches.md` in the review-document layout of the shared `reviews.md` rule:
  - what is reviewed and how;
  - automatic checks: the gate passes, and every `mismatch` case has an item;
  - each mismatch, with source, locator and the three values;
  - "Sources needed";
  - decisions, with options *follow the reference*, *keep metafor* and *investigate*, and a recommendation for each;
  - an empty Answers section.
- [ ] Publish its interactive page, with answers saved to the artifact's database, and put `**Review page:** <URL>` in the document's header
- [ ] If there are no mismatches and no sources needed, write that in the document and skip the page
- [ ] Stop the list for the operator. Once every item has an answer, an operator session reads the page's answers, copies each into the Answers
      section with the date, commits that, and only then restarts the list

**Acceptance Criteria:**

- [ ] Every `mismatch` case in `cases/*.toml` has an item on the page

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy` passes
- [ ] `pytest src/tests/correctness` passes, `slow` tests skipped (`-m 'not slow'`)

**Notes:**

- Depends on Tasks 3 to 12

---

## Task 14: File the answers and close the spec

**Status:** todo
**Priority:** high
**Complexity:** small
**Spec Reference:** Mismatches (follow-up)

**Technical Goal:** each answer becomes recorded work. Fixing metafopy is a follow-up change, not part of this list.

**Subtasks:**

- [ ] Read the answers from the Answers section of `agents/specs/correctness-tests/mismatches.md`, never from the page. Stop as `blocked`, naming the
      missing items, if the section is empty or incomplete
- [ ] *Keep metafor*: correct the case to the convention metafor documents, cite where it documents it, and set `status = "pass"`
- [ ] *Follow the reference*: add a `TODO.md` item for each with the `todo-add` skill. The item names the case, the fix, the R parity fixtures and
      `known.toml` entries to mark, and the `docs/differences.md` section "Where metafor departs from its reference" to add
- [ ] *Investigate*: a `TODO.md` item naming what is in doubt
- [ ] A source the user supplied: write its `published` cases as in Task 9
- [ ] Set the spec's `**Status:**` to `implemented`, and its entry in `agents/specs/index.md` to say so

**Acceptance Criteria:**

- [ ] Every answered item has a case change or a `TODO.md` item

**Definition of Done:**

- [ ] `pre-commit run --all-files` passes
- [ ] `pre-commit run auditme-all --hook-stage manual --all-files` passes
- [ ] `mypy` passes
- [ ] `pytest src/tests/correctness` passes, `slow` tests included

**Notes:**

- Depends on Task 13
