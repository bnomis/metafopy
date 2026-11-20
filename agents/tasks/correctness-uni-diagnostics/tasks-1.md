# Tasks: Correctness tests: rma_uni diagnostics

**Spec:** `agents/specs/correctness-uni-diagnostics/spec.md`
**Created:** 2026-11-20
**Version:** 1

**Running tests in the worktree:** the worktree has no `.venv`, and the main checkout's venv imports `metafopy` from the main checkout's `src/` (its
editable install), so `PATH=/Users/simonb/src/metafopy/.venv/bin:$PATH pytest ...` alone tests the code on `main`, not this list's. Run every pytest
and Python check from the worktree root as `export PATH=/Users/simonb/src/metafopy/.venv/bin:$PATH PYTHONPATH=$PWD/src`, and check once with
`python -c 'import metafopy; print(metafopy.__file__)'` that it prints this worktree's path.

The spec (version 1.1, approved by the operator on 2026-11-20 in `agents/reviews/correctness-uni-diagnostics-spec.md`, every decision answered)
adds published, derived and property correctness cases for the `rma_uni` diagnostics: `cooks_distance`, `dfbetas`, `rstudent`, every `influence`
column, `rstandard`, `ranef` and `blup`. No file under `src/metafopy/` changes; no version bump. The list is modelled on its sibling
`agents/tasks/correctness-mv-diagnostics/tasks-1.md` (references in `formulas.py` that never import metafopy, checks in `test_derived.py`,
`test_published.py` and `test_properties.py`, items in `test_coverage.py`); Task 6 hoists anything the new code duplicates from it.

**Operator stops:** Task 5 stops for the mismatch review only if Tasks 3 and 4 committed a `status = "mismatch"` case; the spec expects none
(every reference measured on 2026-11-20 agrees with metafopy to 1.6e-11 or better). No paper needs supplying: every cited source is in the papers
folder `/Users/simonb/src/metafopy papers/`.

---

## Task 1: The references uni_deletion and uni_blups

**Status:** done
**Priority:** high
**Complexity:** medium
**Spec Reference:** Definitions and references; API (formulas.py); Error handling; Implementation step 1

**Technical Goal:** independent brute-force references for the rma.uni deletion diagnostics and the BLUPs of the true effects exist in
`formulas.py`, with their own harness tests.

**Subtasks:**

- [x] Starting-point check: `grep -nE 'def (uni_deletion|uni_blups|influence_printed|blup_printed|uni_mean_shift|uni_blup)\b|class (UniDeletion|UniBlups)\b|^UNI_(FITS|INFLUENCE) ' src/tests/correctness/*.py`
      prints nothing; stop as `blocked`, naming the match, if it does (the local variable `uni_ranef` in `test_properties.py` is not a clash, and
      the check function `uni_ranef` is added only in Task 2)
- [x] Add `UniDeletion` and `uni_deletion` to `src/tests/correctness/formulas.py` with the fields, signature and docstring of the spec's API
      section, coded from Viechtbauer and Cheung (2010) as its "Definitions and references" section says: for each study, a refit on the kept
      studies with `rma_tau2` (or the fixed `tau2`; `0` for EE), `b(-i)` and `vb(-i)` by weighted least squares, Cook's distance (eq 20) over
      `btt`, DFBETAS (eq 23), the deleted residual, its standard error and `z` (eqs 15, 16), `tau2(-i)`, `QE(-i)` (Cochran's Q of the refit),
      DFFITS (eq 19), COVRATIO (eq 25), the hat values (eq 13) and the internally studentised residual (eq 14). Build `X` with `model_matrix`;
      `numpy` and `scipy` only, no metafopy import
- [x] Add `UniBlups` and `uni_blups`: Henderson's mixed model equations through `mv_blups` with one `~ 1 | study` term (`R = diag(vi)`, `Z = I`,
      `G = tau2 I`); `pred = x_i b + u_i` and its standard error from the inverse coefficient matrix (the `b` block, the `u` block and their
      covariance)
- [x] The errors of the spec's Error handling section: `ValueError` naming the study when a deletion leaves `X` rank deficient
- [x] Harness tests in `src/tests/correctness/test_coverage.py`, beside `test_mv_deletion_without_reestimation_matches_the_rma_uni_closed_forms`:
      on a small inline dataset, an intercept-only fit with `tau2` fixed, `uni_deletion` against eq 24 for DFBETAS and `(mu - mu(-i))^2 sum(w)`
      for Cook's distance; a rank-deficient deletion raises; `uni_blups` against the marginal form `x_i b + lambda_i (y_i - x_i b)` and the
      variance `lambda_i v_i + (1 - lambda_i)^2 x_i vb x_i'`
- [x] Run `pytest src/tests/correctness/test_coverage.py`

**Acceptance Criteria:**

- [x] Both references reproduce their closed forms on the inline data to `1e-10`
- [x] `formulas.py` still imports no metafopy module (the harness's import-ban test passes)

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] The pytest run of the last subtask passes

**Notes:**

- Key files: `src/tests/correctness/formulas.py` (`rma_tau2`, `model_matrix`, `mv_blups`, `RandomTerm`), `src/tests/correctness/test_coverage.py`
- Measured values to expect are in `agents/specs/correctness-uni-diagnostics/measurements.md`; the probe scripts of 2026-11-20 were outside the
  repository and are not needed
- Reuse `mv_blups` by calling it; do not copy its solve

---

## Task 2: The checks, the coverage items and the exemptions

**Status:** done
**Priority:** high
**Complexity:** medium
**Spec Reference:** API (test_derived.py, test_published.py, test_properties.py); Coverage list; Error handling; Implementation step 2

**Technical Goal:** the case checks exist, and the coverage gate names every new item, failing until Task 3 adds the cases.

**Subtasks:**

- [x] In `src/tests/correctness/test_derived.py`, add `uni_deletion`, `uni_ranef` and `uni_blup` with the spec's docstrings: `uni_deletion` refits
      the case (`refit`) and compares `cooks_distance` (with the `btt` of the case's `then`), `dfbetas`, `rstudent`, `rstandard` and every
      `influence` column against `formulas.uni_deletion` at `DERIVED_TOLERANCE`, failing naming the study where metafopy gives NaN and the
      reference a value; `uni_ranef` compares `ranef`'s `pred`, `se` and bounds against `formulas.mv_blups` at metafopy's `tau2` (every value 0 for
      EE fits); `uni_blup` compares `blup`'s `pred`, `se` and bounds against `formulas.uni_blups`; bounds use the normal quantile, or the t quantile
      on `k - p` degrees of freedom when the fit's `test` is `t`, `knha`, `hksj` or `adhoc`
- [x] In `src/tests/correctness/test_published.py`, add `influence_printed` (`hat[i]`, `rstudent[i]`, `cook.d[i]`, `R[i]` = `100 (tau2 -
      tau2.del) / tau2`, `dfbetas[i,j]`, `tau2`) and `blup_printed` (`pred[i]`, `min(pred)`, `max(pred)`), both through `check_values`; an unknown
      printed name raises the loader's existing error
- [x] In `src/tests/correctness/test_properties.py`, add `uni_mean_shift`: for each study, the z of a dummy for it added to `mods` equals its
      `rstudent` `z`; raise `ValueError` for an ML case
- [x] In `src/tests/correctness/test_coverage.py`, add `UNI_FITS = ('ee', 'me', 're', 'tau2_fixed')`, `UNI_INFLUENCE = ('cov_r', 'dffits',
      'hat', 'qe_del', 'tau2_del')` and `uni_diagnostics_items()`, called from `coverage_items()`: `rma_uni.<f>.<fit>` for each `f` in
      `MV_DIAGNOSTICS` and `fit` in `UNI_FITS`; `rma_uni.cooks_distance.btt`; `rma_uni.ranef.{re,me,ee,pi.z,pi.t}`;
      `rma_uni.influence.<column>` for each column in `UNI_INFLUENCE`; `rma_uni.rstandard.marginal`; `rma_uni.blup.{re,me}`; and the exempt items
      `rma_uni.<f>.{knha,unweighted,weights}` for each `f` in `MV_DIAGNOSTICS`
- [x] Create `src/tests/correctness/cases/rma_uni_diagnostics.toml` holding only the `[[exempt]]` entries for the exempt items, each with reason
      "defined only in metafor's code; R parity covers it" (the shape of the `rma_mv.ranef.pi.t` entry in `cases/rma_mv.toml`)
- [x] Run `pytest src/tests/correctness/test_coverage.py`: the gate fails listing exactly the new non-exempt items, and every other test passes

**Acceptance Criteria:**

- [x] The gate names every item of the spec's Coverage list; the only failures are the new items with no case yet
- [x] The knha, unweighted and weights items are exempt with the spec's reason

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] The pytest run of the last subtask behaves as stated there

**Notes:**

- Depends on Task 1
- The commit leaves the coverage gate failing by design until Task 3; say so in the commit body

---

## Task 3: The cases

**Status:** done
**Priority:** high
**Complexity:** medium
**Spec Reference:** Cases; Definitions and references (published values, sources table); Implementation step 3

**Technical Goal:** every case of the spec's Cases table exists with its `source`, and the coverage gate passes.

**Subtasks:**

- [x] In `src/tests/correctness/cases/rma_uni_diagnostics.toml`, add each case of the spec's Cases table with its kind, fit, `then`, check and
      `covers`; REML and ML cases pass `control = { threshold = 1e-13, tol = 1e-13, maxiter = 1000, tau2_max = 1e6 }` as `cases/inference.toml`
      does; DL and EE cases use the defaults
- [x] `uni.diag.published.bcg`: the Table I data of Viechtbauer and Cheung (2010) inline (counts, `ablat_c` = latitude minus 33, `year_c` = year
      minus 1966), `escalc` RR, DL, `~ ablat_c + year_c`; its `expect` holds the printed values of `measurements.md` (study 4 is index 3): `tau2`,
      `hat`, `rstudent`, `cook.d` and `R` of studies 4, 7 and 13, and `dfbetas[3,0]` to `dfbetas[3,2]`, each as the printed string
- [x] `uni.diag.dl.wtl`: the Table II rows inline (the `dat.bangertdrowns2004` rows `measurements.md` names), DL, `~ length + meta + college`
- [x] `uni.blup.published.raudenbush`: `dat.raudenbush1985`, REML; `expect` `pred[3]` "0.21", `min(pred)` "-0.03", `max(pred)` "0.25"
      (Raudenbush 2009, Table 16.2 and section 16.5.6.1); the mixed-model range is not used
- [x] Each case's `source` has `citation`, `doi` where one exists, a `locator` naming the equations or table (from the spec's sources table:
      Viechtbauer and Cheung eqs 13-27, section 5.1 and Table I, section 5.2 and Table II; Henderson 1975 eq 5; Searle et al. 1992 section 7.6,
      eqs 52, 53; Raudenbush 2009 Table 16.2, eqs 16.23-16.25; the metafor 5.2-1 `ranef` help page for the t intervals and EE BLUPs) and `read`
- [x] The property cases `property.uni.mean-shift.dl.bcg` (the paper's fit) and `property.uni.mean-shift.reml.bcg` (`~ ablat`, tight control) in
      `src/tests/correctness/cases/properties.toml`, check `test_properties.uni_mean_shift`
- [x] A case whose check disagrees with metafopy beyond its tolerance is committed with `status = "mismatch"` and a `reason` naming both values,
      the deleted study or column, whether R metafor 5.2-1 agrees with metafopy, and "Recorded on <date>, not yet reviewed"; never loosen a
      tolerance or change a case's data to make it pass
- [x] Run `pytest src/tests/correctness -m 'not slow'`

**Acceptance Criteria:**

- [x] The coverage gate passes: every new item has a case or an exemption
- [x] Every case passes or is a strict expected failure with a reason

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `pytest src/tests/correctness -m 'not slow'` passes

**Notes:**

- Depends on Task 2
- The printed values and the Table II rows are in `agents/specs/correctness-uni-diagnostics/measurements.md`; check each against the PDF
  `/Users/simonb/src/metafopy papers/Outlier and influence diagnostics for meta-analysis.pdf` (`pdftotext -layout` into a scratch directory outside
  the repository) before writing it
- The `uni_deletion` check calls `influence` without `btt`: metafopy's `btt` also restricts `cov.r` to those coefficients, which the paper's eq 25 (all coefficients) does not define; `cooks_distance` keeps the case's `btt`
- None of the cases is `slow` (each measured under 0.2 s with its reference)

---

## Task 4: Sources and the mismatch record

**Status:** todo
**Priority:** medium
**Complexity:** small
**Spec Reference:** Implementation steps 4 and 5 (the record)

**Technical Goal:** every locator is checked against its paper, and `mismatches.md` records the automatic checks and an item for each mismatch case.

**Subtasks:**

- [ ] Check each case's `source.locator` against the paper it names in `/Users/simonb/src/metafopy papers/` (text extracted outside the
      repository); correct any that names the wrong equation or table. No work stops here: every source is in the folder
- [ ] Write `agents/specs/correctness-uni-diagnostics/mismatches.md` in the shape of `agents/specs/correctness-mv-diagnostics/mismatches.md`:
      header (task list, spec), "What is reviewed and how", "Automatic checks" (dated: the `not slow` run, and that every `status = "mismatch"`
      case of this list has an item), one item per mismatch case (case, source, reference, metafopy and R metafor values, cause if found), a
      "Decisions" entry per item with the options *follow the reference*, *keep metafor* and *investigate*, each with its consequence and a
      recommendation, and an empty `## Answers` section. If there is no mismatch case, say so under Automatic checks and write "None: no answer
      needed" under Answers
- [ ] Run `pytest src/tests/correctness -m 'not slow'`

**Acceptance Criteria:**

- [ ] Every locator names the equation or table its value comes from
- [ ] Every `mismatch` case has an item with its figures, options and a recommendation, or the document says there is none

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `pytest src/tests/correctness -m 'not slow'` passes

**Notes:**

- Depends on Task 3

---

## Task 5: The mismatch review

**Status:** todo
**Priority:** high
**Complexity:** small
**Spec Reference:** Implementation step 5

**Technical Goal:** each mismatch case has an answer before the list closes; with none, the task passes straight through.

**Subtasks:**

- [ ] If `agents/specs/correctness-uni-diagnostics/mismatches.md` says there is no mismatch case, mark every remaining subtask done with the note
      "no mismatch case" and set this task `done`
- [ ] Otherwise run `pytest src/tests/correctness -m 'not slow'` (the quick form before a human review)
- [ ] **Stops for the operator: the mismatch review.** Set this task `blocked` with "waiting for the mismatch review in
      agents/specs/correctness-uni-diagnostics/mismatches.md". An operator session publishes the review page, and once every item has an answer it
      reads the page's answers, copies each into the Answers section with the date, commits that, and only then restarts the list

**Acceptance Criteria:**

- [ ] Either the document says there is no mismatch, or its Answers section holds a dated answer for every item

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `pytest src/tests/correctness -m 'not slow'` passes

**Notes:**

- Depends on Task 4

---

## Task 6: Applying the answers, the full test run, and closing

**Status:** todo
**Priority:** high
**Complexity:** small
**Spec Reference:** Implementation step 6

**Technical Goal:** any answers are acted on, duplicated code is hoisted, the whole branch is checked once, the spec is marked implemented and the
`TODO.md` item moves to `DONE.md`.

**Subtasks:**

- [ ] Read the answers from the Answers section of `agents/specs/correctness-uni-diagnostics/mismatches.md`, never from a page; if a mismatch item
      has no answer there, stop as `blocked`, naming what is missing
- [ ] For each *follow the reference* answer, file a `TODO.md` item (the fix, flipping the case, marking the parity fixtures, `known.toml`,
      `docs/differences.md`) at the top of `TODO.md`; for *keep metafor*, add "accepted as a difference from the reference on <date>" to the case's
      `reason`; for *investigate*, file a `TODO.md` item with the figures
- [ ] Hoist anything this list duplicates from the rma_mv diagnostics work to a single owner: grep `src/tests/correctness` for each literal
      collection, constant and one-line helper the new code defines (quantile choice for intervals, the refit-and-compare loop, NaN-naming), and
      import rather than copy. Choose the owner from the existing import graph; prefer a module the consumer already imports over adding a new edge
- [ ] Move the `TODO.md` item "No correctness case covers the rma_uni diagnostics" to the end of `DONE.md`: append its heading and body unchanged,
      then the line `Done: <date> - agents/specs/correctness-uni-diagnostics/spec.md, agents/tasks/correctness-uni-diagnostics/tasks-1.md.`, and
      remove it from `TODO.md`
- [ ] Set the spec's `**Status:**` to `implemented`, and its row in `agents/specs/index.md` from "Approved (2026-11-20)" to "Implemented"
- [ ] Run the final tests from the worktree and save their whole output for the merge gate (the main checkout is the first entry of
      `git worktree list --porcelain`, `/Users/simonb/src/metafopy`):
      `mkdir -p /Users/simonb/src/metafopy/ralpher-logs/correctness-uni-diagnostics/tasks-1 && export PATH=/Users/simonb/src/metafopy/.venv/bin:$PATH PYTHONPATH=$PWD/src && pytest src/tests 2>&1 | tee /Users/simonb/src/metafopy/ralpher-logs/correctness-uni-diagnostics/tasks-1/final-tests.log`

**Acceptance Criteria:**

- [ ] Every answer is acted on as above
- [ ] The spec says `implemented`; the item is in `DONE.md` with its body unchanged and gone from `TODO.md`

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `pre-commit run --all-files` passes
- [ ] `pre-commit run auditme-all --hook-stage manual --all-files` passes
- [ ] `mypy` passes
- [ ] The final test run above passes
- [ ] `git show $(git log --diff-filter=A --format=%H -- agents/tasks/correctness-uni-diagnostics/tasks-1.md):TODO.md > <scratch>/todo-before.md` and
      `python3 ~/.claude/skills/todo-review/verify_bodies.py <scratch>/todo-before.md TODO.md DONE.md` exits 0 (`<scratch>` is a scratch directory
      outside the repository)

**Notes:**

- Depends on Task 5
- The full run can outlast a ten-minute command limit; run it in the background with the longest limit the session allows, and wait for its exit
  status before going on
