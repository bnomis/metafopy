# Tasks: Correctness tests: rma_mv diagnostics

**Spec:** `agents/specs/correctness-mv-diagnostics/spec.md`
**Created:** 2026-11-19
**Version:** 1

**Running tests in the worktree:** the worktree has no `.venv`, and the main checkout's venv imports `metafopy` from the main checkout's `src/` (its
editable install), so `PATH=/Users/simonb/src/metafopy/.venv/bin:$PATH pytest ...` alone tests the code on `main`, not this list's. Run every pytest
and Python check from the worktree root as `export PATH=/Users/simonb/src/metafopy/.venv/bin:$PATH PYTHONPATH=$PWD/src`, and check once with
`python -c 'import metafopy; print(metafopy.__file__)'` that it prints this worktree's path.

The spec (version 1.1, approved by the operator on 2026-11-19 in `agents/reviews/correctness-mv-diagnostics-spec.md`) adds derived and property
correctness cases for the `rma_mv` diagnostics `cooks_distance`, `dfbetas`, `rstudent` and `ranef`. No file under `src/metafopy/` changes; no version
bump. The list follows the sibling harness extensions (`correctness-mv-robust`, `correctness-location-scale`): references in `formulas.py` that never
import metafopy, checks in `test_derived.py` and `test_properties.py`, cases in `cases/rma_mv.toml`, items in `test_coverage.py`.

**Operator stops:** Task 4 stops if any paper the spec marks "supply" is missing from the papers folder `/Users/simonb/src/metafopy papers/`
(Viechtbauer and Cheung 2010; Christensen, Pearson and Johnson 1992; Henderson 1975; Searle, Casella and McCulloch 1992); placing them there before the
run lets it go straight through. Task 5 stops for the review of the mismatch cases (at least `mv.ranef.gen.crossed`, expected by the spec).

---

## Task 1: The references mv_deletion and mv_blups

**Status:** done
**Priority:** high
**Complexity:** medium
**Spec Reference:** Definitions and references; API (formulas.py); Error handling; Implementation step 1

**Technical Goal:** independent brute-force references for the deletion diagnostics and the BLUPs exist in `formulas.py`, with their own harness tests.

**Subtasks:**

- [x] Starting-point check: `grep -nwE 'MVDeletion|mv_deletion|MVBlups|mv_blups' src/tests/correctness/formulas.py src/tests/correctness/test_derived.py`
      prints nothing; stop as `blocked`, naming the match, if it does
- [x] Add `MVDeletion`, `mv_deletion`, `MVBlups` and `mv_blups` to `src/tests/correctness/formulas.py` with the signatures and docstrings of the
      spec's API section, computed as its "Definitions and references" section says: for each deleted set (one row, or the rows of one cluster), a
      refit on the kept rows with `mv_fit` (`reestimate=True`) or the full-data components (`reestimate=False`), then GLS for `b_(S)` and `vb_(S)`,
      Cook's distance over `btt`, DFBETAS with `vb*`, the deleted residual, its standard error, `z` and `X2`; `mv_blups` by Henderson's mixed model
      equations with `R = V`, all terms at once, the prediction error variances from the lower block of the inverse coefficient matrix. `numpy` and
      `scipy` only; no metafopy import
- [x] The errors of the spec's Error handling section: `ValueError` naming the term for a singular `G`, `ValueError` for a deletion that leaves
      `X[K]` rank deficient; each refit's `success` and gradient norm recorded in `converged` and `gradient_norms`
- [x] Harness tests in `src/tests/correctness/test_coverage.py` (beside the existing formula tests): on a small inline dataset, `mv_deletion` with
      `reestimate=False` against the closed-form rma.uni deletion formulas, and `mv_blups` against the marginal form `G Z' M^-1 (y - X b)`
- [x] Run `pytest src/tests/correctness/test_coverage.py`

**Acceptance Criteria:**

- [x] Both references reproduce their closed forms on the inline data to `1e-10`
- [x] `formulas.py` still imports no metafopy module (the harness's import-ban test passes)

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] The pytest run of the last subtask passes

**Notes:**

- Key files: `src/tests/correctness/formulas.py` (`mv_fit`, `RandomTerm`, the existing `rma_mv` helpers), `src/tests/correctness/test_coverage.py`
- Measured values to expect are in `agents/specs/correctness-mv-diagnostics/measurements.md`; the measurement script of 2026-11-19 was outside the
  repository and is not needed

---

## Task 2: The checks, the coverage items and the exemption

**Status:** done
**Priority:** high
**Complexity:** medium
**Spec Reference:** API (test_derived.py); Coverage list; Implementation step 2

**Technical Goal:** the derived and property checks exist, and the coverage gate lists the new items, failing until Task 3 adds their cases.

**Subtasks:**

- [x] Add the checks `mv_deletion` and `mv_ranef` to `src/tests/correctness/test_derived.py` as the spec's API section describes, both built on the
      existing `mv_setup`; groups from `pandas.factorize(frame[cluster], sort=True)`; tolerances `MV_COMPONENT_TOLERANCE` (`reestimate=True`) and
      `MV_TIGHT_TOLERANCE` (`reestimate=False`, `ranef`); a reference refit that did not converge fails the case naming the case, the deleted set and
      the optimiser message, and a NaN from metafopy where the reference converges fails naming the set
- [x] Add the property checks of the spec's Cases section to `src/tests/correctness/test_properties.py` (`PROPERTY_TOLERANCE`): `rma_mv(random = ~ 1 |
      id)` with one row per id on `dat.bcg` (`~ ablat`) gives `rma_uni`'s Cook's distance, DFBETAS, `rstudent` and `ranef` with
      `control={'threshold': 1e-12, 'maxiter': 1000}` on `rma_uni`; `cluster` with one value per row gives the per-row results; a one-row cluster's
      `X2` equals its `z` squared; `reestimate=True` and `False` agree when every component is fixed
- [x] Add `MV_DIAGNOSTICS`, `MV_DELETIONS` and `mv_diagnostics_items()` to `src/tests/correctness/test_coverage.py` and include them in
      `coverage_items()`: the items of the spec's Coverage list; add the `[[exempt]]` entry for `rma_mv.ranef.pi.t` with the spec's reason to
      `src/tests/correctness/cases/rma_mv.toml`
- [x] Run `pytest src/tests/correctness/test_properties.py src/tests/correctness/test_derived.py -k "mv"`: the property cases pass; the coverage
      gate is expected to fail until Task 3, so do not run it here

**Acceptance Criteria:**

- [x] The property cases pass
- [x] `coverage_items()` lists every item of the spec's Coverage list

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] The pytest run of the last subtask passes

**Notes:**

- Depends on Task 1
- `LS_PREDICT` in `test_coverage.py` is the model for constants that no list in the code names

---

## Task 3: The cases

**Status:** done
**Priority:** high
**Complexity:** medium
**Spec Reference:** Cases; Implementation step 3

**Technical Goal:** every new coverage item has a derived case, and the expected mismatch is recorded.

**Subtasks:**

- [x] Add the cases of the spec's Cases table to `src/tests/correctness/cases/rma_mv.toml`, each `kind = "derived"` with the fit, `then` and `covers`
      the table gives, `check = "test_derived.mv_deletion"` or `"test_derived.mv_ranef"`, and a `[case.source]` naming the definition it uses
      (`locator` to be completed by Task 4)
- [x] `mv.ranef.gen.crossed`: if it fails as the spec expects, give it `status = "mismatch"` and a `reason` with metafopy's, R metafor's (from
      `measurements.md`) and the reference's values for the clusters that differ most, "metafopy and R metafor agree; recorded on <date of this
      task>, not yet reviewed"
- [x] Any other case that fails: give it `status = "mismatch"` and a `reason` with the figures, the same way, and list it in this task's Notes
- [x] Run `pytest src/tests/correctness/test_derived.py src/tests/correctness/test_properties.py src/tests/correctness/test_coverage.py`

**Acceptance Criteria:**

- [x] The coverage gate passes: every new item has a case or the exemption
- [x] Every failing case is a strict `mismatch` with its figures in its `reason`

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] The pytest run of the last subtask passes

**Notes:**

- Depends on Task 2
- Mismatch: `mv.ranef.gen.crossed` only (as the spec expects); every other case passes.
- None of the cases is `slow` (measurements.md: no fit took more than a few seconds with its reference)

---

## Task 4: The sources

**Status:** done
**Priority:** high
**Complexity:** small
**Spec Reference:** Definitions and references; Implementation step 4

**Technical Goal:** each case's source names the equation it rests on, read from the paper, and the definitions are checked against the papers.

**Subtasks:**

- [x] Create `agents/specs/correctness-mv-diagnostics/mismatches.md` (modelled on `agents/specs/correctness-mv-robust/mismatches.md`) with a
      "Sources needed" section listing each "supply" row of the spec's reference table and the equations wanted
- [x] Look in `/Users/simonb/src/metafopy papers/` for each of those papers (by author and year in the file name or first page). **Stops for the
      operator if any is missing:** set this task `blocked`, naming each missing paper and that the operator either places it in the papers folder
      or drops it in this task's Notes, and change nothing else
- [x] For each paper present, extract it with `pdftotext -layout` into a scratch directory outside the repository, and check the spec's definitions
      against it (Cook's distance, DFBETAS and the studentised residual of Viechtbauer and Cheung; Christensen et al.'s set deletion; the mixed model
      equations and their inverse). Record each check in this task's Notes. If a definition differs (a factor `1/p` in Cook's distance, another
      variance in DFBETAS), change the reference in `formulas.py` to the paper's, rerun Task 3's tests, and make any new disagreement a `mismatch`
      case as Task 3 does
- [x] Fill each case's `source.locator` with the paper and equation, and mark each row of "Sources needed" as read (or dropped, as the Notes say)
- [x] Run `pytest src/tests/correctness/test_derived.py src/tests/correctness/test_properties.py src/tests/correctness/test_coverage.py`

**Acceptance Criteria:**

- [x] Every case's locator names its paper and equation, or the paper is recorded as dropped by the operator
- [x] The Notes record each definition check

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] The pytest run of the last subtask passes

**Notes:**

- Depends on Task 3
- Robinson (1991, doi:10.1214/ss/1177011926) is open access on Project Euclid; the other papers need the operator
- Read 2026-11-19; each definition check is recorded in `agents/specs/correctness-mv-diagnostics/mismatches.md` (Definition checks). Summary: Cook's
  distance, DFBETAS, `rstudent` and the mixed model equations match the papers; Christensen et al. divide Cook's distance by `p` where Viechtbauer and
  Cheung (and metafor) do not, so the reference keeps Viechtbauer and Cheung's form and no case changed.
- Robinson (1991) could not be fetched (Project Euclid refused the download); it is not a "supply" paper and no case cites it.

---

## Task 5: The mismatch review

**Status:** todo
**Priority:** high
**Complexity:** small
**Spec Reference:** Implementation step 5

**Technical Goal:** each mismatch case has a review item, and the list stops until each has an answer.

**Subtasks:**

- [ ] In `agents/specs/correctness-mv-diagnostics/mismatches.md`, add one item per `mismatch` case (from Tasks 3 and 4): the case, the source, the
      reference's, metafopy's and R metafor's values, and the options *follow the reference*, *keep metafor* and *investigate*, each with its
      consequence and a recommendation, in the shape of the shared `reviews.md` rule; add an empty `## Answers` section
- [ ] Run `pytest src/tests/correctness -m 'not slow'` (the quick form before a human review)
- [ ] **Stops for the operator: the mismatch review.** Set this task `blocked` with "waiting for the mismatch review in
      agents/specs/correctness-mv-diagnostics/mismatches.md". An operator session publishes the review page, and once every item has an answer it
      reads the page's answers, copies each into the Answers section with the date, commits that, and only then restarts the list

**Acceptance Criteria:**

- [ ] Every `mismatch` case has an item with its figures, options and a recommendation

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

**Technical Goal:** the answers are acted on, the whole branch is checked once, the spec is marked implemented, and the `TODO.md` item records that
its `rma_mv` part is done.

**Subtasks:**

- [ ] Read the answers from the Answers section of `agents/specs/correctness-mv-diagnostics/mismatches.md`, never from the page; if it is empty or
      lacks an answer for any item, stop as `blocked`, naming what is missing
- [ ] For each *follow the reference* answer, file a `TODO.md` item (the fix, flipping the case, marking the parity fixtures, `known.toml`,
      `docs/differences.md`), placed before "Correctness coverage the extension specs left for later"; for *keep metafor*, add "accepted as a
      difference from the reference on <date>" to the case's `reason`; for *investigate*, file a `TODO.md` item with the figures
- [ ] In the `TODO.md` item "Correctness coverage the extension specs left for later", add after its last line a dated note: "Note <date>: the
      rma_mv diagnostics part is done (`agents/specs/correctness-mv-diagnostics/spec.md`, `agents/tasks/correctness-mv-diagnostics/tasks-1.md`); the
      location-scale part remains." The item stays in `TODO.md`
- [ ] Set the spec's `**Status:**` to `implemented`, and its row in `agents/specs/index.md` from "Approved (2026-11-19)" to "Implemented"
- [ ] Run the final tests from the worktree and save their whole output for the merge gate (the main checkout is the first entry of
      `git worktree list --porcelain`, `/Users/simonb/src/metafopy`):
      `mkdir -p /Users/simonb/src/metafopy/ralpher-logs/correctness-mv-diagnostics/tasks-1 && export PATH=/Users/simonb/src/metafopy/.venv/bin:$PATH PYTHONPATH=$PWD/src && pytest src/tests 2>&1 | tee /Users/simonb/src/metafopy/ralpher-logs/correctness-mv-diagnostics/tasks-1/final-tests.log`

**Acceptance Criteria:**

- [ ] Every answer is acted on as above
- [ ] The spec says `implemented`; the `TODO.md` item carries the dated note and is unchanged otherwise

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `pre-commit run --all-files` passes
- [ ] `pre-commit run auditme-all --hook-stage manual --all-files` passes
- [ ] `mypy` passes
- [ ] The final test run above passes
- [ ] `git show $(git log --diff-filter=A --format=%H -- agents/tasks/correctness-mv-diagnostics/tasks-1.md):TODO.md > <scratch>/todo-before.md` and
      `python3 ~/.claude/skills/todo-review/verify_bodies.py <scratch>/todo-before.md TODO.md DONE.md` exits 0 (`<scratch>` is a scratch directory
      outside the repository)

**Notes:**

- Depends on Task 5
- The full run can outlast a ten-minute command limit; run it with the longest limit the session allows
- Hoist nothing: the new references share `mv_fit` and `mv_setup` with the existing `rma_mv` checks by calling them, not copying
