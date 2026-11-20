# Tasks: Correctness tests: user weights

**Spec:** `agents/specs/correctness-ls-weights/spec.md`
**Created:** 2026-11-20
**Version:** 1

**Running tests in the worktree:** the worktree has no `.venv`, and the main checkout's venv imports `metafopy` from the main checkout's `src/` (its
editable install), so `PATH=/Users/simonb/src/metafopy/.venv/bin:$PATH pytest ...` alone tests the code on `main`, not this list's. Run every pytest
and Python check from the worktree root as `export PATH=/Users/simonb/src/metafopy/.venv/bin:$PATH PYTHONPATH=$PWD/src`, and check once with
`python -c 'import metafopy; print(metafopy.__file__)'` that it prints this worktree's path.

The spec (version 1.1, approved by the operator on 2026-11-20 in `agents/reviews/correctness-ls-weights-spec.md`, every decision answered) adds
derived and property correctness cases for user `weights` in `rma_uni`, with a scale model and without one (decision D1, included), against Hedges,
Tipton and Johnson (2010) equations 3 and 4, with four exemptions. No file under `src/metafopy/` changes; no version bump. Decision D2 (`optbeta=True`
with `weights`) was filed and fixed separately (`0f1f1fc`); this list does not touch it. The list follows the sibling list
`agents/tasks/correctness-ls-extensions/tasks-1.md` (references in `formulas.py` that never import metafopy, checks in `test_derived.py` and
`test_properties.py`, items in `test_coverage.py`); Task 5 hoists anything the new code duplicates from it.

**Operator stops:** Task 4 stops for the mismatch review only if a mismatch case appears; the spec expects none.

---

## Task 1: User weights in the rma_uni reference

**Status:** done
**Priority:** high
**Complexity:** small
**Spec Reference:** API (formulas.py); Definitions and references; Implementation step 1

**Technical Goal:** `formulas.fixed_effects_fit` takes its user-weights branch (Hedges, Tipton and Johnson 2010, equations 3 and 4) for every method
given `weights`, so the plain `rma_uni` cases can use the existing `expect` formulas.

**Subtasks:**

- [x] Starting-point check: `grep -nwE 'ls_weights|uni_weights_items|test_weights_constant_is_unweighted|test_weights_rescale_invariant' src/tests/correctness/*.py`
      prints nothing; stop as `blocked`, naming the match, if it does (the existing `ls_unweighted` check and `UNI_EXEMPT` constant are not
      clashes)
- [x] In `src/tests/correctness/formulas.py`, change `fixed_effects_fit` so the branch that weights the coefficients by `diag(weights)` and returns
      the sandwich standard errors runs whenever `weights` is given, not only for `GENQ` and `GENQM`; keep the requirement (the `assert`) that
      `GENQ` and `GENQM` are given `weights`; give its docstring the spec's API text and cite Hedges, Tipton and Johnson (2010), Res Synth Methods
      1:39-65, doi:10.1002/jrsm.5, equations 3 and 4, beside the existing Jackson et al. (2014) comment; `rma_tau2` is unchanged
- [x] Run `pytest src/tests/correctness -m 'not slow' -k 'genq or GENQ'` (the existing `tau2.genq.*` and GENQ interval cases must still pass)

**Acceptance Criteria:**

- [x] `fixed_effects_fit(yi, vi, 'REML', weights=w)` returns the user-weighted coefficients and sandwich standard errors, and without `weights` the
      inverse-variance ones, as before
- [x] `formulas.py` still imports no metafopy module (the harness's import-ban test passes)

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] The pytest run of the last subtask passes

**Notes:**

- Key file: `src/tests/correctness/formulas.py` (`fixed_effects_fit`, about line 1370 on 2026-11-20)
- Measured values to expect are in `agents/specs/correctness-ls-weights/measurements.md`

---

## Task 2: The check, the properties, the coverage items and the exemptions

**Status:** done
**Priority:** high
**Complexity:** medium
**Spec Reference:** API (test_derived.py, test_properties.py); Coverage list; Error handling; Implementation step 2

**Technical Goal:** the `ls_weights` check and the two properties exist, and the coverage gate names every new item, failing until Task 3 adds the
cases.

**Subtasks:**

- [x] Add `ls_weights` to `src/tests/correctness/test_derived.py`, with the spec's signature and docstring: `alpha` against `ls_reference(case)` at
      `LS_RELATIVE`; `beta` and `vb` against `correctness.formulas.gls(yi, numpy.diag(vi + tau2_i), X, numpy.diag(weights))` at metafopy's
      `tau2_i` with `numbers_agree`; the weights read from the case's frame by the column the case's `weights` argument names; `va` not compared.
      Model it on `ls_unweighted` and reuse `ls_inputs`
- [x] Add `test_weights_constant_is_unweighted` and `test_weights_rescale_invariant` to `src/tests/correctness/test_properties.py` as the spec's
      API section describes: each fits a plain and a location-scale model (`dat.bcg` RR, `mods = ~ ablat`, and `scale = ~ ablat` for the second),
      hypothesis draws `c` from `floats(1e-3, 1e3)`, `@hypothesis.settings(max_examples=20, deadline=10000)`, compared at `DERIVED_TOLERANCE`;
      reuse the module's existing data helpers where one fits
- [x] In `src/tests/correctness/test_coverage.py`: add `ls.weights`, `ls.weights.fitstats` and `ls.weights.knha` to `ls_items()`, and a new
      `uni_weights_items()` returning `rma_uni.weights`, `rma_uni.weights.fitstats` and `rma_uni.weights.knha`, called from `coverage_items()`
- [x] Add the `[[exempt]]` entries, reason "defined only in metafor's code; R parity covers it": `ls.weights.fitstats` and `ls.weights.knha` in
      `src/tests/correctness/cases/location_scale.toml`, `rma_uni.weights.fitstats` and `rma_uni.weights.knha` in
      `src/tests/correctness/cases/inference.toml`
- [x] Run `pytest src/tests/correctness/test_properties.py -k weights` and `pytest src/tests/correctness/test_coverage.py`

**Acceptance Criteria:**

- [x] Both properties pass
- [x] The coverage gate fails listing exactly `ls.weights` and `rma_uni.weights` as missing (the exempt items are not listed)

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] The property run of the last subtask passes; the coverage run fails only as the acceptance criterion says

**Notes:**

- Depends on Task 1
- Key files: `src/tests/correctness/test_derived.py`, `src/tests/correctness/test_properties.py`, `src/tests/correctness/test_coverage.py`

---

## Task 3: The cases

**Status:** done
**Priority:** high
**Complexity:** medium
**Spec Reference:** Cases; Implementation step 3

**Technical Goal:** the six location-scale and four plain `rma_uni` cases of the spec exist with their sources, and the coverage gate passes.

**Subtasks:**

- [x] In a new section of `src/tests/correctness/cases/location_scale.toml`, add `ls.weights.ML.log.bcg`, `ls.weights.REML.log.bcg`,
      `ls.weights.ML.identity.bcg`, `ls.weights.REML.identity.bcg` (`dat.bcg`, `escalc` RR as the existing `dat.bcg` cases, `weights = "tpos"`,
      `mods = scale = ~ ablat`), `ls.weights.ML.log.bd` and `ls.weights.REML.identity.bd` (`dat.bangertdrowns2004`, `weights = "ni"`,
      `mods = scale = ~ imag`), each `check = "test_derived.ls_weights"`, `covers = ["ls.weights"]`
- [x] In `src/tests/correctness/cases/inference.toml`, add `uni.weights.dl.bcg` (DL), `uni.weights.dl.bcg.ablat` (DL, `~ ablat`),
      `uni.weights.reml.bcg.ablat` (REML, `~ ablat`) on `dat.bcg` RR with `weights = "tpos"`, and `uni.weights.reml.bd` (REML) on
      `dat.bangertdrowns2004` with `weights = "ni"`, each with `expect = { tau2 = "formulas.rma_tau2", beta = "formulas.rma_beta", se =
      "formulas.rma_se" }` and `covers = ["rma_uni.weights"]`; the REML ones pass `control = { threshold = 1e-13, tol = 1e-13, maxiter = 1000,
      tau2_max = 1e6 }`
- [x] Each case's `source` has `citation` (Hedges LV, Tipton E, Johnson MC (2010). Robust variance estimation in meta-regression with dependent
      effect size estimates. Res Synth Methods 1:39-65; for the location-scale cases also Viechtbauer W, Lopez-Lopez JA (2022). Location-scale
      models for meta-analysis. Res Synth Methods 13:697-715), `doi`, the `locator` of the spec's Implementation step 3, and `read = 2026-11-20`
- [x] A case whose check disagrees with metafopy beyond its tolerance is committed with `status = "mismatch"` and a `reason` naming both values,
      whether R metafor 5.2-1 agrees with metafopy, and "Recorded on <date>, not yet reviewed"; the spec expects none. Never loosen a tolerance or
      change a case's data to make it pass
- [x] Check each locator against the PDFs in `/Users/simonb/src/metafopy papers/` (`Robust variance estimation in meta-regression with dependent
      effect size estimates.pdf`, equations 3 and 4; `Viechtbauer_2022_Locationscale_models_for_metaanalysis.pdf`, equations 5 and 9)
- [x] Run `pytest src/tests/correctness -m 'not slow' -k 'weights'` and `pytest src/tests/correctness/test_coverage.py`

**Acceptance Criteria:**

- [x] Every case passes or is a strict expected failure with a reason
- [x] The coverage gate passes

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] The pytest runs of the last subtask pass

**Notes:**

- Depends on Task 2
- Expected agreement (measurements.md, 2026-11-20): `alpha` to 4.4e-08, `beta` and `vb` to 1e-15, plain REML `tau2` exact under the tight control

---

## Task 4: The mismatch review

**Status:** todo
**Priority:** high
**Complexity:** small
**Spec Reference:** Implementation step 4

**Technical Goal:** each mismatch case has a review item, and the list stops until each has an answer; with no mismatch case it goes straight on.

**Subtasks:**

- [ ] Write `agents/specs/correctness-ls-weights/mismatches.md` in the shape of `agents/specs/correctness-ls-extensions/mismatches.md`: header
      (task list, spec, review page line), "What is reviewed and how", "Automatic checks" (dated: the `not slow` run, and that every
      `status = "mismatch"` case of this list has an item), one item per mismatch case (case, source, reference, metafopy and R metafor values,
      cause if found), a "Decisions" entry per item with the options *follow the reference*, *keep metafor* and *investigate*, each with its
      consequence and a recommendation, and an empty `## Answers` section. If there is no mismatch case, say so under Automatic checks, write
      "None: no answer needed" under Answers, mark the remaining subtasks done with the note "no mismatch case" and set this task `done`
- [ ] Run `pytest src/tests/correctness -m 'not slow'` (the quick form before a human review)
- [ ] **Stops for the operator: the mismatch review** (only if a mismatch case exists). Set this task `blocked` with "waiting for the mismatch
      review in agents/specs/correctness-ls-weights/mismatches.md". An operator session publishes the review page, and once every item has an answer
      it reads the page's answers, copies each into the Answers section with the date, commits that, and only then restarts the list

**Acceptance Criteria:**

- [ ] Every `mismatch` case has an item with its figures, options and a recommendation, or the document says there is none

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `pytest src/tests/correctness -m 'not slow'` passes

**Notes:**

- Depends on Task 3

---

## Task 5: Applying the answers, the full test run, and closing

**Status:** todo
**Priority:** high
**Complexity:** small
**Spec Reference:** Implementation step 5

**Technical Goal:** any answers are acted on, duplicated code is hoisted, the whole branch is checked once, the spec is marked implemented and the
`TODO.md` item moves to `DONE.md`.

**Subtasks:**

- [ ] Read the answers from the Answers section of `agents/specs/correctness-ls-weights/mismatches.md`, never from a page; if a mismatch item has no
      answer there, stop as `blocked`, naming what is missing ("None: no answer needed" is complete)
- [ ] For each *follow the reference* answer, file a `TODO.md` item (the fix, flipping the case, marking the parity fixtures, `known.toml`,
      `docs/differences.md`) at the top of `TODO.md`; for *keep metafor*, add "accepted as a difference from the reference on <date>" to the case's
      `reason`; for *investigate*, file a `TODO.md` item with the figures
- [ ] Hoist anything this list duplicates from the location-scale extensions work to a single owner: grep `src/tests/correctness` for each literal
      collection, constant and one-line helper the new code defines (the weights column read, the sandwich built in `ls_weights` beside
      `ls_unweighted`'s), and import rather than copy. Choose the owner from the existing import graph; prefer a module the consumer already
      imports over adding a new edge
- [ ] Move the `TODO.md` item "No correctness case covers user weights with a scale model" to the end of `DONE.md`: append its heading and body
      unchanged, then the line `Done: <date> - agents/specs/correctness-ls-weights/spec.md, agents/tasks/correctness-ls-weights/tasks-1.md.`, and
      remove it from `TODO.md`
- [ ] Set the spec's `**Status:**` to `implemented`, and its row in `agents/specs/index.md` from "Approved (2026-11-20)" to "Implemented"
- [ ] Run the final tests from the worktree and save their whole output for the merge gate (the main checkout is the first entry of
      `git worktree list --porcelain`, `/Users/simonb/src/metafopy`):
      `mkdir -p /Users/simonb/src/metafopy/ralpher-logs/correctness-ls-weights/tasks-1 && export PATH=/Users/simonb/src/metafopy/.venv/bin:$PATH PYTHONPATH=$PWD/src && pytest src/tests 2>&1 | tee /Users/simonb/src/metafopy/ralpher-logs/correctness-ls-weights/tasks-1/final-tests.log`

**Acceptance Criteria:**

- [ ] Every answer is acted on as above
- [ ] The spec says `implemented`; the item is in `DONE.md` with its body unchanged and gone from `TODO.md`

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `pre-commit run --all-files` passes
- [ ] `pre-commit run auditme-all --hook-stage manual --all-files` passes
- [ ] `mypy` passes
- [ ] The final test run above passes
- [ ] `git show $(git log --diff-filter=A --format=%H -- agents/tasks/correctness-ls-weights/tasks-1.md):TODO.md > <scratch>/todo-before.md` and
      `python3 ~/.claude/skills/todo-review/verify_bodies.py <scratch>/todo-before.md TODO.md DONE.md` exits 0 (`<scratch>` is a scratch directory
      outside the repository)

**Notes:**

- Depends on Task 4
- The full run can outlast a ten-minute command limit; run it in the background with the longest limit the session allows, and wait for its exit
  status before going on
