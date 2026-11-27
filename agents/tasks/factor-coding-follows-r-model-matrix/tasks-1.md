# Tasks: Factor coding follows R's model matrix

**Spec:** `agents/specs/factor-coding-follows-r-model-matrix/spec.md`
**Created:** 2026-11-27
**Version:** 1

The spec was approved on 2026-11-27 with every item approved (S1-S6 in
`agents/reviews/factor-coding-follows-r-model-matrix-spec.md`). Approving S1 confirms the output change the spec lists under Output changes: the
coefficients of a formula whose R matrix is rank-deficient (`~ f:g`, `~ f:g:h`, `~ f:g + g:h`, `~ 0 + f:g + h`), and `emmprep` raising on such a
fit. It confirms nothing else: a task that would change any other name, value or printout stops as `blocked` and names it. Approving S4 is the
instruction to add `interface-meta` to `pyproject.toml` and run `uv lock`; no other change to `uv.lock` is approved.

How to run things, in every task (from the worktree root):

- **Tests:** `export PATH=/Users/simonb/src/metafopy/.venv/bin:$PATH PYTHONPATH=$PWD/src && pytest <files>`
- **Type check:** `mypy <files>`, with the same `PATH`.
- **R:** `LC_ALL=C.UTF-8 R_LIBS=/Users/simonb/src/metafopy/.audit/Rlib Rscript <script>` (the pinned library holds metafor 5.2-1, jsonlite and
  emmeans).

---

## Task 1: Starting checks

**Status:** done
**Priority:** high
**Complexity:** small
**Spec Reference:** Motivation; Module location

**Technical Goal:** confirm the work has not started and the code is as the spec measured it, so later tasks start from the measured state.

**Subtasks:**

- [x] `grep -rnwE 'RPandasMaterializer|r_model_matrix' src/metafopy --include='*.py'` prints nothing; otherwise stop as `blocked` naming the match
- [x] `src/tests/test_formula_rank.py`, `src/tests/r/generate_formula_rank.R` and `src/tests/reference/formula_rank.json` do not exist; otherwise
      stop as `blocked` naming the file
- [x] `grep -rn 'formulaic.model_matrix(' src/metafopy --include='*.py'` prints exactly two calls, in `src/metafopy/formula.py` (`design_matrix`)
      and `src/metafopy/methods/emm_grid.py` (`reference_grid`); otherwise stop as `blocked` naming the extra or missing call
- [x] Run `pytest src/tests/test_formula.py src/tests/test_formula_names.py src/tests/test_emmeans_nesting.py`: it passes; otherwise stop as
      `blocked` naming the failures

**Acceptance Criteria:**

- [x] Every check above holds, and the task's Notes record the test run's pass and skip counts with the date

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `pytest src/tests/test_formula.py src/tests/test_formula_names.py src/tests/test_emmeans_nesting.py` passes

**Notes:**

- Run on 2026-11-27: 84 passed, 0 skipped
- Key files: `src/metafopy/formula.py`, `src/metafopy/methods/emm_grid.py`

---

## Task 2: Declare the `interface-meta` dependency

**Status:** todo
**Priority:** high
**Complexity:** small
**Spec Reference:** External dependencies

**Technical Goal:** metafopy will import `interface_meta` directly, so it is declared rather than reached through formulaic.

**Subtasks:**

- [ ] Add `interface-meta>=2.0` to `[project] dependencies` in `pyproject.toml`, in the list's existing order
- [ ] Run `uv lock` (from the worktree root, with `direnv exec /Users/simonb/src/metafopy`)
- [ ] `git diff uv.lock` shows only the metafopy package's dependency list gaining `interface-meta`, and the locked `interface-meta` version is
      unchanged (2.0.1 on 2026-11-27); any other change stops the task as `blocked`, naming it, and the lock is restored with `git checkout uv.lock`

**Acceptance Criteria:**

- [ ] `python -c 'import interface_meta; print(interface_meta.__version__)'` in the project venv prints the locked version
- [ ] `uv.lock` differs from its starting state only as above

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `uv lock --check` exits 0

**Notes:**

- Depends on Task 1
- Do not edit `uv.lock` by hand (`AGENTS.md`)

---

## Task 3: R's coding rule in `design_matrix`

**Status:** todo
**Priority:** high
**Complexity:** large
**Spec Reference:** API; Implementation (Scoping, No intercept, One builder); Error handling; Tests (`RPandasMaterializer`)

**Technical Goal:** `design_matrix` codes each factor of each term as R's `model.matrix` does, through one shared builder.

**Subtasks:**

- [ ] Write the failing tests first in `src/tests/test_formula.py`: on a frame with factors `f` (three levels), `g` (two), `h` (two) and numerics
      `x`, `z`, the column names of `design_matrix` for `~ f:g` (`intrcpt`, then all six `f`-by-`g` products, R's order), `~ 0 + f + g`
      (`fa fb fc gq` or the frame's level names), `~ 0 + x + f` (`x` then every level of `f`); one scoped term per formula term; a constant factor
      kept as the scoped term's `scale`; `ensure_full_rank=False` giving the same scoped terms
- [ ] Add `RPandasMaterializer` to `src/metafopy/formula.py` with the docstring in the spec's API section, overriding `_get_scoped_terms` with
      `@interface_meta.override`, by the rule in Implementation steps 1 and 2
- [ ] Add `r_model_matrix(rhs, data, context)` with the spec's docstring: `formulaic.model_matrix(r_power(r_term_order(rhs)), data,
      na_action='ignore', context=context, materializer=RPandasMaterializer)`
- [ ] Make `design_matrix` call `r_model_matrix` where it now calls `formulaic.model_matrix`; leave its signature, its errors and its naming as
      they are
- [ ] Run `src/tests/test_formula_names.py`: every existing entry still passes. A failure stops the task as `blocked`, naming the formula, since
      the spec says no formula outside the rank-deficient shapes changes

**Acceptance Criteria:**

- [ ] The new tests pass, and `design_matrix('~ deltype:outcome', ...)` on the spec's 12-row data gives R's seven columns
- [ ] `formulaic.model_matrix(` appears in `src/metafopy/formula.py` only inside `r_model_matrix`

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy src/metafopy/formula.py src/tests/test_formula.py` passes
- [ ] `pytest src/tests/test_formula.py src/tests/test_formula_names.py src/tests/test_formula_rewriting_properties.py` passes

**Notes:**

- Depends on Task 2
- The prototype measured on 2026-11-27 matched R's names and rank on all 20 formulas of the spec's Motivation; it took the evaluated factors as
  the base method does and yielded `ScopedTerm(factors=..., scale=...)` with `ScopedFactor(f, reduced=...)`
- formulaic 1.2.2: `PandasMaterializer` from `formulaic.materializers`, `ScopedTerm` and `ScopedFactor` from `formulaic.materializers.types`,
  `Factor.Kind` for categorical and constant factors

---

## Task 4: Extend the R names oracle

**Status:** todo
**Priority:** high
**Complexity:** medium
**Spec Reference:** Tests (R names oracle)

**Technical Goal:** R checks the new coding on every no-main-effect shape the spec names.

**Subtasks:**

- [ ] Save the current `src/tests/reference/formula_names.json` to the scratch directory
- [ ] Add a factor `arm` (two levels) to the data frame in `src/tests/r/generate_formula_names.R` and the same column to `DATA` in
      `src/tests/test_formula_names.py`
- [ ] Add the spec's formulas: `~ deltype:outcome`, `~ outcome:deltype`, `~ deltype:outcome:arm`, `~ deltype:outcome + outcome:arm`,
      `~ 0 + deltype:outcome + arm`, `~ 0 + deltype + outcome`, `~ 0 + x + deltype`, `~ x:deltype`, `~ outcome + deltype:outcome`
- [ ] Regenerate the JSON with the R command above
- [ ] Compare the saved and new JSON on the saved file's keys: every old entry is unchanged; otherwise stop as `blocked`, naming the entry

**Acceptance Criteria:**

- [ ] `formula_names.json` holds every old entry unchanged and the nine new formulas
- [ ] `src/tests/test_formula_names.py` passes on every entry, new ones included

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] The generator runs with the R command above and exits 0
- [ ] `pytest src/tests/test_formula_names.py` passes

**Notes:**

- Depends on Task 3

---

## Task 5: `emmprep` builds its grid through the same builder

**Status:** todo
**Priority:** high
**Complexity:** small
**Spec Reference:** API; Implementation (One builder, `emmprep` on rank-deficient fits); Tests (`emmprep` with an out-of-order interaction)

**User Story:** As an analyst, I want `emmprep` to accept a fit whose interaction is written in any order, so that `~ g + f:g` works as in
metafor.

**Subtasks:**

- [ ] Write the failing test in `src/tests/test_emmeans_nesting.py`: `rma_uni(mods='~ grp + lat:grp')` on its nested data; `emmprep` succeeds and
      `emmeans(grid, 'lat')` gives the same estimates as the fit written `~ grp + grp:lat`
- [ ] Add a test that `emmprep` on a `~ f:g` fit raises metafor's "Cannot use function when some redundant predictors were dropped from the
      model." (`emm_grid.py:255` on 2026-11-27)
- [ ] In `src/metafopy/methods/emm_grid.py`, replace `reference_grid`'s `formulaic.model_matrix(...)` call with
      `metafopy.formula.r_model_matrix(metafopy.formula.r_logical_formula(rhs), data, context)`; keep `spec.get_model_matrix` as it is

**Acceptance Criteria:**

- [ ] Both new tests pass, and every existing test in `src/tests/test_emmeans_nesting.py` still passes
- [ ] `grep -rn 'formulaic.model_matrix(' src/metafopy --include='*.py'` prints only the call inside `r_model_matrix`

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy src/metafopy/methods/emm_grid.py src/tests/test_emmeans_nesting.py` passes
- [ ] `pytest src/tests/test_emmeans_nesting.py src/tests/test_emmeans.py src/tests/test_emmeans_inference.py` passes

**Notes:**

- Depends on Task 3

---

## Task 6: The R fits oracle

**Status:** todo
**Priority:** high
**Complexity:** medium
**Spec Reference:** Tests (R fits oracle); Implementation (Redundant columns are already handled); Output changes

**Technical Goal:** R checks that `rma`, `rma.mv` and the scale model drop the redundant column and warn as metafor does.

**Subtasks:**

- [ ] Write `src/tests/r/generate_formula_rank.R`: the 12-row data set as literal values (not R's RNG), `deltype` three levels and `outcome` two,
      plus `year` and `id`; the fits `rma(yi, vi, mods = ~ deltype:outcome, method = "FE")`, `rma.mv(yi, vi, mods = ~ deltype:outcome,
      random = ~ 1 | id)` and `rma(yi, vi, mods = ~ year, scale = ~ deltype:outcome)`; for each, the coefficient names, estimates, standard errors
      and metafor's warning; and the R and metafor versions
- [ ] Choose `year` and the data so the location-scale fit converges in R; if no choice found converges, stop as `blocked` with what was tried
- [ ] Generate `src/tests/reference/formula_rank.json` with the R command above
- [ ] Write `src/tests/test_formula_rank.py`: fit each in metafopy and check the names, the estimates and standard errors (`rtol=1e-6`), and the
      warning text with `pytest.warns`
- [ ] A difference from R in a value stops the task as `blocked`, naming the fit and the value; the spec says no model changes

**Acceptance Criteria:**

- [ ] `src/tests/test_formula_rank.py` passes on all three fits

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] The generator runs with the R command above and exits 0
- [ ] `mypy src/tests/test_formula_rank.py` passes
- [ ] `pytest src/tests/test_formula_rank.py` passes

**Notes:**

- Depends on Task 3
- The location-scale fit's warning is "Redundant predictors dropped from the scale model." (measured 2026-11-27)

---

## Task 7: The user guide sentence

**Status:** todo
**Priority:** medium
**Complexity:** small
**Spec Reference:** Documentation

**User Story:** As an analyst, I want the guide to say how a rank-deficient `mods` formula is fitted, so that the warning does not surprise me.

**Subtasks:**

- [ ] Add one sentence to `docs/guide/models.md` where it describes `mods` (the paragraph beginning "Moderators are given as a formula in
      `mods`"): a formula whose matrix is rank-deficient is fitted as in metafor, with the redundant columns dropped and a warning
- [ ] `docs/differences.md` is not changed

**Acceptance Criteria:**

- [ ] The sentence is in `docs/guide/models.md` and names no count

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)

**Notes:**

- Depends on Task 6

---

## Task 8: The full local check and closing

**Status:** todo
**Priority:** high
**Complexity:** medium
**Spec Reference:** the whole spec

**Technical Goal:** the branch passes every check before the operator merges it, the spec says it is implemented, and the `TODO.md` item moves
to `DONE.md`.

**Subtasks:**

- [ ] Run the final tests and save their whole output for the merge gate. The main checkout is the first entry of
      `git worktree list --porcelain`, `/Users/simonb/src/metafopy`:
      `mkdir -p /Users/simonb/src/metafopy/ralpher-logs/factor-coding-follows-r-model-matrix/tasks-1 && export PATH=/Users/simonb/src/metafopy/.venv/bin:$PATH PYTHONPATH=$PWD/src && pytest src/tests 2>&1 | tee /Users/simonb/src/metafopy/ralpher-logs/factor-coding-follows-r-model-matrix/tasks-1/final-tests.log`
- [ ] Any failure stops the task as `blocked`, naming the test: the spec says no formula in the repository's tests changes
- [ ] Set the spec's `**Status:**` to `implemented`, and its row in `agents/specs/index.md` to say so
- [ ] Move the `TODO.md` item "A two-factor interaction with neither main effect gets other columns than in R" to the end of `DONE.md`: append its
      heading and body unchanged, then `Done: <date> - agents/specs/factor-coding-follows-r-model-matrix/spec.md,
      agents/tasks/factor-coding-follows-r-model-matrix/tasks-1.md.`, and remove it from `TODO.md`

**Acceptance Criteria:**

- [ ] The final test run passes, the spec says `implemented`, and the item is in `DONE.md` with its `Done:` line

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `pre-commit run --all-files` passes
- [ ] `pre-commit run auditme-all --hook-stage manual --all-files` passes
- [ ] `mypy src/metafopy` passes
- [ ] The final test run above passes
- [ ] `git show $(git log --diff-filter=A --format=%H -- agents/tasks/factor-coding-follows-r-model-matrix/tasks-1.md):TODO.md > <scratch>/todo-before.md`
      and `python3 ~/.claude/skills/todo-review/verify_bodies.py <scratch>/todo-before.md TODO.md DONE.md` exits 0

**Notes:**

- Depends on Tasks 1-7
- The list does not push
- The full run can outlast a ten-minute command limit; run it in the background with the longest limit the session allows, and wait for its
  exit status before going on
