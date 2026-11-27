# Tasks: Coefficient names and print tests follow R

**Spec:** `agents/specs/coefficient-names-and-print-tests-follow-r/spec.md`
**Created:** 2026-11-27
**Version:** 1

The spec was approved on 2026-11-27 with every item approved (S1-S6 in
`agents/reviews/coefficient-names-and-print-tests-follow-r-spec.md`). Approving S1 confirms the one output change the spec lists: the names, and
for two factors the column order, of an interaction term written out of its variables' first-appearance order. It confirms nothing else: a task
that would change any other name, value or printout stops as `blocked` and names it. `C(x)` keeps naming its columns like a plain factor (the
operator's decision of 2026-11-27); no task changes that.

How to run things, in every task (from the worktree root):

- **Tests:** `export PATH=/Users/simonb/src/metafopy/.venv/bin:$PATH PYTHONPATH=$PWD/src && pytest <files>`
- **Type check:** `mypy <files>`, with the same `PATH`.
- **R:** `LC_ALL=C.UTF-8 R_LIBS=/Users/simonb/src/metafopy/.audit/Rlib Rscript <script>` (the pinned library holds jsonlite).

---

## Task 1: Starting checks

**Status:** done
**Priority:** high
**Complexity:** small
**Spec Reference:** Motivation; Module location

**Technical Goal:** confirm the work has not started and the skips the spec clears are as it measured them, so later tasks start from the
measured state.

**Subtasks:**

- [x] `grep -rnwE 'r_term_order' src/metafopy` prints nothing; otherwise stop as `blocked` naming the match
- [x] `src/tests/test_formula_names.py`, `src/tests/r/generate_formula_names.R` and `src/tests/reference/formula_names.json` do not exist;
      otherwise stop as `blocked` naming the file
- [x] `NAMES_DIFFER` in `src/tests/test_rma_mv.py` is `{'ass_mods', 'ass_mods_t', 'ber_un_mods', 'kon_btt'}`, `ass_mods` and `ass_mods_t` write
      `C(deltype)`, and `src/tests/test_robust.py`'s `test_print` skips `uni_cs_btt`; otherwise stop as `blocked` naming what differs
- [x] Run `pytest src/tests/test_formula.py src/tests/test_rma_mv.py src/tests/test_robust.py`: it passes; otherwise stop as `blocked` naming
      the failures

**Acceptance Criteria:**

- [x] Every check above holds, and the task's Notes record the test run's pass and skip counts with the date

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `pytest src/tests/test_formula.py src/tests/test_rma_mv.py src/tests/test_robust.py` passes

**Notes:**

- Key files: `src/metafopy/formula.py`, `src/tests/test_rma_mv.py`, `src/tests/test_robust.py`
- Run on 2026-11-27: 438 passed, 12 skipped (4 + 7 in `test_rma_mv.py`, 1 in `test_robust.py`)
- The commit holds only this task file's ticks and Notes

---

## Task 2: The R names oracle

**Status:** done
**Priority:** high
**Complexity:** small
**Spec Reference:** Tests (R names oracle); External dependencies

**Technical Goal:** an R-generated fixture of `model.matrix` names and matrices, so the naming rule is checked against R rather than against
names written by hand.

**Subtasks:**

- [x] Write `src/tests/r/generate_formula_names.R` in the style of the other generators (a header saying what it writes and its usage line,
      `options(useFancyQuotes = FALSE)`, `library(jsonlite)`): one data frame of six rows with numeric `x`, `z` and `year`, character `deltype`
      (levels `covert`, `general`, `overt`, each present) and `outcome` (`AL`, `PD`); for each formula, `colnames(model.matrix(...))` and the
      matrix by column
- [x] Its formulas: `~ x`, `~ factor(deltype) + year`, `~ deltype + year`, `~ I(x^2)`, `~ x*outcome`, `~ outcome*x`, `~ x:outcome`,
      `~ outcome:x`, `~ 0 + outcome + outcome:x`, `~ 0 + outcome + I(year - 1985):outcome`, `~ outcome:I(year - 1985)`, `~ x + z + z:x`,
      `~ z:x + x + z`, `~ x:z`, `~ factor(deltype):x`, `~ deltype + outcome + outcome:deltype`, `~ x + z + outcome + outcome:z:x`, `~ x - 1`
      (not `~ deltype:outcome`, which the spec leaves out)
- [x] Run it and write `src/tests/reference/formula_names.json`; check the names it records for `~ 0 + outcome + I(year - 1985):outcome` are
      `outcomeAL`, `outcomePD`, `outcomeAL:I(year - 1985)`, `outcomePD:I(year - 1985)`, as the spec measured; otherwise stop as `blocked`

**Acceptance Criteria:**

- [x] The fixture holds names and a matrix for every formula above, and running the script again gives the same file

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] The generator runs with the R command above and exits 0

**Notes:**

- Depends on Task 1
- `Rscript` with `R_LIBS` as above; the default library has no jsonlite (checked 2026-11-27)
- Write numbers with `jsonlite::toJSON(..., digits = NA)` so the matrices are exact
- No test reads the fixture yet: Task 3 adds the test together with the rule it checks

---

## Task 3: Interaction terms order their variables as R does

**Status:** done
**Priority:** high
**Complexity:** medium
**Spec Reference:** API; Implementation; Error handling; Tests (R names oracle, `r_term_order`); Output changes

**User Story:** As an analyst moving from metafor, I want an interaction's coefficients named as R names them, so that I can match a metafopy
fit with its R counterpart by name.

**Subtasks:**

- [x] Write `src/tests/test_formula_names.py`: for every entry of `src/tests/reference/formula_names.json`, `design_matrix`'s names equal R's
      (with `(Intercept)` read as `intrcpt`) and its matrix equals R's column for column (exact, or `numpy.testing.assert_allclose` at
      `rtol=1e-15`); run it and see the entries written against first-appearance order fail
- [x] Add direct tests of `r_term_order` to `src/tests/test_formula.py`: no interaction comes back unchanged; `0 +` and `- 1` survive; a
      three-way interaction is reordered; `I(year - 1985)` is found; a factor missing from the text raises `metafopy.errors.InternalError`
- [x] Add `r_term_order(rhs: str) -> str` to `src/metafopy/formula.py` with the spec's docstring (not in `__all__`), as the spec's
      Implementation 1-2 describe: formulaic's parsed terms, each interaction's factors ordered by first position in `r_tokens(rhs)`,
      rebuilt with the original intercept and the parsed term order, and returned unchanged when there is no interaction
- [x] Call it in `design_matrix` before `r_power`
- [x] Say in `r_column_name`'s docstring that `C(x)` names its columns like a plain factor, by the operator's decision of 2026-11-27, and
      that R names them `C(x)level`
- [x] Run `pytest src/tests/test_rma_mv.py -k "test_fit and ber_un_mods"` with `ber_un_mods` taken out of `NAMES_DIFFER` for the run only:
      its `coef_names` now equal R's; put `NAMES_DIFFER` back (Task 4 changes it)

**Acceptance Criteria:**

- [x] `pytest src/tests/test_formula_names.py` passes; it failed before the rule
- [x] No other test in the Definition of Done changes its result

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy src/metafopy/formula.py src/tests/test_formula.py src/tests/test_formula_names.py` passes
- [x] `pytest src/tests/test_formula.py src/tests/test_formula_names.py src/tests/test_formula_rewriting_properties.py src/tests/test_rma_mv.py`
      passes

**Notes:**

- Depends on Task 2
- formulaic 1.2.2: `list(formulaic.Formula('0 + outcome + I(year - 1985):outcome'))` gives the terms `outcome` and
  `I(year - 1985):outcome`, each with `.factors`; the `0 +` is not among the terms, so take the intercept from the parsed formula, not the
  term list (checked 2026-11-27)
- The rewritten formula feeds `r_power` and `r_logical_terms`'s output, so call `r_term_order` on the text `design_matrix` passes to
  `formulaic.model_matrix`; check that a formula using R's logical operators still names its columns as before (`src/tests/test_formula.py`)
- A guess needs positive evidence: never fall back to the written order when a factor cannot be placed; raise

---

## Task 4: The rma_mv and robust print tests compare like with like

**Status:** done
**Priority:** high
**Complexity:** small
**Spec Reference:** Tests (`test_rma_mv.py`, `test_robust.py`, `conftest.py`)

**Technical Goal:** every skipped print case is compared with R at the width R printed it, so the print tests check the names again.

**Subtasks:**

- [x] In `src/tests/test_rma_mv.py`: `ass_mods` and `ass_mods_t` write `factor(deltype)`, as `src/tests/r/generate_rma_mv.R` does
- [x] Remove `NAMES_DIFFER` and both its uses
- [x] Add `ber_un_mods` to `PRINT_UNSTABLE`, with a comment giving the measured reason: on 2026-11-27 its REML log-likelihood equalled R's to
      4e-12 while `QM` was 76.708512 against R's 76.708353, either side of the rounding point 76.70845
- [x] Add a module fixture that sets the print width to 80 (`monkeypatch.setitem(metafopy.printing.SETTINGS, 'width', 80)`), with a docstring
      saying `generate_rma_mv.R` prints at R's default width
- [x] In `src/tests/test_robust.py`'s `test_print`, remove the `uni_cs_btt` skip
- [x] In `src/tests/conftest.py`, correct `wide_console`'s docstring: the generators that print wide tables set `width = 200`, and a module
      whose generator does not sets 80 for its print tests

**Acceptance Criteria:**

- [x] `test_print` runs and passes for `ass_mods`, `ass_mods_t`, `kon_btt` and `uni_cs_btt`, and `test_fit` checks `coef_names` for every case
- [x] Every `rma_mv` print case that passed before still passes at width 80

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy src/tests/test_rma_mv.py src/tests/test_robust.py src/tests/conftest.py` passes
- [x] `pytest src/tests/test_rma_mv.py src/tests/test_robust.py` passes

**Notes:**

- Depends on Task 3 (`ber_un_mods`'s `coef_names` match R only after it)
- On 2026-11-27 every `rma_mv` print case not in `PRINT_UNSTABLE` matched R at both 80 and 200 apart from the three this task fixes
- The fixture in `test_rma_mv.py` runs after `conftest.py`'s `wide_console` (autouse fixtures of the same scope run in conftest-then-module
  order); check by running one wide case

---

## Task 5: docs/differences.md records the C() naming

**Status:** todo
**Priority:** medium
**Complexity:** small
**Spec Reference:** Documentation

**User Story:** As an analyst moving from metafor, I want the one remaining naming difference written down, so that `C(alloc)` does not
surprise me.

**Subtasks:**

- [ ] Add a section "`C()` names its columns like a plain factor" under "By design" in `docs/differences.md`, after "Names and 0-based
      indices", in the page's R-and-Python comparison layout: R `model.matrix(~ C(alloc), dat.bcg)` names `C(alloc)random`, metafopy
      `allocrandom`; why (in metafopy `C()` is formulaic's categorical marker, not R's contrasts setter); `factor(alloc)` and plain `alloc` name
      as R does
- [ ] Fill each side's output by running it: R with the R command above (it needs metafor's `dat.bcg`, in the pinned library), Python through
      the page's own example runner

**Acceptance Criteria:**

- [ ] The section's outputs are the outputs the two code blocks print

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `pytest src/tests/test_docs_examples.py` passes

**Notes:**

- Depends on Task 3 (the section's claim that `factor()` matches R rests on the oracle test)
- `docs/guide/models.md` stays as it is: its `'~ C(alloc)'` example and `allocrandom` output are still right

---

## Task 6: The full local check and closing

**Status:** todo
**Priority:** high
**Complexity:** medium
**Spec Reference:** the whole spec

**Technical Goal:** the branch passes every check before the operator merges it, the spec says it is implemented, and the `TODO.md` item moves
to `DONE.md`.

**Subtasks:**

- [ ] Run the final tests and save their whole output for the merge gate. The main checkout is the first entry of
      `git worktree list --porcelain`, `/Users/simonb/src/metafopy`:
      `mkdir -p /Users/simonb/src/metafopy/ralpher-logs/coefficient-names-and-print-tests-follow-r/tasks-1 && export PATH=/Users/simonb/src/metafopy/.venv/bin:$PATH PYTHONPATH=$PWD/src && pytest src/tests 2>&1 | tee /Users/simonb/src/metafopy/ralpher-logs/coefficient-names-and-print-tests-follow-r/tasks-1/final-tests.log`
- [ ] A test that hard-codes a coefficient name and now fails is fixed only when the new name is the one R gives for the same formula (an
      interaction written out of first-appearance order); any other failure stops the task as `blocked`, naming the test
- [ ] Set the spec's `**Status:**` to `implemented`, and its row in `agents/specs/index.md` to say so
- [ ] Move the `TODO.md` item "`rma_mv` prints some coefficient labels unlike R, and the print tests skip them for a partly stale reason" to the
      end of `DONE.md`: append its heading and body unchanged, then `Done: <date> - agents/specs/coefficient-names-and-print-tests-follow-r/spec.md,
      agents/tasks/coefficient-names-and-print-tests-follow-r/tasks-1.md.`, and remove it from `TODO.md`

**Acceptance Criteria:**

- [ ] The final test run passes, the spec says `implemented`, and the item is in `DONE.md` with its `Done:` line

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `pre-commit run --all-files` passes
- [ ] `pre-commit run auditme-all --hook-stage manual --all-files` passes
- [ ] `mypy src/metafopy` passes
- [ ] The final test run above passes
- [ ] `git show $(git log --diff-filter=A --format=%H -- agents/tasks/coefficient-names-and-print-tests-follow-r/tasks-1.md):TODO.md > <scratch>/todo-before.md`
      and `python3 ~/.claude/skills/todo-review/verify_bodies.py <scratch>/todo-before.md TODO.md DONE.md` exits 0

**Notes:**

- Depends on Tasks 1-5
- The list does not push
- The full run can outlast a ten-minute command limit; run it in the background with the longest limit the session allows, and wait for its
  exit status before going on
- The `TODO.md` item "A two-factor interaction with neither main effect gets other columns than in R" stays: the spec leaves it out of scope
