# Tasks: Stijnen et al. (2010) Table VII treated arm, PN

**Source:** TODO.md, "Stijnen et al. (2010) Table VII treated arm, PN, differs from metafopy and R metafor"
**Created:** 2026-11-17
**Version:** 1

**Waits for:** row 4 of the run order `agents/tasks/order-todo-items-1-10.md`, the list `agents/tasks/simmonds2016-fig1/tasks-1.md`, which edits
`src/tests/correctness/cases/glmm_published.toml` as this list does. Task 1 stops the list if that list is not merged on main.

The item comes from the published-values review of `agents/specs/correctness-glmm/mismatches.md`, where the user answered *investigate* for the case
`glmm.published.stijnen2010.table7.pn.treated.mismatch` on 2026-11-14. The plan approved on 2026-11-17: check the transcription of the treated rows
of Table VI against the paper, then the Poisson log-normal likelihood with an offset (`measure='IRLN'`, `method='ML'`) with
`src/tests/correctness/glmm_formulas.py`; then correct the data, or record the source difference with its reason, or - if metafopy is the one that is
wrong - stop for the operator with the figures. No file under `src/metafopy/` changes in this list.

<!-- doc-counts: accept-start probed on 2026-11-17 - a dated record, not a claim about the present -->
Probed on 2026-11-17 (metafopy at `c75f3d0`, R metafor 5.2-1 from `/Users/simonb/src/metafopy/.audit/Rlib`, the paper read with `pdftotext -layout`
from `/Users/simonb/src/metafopy papers/Random effects meta-analysis of event outcome in the framework of the generalized linear mixed model with
applications in sparse data.pdf`, journal page 3054, PDF page 9):

- The case's treated rows match Table VI as printed: events 7, 8, 3, 6, 1, 1, 17, 3, 2 and catheter-days 1491, 1638, 11 484, 1446, 370, 786, 6840,
  1156, 400. On these, metafopy, R metafor and `glmm_formulas.one_group_glmm_fit(..., 'poisson', 'log', 'ML')` all give log rate 0.81997, SE 0.31962
  and `tau2` 0.59899 (0.59898 in R), so the transcription and the likelihood are not the cause.
- metafor's `dat.nielweise2008` (from Niel-Weise et al. 2008, the source Table VI cites; its help page says Stijnen et al. 2010 used it) has the same
  treated events but other treated catheter-days: 1344, 1600, 12 012, 1536, 370, 729, 6760, 1107, 320. On these (divided by 1000), metafopy gives log
  rate 0.849404, SE 0.330266 and `tau2` 0.654298; R metafor 0.8494055, 0.3302667 and 0.6543036; the formulas 0.849404, 0.330266 and 0.654298. The
  printed SE 0.330 and `tau2` 0.654 are reproduced, and `exp(0.8494)` is the printed incidence rate 2.34. So Table VII was computed from the
  original data, and Table VI misprints the treated catheter-days.
- The printed log rate 0.850 is 0.0006 from 0.8494, one unit of the last digit, which is outside the comparison's half-unit tolerance
  (`test_published_tolerance_is_half_the_last_digit` in `src/tests/correctness/test_coverage.py`). That is the size of the source differences the user
  accepted on 2026-11-14 (the control arm's `tau2`, for one), so it is recorded as one.
- The control rows of Table VI also differ from `dat.nielweise2008` by one catheter-day in studies 2, 4 and 5; both give the printed control
  values within the same accepted `tau2` difference, so the control cases stay as they are.
<!-- doc-counts: accept-end -->

So the expected path is Task 1 confirming the probe, Task 2 correcting the treated data and splitting the case into a passing case (SE, `tau2`)
and a mismatch case for the log rate recorded as a source difference, and Task 3 closing the item. The project has no affected-tests tool, so each
task names its test files, and the last task runs `pytest src/tests` as the run order requires. Every R call sets
`METAFOPY_AUDIT_RLIB=/Users/simonb/src/metafopy/.audit/Rlib` (with `R_LIBS` set to it): a worktree has no `.audit/Rlib` of its own. No version
bump: no library output changes.

**Operator stop (conditional):** Task 1, "stops for the operator: metafopy differs from the formulas or R metafor on the PN fit". It stops only if
Task 1's run disagrees with the probe in that way; otherwise the list runs to the end without the operator.

---

## Task 1: Re-check the starting point, the transcription and the likelihood

**Status:** todo
**Priority:** high
**Complexity:** small
**Source Reference:** the item's fix direction (check the transcription of Table VI's treated rows, then the likelihood)

**Technical Goal:** which data Table VII's treated PN row was computed from, and whether metafopy fits the Poisson log-normal model correctly on
them, is known and recorded in this task's Notes with the figures of metafopy, R metafor and the formulas.

**Subtasks:**

- [ ] Check that row 4 of the run order is merged on main: in the main checkout (the first entry of `git worktree list --porcelain`,
      `main=$(git worktree list --porcelain | sed -n '1s/^worktree //p')`), `git -C "$main" show main:DONE.md` holds the heading
      `## Simmonds and Higgins (2016) Fig 1 one-stage random effects differs from metafopy and R metafor`, `git -C "$main" show main:TODO.md` does
      not, and `git merge-base --is-ancestor main HEAD` exits 0. Stop as `blocked`, naming what is missing, if not
- [ ] Re-check the starting point: `src/tests/correctness/cases/glmm_published.toml` holds the case
      `glmm.published.stijnen2010.table7.pn.treated.mismatch` with `status = "mismatch"` and the treated data
      `xi = [7, 8, 3, 6, 1, 1, 17, 3, 2]`, `ti = [1.491, 1.638, 11.484, 1.446, 0.370, 0.786, 6.840, 1.156, 0.400]`, and no case id
      `glmm.published.stijnen2010.table7.pn.treated` (without `.mismatch`) exists. Stop as `blocked`, naming what differs, if not
- [ ] Transcription: extract page 3054 of the paper (path in the header) with `pdftotext -layout -f 9 -l 9` (or the page that holds "Table VI. Data
      of example 4") into the scratch directory, and compare the treated columns of Table VI with the case's `xi` and `ti * 1000`. Record any
      difference
- [ ] Original data: print `dat.nielweise2008` from R metafor 5.2-1
      (`export METAFOPY_AUDIT_RLIB=/Users/simonb/src/metafopy/.audit/Rlib; LC_ALL=C.UTF-8 R_LIBS=$METAFOPY_AUDIT_RLIB Rscript <script>`; stop as
      `blocked` if metafor does not load or `packageVersion("metafor")` is not 5.2.1), and compare its `x1i`, `t1i` with Table VI's treated columns
- [ ] Likelihood: for each data set - Table VI's treated rows, and `dat.nielweise2008`'s `x1i` with `t1i / 1000` - fit
      `rma.glmm(xi=, ti=, measure="IRLN", method="ML")` in R, `metafopy.rma_glmm(xi=, ti=, measure='IRLN', method='ML')` (after
      `source bin/activate.sh`), and `glmm_formulas.one_group_glmm_fit(xi, ti, 'poisson', 'log', 'ML')` (importing `tests.correctness.glmm_formulas`
      with `src` on `sys.path`), printing `beta`, `se` and `tau2` of each to at least six digits. Scripts stay in the scratch directory
- [ ] Record in this task's Notes, dated: the transcription result, the figures of the three programs on each data set, and one verdict:
      - "Table VI misprints the treated catheter-days; `dat.nielweise2008` reproduces the printed SE and `tau2`" (the probe's finding), or
      - "neither data set reproduces the printed values, and metafopy, R metafor and the formulas agree" (a source difference on the case's data), or
      - "metafopy differs from the formulas or R metafor" (by more than 1e-4 relative on `beta`, `se` or `tau2` on either data set)
- [ ] If the verdict is the third, **stops for the operator: metafopy differs from the formulas or R metafor on the PN fit**: stop as `blocked` with
      the figures in the Notes and change nothing else; a change to `src/metafopy/` needs the user (`claude/rules/metafopy.md`)

**Acceptance Criteria:**

- [ ] This task's Notes record the transcription result, the three programs' figures on both data sets, and one of the three verdicts

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)

**Notes:**

- Only this task list file changes; no code, case or test changes
- The 2026-11-17 probe in the header expects the first verdict
- Key files: `src/tests/correctness/cases/glmm_published.toml` (the `glmm.published.stijnen2010.table7.*` cases),
  `src/tests/correctness/glmm_formulas.py` (`one_group_problem`, `one_group_glmm_fit`)

---

## Task 2: Correct the treated data, and record what remains as a source difference

**Status:** todo
**Priority:** high
**Complexity:** small
**Source Reference:** the item's fix direction (correct the data, or record the source difference with its reason)

**Technical Goal:** the treated PN case fits the data Table VII was computed from, the printed values it reproduces are checked as a passing case,
and any printed value it still misses is a mismatch case whose reason says why, as the accepted source differences of the Stijnen cases do.

**Subtasks:**

- [ ] Read Task 1's verdict. If it is missing, or is the third, stop as `blocked`
- [ ] **If the verdict is the first (Table VI misprints the treated catheter-days):**
      - in `src/tests/correctness/cases/glmm_published.toml`, set the treated data of the case to `dat.nielweise2008`'s, divided by 1000 as the
        comment above the Table VII cases says: `xi = [7, 8, 3, 6, 1, 1, 17, 3, 2]`,
        `ti = [1.344, 1.600, 12.012, 1.536, 0.370, 0.729, 6.760, 1.107, 0.320]` (or the values Task 1 recorded, if they differ)
      - split the case: a passing case `glmm.published.stijnen2010.table7.pn.treated` (no `status`) with the printed values the new data reproduce
        within half a unit of the last digit (by the probe, `printed = { "se[0]" = "0.330", tau2 = "0.654" }`), and
        `glmm.published.stijnen2010.table7.pn.treated.mismatch` with only the values it still misses (by the probe, `"beta[0]" = "0.850"`) and the
        reason, in the form of the other accepted Stijnen cases: `"beta[0] printed 0.850, metafopy <value>, R metafor 5.2-1 <value>; accepted as a
        source difference on <date> (Table VI misprints the treated catheter-days; on the original data of dat.nielweise2008 the source used other
        software, metafopy and R metafor agree)"`, with Task 1's figures to five digits. If every printed value passes, keep only the passing case
      - in both cases' `[case.source]`, set `locator = "Table VII, treated, PN row (data of dat.nielweise2008, Niel-Weise et al. 2008: Table VI
        misprints the treated catheter-days)"` and `read` to the date of this task
      - extend the comment above the Table VII cases: the treated cases use `dat.nielweise2008`'s catheter-days, because Table VI's treated
        catheter-days are not the data Table VII was computed from (checked on the date of this task, `agents/tasks/stijnen2010-pn/tasks-1.md`)
      - leave the control cases and `glmm.published.stijnen2010.table8.bn` unchanged
- [ ] **If the verdict is the second (neither data set reproduces the printed values; the three programs agree):** keep the data, and replace the
      case's `reason` with one in the form above that records the values as a source difference accepted on the date of this task, naming the data
      sets tried. Mark the first branch's subtasks not applicable
- [ ] In `agents/specs/correctness-glmm/mismatches.md`, add a dated paragraph after the paragraph that follows the published-values table (the one
      beginning "Several rows differ by one unit"), without changing the table or the Answers: the result of the investigation of the Table VII treated
      row, the data now used, the values that now pass and the one recorded as a source difference, and this task list's path
- [ ] Run `pytest src/tests/correctness/test_published.py src/tests/correctness/test_coverage.py`; the new passing case passes and the mismatch case
      is a strict expected failure. If the passing case fails, move the failing value to the mismatch case with its figures in the reason

**Acceptance Criteria:**

- [ ] The treated PN case uses the data Task 1 found Table VII was computed from, or keeps Table VI's with the reason naming the data sets tried
- [ ] Every printed value of the treated PN row is in a passing case or in the mismatch case, and the mismatch case's reason names the printed value,
      metafopy's and R metafor's, and why it is accepted
- [ ] No file under `src/metafopy/` changes

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `pytest src/tests/correctness/test_published.py src/tests/correctness/test_coverage.py` passes

**Notes:**

- Depends on Task 1
- Key files: `src/tests/correctness/cases/glmm_published.toml`, `src/tests/correctness/test_published.py` (`glmm_printed`, `check_values`),
  `agents/specs/correctness-glmm/mismatches.md`
- The comparison of a printed value allows half a unit of its last digit; a `mismatch` case becomes `xfail(strict=True)`
  (`src/tests/correctness/cases.py`, `case_param`)

---

## Task 3: The full test run and closing the item

**Status:** todo
**Priority:** high
**Complexity:** small
**Source Reference:** the item as a whole

**Technical Goal:** the whole branch is checked once, and the item moves to `DONE.md` with the result.

**Subtasks:**

- [ ] Move the item from `TODO.md` to `DONE.md`: append its heading and body unchanged at the end of `DONE.md`, then a line
      `Done: 2026-MM-DD, agents/tasks/stijnen2010-pn/tasks-1.md; <outcome>` with the date of the commit, where `<outcome>` is, as Task 2 did,
      "Table VI misprints the treated catheter-days; on dat.nielweise2008's the printed SE and tau2 pass, and the log rate (printed 0.850, metafopy
      <value>) is recorded as a source difference" or "no data set reproduces the printed values; recorded as a source difference, metafopy, R
      metafor and the formulas agree"
- [ ] Run the final tests from the worktree and save their whole output for the merge gate (the main checkout is the first entry of
      `git worktree list --porcelain`, `/Users/simonb/src/metafopy`):
      `mkdir -p /Users/simonb/src/metafopy/ralpher-logs/stijnen2010-pn/tasks-1 && source bin/activate.sh && pytest src/tests 2>&1 | tee /Users/simonb/src/metafopy/ralpher-logs/stijnen2010-pn/tasks-1/final-tests.log`

**Acceptance Criteria:**

- [ ] The item is in `DONE.md` with its body unchanged and a `Done:` line naming this list and the outcome, and no longer in `TODO.md`

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `pre-commit run --all-files` passes
- [ ] `pre-commit run auditme-all --hook-stage manual --all-files` passes
- [ ] `mypy` passes
- [ ] The final test run above passes
- [ ] `git show $(git log --diff-filter=A --format=%H -- agents/tasks/stijnen2010-pn/tasks-1.md):TODO.md > <scratch>/todo-before.md` and
      `python3 ~/.claude/skills/todo-review/verify_bodies.py <scratch>/todo-before.md TODO.md DONE.md` exits 0 (`<scratch>` is the session's scratch
      directory, outside the repository)

**Notes:**

- Depends on Task 2
- No version bump: no library output changes
- The full `pytest src/tests` run is the run order's required final check (its checkpoint command)

---
