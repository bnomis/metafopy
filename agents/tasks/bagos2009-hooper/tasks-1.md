# Tasks: Bagos and Nikolopoulos (2009) Table 5, the Hooper column

**Source:** TODO.md, "Find the Hooper data behind Table 5 of Bagos and Nikolopoulos (2009)"
**Created:** 2026-11-19
**Version:** 1

**Running tests in the worktree:** the worktree has no `.venv`, and the main checkout's venv imports `metafopy` from the main checkout's `src/` (its
editable install), so `PATH=/Users/simonb/src/metafopy/.venv/bin:$PATH pytest ...` alone tests the code on `main`, not this list's. Run every pytest
and Python check from the worktree root as `export PATH=/Users/simonb/src/metafopy/.venv/bin:$PATH PYTHONPATH=$PWD/src`, and check once with
`python -c 'import metafopy; print(metafopy.__file__)'` that it prints this worktree's path.

This list writes published cases for the Hooper column of Table 5 (dietary fat and combined cardiovascular events), beside the ER-visit and
school-absence cases that `agents/tasks/bagos2009-table5/tasks-1.md` wrote, with the same three rows: the fixed-effects Poisson row, the random effects
with fixed study effects row and the random effects with random study effects, `z=0/1, cov!=0`, row. It then closes the item. No file under
`src/metafopy/` changes; no version bump.

## The data, and how it was found

<!-- doc-counts: accept-start probed on 2026-11-19 - a dated record, not a claim about the present -->
Found on 2026-11-19 by the operator session. Bagos and Nikolopoulos cite Hooper et al. (2001), BMJ 322(7289):757-763 (doi:10.1136/bmj.322.7289.757),
and print the totals intervention 6636 people, 10649 person-years, 553 events; control 6631, 10426, 653 (section 5). The BMJ article's data supplement,
"Table A Characteristics of included trials" (posted as supplied by the authors; saved in the papers folder as
`/Users/simonb/src/metafopy papers/Dietary fat intake and prevention of cardiovascular disease_ systematic review _ The BMJ.html`), gives each trial's
control and intervention numbers. The ten studies below, with the London corn/olive trial split into its corn and olive arms each with the shared
control group of 26, add to 6636 and 6631 people; their person-years from the article's Table 1
(`/Users/simonb/src/metafopy papers/Hooper_2001_Dietary_fat_intake_and_prevention_of_cardiovascular_disease.pdf`) add to 10649 and 10426. Their events
are Table 1's "Combined CVD events" (printed control/intervention), except the corn and olive arms: Table 1 prints them as 5.5/15 and 5.5/11, and
Bagos and Nikolopoulos read them with the arms reversed and 5.5 rounded to 6, which gives exactly 553 and 653.

| Study | x1i (intervention events) | t1i (intervention person-years) | x2i (control events) | t2i (control person-years) | people, intervention/control (Table A) |
|---|---|---|---|---|---|
| DART | 136 | 1925 | 147 | 1917 | 1018/1015 |
| London corn | 6 | 41 | 15 | 22 | 28/26 |
| London olive | 6 | 38 | 11 | 22 | 26/26 |
| London low fat | 38 | 374 | 42 | 393 | 123/129 |
| Minnesota Coronary | 134 | 4823 | 129 | 4715 | 4516/4516 |
| MRC Soya | 62 | 751 | 74 | 715 | 199/194 |
| Oslo Diet-Heart | 66 | 895 | 91 | 885 | 206/206 |
| STARS | 8 | 91 | 20 | 88 | 30/30 |
| Veterans Administration | 97 | 1588 | 122 | 1544 | 424/422 |
| Veterans Skin Cancer | 0 | 123 | 2 | 125 | 66/67 |

Probe on 2026-11-19 (metafopy at `866fe24`, R metafor 5.2.1), `exp` of `beta`, `ci_lb`, `ci_ub`:

| Row | Arguments | Printed | metafopy | R metafor 5.2.1 |
|---|---|---|---|---|
| Fixed effects | `model = "UM.FS", method = "FE"` | 0.819 (0.731, 0.917) | 0.818536 (0.730773, 0.916839) | 0.818536 (0.730773, 0.916839) |
| Random effects (fixed study effects) | `model = "UM.FS", method = "ML", coding = 1` | 0.819 (0.731, 0.917) | 0.818536 (0.730773, 0.916839) | 0.818536 (0.730773, 0.916839) |
| Random effects (random study effects, z=0/1, cov!=0) | `model = "UM.RS", method = "ML", coding = 1, cor = true` | 0.746 (0.592, 0.940) | 0.746103 (0.597097, 0.932294) | 0.746101 (0.597095, 0.932292) |

The summary-method rows also reproduce in R (inverse variance 0.825 (0.736, 0.924) and DerSimonian-Laird 0.754 (0.617, 0.921) exactly), which
confirms the data; as in the other columns they are `rma_uni` fits and get no case here.
<!-- doc-counts: accept-end -->

---

## Task 1: Check the data against the sources, then write the published cases

**Status:** done
**Priority:** high
**Complexity:** medium
**Source Reference:** the item's fix direction (write the Hooper column's cases beside the `glmm.published.bagos2009.table5.*` cases)

**Technical Goal:** the Hooper column of Table 5 has published cases in `glmm_published.toml`, on data checked against the BMJ article's Table 1 and
Table A and against Bagos and Nikolopoulos's printed totals, each compared with the printed IRR and 95% CI.

**Subtasks:**

- [x] Starting-point check: `grep -rnw 'glmm.published.bagos2009.table5.hooper' src/tests/correctness/cases` finds nothing; stop as `blocked`,
      naming the match, if it does
- [x] Extract the BMJ PDF (path in the header) with `pdftotext -layout`, and Table A from the saved BMJ page (the text after "Table A
      Characteristics of included trials"), into a scratch directory outside the repository. Check every row of the header's data table against
      them: person-years and the printed combined events of Table 1 (control/intervention), the people of Table A, and the corn and olive rows as
      the header describes. Check the column sums against Bagos and Nikolopoulos's totals in the header. Record the result in this task's Notes
- [x] Fit the data with the three argument sets of the probe table in metafopy (`metafopy.rma_glmm(measure='IRR', x1i=, t1i=, x2i=, t2i=, ...)`) and
      in R metafor 5.2-1 (`LC_ALL=C.UTF-8 R_LIBS=/Users/simonb/src/metafopy/.audit/Rlib Rscript`; stop as `blocked` if `packageVersion("metafor")` is
      not 5.2.1), printing `exp` of `beta`, `ci_lb` and `ci_ub` to six digits; scripts stay in the scratch directory. Record both programs' figures
- [x] **Stops for the operator if the data or a row does not reproduce as the probe found:** if a transcription differs from the header, a fit
      differs from the probe table by more than 1e-4, or a printed value other than the two random-study-effects CI bounds falls outside half a unit
      of its last printed digit, stop as `blocked` with the figures in the Notes and change nothing else
- [x] In `src/tests/correctness/cases/glmm_published.toml`, after the last `glmm.published.bagos2009.table5.school.*` case, add a comment naming the
      source and the data's origin (the trials chosen by Table A's numbers, person-years and events from Table 1, the corn and olive events read with
      the arms reversed and 5.5 rounded to 6, which gives Bagos and Nikolopoulos's totals), then these cases, each with `kind = "published"`,
      `function = "rma_glmm"`, inline `data` (`x1i`, `t1i`, `x2i`, `t2i` in the header table's order), `args` with
      `x1i = "x1i", t1i = "t1i", x2i = "x2i", t2i = "t2i", measure = "IRR"` and the row's arguments, `check = "test_published.glmm_printed"`, and
      `printed` with `"exp_beta[0]"`, `"exp_ci_lb[0]"`, `"exp_ci_ub[0]"` as printed:
      - `glmm.published.bagos2009.table5.hooper.fe` (`model = "UM.FS", method = "FE"`), covers `["rma_glmm.fit.IRR.UM.FS.FE"]`
      - `glmm.published.bagos2009.table5.hooper.fs` (`model = "UM.FS", method = "ML", coding = 1`), covers `["rma_glmm.fit.IRR.UM.FS.ML"]`
      - `glmm.published.bagos2009.table5.hooper.rs` (`model = "UM.RS", method = "ML", coding = 1, cor = true`), `printed` with `"exp_beta[0]"`
        only, covers `["rma_glmm.fit.IRR.UM.RS.ML"]`
      - `glmm.published.bagos2009.table5.hooper.rs.mismatch`, the same fit with `status = "mismatch"`, `printed` with `"exp_ci_lb[0]" = "0.592"`
        and `"exp_ci_ub[0]" = "0.940"`, and a `reason` naming each printed bound, metafopy's and R metafor's (from this task's Notes), "the source
        fitted with Stata gllamm; metafopy and R metafor agree; recorded on <date of this task>, not yet reviewed"
- [x] Give each case a `[case.source]` with `citation` and `doi` as the other `bagos2009.table5` cases, `read` the date of this task, and a `locator`
      naming the Table 5 row and the "Dietary fat intake / combined cardiovascular events (Hooper et al, 2001)" column, and the data's origin: Hooper L
      et al. (2001), BMJ 322:757-763, doi:10.1136/bmj.322.7289.757, Table 1 and the data supplement Table A, corn and olive events reversed
- [x] In the file's header comment, the sentence beginning "Bagos and Nikolopoulos (2009) Table 5:" (it runs over two comment lines and ends
      "its data not found (a TODO.md item).") becomes "Bagos and Nikolopoulos (2009) Table 5: the ER-visit, school-absence and Hooper columns
      have cases (see the comments above them)."; in the comment above the ER-visit and school-absence cases, replace "The Hooper column has none: its data
      were not found (a TODO.md item)." with "The Hooper column's cases follow the school-absence cases."
- [x] Run `pytest src/tests/correctness/test_published.py src/tests/correctness/test_coverage.py`: the three passing cases pass and the mismatch case
      is a strict expected failure. If a passing case fails, stop as for the operator stop above

**Acceptance Criteria:**

- [x] This task's Notes record the transcription checks against the BMJ article and Table A, the column sums, and both programs' figures
- [x] Every printed IRR and CI bound of the three rows of the Hooper column is in a passing case or in the mismatch case, whose reason gives the figures
- [x] No file under `src/metafopy/` changes

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `pytest src/tests/correctness/test_published.py src/tests/correctness/test_coverage.py` passes

**Notes:**

- Key files: `src/tests/correctness/cases/glmm_published.toml` (the `glmm.published.bagos2009.table5.*` cases are the exemplar),
  `src/tests/correctness/test_published.py` (`glmm_printed`), `src/tests/correctness/cases.py` (`check_printed`, `published_tolerance`; a
  `mismatch` case is `xfail(strict=True)`)
- The probe expects no operator stop
- Result, 2026-11-19: Table 1 of the BMJ PDF (`pdftotext -layout`) matches every row of the header table for person-years and combined events
  (control/intervention), the corn and olive arms read reversed with 5.5 rounded to 6, and Veterans Skin Cancer 2/0 events, 125/123 person-years.
  Table A (the saved BMJ page) matches the people of every row (corn 26/28 and olive 26/26 with the shared control group of 26). Column sums: people
  6636/6631, person-years 10649/10426, events 553/653, equal to Bagos and Nikolopoulos's totals.
- Fits (`exp` of `beta`, `ci_lb`, `ci_ub`; metafopy from this worktree, R metafor 5.2.1): `UM.FS`/`FE` 0.818536 (0.730773, 0.916839) in both;
  `UM.FS`/`ML`/coding 1 the same in both; `UM.RS`/`ML`/coding 1/`cor` metafopy 0.746103 (0.597097, 0.932294), R 0.746101 (0.597095, 0.932292), as
  the probe found. R warned of non-convergence on the `UM.RS` row. Printed: 0.819 (0.731, 0.917) twice and 0.746 (0.592, 0.940); only the
  `UM.RS` bounds differ, and they are the mismatch case. No operator stop.
- Tests: `pytest src/tests/correctness/test_published.py src/tests/correctness/test_coverage.py` passes, the new mismatch case an expected failure.

---

## Task 2: Records, the full test run, and closing the item

**Status:** done
**Priority:** high
**Complexity:** small
**Source Reference:** the item as a whole

**Technical Goal:** the GLMM mismatch record names the new mismatch, the whole branch is checked once, and the item moves to `DONE.md`.

**Subtasks:**

- [x] In `agents/specs/correctness-glmm/mismatches.md`, add a row to the table that lists the Bagos and Nikolopoulos school-absence row: "Bagos and
      Nikolopoulos (2009) Table 5, dietary fat, random study effects, z=0/1, cov!=0 (`UM.RS`, `ML`, `coding` 1, `cor`;
      `glmm.published.bagos2009.table5.hooper.rs.mismatch`; recorded <date>, not yet reviewed)", with the printed CI and both programs' bounds from
      Task 1's Notes
- [x] Move the item "Find the Hooper data behind Table 5 of Bagos and Nikolopoulos (2009)" from `TODO.md` to `DONE.md`: append its heading and body
      unchanged at the end of `DONE.md`, then a line
      `Done: 2026-MM-DD, agents/tasks/bagos2009-hooper/tasks-1.md; the data are the BMJ article's Table 1 for the trials its data supplement Table A gives, with the corn and olive events reversed; three rows of the Hooper column are published cases and the random-study-effects CI bounds a recorded mismatch`
      with the date of the commit
- [x] Run the final tests from the worktree and save their whole output for the merge gate (the main checkout is the first entry of
      `git worktree list --porcelain`, `/Users/simonb/src/metafopy`):
      `mkdir -p /Users/simonb/src/metafopy/ralpher-logs/bagos2009-hooper/tasks-1 && export PATH=/Users/simonb/src/metafopy/.venv/bin:$PATH PYTHONPATH=$PWD/src && pytest src/tests 2>&1 | tee /Users/simonb/src/metafopy/ralpher-logs/bagos2009-hooper/tasks-1/final-tests.log`

**Acceptance Criteria:**

- [x] `mismatches.md` lists the new mismatch case with its figures
- [x] The item is in `DONE.md` with its body unchanged and the `Done:` line, and no longer in `TODO.md`

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `pre-commit run --all-files` passes
- [x] `pre-commit run auditme-all --hook-stage manual --all-files` passes
- [x] `mypy` passes
- [x] The final test run above passes
- [x] `git show $(git log --diff-filter=A --format=%H -- agents/tasks/bagos2009-hooper/tasks-1.md):TODO.md > <scratch>/todo-before.md` and
      `python3 ~/.claude/skills/todo-review/verify_bodies.py <scratch>/todo-before.md TODO.md DONE.md` exits 0 (`<scratch>` is a scratch directory
      outside the repository)

**Notes:**

- Depends on Task 1
- The full run can outlast a ten-minute command limit; run it with the longest limit the session allows
