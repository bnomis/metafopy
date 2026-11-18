# Tasks: Bagos and Nikolopoulos (2009) Table 5, ER visits and school absences

**Source:** TODO.md, "Find the data behind Table 5 of Bagos and Nikolopoulos (2009)"
**Created:** 2026-11-18
**Version:** 1

**Running tests in the worktree (added 2026-11-18, after row 2):** the worktree has no `.venv`, and the main checkout's venv imports `metafopy`
from the main checkout's `src/` (its editable install), so `PATH=/Users/simonb/src/metafopy/.venv/bin:$PATH pytest ...` alone tests the code on
`main`, not this list's. Run every pytest and Python check from the worktree root as
`export PATH=/Users/simonb/src/metafopy/.venv/bin:$PATH PYTHONPATH=$PWD/src`, and check once with `python -c 'import metafopy; print(metafopy.__file__)'`
that it prints this worktree's path.

This list is row 4 of the run order `agents/tasks/order-todo-items-1-5.md` and waits for no other row. The plan was approved on 2026-11-18 (item 1, row 4
of `agents/reviews/order-todo-items-1-5-specs.md`), with its decision D2 answered *(a) narrow the item to Hooper*: write published cases in
`src/tests/correctness/cases/glmm_published.toml` for the ER-visit and school-absence columns of Table 5 (the fixed-effects Poisson row, the random
effects with fixed study effects row and the random effects with random study effects, `z=0/1, cov!=0`, row), each against the printed IRR and 95% CI;
correct the "Sources needed" record of `agents/specs/correctness-glmm/mismatches.md` with a dated note; write no Hooper case, and replace the item with a
smaller `TODO.md` item for the Hooper column. No file under `src/metafopy/` changes in this list.

Sources (all read with `pdftotext -layout` into a scratch directory outside the repository):

- Bagos PG, Nikolopoulos GK (2009). Mixed-effects Poisson regression models for meta-analysis of follow-up studies with constant or varying durations.
  Int J Biostat 5(1), article 21. doi:10.2202/1557-4679.1168. PDF: `/Users/simonb/src/metafopy papers/Mixed-Effects Poisson Regression Models for
  Meta-Analysis of Follow-Up Studies with Constant or Varying Durations.pdf`. Table 5 (the IRR columns "School absences" and "ER visits") and
  Appendix III (the Stata `input` block of the ER-visit data, "we used the data of Wolf et al").
- Guevara JP, Berlin JA, Wolf FM (2004). Meta-analytic methods for pooling rates when follow-up duration varies: a case study. BMC Med Res Methodol
  4:17. doi:10.1186/1471-2288-4-17, PMC481068. Table 1 (school absences) and Table 2 (ER visits). Not in the papers folder; its full text is at
  `https://www.ebi.ac.uk/europepmc/webservices/rest/PMC481068/fullTextXML` (Europe PMC, open access). The licence the article's record states, read
  2026-11-18: "This is an Open Access article: verbatim copying and redistribution of this article are permitted in all media for any purpose, provided
  this notice is preserved along with the article's original URL" (BioMed Central's 2004 licence; the XML does not name a Creative Commons licence,
  although the approved review calls it CC-BY).

<!-- doc-counts: accept-start probed on 2026-11-18 - a dated record, not a claim about the present -->
The data, copied on 2026-11-18. Group 1 (`x1i`, `t1i`) is the intervention, group 2 (`x2i`, `t2i`) the control, so the IRR is intervention against
control; person-time is in person-months.

ER visits, Appendix III of Bagos and Nikolopoulos as printed (`id personmonth1 events1 personmonth0 events0`). Each row is also `round(N * mean)` events
and `N * duration` person-months of a row of Guevara et al. Table 2 (row 1 is Alexander: 11 x 12 = 132, 11 x 0.60 = 6.6 -> 7; 10 x 12 = 120,
10 x 2.40 = 24):

| id | t1i | x1i | t2i | x2i |
|---|---|---|---|---|
| 1 | 132 | 7 | 120 | 24 |
| 2 | 324 | 8 | 180 | 3 |
| 3 | 1908 | 273 | 876 | 182 |
| 4 | 156 | 1 | 156 | 13 |
| 5 | 528 | 20 | 540 | 27 |
| 6 | 576 | 110 | 336 | 104 |
| 7 | 84 | 13 | 84 | 52 |
| 8 | 90 | 5 | 90 | 18 |
| 9 | 1368 | 8 | 1140 | 22 |
| 10 | 1212 | 55 | 1248 | 40 |
| 11 | 75 | 11 | 75 | 27 |
| 12 | 378 | 95 | 306 | 85 |

School absences, rebuilt from Guevara et al. Table 1 (intervention N and mean, control N and mean, duration in months). Events are `N * mean` rounded
half up to an integer (the `raw` columns), person-months `N * duration`:

| study | n1 | mean1 | n2 | mean2 | months | t1i | raw1 | x1i | t2i | raw2 | x2i |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Charlton | 42 | 2.10 | 37 | 4.70 | 12 | 504 | 88.2 | 88 | 444 | 173.9 | 174 |
| Christiansen | 27 | 2.39 | 15 | 2.98 | 12 | 324 | 64.53 | 65 | 180 | 44.7 | 45 |
| Colland | 45 | 0.98 | 34 | 0.53 | 6 | 270 | 44.1 | 44 | 204 | 18.02 | 18 |
| Dahl | 9 | 0.80 | 10 | 0.90 | 1 | 9 | 7.2 | 7 | 10 | 9.0 | 9 |
| Deaves | 32 | 3.69 | 31 | 5.19 | 12 | 384 | 118.08 | 118 | 372 | 160.89 | 161 |
| Evans | 117 | 19.40 | 87 | 19.70 | 12 | 1404 | 2269.8 | 2270 | 1044 | 1713.9 | 1714 |
| Fireman | 13 | 0.50 | 13 | 4.60 | 12 | 156 | 6.5 | 7 | 156 | 59.8 | 60 |
| Hill | 211 | 5.43 | 193 | 6.23 | 4 | 844 | 1145.73 | 1146 | 772 | 1202.39 | 1202 |
| Hughes | 44 | 10.70 | 45 | 16.00 | 12 | 528 | 470.8 | 471 | 540 | 720.0 | 720 |
| Mitchell | 133 | 7.92 | 126 | 8.48 | 6 | 798 | 1053.36 | 1053 | 756 | 1068.48 | 1068 |
| Perrin | 29 | 0.24 | 27 | 0.22 | 1 | 29 | 6.96 | 7 | 27 | 5.94 | 6 |
| Persaud | 18 | 6.40 | 18 | 7.60 | 5 | 90 | 115.2 | 115 | 90 | 136.8 | 137 |
| Rubin | 29 | 11.90 | 25 | 15.40 | 12 | 348 | 345.1 | 345 | 300 | 385.0 | 385 |
| Talabere | 25 | 1.36 | 25 | 2.60 | 3 | 75 | 34.0 | 34 | 75 | 65.0 | 65 |
| Toelle | 63 | 2.62 | 51 | 2.67 | 6 | 378 | 165.06 | 165 | 306 | 136.17 | 136 |
| Wilson | 30 | 0.80 | 29 | 1.40 | 1 | 30 | 24.0 | 24 | 29 | 40.6 | 41 |

The printed IRR (95% CI) of Table 5 and the probe's fits on these data (metafopy at `552f50e`, R metafor 5.2-1 from
`/Users/simonb/src/metafopy/.audit/Rlib`), `exp` of `beta`, `ci_lb`, `ci_ub` to six digits:

| Column | Table 5 row | rma_glmm arguments | Printed | metafopy | R metafor |
|---|---|---|---|---|---|
| ER visits | Fixed effects | `model='UM.FS', method='FE'` | 0.659 (0.588, 0.739) | 0.659434 (0.588084, 0.739441) | 0.659434 (0.588084, 0.739441) |
| ER visits | Random effects (fixed study effects) | `model='UM.FS', coding=1` | 0.516 (0.361, 0.739) | 0.516040 (0.360537, 0.738614) | 0.516032 (0.360532, 0.738602) |
| ER visits | Random effects (random study effects, z=0/1, cov!=0) | `model='UM.RS', coding=1, cor=True` | 0.525 (0.361, 0.764) | 0.525159 (0.361145, 0.763661) | 0.525160 (0.361144, 0.763663) |
| School absences | Fixed effects | `model='UM.FS', method='FE'` | 0.864 (0.834, 0.896) | 0.864114 (0.833555, 0.895793) | 0.864114 (0.833555, 0.895793) |
| School absences | Random effects (fixed study effects) | `model='UM.FS', coding=1` | 0.727 (0.587, 0.900) | 0.727009 (0.587168, 0.900156) | 0.727005 (0.587162, 0.900153) |
| School absences | Random effects (random study effects, z=0/1, cov!=0) | `model='UM.RS', coding=1, cor=True` | 0.721 (0.579, 0.897) | 0.721018 (0.579941, 0.896414) | 0.721015 (0.579940, 0.896409) |

Every value is within half a unit of its last printed digit (0.0005, the comparison's tolerance, `published_tolerance` in
`src/tests/correctness/cases.py`) except the two CI bounds of the school-absence random-study-effects row: 0.579941 against 0.579 and 0.896414 against
0.897, each about one unit of the last digit, in both programs alike (the printed interval is slightly wider: its implied SE on the log scale is
about 0.1117 against the fit's 0.1111). `nAGQ` from 1 to 20 does not change the R fit. Bagos and Nikolopoulos fitted the models with Stata's `gllamm`.
So those two bounds go in a mismatch case that records the figures, beside the passing case of the row's IRR. Two other ER bounds are close to the
tolerance (0.360537 against 0.361, 0.000463 away; 0.833555 against 0.834, 0.000445 away).
<!-- doc-counts: accept-end -->

The other rows of Table 5 get no case: the summary-method rows (IV, MM-DL, ML, REML) are `rma_uni` fits of the log IRR, not `rma_glmm`; the bivariate
rows are not `rma_glmm` fits; the random-study-effects `z=+/-1/2, cov=0` row and the IRD columns are outside the approved plan. The Hooper column gets no
case (D2 (a)): its data are not in the PDFs present.

The project has no affected-tests tool, so each task names its test files, and the last task runs `pytest src/tests` as the run order requires. Every
R call runs as `export METAFOPY_AUDIT_RLIB=/Users/simonb/src/metafopy/.audit/Rlib; LC_ALL=C.UTF-8 R_LIBS=$METAFOPY_AUDIT_RLIB Rscript <script>`: a
worktree has no `.audit/Rlib` of its own. Scratch scripts and extractions stay in a scratch directory outside the repository (`<scratch>` below). No
version bump: no library output changes.

**Operator stop (conditional):** Task 1, "stops for the operator: a printed row of Table 5 does not reproduce as the probe found". It stops only if a
transcription differs from the sources, or a fit differs from the probe's table above by more than 1e-4 in either program, or a value other than the
two school-absence random-study-effects bounds falls outside the printed tolerance.

---

## Task 1: Check the data and printed values against the sources, then write the published cases

**Status:** todo
**Priority:** high
**Complexity:** medium
**Source Reference:** the item's fix direction (find the per-study counts and person-times that reproduce Table 5, then write the published cases)

**Technical Goal:** the ER-visit and school-absence columns of Table 5 have published cases in `glmm_published.toml`, on data checked row by row
against Bagos and Nikolopoulos (2009) Appendix III and Guevara et al. (2004) Tables 1 and 2, each compared with the printed IRR and 95% CI.

**Subtasks:**

- [ ] Re-check the starting point: `grep -rnw 'bagos2009' src/tests/correctness/cases` finds nothing. Stop as `blocked`, naming what it finds, if not
- [ ] Bagos and Nikolopoulos: extract the PDF (path in the header) with `pdftotext -layout` into `<scratch>`; compare the Appendix III `input` block
      with the ER-visit table in the header, and the IRR column entries of the "Fixed effects", "Random effects (fixed study effects)" and "Random
      effects (random study effects, z=0/1, cov!=0)" rows of Table 5 (school absences, ER visits) with the "Printed" column of the probe table. Record
      any difference
- [ ] Guevara et al.: fetch `https://www.ebi.ac.uk/europepmc/webservices/rest/PMC481068/fullTextXML` into its own new directory in `<scratch>`, and
      compare its Table 1 (N, mean, duration of each arm) with the school-absence table in the header, and its Table 2 with the ER-visit rows
      (`N * duration` and `round(N * mean)`). Check that the events and person-months of the school table follow from Table 1 by the rule in the
      header. Read the licence statement in the XML's `<permissions>` and record it as stated. If the URL cannot be fetched, compare against the
      header's table (copied from that XML on 2026-11-18) and say so in this task's Notes
- [ ] Fit both data sets with each of the three argument sets of the probe table, in metafopy (`source bin/activate.sh`,
      `metafopy.rma_glmm(measure='IRR', x1i=, t1i=, x2i=, t2i=, ...)`) and in R metafor 5.2-1 (`rma.glmm(measure="IRR", ...)`; stop as `blocked` if
      metafor does not load or `packageVersion("metafor")` is not 5.2.1), printing `exp` of `beta`, `ci_lb` and `ci_ub` to six digits. Scripts stay in
      `<scratch>`
- [ ] Record the results in this task's Notes, dated: the transcription checks, the licence statement, and the figures of both programs
- [ ] If a transcription differs, or a fit differs from the probe table by more than 1e-4, or a value other than the two school-absence
      random-study-effects bounds falls outside half a unit of its last printed digit, **stops for the operator: a printed row of Table 5 does not
      reproduce as the probe found**: stop as `blocked` with the figures in the Notes and change nothing else
- [ ] In `src/tests/correctness/cases/glmm_published.toml`, after the Stijnen et al. (2010) cases, add a comment naming the source, the data's origin
      (ER visits from Appendix III; school absences rebuilt from Guevara et al. Table 1 by the rule in this list's header) and why the other rows have
      no case (as in the header), then these cases, each with `kind = "published"`, `function = "rma_glmm"`, inline `data` (`x1i`, `t1i`, `x2i`,
      `t2i` in study order of the header tables), `args` with `x1i = "x1i", t1i = "t1i", x2i = "x2i", t2i = "t2i", measure = "IRR"` and the row's
      arguments, `check = "test_published.glmm_printed"`, and `printed` with `"exp_beta[0]"`, `"exp_ci_lb[0]"`, `"exp_ci_ub[0]"` as printed:
      - `glmm.published.bagos2009.table5.er.fe` (`model = "UM.FS", method = "FE"`), covers `["rma_glmm.fit.IRR.UM.FS.FE"]`
      - `glmm.published.bagos2009.table5.er.fs` (`model = "UM.FS", method = "ML", coding = 1`), covers `["rma_glmm.fit.IRR.UM.FS.ML"]`
      - `glmm.published.bagos2009.table5.er.rs` (`model = "UM.RS", method = "ML", coding = 1, cor = true`), covers `["rma_glmm.fit.IRR.UM.RS.ML"]`
      - `glmm.published.bagos2009.table5.school.fe`, `.school.fs` and `.school.rs`, the same three on the school data, except that
        `.school.rs` prints `"exp_beta[0]"` only
      - `glmm.published.bagos2009.table5.school.rs.mismatch`, the school random-study-effects fit with `status = "mismatch"`, `printed` with
        `"exp_ci_lb[0]" = "0.579"` and `"exp_ci_ub[0]" = "0.897"`, and `reason` naming each printed bound, metafopy's and R metafor's (from this task's
        Notes), "one unit of the last printed digit; the source fitted with Stata gllamm; metafopy and R metafor agree; recorded on <date of this
        task>, not yet reviewed"
- [ ] Give each case a `[case.source]` with `citation = "Bagos PG, Nikolopoulos GK (2009). Mixed-effects Poisson regression models for meta-analysis
      of follow-up studies with constant or varying durations. Int J Biostat 5(1), article 21"`, `doi = "10.2202/1557-4679.1168"`, `read` the date of
      this task, and a `locator` naming the Table 5 row and column and the data's origin; for the school cases it names Guevara JP, Berlin JA, Wolf FM
      (2004), BMC Med Res Methodol 4:17, doi:10.1186/1471-2288-4-17, PMC481068, Table 1, and the licence as recorded in this task's Notes
- [ ] Replace the header comment's sentences on Bagos and Nikolopoulos (the ones beginning "Bagos and Nikolopoulos (2009) prints no per-study data"
      and "Task 11 read the two reviews") with one that says the ER-visit and school-absence columns of Table 5 have cases (see the comment above them)
      and the Hooper column has none, its data not found (a `TODO.md` item)
- [ ] Run `pytest src/tests/correctness/test_published.py src/tests/correctness/test_coverage.py`: the six passing cases pass and the mismatch case
      is a strict expected failure. If a passing case fails, stop as for the operator stop above

**Acceptance Criteria:**

- [ ] This task's Notes record the transcription checks against both sources, the licence statement, and both programs' figures
- [ ] Every printed IRR and CI bound of the three rows in both columns is in a passing case or in the mismatch case, whose reason gives the figures
- [ ] No file under `src/metafopy/` changes

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `pytest src/tests/correctness/test_published.py src/tests/correctness/test_coverage.py` passes

**Notes:**

- Key files: `src/tests/correctness/cases/glmm_published.toml`, `src/tests/correctness/test_published.py` (`test_published`, `glmm_printed`),
  `src/tests/correctness/cases.py` (`check_printed`, `published_tolerance`, `case_param`: a `mismatch` case is `xfail(strict=True)`)
- The exemplar for published Poisson cases is `agents/tasks/stijnen2010-pn/tasks-1.md` and its cases `glmm.published.stijnen2010.*`
- `coding` and `cor` are `rma_glmm` arguments (`src/metafopy/models/glmm_fit.py`); a `glmm.toml` case already passes `coding` in `args`
- The probe expects no operator stop

---

## Task 2: Correct the "Sources needed" record in mismatches.md

**Status:** todo
**Priority:** medium
**Complexity:** small
**Source Reference:** the approved plan (correct the "Sources needed" record of `agents/specs/correctness-glmm/mismatches.md`)

**Technical Goal:** the record that says Table 5 has no data is corrected by a dated note, so a reader learns that the ER-visit and school-absence
data were found and where, without the dated text of 2026-11-14 being rewritten.

**Subtasks:**

- [ ] In `agents/specs/correctness-glmm/mismatches.md`, section "Sources needed", add a paragraph after the one beginning "Neither source gives
      per-study counts", without changing any existing text: "**Corrected <date of this task>** (`agents/tasks/bagos2009-table5/tasks-1.md`):" the
      ER-visit data are in Appendix III of Bagos and Nikolopoulos, and the school-absence data rebuild from Table 1 of Guevara, Berlin and Wolf (2004),
      BMC Med Res Methodol 4:17 (PMC481068); on them the fixed-effects, fixed-study-effects and random-study-effects rows of Table 5 reproduce in
      metafopy and R metafor, and have cases `glmm.published.bagos2009.table5.*` in `glmm_published.toml`, except the two school random-study-effects
      CI bounds, one unit of the last digit off, recorded as `glmm.published.bagos2009.table5.school.rs.mismatch`; the Hooper column still has no data
      that reproduce it, and has its own `TODO.md` item
- [ ] Add the mismatch case `glmm.published.bagos2009.table5.school.rs.mismatch` to the file's "Mismatches" section as an item, in the form of the
      section's other published-value items, with the figures from tasks-1.md Task 1's Notes, marked as written after the review of 2026-11-14 and not
      yet reviewed

**Acceptance Criteria:**

- [ ] The dated note and the mismatch item are present, and `git diff` of the file shows only added lines

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)

**Notes:**

- Depends on Task 1
- The "Automatic checks" section of the file says every mismatch case has an item in "Mismatches"; the second subtask keeps that true

---

## Task 3: The full test run, and replacing the item

**Status:** todo
**Priority:** high
**Complexity:** small
**Source Reference:** the item as a whole; D2 (a) of `agents/reviews/order-todo-items-1-5-specs.md`

**Technical Goal:** the whole branch is checked once; the item moves to `DONE.md` with its outcome, and a smaller `TODO.md` item for the Hooper column
takes its place.

**Subtasks:**

- [ ] Move the item from `TODO.md` to `DONE.md`: append its heading and body unchanged at the end of `DONE.md`, then a line
      `Done: 2026-MM-DD, agents/tasks/bagos2009-table5/tasks-1.md; ER-visit and school-absence rows of Table 5 have published cases (data from
      Appendix III and Guevara et al. 2004 Table 1); the school random-study-effects CI bounds are a recorded mismatch; Hooper narrowed to a new item`
      with the date of the commit
- [ ] In `TODO.md`, where the old item was, add this item (fill in the date of the task):

      ```markdown
      ## Find the Hooper data behind Table 5 of Bagos and Nikolopoulos (2009)

      <!-- doc-counts: accept-start probed on 2026-11-18 - a dated record, not a claim about the present -->
      Narrowed on <date> from "Find the data behind Table 5 of Bagos and Nikolopoulos (2009)" (`agents/tasks/bagos2009-table5/tasks-1.md`; decision
      D2 (a) of `agents/reviews/order-todo-items-1-5-specs.md`, answered 2026-11-18): the ER-visit and school-absence columns now have cases; the
      Hooper column (dietary fat and combined cardiovascular events) does not. Probed on 2026-11-18: the ten trials of Table 1 of Hooper et al. (2001,
      BMJ 322:757) match Bagos and Nikolopoulos's person-years but carry 567 intervention and 638 control events against their 553 and 653, and a
      fixed-effects Poisson fit gives an IRR of 0.858 against the printed 0.819; no swap of arms or rounding of the London corn/olive control events
      reproduces it.
      <!-- doc-counts: accept-end -->

      Fix direction: find the full Cochrane review, Hooper L et al. (2001), Dietary fat intake and prevention of cardiovascular disease, Cochrane
      Database Syst Rev CD002137, DOI 10.1002/14651858.CD002137 (the 2001 issue), and its per-trial events and person-years; repository copies at UEA
      ePrints and LSHTM Research Online were not checked. If they reproduce Table 5's Hooper column, write its cases in
      `src/tests/correctness/cases/glmm_published.toml` beside the `glmm.published.bagos2009.table5.*` cases.
      ```

- [ ] Run the final tests from the worktree and save their whole output for the merge gate (the main checkout is the first entry of
      `git worktree list --porcelain`, `/Users/simonb/src/metafopy`):
      `mkdir -p /Users/simonb/src/metafopy/ralpher-logs/bagos2009-table5/tasks-1 && source bin/activate.sh && pytest src/tests 2>&1 | tee /Users/simonb/src/metafopy/ralpher-logs/bagos2009-table5/tasks-1/final-tests.log`

**Acceptance Criteria:**

- [ ] The old item is in `DONE.md` with its body unchanged and a `Done:` line naming this list and the outcome, and no longer in `TODO.md`
- [ ] The Hooper item is in `TODO.md` in the old item's place

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `pre-commit run --all-files` passes
- [ ] `pre-commit run auditme-all --hook-stage manual --all-files` passes
- [ ] `mypy` passes
- [ ] The final test run above passes
- [ ] `git show $(git log --diff-filter=A --format=%H -- agents/tasks/bagos2009-table5/tasks-1.md):TODO.md > <scratch>/todo-before.md` and
      `python3 ~/.claude/skills/todo-review/verify_bodies.py <scratch>/todo-before.md TODO.md DONE.md` exits 0 (`<scratch>` is the session's scratch
      directory, outside the repository)

**Notes:**

- Depends on Task 2
- No version bump: no library output changes
- The full `pytest src/tests` run is the run order's required final check (its checkpoint command)

---
