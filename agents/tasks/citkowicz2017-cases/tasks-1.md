# Tasks: Citkowicz and Vevea (2017) printed Baskerville estimates

**Source:** TODO.md, "Add the printed Baskerville estimates of Citkowicz and Vevea (2017) as published cases"
**Created:** 2026-11-19
**Version:** 1

**Running tests in the worktree:** the worktree has no `.venv`, and the main checkout's venv imports `metafopy` from the main checkout's `src/` (its
editable install), so `PATH=/Users/simonb/src/metafopy/.venv/bin:$PATH pytest ...` alone tests the code on `main`, not this list's. Run every pytest
and Python check from the worktree root as `export PATH=/Users/simonb/src/metafopy/.venv/bin:$PATH PYTHONPATH=$PWD/src`, and check once with
`python -c 'import metafopy; print(metafopy.__file__)'` that it prints this worktree's path.

This list is row 1 of the run order `agents/tasks/order-citkowicz-cases-and-identity-profile.md` and waits for no other row. It writes the printed
estimates of the article's Baskerville example as published cases in `src/tests/correctness/cases/selmodel_published.toml`: each value metafopy
reproduces as a passing case, each value it does not as a strict expected failure (`status = "mismatch"`) recorded "not yet reviewed" for the user
to judge, as the user chose on 2026-11-19. It corrects the records that call the article unavailable. No file under `src/metafopy/` changes; no
version bump.

Source: Citkowicz M, Vevea JL (2017). A parsimonious weight function for modeling publication bias. Psychological Methods 22(1):28-41.
doi:10.1037/met0000119. PDF: `/Users/simonb/src/metafopy papers/A Parsimonious Weight Function for Modeling Publication Bias paper.pdf` (APA; read
for its printed values only). Read it with `pdftotext` into a scratch directory outside the repository (`<scratch>` below).

<!-- doc-counts: accept-start probed on 2026-11-19 - a dated record, not a claim about the present -->
The probe of 2026-11-19 (metafopy at `1842a93`, R metafor 5.2-1 from `/Users/simonb/src/metafopy/.audit/Rlib`). Data: the bundled
`dat.baskerville2012`, all 23 rows, `yi = smd`, `vi = se**2`, `method='ML'`; the moderator model is `mods = '~ factor(blind) - 1'`, so `beta[0]` is
the studies with `blind = 0` (not blinded, 9 studies) and `beta[1]` those with `blind = 1` (blinded, 14). The supplement's 19 effects, used by the two
existing `published.citkowicz2017.baskerville2012.*` cases, are these 23 without four blinded studies: Frijling et al., Margolis et al., Mold et al.
and Hogg et al.

| Model | Value | Printed (section "Example") | metafopy and R |
|---|---|---|---|
| mean only, unadjusted | `beta[0]`, `se[0]`, `tau2` | 0.555, 0.069, 0.016 | 0.5552, 0.0633, 0.0162 |
| mean only, `type='beta'` | `delta[0]` (a), `delta[1]` (b), `LRT`, `tau2` | 0.473, 4.459, 7.847, 0.0 | 0.4731, 4.4613, 7.8469, 0.0000 |
| mean only, `type='beta'` | `beta[0]`, `se[0]` | 0.115, 0.166 | 0.1147, 0.1664 |
| moderator, unadjusted | `beta[0]`, `se[0]`, `beta[1]`, `se[1]`, `tau2` | 0.574, 0.098, 0.538, 0.090, 0.017 | 0.5731, 0.0925, 0.5397, 0.0868, 0.0163 |
| moderator, `type='beta'` | `delta[0]`, `delta[1]`, `LRT`, `tau2` | 0.420, 5.097, 9.045, 0.0 | 0.4201, 5.0960, 9.0432, 0.0000 |
| moderator, `type='beta'` | `beta[0]`, `se[0]`, `beta[1]`, `se[1]` | 0.134, 0.171, -0.002, 0.199 | 0.1342, 0.1708, -0.0021, 0.1991 |

Outside half a unit of the last printed digit (`published_tolerance` in `src/tests/correctness/cases.py`), in both programs alike: the mean-only
unadjusted `se[0]`; the mean-only `delta[1]`; all five values of the unadjusted moderator model; the moderator `delta[1]` and `LRT` (9.0443 with
`mods = '~ blind'` coding). Every other value is inside it. The printed z statistics (8.016, 0.275) follow from the printed standard errors and get
no case.
<!-- doc-counts: accept-end -->

The project has no affected-tests tool, so each task names its test files, and the last task runs `pytest src/tests` as the run order requires. Every
R call runs as `export METAFOPY_AUDIT_RLIB=/Users/simonb/src/metafopy/.audit/Rlib; LC_ALL=C.UTF-8 R_LIBS=$METAFOPY_AUDIT_RLIB Rscript <script>`: a
worktree has no `.audit/Rlib` of its own.

**Operator stop (conditional):** Task 1, "stops for the operator: a printed value does not reproduce as the probe found". It stops only if a printed
value differs from the probe table's "Printed" column, the dataset's `blind` coding is not 1 for blinded studies, a fit differs from the probe table
by more than 1e-4 in either program, or a value the probe found inside the tolerance falls outside it.

---

## Task 1: Check the printed values and write the published cases

**Status:** done
**Priority:** high
**Complexity:** medium
**Source Reference:** the item's fix direction (the matching values as published cases, each miss a strict expected failure)

**Technical Goal:** the article's Baskerville estimates are in `selmodel_published.toml`, checked against the printed text and against both programs,
so the beta weight function is tested against its defining publication's own numbers.

**Subtasks:**

- [x] Re-check the starting point: `grep -n 'citkowicz2017.baskerville2012.article' src/tests/correctness/cases/selmodel_published.toml` finds
      nothing. Stop as `blocked`, naming what it finds, if not
- [x] Extract the PDF (path in the header) with `pdftotext` into `<scratch>` and compare the "Example" section's figures (the paragraphs on the
      random-effects analysis, Figure 3's weight function and the blinding moderator) with the "Printed" column of the probe table. Record any
      difference
- [x] Check the coding of `blind` in R's help page for `dat.baskerville2012` (`?dat.baskerville2012` in R metafor 5.2-1): 1 must mean the study was
      blinded. Record what it says
- [x] Fit the six models of the probe table in metafopy and in R metafor 5.2-1 (stop as `blocked` if `packageVersion("metafor")` is not 5.2.1),
      printing each value to six digits. Scripts stay in `<scratch>`
- [x] Record the results in this task's Notes, dated: the transcription check, the `blind` coding and both programs' figures
- [x] If a printed value differs, the coding is not as stated, a fit differs from the probe table by more than 1e-4, or a value the probe found inside
      the tolerance falls outside it, **stops for the operator: a printed value does not reproduce as the probe found**: stop as `blocked` with the
      figures in the Notes and change nothing else
- [x] In `selmodel_published.toml`, after the two `published.citkowicz2017.baskerville2012.*` cases, add a comment naming the article's example and
      its data (the 23 rows of `dat.baskerville2012`, of which the supplement's 19 leave out the four blinded studies named in the header), then these
      cases, each on `dataset = "dat.baskerville2012"` with `yi = "smd"` and the sampling variances from `se` (use `sei = "se"` if the harness passes
      it to `rma_uni`; otherwise inline `data` with `yi`, `vi = se**2` to four decimals and `blind`, as the existing Citkowicz cases inline theirs),
      `method = "ML"`, and `expect` holding the printed values as strings:
      - `published.citkowicz2017.baskerville2012.article.unadjusted`: `rma_uni`, `beta[0]` and `tau2`
      - `published.citkowicz2017.baskerville2012.article.beta`: `selmodel`, `type = "beta"`, `delta[0]`, `LRT`, `tau2`, `beta[0]`, `se[0]`; covers
        `["selmodel.beta", "selmodel.LRT"]`
      - `published.citkowicz2017.baskerville2012.article.blind.beta`: `selmodel` on `mods = "~ factor(blind) - 1"`, `type = "beta"`, `delta[0]`,
        `tau2`, `beta[0]`, `se[0]`, `beta[1]`, `se[1]`; covers `["selmodel.beta"]`
- [x] Add one mismatch case for each group of misses, with `status = "mismatch"` and a `reason` naming each printed value, metafopy's and R metafor's
      (from this task's Notes), "metafopy and R metafor agree; the article fitted with its own nlminb code; recorded on <date of this task>, not yet
      reviewed":
      - `published.citkowicz2017.baskerville2012.article.unadjusted.se`: `se[0]` = "0.069"
      - `published.citkowicz2017.baskerville2012.article.beta.b`: `delta[1]` = "4.459"
      - `published.citkowicz2017.baskerville2012.article.blind.unadjusted`: the five values of the unadjusted moderator model
      - `published.citkowicz2017.baskerville2012.article.blind.beta.b_lrt`: `delta[1]` = "5.097", `LRT` = "9.045" (the reason also gives the 9.0443 of
        the `~ blind` coding)
- [x] Give each new case a `[case.source]` with the existing Citkowicz `citation` without its "supplement" clause, `doi = "10.1037/met0000119"`,
      `read` the date of this task, and a `locator` naming the "Example" paragraph the values come from (and Figure 3 for a and b)
- [x] In the `locator` of the two existing supplement cases, add that the supplement's data are 19 of the article's 23 effects, without the four
      blinded studies named in the header
- [x] Replace the header comment's clause "(the 2017 article itself was not available)" with one saying the article's own printed values have the
      `...article.*` cases on the bundled 23-effect data
- [x] Run `pytest src/tests/correctness/test_published.py src/tests/correctness/test_coverage.py`: the passing cases pass and each mismatch case is a
      strict expected failure. If a passing case fails, stop as for the operator stop above

**Acceptance Criteria:**

- [x] This task's Notes record the transcription check, the `blind` coding and both programs' figures
- [x] Every printed value of the probe table is in a passing case or in a mismatch case whose reason gives the figures
- [x] No file under `src/metafopy/` changes

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `pytest src/tests/correctness/test_published.py src/tests/correctness/test_coverage.py` passes

**Notes:**

- Key files: `src/tests/correctness/cases/selmodel_published.toml`, `src/tests/correctness/test_published.py`, `src/tests/correctness/cases.py`
  (`published_tolerance`, `case_param`: a `mismatch` case is `xfail(strict=True)`)
- `delta[0]` and `delta[1]` are the beta weight function's a and b (`sel.delta`); `src/tests/correctness/cases/other_bias.toml` uses `delta[i]` keys
- The exemplar for published cases with mismatches is `agents/tasks/bagos2009-table5/tasks-1.md` Task 1
- The probe expects no operator stop
- Results, 2026-11-19 (no operator stop): every printed value in the article's "Example" section matches the probe table's "Printed" column
  (pdftotext of the PDF); `?dat.baskerville2012` in R metafor 5.2-1 says `blind`: single- or double-blind study (0 = no, 1 = yes); R is 5.2.1.
  metafopy and R agree to six digits: unadjusted 0.555211, se 0.063271, tau2 0.016219; beta a 0.473115, b 4.461302 (R 4.461301), LRT 7.846904,
  tau2 0.000002, beta 0.114748, se 0.166414; moderator unadjusted 0.573060, 0.092547, 0.539653, 0.086767, tau2 0.016278; moderator beta a 0.420109,
  b 5.095973, LRT 9.043166, tau2 0.000009, beta 0.134247, se 0.170787, -0.002094, se 0.199132; `mods = ~ blind` coding: b 5.095971, LRT 9.044252
  (R) and 9.044252 (metafopy). Inline data are used because the bundled dataset has no `vi`

---

## Task 2: Correct the records that call the article unavailable

**Status:** done
**Priority:** medium
**Complexity:** small
**Source Reference:** the item's fix direction (reverse the 2026-11-15 "dropped" entry and update its records)

**Technical Goal:** each record that says the article could not be read says, in a dated note, that it was found and where its values are tested,
without rewriting the dated text of 2026-11-15.

**Subtasks:**

- [x] `agents/specs/correctness-publication-bias/mismatches.md`, "Sources needed", table "`selmodel` sources": after the table, add a paragraph
      "**Corrected <date of this task>** (`agents/tasks/citkowicz2017-cases/tasks-1.md`):" the article is in the papers folder; its Baskerville
      example uses the 23 effects of `dat.baskerville2012` (the supplement's 19 leave out four blinded studies); its printed values have the
      `published.citkowicz2017.baskerville2012.article.*` cases, the misses recorded as mismatch cases. Leave the table row's text unchanged
- [x] In the same file's "Mismatches" section, add an item for the article's mismatch cases in the form of the section's other published-value items
      ("Published selection-model values that differ from metafopy (Task 9)"), with the figures from Task 1's Notes, marked as written after the review
      of 2026-11-15 and not yet reviewed
- [x] `agents/specs/correctness-publication-bias/spec.md`: after the paragraph containing "`dat.baskerville2012` has a flat likelihood, so Citkowicz
      and Vevea's values may not be reproducible", add a dated note that on the article's 23 effects the beta-model means, standard errors, a and the
      mean-only likelihood ratio reproduce, and b and the moderator likelihood ratio are off in the last digit, recorded in `mismatches.md`. Change no
      existing text and not the spec's version
- [x] `NOTES.md`: at the end of the entry "Citkowicz and Vevea (2017): revisit if the paper turns up", add one paragraph "Found <date of this task>:"
      the PDF is in the papers folder and its values are tested by the `...article.*` cases (`agents/tasks/citkowicz2017-cases/tasks-1.md`). Change
      nothing else in the file

**Acceptance Criteria:**

- [x] `git diff` of the three files shows only added lines
- [x] Every mismatch case added by Task 1 has an item in the "Mismatches" section of `mismatches.md`

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)

**Notes:**

- Depends on Task 1
- The "Automatic checks" section of `mismatches.md` says every mismatch case has an item in "Mismatches"; the second subtask keeps that true

---

## Task 3: The full test run, and closing the item

**Status:** done
**Priority:** high
**Complexity:** small
**Source Reference:** the item as a whole

**Technical Goal:** the whole branch is checked once, and the item moves to `DONE.md` with its outcome.

**Subtasks:**

- [x] Move the item from `TODO.md` to `DONE.md`: append its heading and body unchanged at the end of `DONE.md`, then a line
      `Done: 2026-MM-DD, agents/tasks/citkowicz2017-cases/tasks-1.md; the article's Baskerville values have published cases on the 23 bundled effects; the misses are recorded mismatch cases, not yet reviewed`
      with the date of the commit
- [x] Run the final tests from the worktree and save their whole output for the merge gate (the main checkout is the first entry of
      `git worktree list --porcelain`, `/Users/simonb/src/metafopy`):
      `mkdir -p /Users/simonb/src/metafopy/ralpher-logs/citkowicz2017-cases/tasks-1 && export PATH=/Users/simonb/src/metafopy/.venv/bin:$PATH PYTHONPATH=$PWD/src && pytest src/tests 2>&1 | tee /Users/simonb/src/metafopy/ralpher-logs/citkowicz2017-cases/tasks-1/final-tests.log`

**Acceptance Criteria:**

- [x] The item is in `DONE.md` with its body unchanged and a `Done:` line naming this list and the outcome, and no longer in `TODO.md`

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `pre-commit run --all-files` passes
- [x] `pre-commit run auditme-all --hook-stage manual --all-files` passes
- [x] `mypy` passes
- [x] The final test run above passes
- [x] `git show $(git log --diff-filter=A --format=%H -- agents/tasks/citkowicz2017-cases/tasks-1.md):TODO.md > <scratch>/todo-before.md` and
      `python3 ~/.claude/skills/todo-review/verify_bodies.py <scratch>/todo-before.md TODO.md DONE.md` exits 0

**Notes:**

- Depends on Task 2
- No version bump: no library output changes
- The full `pytest src/tests` run is the run order's required final check (its checkpoint command)

---
