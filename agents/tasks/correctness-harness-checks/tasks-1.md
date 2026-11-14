# Tasks: Correctness harness checks

**Source:** TODO.md, "The correctness harness cannot check exp(beta), prediction intervals or test-based tau2 and I2 intervals"
**Created:** 2026-11-14
**Version:** 1

Checked against the code on 2026-11-14 (metafopy at `1bd2712`): `check_case` in `src/tests/correctness/cases.py` compares each `expect` key with
`fetch(result, key)` (an attribute, frame column or mapping entry of the result of `metafopy.<function>(**args)`), then calls the optional
`check = "<test module>.<function>"`. A `published` value printed on another scale (`exp(beta)`), or held by no attribute of the fit (a `predict`
interval, a `confint` bound), therefore cannot be a published case. `src/tests/correctness/test_derived.py` already has check functions for
`derived` cases (`confint_ht`, `predict_pi`, ...), which compute their reference from `formulas.py`; a `published` check needs the printed value in
the case instead, and the case format has no key for it (`CASE_KEYS` in `cases.py`).

**Design chosen for the list** (not in the item, decided here because the item's fix direction leaves the mechanism open): one optional case key
`printed`, a table of names to values written as printed (strings), allowed only with `kind = "published"` and a `check`, validated like `expect` and
compared at `published_tolerance`. The check functions live in `src/tests/correctness/test_published.py` and read it. `expect` keeps its meaning,
so no existing case changes.

**Recording a mismatch (Tasks 2 and 3):** when a case's reference disagrees with metafopy beyond the published tolerance, commit it with
`status = "mismatch"` and a `reason` naming both values, and add an item to `agents/reviews/correctness-harness-checks-mismatches.md`, written in
the layout of `agents/specs/correctness-tests/mismatches.md` (Task 4 says what it holds). That item gives the source, locator, printed value,
metafopy's value, and R metafor's value where an R parity fixture or the audit's R oracle gives one. Per `claude/rules/metafopy.md`, metafopy's
output changes only after the user confirms it, so never change metafopy's output in this list. A source in the papers directory that does not
print the value wanted is noted in the same file under "Sources without a usable value", and its case is not written.

**Prerequisite outside this repository:** the supplied sources are PDFs in `/Users/simonb/src/metafopy papers`. Read them with `pdftotext`
(`/opt/homebrew/bin/pdftotext`); never copy them, or text or tables from them, into the repository; copy only the numbers used. If that directory
cannot be read from the queue's session, stop as `blocked` and say so. Readable on 2026-11-14: "Introduction to Meta-Analysis.pdf" (Borenstein et al.
2009, chapter 17 "Prediction intervals"), "Statistical Aspects of the Analysis of Data From Retrospective Studies of Disease.pdf" (Mantel and
Haenszel 1959, "R = 10.68"), "Quantifying heterogeneity in a meta-analysis.pdf" (Higgins and Thompson 2002) and "Beta blockade during and after
myocardial infarction- An overview of the randomized trials.pdf" (Yusuf et al. 1985).

The project has no affected-tests tool. This list changes no module under `src/metafopy/`, so each task names its test files and the last task
runs `src/tests/correctness`, `slow` cases included.

---

## Task 1: The `printed` key and the three check functions

**Status:** done
**Priority:** high
**Complexity:** medium
**Source Reference:** the item's fix direction (the `check` functions)

**Technical Goal:** a `published` case can compare a back-transformed estimate, a `predict` interval and the `confint` I2 and H2 bounds with the
values a source prints.

**Subtasks:**

- [x] Re-check the starting point: `grep -nw 'printed' src/tests/correctness/cases.py` finds nothing, and `CASE_KEYS` has no `printed`. Stop as
      `blocked` if it does
- [x] In `src/tests/correctness/cases.py`: add `printed` to `CASE_KEYS` and a `printed: dict[str, str]` field to `Case`; validate it in
      `parse_case` (a table of string values; only with `kind = "published"` and a `check`; every value parses as a number), raising `ValueError`
      naming the file and case id as the other validations do. Add a helper `check_printed(case, name, actual)` that compares `actual` with
      `float(case.printed[name])` at `published_tolerance` and raises `AssertionError` naming the case, the name, both values and the source, as
      `check_case` does
- [x] In `src/tests/correctness/test_published.py`, three check functions, each comparing every `printed` name it knows and refusing an unknown
      one:
  - `back_transformed`: `exp(beta)`, `exp(ci_lb)` and `exp(ci_ub)` of the fit (names `exp_beta`, `exp_ci_lb`, `exp_ci_ub`);
  - `prediction_interval`: `metafopy.predict(result)` (names `pred`, `pi_lb`, `pi_ub`, and their `exp_` forms for a ratio measure);
  - `confint_ht`: the bounds of `metafopy.confint(result, type='ht')` for `tau2`, `I2` and `H2` (names `tau2_lb`, `tau2_ub`, `I2_lb`, `I2_ub`,
    `H2_lb`, `H2_ub`, and `H_lb`, `H_ub` as the square roots of the H2 bounds, since Higgins and Thompson print H). The interval type is in the
    function's name because a case's `args` go to the fitting function, not to `confint`; add a sibling for another type only when a source
    prints one
- [x] Unit tests of the harness on inline TOML strings, beside the existing ones in `src/tests/correctness/test_coverage.py`: `printed` without a
      `check`, with `kind = "derived"`, and with a non-numeric value each raise `ValueError` naming the file; `check_printed` passes inside and
      fails outside the published tolerance

**Acceptance Criteria:**

- [x] `pytest src/tests/correctness` passes, and every existing case loads unchanged
- [x] Each check function names its source equation or printed table in its docstring

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy` passes
- [x] `pytest src/tests/correctness/test_coverage.py src/tests/correctness/test_published.py` passes

**Notes:**

- Key files: `src/tests/correctness/cases.py`, `src/tests/correctness/test_published.py`, `src/tests/correctness/test_coverage.py`
- `check` already accepts `"<test module>.<function>"` in `src/tests/correctness/` (`CHECK_PATTERN`), so `test_published.back_transformed` needs no
  change to the resolver

---

## Task 2: Published cases for back-transformed estimates

**Status:** done
**Priority:** high
**Complexity:** medium
**Source Reference:** the item (Mantel and Haenszel 1959, Yusuf et al. 1985)

**Technical Goal:** the pooled ratios the two sources print are published cases.

**Subtasks:**

- [x] Check that `/Users/simonb/src/metafopy papers` can be read (`ls`); stop as `blocked` if not
- [x] Mantel and Haenszel (1959): the pooled relative risk `R = 10.68` of Table 1, as `published.mh1959.table1.or` beside
      `published.mh1959.table1.chisq` in `src/tests/correctness/cases/published.toml`, reusing that case's inline strata, `function = "rma_mh"`,
      `check = "test_published.back_transformed"`, `printed = { exp_beta = "10.68" }`
- [x] Yusuf et al. (1985): the pooled odds ratios the tables print for the `dat.yusuf1985` subsets the existing `published.yusuf1985.*` cases use,
      with `function = "rma_peto"`, one case per table, with the confidence bounds where printed
- [x] Record any mismatch as the header says

**Acceptance Criteria:**

- [x] Every new case has `citation`, `locator` and `read`, and its values as printed
- [x] No source text or table is copied into the repository beyond the numbers used

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy` passes
- [x] `pytest src/tests/correctness/test_published.py src/tests/correctness/test_coverage.py` passes

**Notes:**

- Depends on Task 1
- The strata of Mantel and Haenszel's Table 1 are already inline in `published.mh1959.table1.chisq` (rebuilt from the garbled extraction and checked
  against the printed totals); copy them, do not rebuild them
- A value printed with few digits gives a loose tolerance; prefer the table with the most digits

---

## Task 3: Published cases for prediction intervals and test-based intervals

**Status:** done
**Priority:** high
**Complexity:** medium
**Source Reference:** the item (prediction intervals; Higgins and Thompson 2002; Borenstein et al. 2009)

**Technical Goal:** the prediction intervals and the test-based tau2, I2 and H intervals the supplied sources print are published cases.

**Subtasks:**

- [x] Check that `/Users/simonb/src/metafopy papers` can be read (`ls`); stop as `blocked` if not
- [x] Borenstein et al. (2009), chapter 17: the worked prediction interval, on the metafopy dataset that holds the chapter's data or inline data with
      its locator, `check = "test_published.prediction_interval"`
- [x] Borenstein et al. (2009), chapter 16 (equations 16.14 to 16.19): the worked intervals of tau2, I2 and H, `check = "test_published.confint_ht"`
- [x] Higgins and Thompson (2002): any example whose study data are printed or held by a metafopy dataset, with its H and I2 intervals. The
      sclerotherapy example of Table I prints no study data (see the comment above `published.higgins2002.sclerotherapy`), so it cannot be a case;
      note each such example under "Sources without a usable value"
- [x] Record any mismatch as the header says

**Acceptance Criteria:**

- [x] Every new case has `citation`, `locator` and `read`, and its values as printed
- [x] Each source named here has cases or a "Sources without a usable value" entry

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy` passes
- [x] `pytest src/tests/correctness/test_published.py src/tests/correctness/test_coverage.py` passes

**Notes:**

- Depends on Task 1 and on `agents/tasks/confint-ht-centre/tasks-1.md` (the `ht` interval is centred on H from Q by then, as Borenstein et al.
  centre it; before that list a REML fit would be a known mismatch)
- Check which tau2 estimator a source uses (Borenstein et al. use DerSimonian and Laird) and fit with that `method`

---

## Task 4: Review the new mismatches (stop for the operator only if there is one)

**Status:** done
**Priority:** high
**Complexity:** small
**Source Reference:** `claude/rules/metafopy.md` (metafopy's output changes only after the user confirms it)

**Technical Goal:** the user decides each mismatch Tasks 2 and 3 recorded. With none, the task says so and the list goes on without stopping.

**Subtasks:**

- [x] Count the cases with `status = "mismatch"` that Tasks 2 and 3 added (`git diff main -- src/tests/correctness/cases/` on the branch)
- [x] If there are none: write `agents/reviews/correctness-harness-checks-mismatches.md` with the automatic checks, the line "No mismatch: every new
      published case agrees with metafopy", and any "Sources without a usable value"; mark this task done without stopping, and say in its Notes
      that no stop was needed
- [x] If there is at least one: write `agents/reviews/correctness-harness-checks-mismatches.md` in the layout of
      `agents/specs/correctness-tests/mismatches.md`:
  - what is reviewed and how;
  - automatic checks: `pytest src/tests/correctness` passes, every `mismatch` case has an item;
  - each mismatch, with source, locator, the printed value, metafopy's value and R metafor's value;
  - "Sources without a usable value";
  - decisions, one per mismatch, with options *follow the reference*, *keep metafor*, *treat it as a misprint* and *investigate*, each with its
    consequence, and a recommendation with its reason;
  - an empty Answers section.
- [x] With a mismatch: publish its interactive page (answers saved to the artifact's database as each is clicked, an accept-all button for the
      recommendations), and put `**Review page:** <URL>` in the document's header. If the queue session has no tool to publish a page, leave the
      header line as `**Review page:** to be published by the operator session` and say so in the stop message
- [x] With a mismatch: stop the list for the operator (`blocked`, waiting for the answers to every decision in
      `agents/reviews/correctness-harness-checks-mismatches.md`). Once every item has an answer, an operator session reads the page's answers,
      copies each into the document's Answers section with the date, commits that, and only then restarts the list

**Acceptance Criteria:**

- [x] Every new `mismatch` case has an item and a decision in the review document, or the document says there is none

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy` passes
- [x] `pytest src/tests/correctness` passes, `slow` tests skipped (`-m 'not slow'`)

**Notes:**

- Depends on Tasks 2 and 3
- One mismatch was recorded (the I2 upper limit of Borenstein et al. Dataset 2), so the list stops here for the operator.
- This is the list's only stop for the operator, and it happens only when Tasks 2 or 3 recorded a mismatch

---

## Task 5: Apply the answers, the full test run and closing the item

**Status:** done
**Priority:** high
**Complexity:** small
**Source Reference:** the item as a whole

**Technical Goal:** each answer becomes a case change or recorded work, the whole branch is checked once, and the item moves to `DONE.md`.

**Subtasks:**

- [x] If Task 4 recorded no mismatch, skip to the `TODO.md` subtask. Otherwise read the answers from the Answers section of
      `agents/reviews/correctness-harness-checks-mismatches.md`, never from the page; stop as `blocked`, naming the missing items, if the section is
      empty or incomplete
- [x] *Keep metafor*: correct the case to the convention metafor documents, cite where it documents it, and set `status = "pass"`
- [x] *Treat it as a misprint*: remove the case and record the answer in the review document, as `agents/specs/correctness-tests/mismatches.md` does
- [x] *Follow the reference* or *investigate*: add a `TODO.md` item for each with the `todo-add` skill, naming the case, the fix or what is in
      doubt, the R parity fixtures and `known.toml` entries to mark, and the `docs/differences.md` section "Where metafor departs from its
      reference". The case stays `status = "mismatch"` until that item is done
- [x] Move the item from `TODO.md` to `DONE.md`: append its heading and body unchanged at the end of `DONE.md`, then a line
      `Done: 2026-MM-DD, agents/tasks/correctness-harness-checks/tasks-1.md` with the date of the commit
- [x] Run the final tests from the worktree and save their output for the merge gate:
      `main=$(git worktree list --porcelain | sed -n '1s/^worktree //p'); mkdir -p "$main/ralpher-logs/correctness-harness-checks/tasks-1" && pytest src/tests/correctness 2>&1 | tee "$main/ralpher-logs/correctness-harness-checks/tasks-1/final-tests.log"`

**Acceptance Criteria:**

- [x] Every answered item has a case change or a `TODO.md` item
- [x] The item is in `DONE.md` with its body unchanged, and no longer in `TODO.md`

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `pre-commit run --all-files` passes
- [x] `pre-commit run auditme-all --hook-stage manual --all-files` passes
- [x] `mypy` passes
- [x] The final test run above passes, `slow` tests included
- [x] `git show $(git log --diff-filter=A --format=%H -- agents/tasks/correctness-harness-checks/tasks-1.md):TODO.md > <scratch>/todo-before.md`
      and `python3 ~/.claude/skills/todo-review/verify_bodies.py <scratch>/todo-before.md TODO.md DONE.md` exits 0 (`<scratch>` is the session's
      scratch directory, outside the repository)

**Notes:**

- Depends on Task 4
- No module under `src/metafopy/` changes, so `src/tests/correctness` is the whole affected set

---
