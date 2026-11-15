# Tasks: audit entries for the no-intercept QE and tau2 difference

**Source:** TODO.md, "No audit entries record the no-intercept QE and tau2 difference"
**Created:** 2026-11-15
**Version:** 1

The difference itself is decided and shipped: the user answered *file it and fix it in this order* on 2026-11-13
(`agents/reviews/order-correctness-follow-ups-specs.md`, D1), commit `64bc719` made QE and every tau2 estimator use `y' P y` with the uncentred `yi`,
and `docs/differences.md` records it under "Where metafor departs from its reference", subsection "QE and tau2 of a model without an intercept",
including `vif(reestimate=True)` of an `rma_uni` fit. This list adds no code and changes no output; it accepts in `src/audit/data/known.toml` only
the audit findings that difference explains, and files every other unaccepted finding as its own `TODO.md` item.

Checked on 2026-11-15 (metafopy at `89ec047`): `src/audit/data/known.toml` has no entry that cites the change (`grep -n -i 'reestimate'` finds only
`options/vif/ls_year/reestimate=true`, a print difference of the flat scale intercept of `ls_year`, which predates it). The only full audit run under
`/Users/simonb/src/metafopy/.audit/runs/` is `20261102-223017` (`"full": true` in its `findings.json`), from before the change; the later runs are
slices. Earlier runs name the cases this list is about `construct/<fixture>/mods={"$formula":"~ ablat - 1"}` (the `- 1` domain of `mods` in
`src/audit/data/domains.toml`), `construct/<fixture>/intercept=false` and `options/vif/<fixture>/reestimate=true`.

How every audit command of this list runs (decided for this list, see the Notes of Task 1):

- From the worktree, with `main=$(git worktree list --porcelain | sed -n '1s/^worktree //p')` (the main checkout is the first entry of
  `git worktree list --porcelain`; it is `/Users/simonb/src/metafopy`), and with both `METAFOPY_AUDIT_RLIB="$main/.audit/Rlib"` (a worktree has no
  `.audit/Rlib`; metafor 5.2-1 loads from there, checked 2026-11-15) and `METAFOPY_AUDIT_WORK="$main/.audit"` (so the `scripts` step reuses the
  verified metafor source in `$main/.audit/source` instead of downloading it, and `--recall` reuses the checkouts in `$main/.audit/worktrees`).
- A full run or a recall run is long: start it with the Bash tool's `run_in_background` and wait for the notice that it exited; do not poll it.
- If `check-r` reports that the R library is missing or differs from `src/audit/data/r-versions.json`, stop as `blocked` and say so. Never pass
  `--allow-r-drift`, and never change `src/audit/data/r-versions.json`.
- An R call made to check a reason runs as `R_LIBS="$main/.audit/Rlib" LC_ALL=C.UTF-8 Rscript -e '<code>'`.

No task stops for the operator. A task stops as `blocked` only on the guards it names. The project has no version-bump rule (`AGENTS.md` and
`claude/rules/metafopy.md` name none), so no task changes `pyproject.toml`. The project has no affected-tests tool: each task names the test files of
what it changes (`known.toml` is read by `src/tests/test_harness_accounts.py` and `src/tests/test_harness_compare.py`), and the last task runs the full
suite, as the run order requires.

---

## Task 1: The full audit and the known.toml entries for the no-intercept difference

**Status:** todo
**Priority:** high
**Complexity:** medium
**Source Reference:** the item's fix direction (run `bin/metafor-audit.sh`, add a `known.toml` entry for each finding the change explains)

**Technical Goal:** a full audit run of the current code against R metafor 5.2-1, with every finding the no-intercept change explains accepted in
`src/audit/data/known.toml` with its reason and `ref = "docs/differences.md"`, and the remaining findings listed for Task 2.

**Subtasks:**

- [ ] Re-check the starting point, and stop as `blocked`, naming what differs, if any of these is not so:
      `grep -n -i -E 'without an intercept|no-intercept|centres? yi|ymci' src/audit/data/known.toml` finds nothing;
      `grep -n '^### QE and tau2 of a model without an intercept$' docs/differences.md` finds the heading; and the item
      "No audit entries record the no-intercept QE and tau2 difference" in `TODO.md` has a `tasks:` line naming
      `agents/tasks/audit-no-intercept/tasks-1.md`
- [ ] Record the baseline for the closing check: write the output of `git rev-parse HEAD` into this task's Notes as "Baseline commit" (the commit
      whose `TODO.md` the last task compares with; the coordinator's commit that adds the `tasks:` line comes before the list starts)
- [ ] Run the full audit in the background, as the header says:
      `METAFOPY_AUDIT_RLIB="$main/.audit/Rlib" METAFOPY_AUDIT_WORK="$main/.audit" bin/metafor-audit.sh`, and record the run directory it prints
      (`$main/.audit/runs/<timestamp>`) in this task's Notes
- [ ] Read the run's `wrong-or-missing-full.md` and the `new` list of its `findings.json`, and sort every finding, and every entry the report lists as
      stale, into two lists in this task's Notes, each with its case id, `kind`, `path` and one line of detail:
      - **Explained by the change:** a finding on `QE`, `QEp`, `I2`, `H2`, `tau2` or a value that follows from tau2 (`se`, `zval`, `pval`, `ci.lb`,
        `ci.ub`, `beta` of a random-effects fit, `fit.stats`, `vb`, the printed lines that show them) of an `rma_uni` or `rma_ls` fit whose location
        design does not span the constant (`intercept=false` with a matrix `mods`, or `mods = ~ ablat - 1`), or a finding of `vif(reestimate=true)` on
        an `rma_uni` fixture (`docs/differences.md`: each refit uses only the columns of one coefficient set, without an intercept)
      - **Not explained:** everything else, including a stale entry
- [ ] Stop as `blocked`, naming the case, for a finding of the first kind on a fixture whose location design spans the constant (an intercept, or
      a full set of dummy columns such as `~ 0 + alloc`): `y' P y` equals metafor's centred form there, so such a finding is a bug in `64bc719`, not
      the accepted difference. `rma_mv` and `rma_glmm` do not centre `yi`, so a finding on an `mv_*` or `glmm_*` fixture is never of the first kind
- [ ] For each finding explained by the change, reproduce R's value with an R call (the README's rule: a reason that says what R does has been
      reproduced in R) and show metafopy's value from Python; record both in this task's Notes
- [ ] Add `[[known]]` entries to `src/audit/data/known.toml`, grouped by cause, each naming its fixtures and paths (`match` and `path` as lists where
      needed, never a glob wider than the reason, such as `*/uni_*`): `kind`, `path`, a `reason` (metafor centres `yi` once, `ymci` in `rma.uni`, so
      its QE is not Cochran's Q when the design does not span the constant and its tau2 differs with it; metafopy uses `y' P y`, for
      `vif(reestimate=TRUE)` each refit has no intercept; the R and Python values measured, with the date and "R 4.6.2, metafor 5.2-1") and
      `ref = "docs/differences.md"`. Escape a literal `[` or `]` in a glob as `[[]` and `[]]`. Add no entry for a finding the run does not report
- [ ] Re-render the report with the new ledger (seconds, no R):
      `METAFOPY_AUDIT_WORK="$main/.audit" bin/metafor-audit.sh --rerender "$main/.audit/runs/<timestamp>"`, and check that it lists no finding of
      the first list and no new entry as stale, and that the findings it lists are exactly the second list

**Acceptance Criteria:**

- [ ] The re-rendered report of the full run lists no finding the no-intercept change explains
- [ ] Every new `known.toml` entry matches at least one finding of the full run, and none matches a finding of the "Not explained" list
- [ ] This task's Notes hold the baseline commit, the run directory and both lists

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy` passes
- [ ] `pytest src/tests/test_harness_accounts.py src/tests/test_harness_compare.py` passes

**Notes:**

- Key files: `src/audit/data/known.toml`, `src/audit/README.md` (commands, the `known.toml` rules), `docs/differences.md` ("QE and tau2 of a model
  without an intercept"), `src/audit/data/domains.toml` (the `- 1` and `intercept = [false]` domains)
- Prerequisite outside this repository: R 4.6.2 with metafor 5.2-1 and the pinned packages in `/Users/simonb/src/metafopy/.audit/Rlib` (present on
  2026-11-15), and the metafor source in `/Users/simonb/src/metafopy/.audit/source`. `.audit/` is ignored by git and exists only in the main checkout
- Why `METAFOPY_AUDIT_WORK` as well as `METAFOPY_AUDIT_RLIB`: `src/audit/config.py` puts the run directory, the metafor source
  (`scripts.fetch_source`, downloaded from CRAN when missing) and the recall checkouts under the work directory, which defaults to the worktree's own
  `.audit/`. Pointing it at the main checkout keeps the run off the network, as the earlier lists `agents/tasks/confint-ht-centre/tasks-1.md` and
  `agents/tasks/qe-no-intercept/tasks-1.md` did
- `construct/*/intercept=false` fixtures that pass `mods` as a formula ignore `intercept=` (as metafor does), so their designs hold the constant and
  give no finding from this change (Task 3 of `agents/tasks/qe-no-intercept/tasks-1.md`, 2026-11-14)
- The existing entry `options/vif/ls_year/reestimate=true` (a print difference of the flat scale intercept) stays as it is

---

## Task 2: File the other findings and the recall check

**Status:** todo
**Priority:** high
**Complexity:** small
**Source Reference:** the item's fix direction ("fix any other finding as its own item") and the run order's requirement that `--recall` passes

**Technical Goal:** every unaccepted finding of the full run that the no-intercept change does not explain has its own `TODO.md` item, and the
harness still reproduces what the earlier audits found.

**Subtasks:**

- [ ] For each cause in Task 1's "Not explained" list, file it with the `todo-add` skill (verified against the run's `findings.json`, dated, the run
      directory named), never fixed here. Where `todo-add` merges a cause into an existing item, record that item's title in this task's Notes: its
      body changes on purpose, and the closing check of Task 3 must name only those titles. If the list is empty, write "none" in the Notes
- [ ] Run the recall check in the background, as the header says:
      `METAFOPY_AUDIT_RLIB="$main/.audit/Rlib" METAFOPY_AUDIT_WORK="$main/.audit" bin/metafor-audit.sh --recall`, and record the recall report it
      prints in this task's Notes. If it exits non-zero, stop as `blocked`, naming each item it missed: a miss is a hole in the harness, fixed in
      `src/audit/data/` (`AGENTS.md`), which is outside this item; the recall run passes `--no-ledger`, so no entry of Task 1 can cause it

**Acceptance Criteria:**

- [ ] Every cause in Task 1's "Not explained" list is in `TODO.md`, as its own item or merged into one by `todo-add`
- [ ] `bin/metafor-audit.sh --recall` exits 0

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy` passes
- [ ] `pytest src/tests/test_harness_accounts.py src/tests/test_harness_compare.py` passes

**Notes:**

- Depends on Task 1
- Key files: `TODO.md`, `src/audit/data/recall.toml`

---

## Task 3: The full test run and closing the item

**Status:** todo
**Priority:** high
**Complexity:** small
**Source Reference:** the item as a whole

**Technical Goal:** the whole branch is checked once, and the item moves to `DONE.md`.

**Subtasks:**

- [ ] Move the item "No audit entries record the no-intercept QE and tau2 difference" from `TODO.md` to `DONE.md`: append its heading and body
      unchanged at the end of `DONE.md`, then a line `Done: 2026-MM-DD, agents/tasks/audit-no-intercept/tasks-1.md` with the date of the commit
- [ ] Run the full test suite from the worktree and save its whole output for the merge gate. The main checkout is the first entry of
      `git worktree list --porcelain` (`/Users/simonb/src/metafopy`):
      `main=$(git worktree list --porcelain | sed -n '1s/^worktree //p'); mkdir -p "$main/ralpher-logs/audit-no-intercept/tasks-1" && source bin/activate.sh && pytest src/tests 2>&1 | tee "$main/ralpher-logs/audit-no-intercept/tasks-1/final-tests.log"`

**Acceptance Criteria:**

- [ ] The item is in `DONE.md` with its body unchanged, and no longer in `TODO.md`

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `pre-commit run --all-files` passes
- [ ] `pre-commit run auditme-all --hook-stage manual --all-files` passes
- [ ] `mypy` passes
- [ ] The full test run above passes, `slow` tests included, judged by its exit status (`pytest`'s, not `tee`'s: run it with `pipefail`, the Bash
      tool's default) and its summary line, not by a tail of the log
- [ ] `git show <the baseline commit in Task 1's Notes>:TODO.md > <scratch>/todo-before.md` and
      `python3 ~/.claude/skills/todo-review/verify_bodies.py <scratch>/todo-before.md TODO.md DONE.md` exits 0, or names as `LOST OR ALTERED` only
      the items Task 2's Notes record as merged by `todo-add` (`<scratch>` is the session's scratch directory, outside the repository)

**Notes:**

- Depends on Task 2
- No version bump: the project's rules require none

---
