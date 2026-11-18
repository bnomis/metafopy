# Tasks: the stale ledger entries of known.toml

**Source:** TODO.md, "Six ledger entries of known.toml match nothing in the audit"
**Created:** 2026-11-18
**Version:** 1

**Running tests in the worktree (added 2026-11-18, after row 2):** the worktree has no `.venv`, and the main checkout's venv imports `metafopy`
from the main checkout's `src/` (its editable install), so `PATH=/Users/simonb/src/metafopy/.venv/bin:$PATH pytest ...` alone tests the code on
`main`, not this list's. Run every pytest and Python check from the worktree root as
`export PATH=/Users/simonb/src/metafopy/.venv/bin:$PATH PYTHONPATH=$PWD/src`, and check once with `python -c 'import metafopy; print(metafopy.__file__)'`
that it prints this worktree's path.

The item was filed on 2026-11-18 by Task 5 of `agents/tasks/audit-ledger-gaps/tasks-1.md`, from the stale list of the full audit run
`/Users/simonb/src/metafopy/.audit/runs/20261118-042436` (`wrong-or-missing.md`, "Not covered", "Stale ledger entries"): each is an entry of
`src/audit/data/known.toml` that matches no finding of that run. This list is row 5 of `agents/tasks/order-todo-items-1-5.md`. The plan was approved
on 2026-11-18 (item 1 of `agents/reviews/order-todo-items-1-5-specs.md`): for each entry, run its case in R and metafopy; remove the entry where the
two now agree, narrow its `detail` where they differ in another way; then one full audit and the recall check, so the report lists no stale entry.

**Notes:** this list runs after rows 1 (`ls-profile-starts`, which changes the profile bounds of location-scale models, `agents/specs/ls-profile-starts/spec.md`)
and 2 (`permutest-exact-limit`, which makes the exact `permutest` of `uni_mods` and `ls_bcg` refuse and adds the `uni_mods` case to the `r-killed`
entry of `options/permutest/ls_bcg/exact=true`, `agents/specs/permutest-exact-limit/spec.md`) are merged, so the one full audit of Task 3 sees their
output changes. Findings they cause are theirs, not this list's (Task 3 says how they are told apart).

The six entries, as the item names them (each found by its own text in `src/audit/data/known.toml`, never by line number):

- **E1:** `match = "scripts/misc_selmodel/*"`, `kind = "value-diff"`, `path = "$*"`, `detail = "*sav <- selmodel(res, type = \"beta\")*"` (the
  `selmodel(type='beta')` stop on `baskerville2012`). Two other entries share its `match` (`*optimizer = \"mads\"*` and the `*(relative *` one); they
  are not stale and stay unchanged.
- **E2:** `match = "scripts/misc_coef_se/*"`, `detail = "*sel <- selmodel(res, type = \"beta\")*"`. Its `reason` refers to E1 ("Same as the
  selmodel(type='beta') entry of misc_selmodel"), so if only one of E1 and E2 is removed the other's `reason` must stand on its own.
- **E3:** `match = "scripts/misc_selmodel/test/41"`, `detail = "*[[]out <- capture.output(print(sav))[]]*"` (the printed lower limit of `delta.2`).
- **E4:** `match = "options/predict/ls_na/newscale=*"`, `path = "$slab"` (the label of a prediction with a named `newscale`).
- **E5:** `match = ["*/ls_year", "*/ls_year/*", "*/permutest/ls_year", "*/permutest/ls_year/*", "*/permutest_ls*"]`, `kind = "print-diff"`,
  `detail = "*QM(df = 2) = 15.610[45]*"` (the printed QM of the flat location-scale fit `ls_year`).
- **E6:** `match = ["*/print/sel_bcg", "*/print/sel_bcg/*", "*/print/sel_bcg#*"]`, `kind = "value-diff"`, the `98.1051` against `98.1050` step table of
  `selmodel(type = 'stepfun')` on `dat.bcg`. `docs/differences.md` ("selmodel on a flat likelihood", its last paragraph) describes the same difference.

<!-- doc-counts: accept-start measured on 2026-11-18 - a dated record, not a claim about the present -->
The full run `20261118-042436` took about 37 minutes (started 03:24:34 UTC, ended 04:01:46 UTC, `agents/tasks/audit-ledger-gaps/tasks-1.md` Task 1).
<!-- doc-counts: accept-end -->

How every audit command of this list runs (as in `agents/tasks/audit-ledger-gaps/tasks-1.md`):

- From the worktree, with `main=$(git worktree list --porcelain | sed -n '1s/^worktree //p')` (the main checkout is the first entry of
  `git worktree list --porcelain`; it is `/Users/simonb/src/metafopy`), and with both `METAFOPY_AUDIT_RLIB="$main/.audit/Rlib"` (a worktree has no
  `.audit/Rlib`) and `METAFOPY_AUDIT_WORK="$main/.audit"` (so the `scripts` step reuses the metafor source in `$main/.audit/source`, `--recall` reuses
  the checkouts in `$main/.audit/worktrees`, and runs land in `$main/.audit/runs/`).
- A full run, a recall run, or a slice that has not finished in the foreground limit is started with the Bash tool's `run_in_background`; wait for the
  notice that it exited; do not poll it.
- If `check-r` reports that the R library is missing or differs from `src/audit/data/r-versions.json`, stop as `blocked` and say so. Never pass
  `--allow-r-drift`, and never change `src/audit/data/r-versions.json`.
- An R call made by hand runs as
  `export METAFOPY_AUDIT_RLIB=/Users/simonb/src/metafopy/.audit/Rlib; LC_ALL=C.UTF-8 R_LIBS=$METAFOPY_AUDIT_RLIB Rscript <script>`, with the script in
  the session's scratch directory, outside the repository; so do the Python probe scripts.
- A fresh worktree has no `.venv`; `uv sync --frozen` (it does not touch `uv.lock`) makes one before the audit can import numpy.

Rules for every `known.toml` entry this list narrows (`src/audit/README.md`): its `match`, `kind` and `path` stay as they are or become narrower, never
wider; its `detail` is a glob that matches the current finding and not a larger value (copied from the shape of the finding's detail in
`findings.json`, a literal `[` or `]` written `[[]` and `[]]`); its `reason` gives both programs' current values, the date and "R 4.6.2, metafor
5.2-1"; it keeps `ref = "docs/differences.md"`.

Stops: no task changes `src/metafopy/`. Two stops for the operator: in Task 1, **stops for the operator: an entry whose case now differs in a way that
is metafopy's output wrong** (Task 1 defines the test); in Task 3, **stops for the operator: a new unaccepted finding of the full audit that this list
did not cause and that is not one of rows 1 and 2's expected changes**. Every other task stops as `blocked` only on the guards it names. No version
bump (the project is at 0.0.1). The project has no affected-tests tool: each task names the test files of what it changes (`known.toml` is read by
`src/tests/test_harness_accounts.py` and `src/tests/test_harness_compare.py`; `docs/` by `src/tests/test_docs_examples.py`), and the last task runs the
full suite, as the run order requires.

---

## Task 1: Run each entry's cases in R and metafopy

**Status:** todo
**Priority:** high
**Complexity:** medium
**Source Reference:** the item's "Fix direction" (run each entry's case in R and metafopy)

**Technical Goal:** for each of E1 to E6, the audit's own R and metafopy calls of the cases the entry matches are run and compared, and each entry is
sorted into "remove" or "narrow", with the values behind the choice recorded; an entry that now hides metafopy's output wrong stops for the operator.

**Subtasks:**

- [ ] Re-check the starting point, and stop as `blocked`, naming what differs, if any of these is not so:
      - `agents/tasks/ls-profile-starts/tasks-1.md` and `agents/tasks/permutest-exact-limit/tasks-1.md` exist on `HEAD` and neither has a line
        `**Status:** todo` or `**Status:** in-progress` (rows 1 and 2 are merged)
      - the item is in `TODO.md` with a `tasks:` line naming `agents/tasks/audit-stale-entries/tasks-1.md`
      - each of E1 to E6 is in `src/audit/data/known.toml` with the `match` and the `detail` (E4: the `path`) quoted in the header, found by
        `grep -n -F` of its `detail` string (E4: of `options/predict/ls_na/newscale=*`). Other entries that share a `match` glob, or a wording change in
        a `reason`, are not a difference
- [ ] For each entry, run two slices of the audit over the cases its `match` names, as the header says, both with `--only` set to each glob of the
      entry's `match` (repeated) and `--steps` set to the step that makes those cases (`scripts` for E1 to E3, `options` for E4, and for E5 and E6,
      whose globs start with `*/`, the steps whose case ids match them in the run `20261118-042436`, found with `grep -c` over its `findings.json`
      and its step results, or `options,construct,matrix` when that shows none):
      - with the ledger: `METAFOPY_AUDIT_RLIB="$main/.audit/Rlib" METAFOPY_AUDIT_WORK="$main/.audit" bin/metafor-audit.sh --only '<glob>' --steps <steps>`
      - without it: the same with `--no-ledger`, so every difference of those cases is reported, accepted or not
      Record both run directories in this task's Notes
- [ ] Sort each entry, recording the reason and the values in this task's Notes, dated:
      - **No case:** the slice runs no case its `match` names (the case ids changed or were removed). Remove it
      - **Agree:** the ledger slice lists no unaccepted finding on its cases, and the `--no-ledger` slice has no finding of the entry's `kind` and
        `path` whose detail is the difference its `reason` describes (another entry accepts every finding it has). Record both programs' values at
        the point the `reason` names (E1, E2: `tau2` and the log-likelihood; E3: the printed lower limit of `delta.2`; E4: the label; E5: the printed
        QM and the log-likelihood; E6: the third `delta` and the printed step table), read from the slice's raw results or printed by a scratch R
        and Python call of the case. Remove it
      - **Moved:** the ledger slice lists an unaccepted finding of the entry's `kind` and `path` on its cases. Print both programs' values by hand
        (the case's call from the slice) and decide whether it is the cause the `reason` describes: the same element or printed line, and the
        `reason`'s own test still holding (E1, E2, E5: log-likelihoods agree to 1e-6 relative or metafopy's is the higher; E3, E5, E6: a last-digit
        print of a value that agrees to 1e-5 relative; E4: the same label rule). If it is, narrow it in Task 2
      - **Otherwise** (a moved finding that is not the cause in the `reason`): it shows metafopy's output wrong unless one of these holds, recorded
        beside it: (a) the spec of row 1 or 2 lists the change as intended; (b) R gives no value to compare (`r-killed`, an R timeout, an R error that
        metafopy reports as an error too); (c) it is a stop point of an optimiser on a flat likelihood, shown by an R call and a Python call whose
        log-likelihoods agree to 1e-6 or where metafopy's is the higher. Where one holds, the entry is removed in Task 2 and the finding is filed in
        Task 3. Where none holds, or the task cannot tell: **stops for the operator: an entry whose case now differs in a way that is metafopy's
        output wrong** - stop as `blocked`, naming the entry, the case, R's and metafopy's values and why none of (a) to (c) holds

**Acceptance Criteria:**

- [ ] Each of E1 to E6 is in exactly one of "No case", "Agree", "Moved" or "Otherwise" in this task's Notes, with its slice run directories and the
      values behind the choice, or the task stopped for the operator
- [ ] No file but this task list changes

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)

**Notes:**

- Key files: `src/audit/README.md` (commands, kinds, the `known.toml` rules), `src/audit/data/known.toml`, `src/audit/data/fixtures.json`
- Prerequisite outside this repository: R 4.6.2 with metafor 5.2-1 and the pinned packages in `/Users/simonb/src/metafopy/.audit/Rlib`, and the
  metafor source in `/Users/simonb/src/metafopy/.audit/source`; `.audit/` is ignored by git and exists only in the main checkout
- A ledger entry that matches nothing hides nothing (the item), so "Moved" and "Otherwise" are expected only where rows 1 or 2 changed an output

---

## Task 2: Remove or narrow the six entries

**Status:** todo
**Priority:** high
**Complexity:** small
**Source Reference:** the item's "Fix direction" (remove where the two now agree, narrow `detail` where they differ in another way)

**Technical Goal:** each of E1 to E6 is removed from `src/audit/data/known.toml` or narrowed to the finding it now accepts, as Task 1 sorted it, and
no record in `docs/differences.md` still describes a difference that is gone.

**Subtasks:**

- [ ] Remove each entry Task 1 sorted as "No case", "Agree" or "Otherwise" (not stopped), found by its own text; remove nothing else. If E1 is
      removed and E2 stays, rewrite E2's `reason` to stand on its own with E2's values from Task 1; if E2 is removed and E1 stays, nothing else changes
- [ ] Narrow each entry Task 1 sorted as "Moved", following the header's rules: its `detail` to the current finding, its `reason` to the current
      values and the date
- [ ] For each removed entry, search `docs/differences.md` for the difference it accepted (for E6 the `98.1051` paragraph of "selmodel on a flat
      likelihood"; for the others its values and call, with `grep -n -F`). Where a passage describes only that difference as current behaviour, remove
      it, or rewrite it with Task 1's values where the passage also holds other content; a narrowed entry's passage gets the current values and date.
      A passage that names no value of the removed entry stays. Record each change, or "none", in this task's Notes
- [ ] Re-render the report of Task 1's ledger slices with the new ledger (seconds, no R):
      `METAFOPY_AUDIT_WORK="$main/.audit" bin/metafor-audit.sh --rerender "$main/.audit/runs/<slice>"`, and check that no finding becomes unaccepted
      on the slices' cases and that a narrowed entry matches its finding

**Acceptance Criteria:**

- [ ] Every entry of E1 to E6 is removed or narrowed as Task 1 sorted it, and no other `[[known]]` entry changes except E2's `reason` as above
- [ ] The re-rendered slice reports list no unaccepted finding that the slices with the old ledger did not list
- [ ] No file under `src/metafopy/` changes

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `pytest src/tests/test_harness_accounts.py src/tests/test_harness_compare.py src/tests/test_docs_examples.py` passes

**Notes:**

- Depends on Task 1
- Key files: `src/audit/data/known.toml`, `docs/differences.md`
- The two other `scripts/misc_selmodel/*` entries (`*optimizer = \"mads\"*` and `*(relative *`) are not part of the item and stay unchanged

---

## Task 3: The full audit and the recall check

**Status:** todo
**Priority:** high
**Complexity:** medium
**Source Reference:** the item's "Fix direction" (rerun `bin/metafor-audit.sh` to see the report list no stale entry); the approved plan (`--recall`)

**Technical Goal:** one full audit of the current code (rows 1 and 2 merged, Task 2's ledger) against R metafor 5.2-1 lists none of E1 to E6 as stale
and no unaccepted finding this list caused, and the harness still reproduces what the earlier audits found.

**Subtasks:**

- [ ] Run the full audit in the background, as the header says:
      `METAFOPY_AUDIT_RLIB="$main/.audit/Rlib" METAFOPY_AUDIT_WORK="$main/.audit" bin/metafor-audit.sh`, and record the run directory it prints
      (`$main/.audit/runs/<timestamp>`) in this task's Notes, with its start and end time
- [ ] Read the run's `wrong-or-missing-full.md` and the `new` list of its `findings.json`, and its stale-entry list. Compare with
      `/Users/simonb/src/metafopy/.audit/runs/20261118-042436` (the last full run) and sort, into lists in this task's Notes, each line with its case
      id, `kind`, `path` and one line of detail:
      - **This list:** a finding on a case of an entry Task 2 removed or narrowed, or one of E1 to E6 listed as stale. Fix it here in
        `src/audit/data/known.toml` (restore or narrow the entry with the header's rules, after printing both programs' values as Task 1 does), then
        re-render this run as in Task 2 and check it is gone. A case that shows metafopy's output wrong by Task 1's test stops as Task 1 says
      - **Row 1:** a finding on a `confint` or `profile` of a location-scale fit whose value is one that "Every output that moves" of
        `agents/specs/ls-profile-starts/spec.md` lists (the ML `dat.bcg` log-link `alpha.2` upper bound, the identity-link lower bounds, the two
        log-link profile points)
      - **Row 2:** `options/permutest/uni_mods/exact=true` or `options/permutest/ls_bcg/exact=true`
      - **Stale, not this list's:** an entry listed as stale that is not one of E1 to E6
      - **Other:** everything else new since `20261118-042436`. **Stops for the operator: a new unaccepted finding of the full audit that this list
        did not cause and that is not one of rows 1 and 2's expected changes** - stop as `blocked`, naming the case, R's and metafopy's values and the
        step
- [ ] File each "Row 1", "Row 2" and "Stale, not this list's" line, and each finding Task 1 sent here ("Otherwise", not stopped), with the `todo-add`
      skill (verified against the run's `findings.json`, dated, the run directory named), never fixed here. Where `todo-add` merges one into an
      existing item, record that item's title in this task's Notes: its body changes on purpose, and the closing check of Task 4 must allow for it.
      If there is nothing to file, write "none" in the Notes
- [ ] Run the recall check in the background, as the header says:
      `METAFOPY_AUDIT_RLIB="$main/.audit/Rlib" METAFOPY_AUDIT_WORK="$main/.audit" bin/metafor-audit.sh --recall`, and record the recall report it
      prints in this task's Notes. If it exits non-zero, stop as `blocked`, naming each item it missed: a miss is a hole in the harness, fixed in
      `src/audit/data/` (`AGENTS.md`), which is outside this item; the recall run passes `--no-ledger`, so no entry change of this list can cause it

**Acceptance Criteria:**

- [ ] The full run's report (re-rendered after any fix here) lists none of E1 to E6 as stale and no finding in "This list"
- [ ] Every finding and stale entry of the report is in exactly one list in this task's Notes, and every line of "Row 1", "Row 2" and "Stale, not
      this list's" is in a `TODO.md` item
- [ ] `bin/metafor-audit.sh --recall` exits 0

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `pytest src/tests/test_harness_accounts.py src/tests/test_harness_compare.py` passes

**Notes:**

- Depends on Task 2
- Key files: `src/audit/data/known.toml`, `src/audit/data/recall.toml`, `TODO.md`
- An item filed here changes `TODO.md` by an addition only, which the closing body check allows

---

## Task 4: The full test run and closing the item

**Status:** todo
**Priority:** high
**Complexity:** small
**Source Reference:** the item as a whole

**Technical Goal:** the whole branch is checked once, and the item moves to `DONE.md`.

**Subtasks:**

- [ ] Move the item from `TODO.md` to `DONE.md`: append its heading and body unchanged at the end of `DONE.md`, then a line
      `Done: 2026-MM-DD, agents/tasks/audit-stale-entries/tasks-1.md; <outcome>` with the date of the commit, where `<outcome>` says in one line
      what the six entries ended as (for example "removed E1, E3 to E6 (the two now agree), narrowed E2; full run <timestamp> lists no stale entry")
- [ ] Run the final tests from the worktree and save their whole output for the merge gate (the main checkout is the first entry of
      `git worktree list --porcelain`, `/Users/simonb/src/metafopy`):
      `mkdir -p /Users/simonb/src/metafopy/ralpher-logs/audit-stale-entries/tasks-1 && source bin/activate.sh && pytest src/tests 2>&1 | tee /Users/simonb/src/metafopy/ralpher-logs/audit-stale-entries/tasks-1/final-tests.log`

**Acceptance Criteria:**

- [ ] The item is in `DONE.md` with its body unchanged and a `Done:` line naming this list and the outcome, and is no longer in `TODO.md`
- [ ] The items filed by Task 3 are in `TODO.md`

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `pre-commit run --all-files` passes
- [ ] `pre-commit run auditme-all --hook-stage manual --all-files` passes
- [ ] `mypy` passes
- [ ] The final test run above passes
- [ ] `git show $(git log --diff-filter=A --format=%H -- agents/tasks/audit-stale-entries/tasks-1.md):TODO.md > <scratch>/todo-before.md` and
      `python3 ~/.claude/skills/todo-review/verify_bodies.py <scratch>/todo-before.md TODO.md DONE.md` exits 0 (`<scratch>` is the session's scratch
      directory, outside the repository); a body changed on purpose by a `todo-add` merge in Task 3 is named in its Notes

**Notes:**

- Depends on Task 3
- No version bump: no task changes library output
- The full `pytest src/tests` run is the run order's required final check (its checkpoint command)

---
