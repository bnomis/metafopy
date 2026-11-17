# Tasks: the four audit ledger gaps

**Source:** TODO.md, "The audit lists the exact permutest of uni_mods as an R kill and has no entry for it"; "The audit lists a selmodel logistic
Hessian element that is zero in both R and metafopy"; "The audit lists printed fits of flat location-scale models that differ in the last digit";
"The audit has no entry for ls_na with an interaction formula that R refuses"
**Created:** 2026-11-17
**Version:** 1

The four items were filed on 2026-11-15 by Task 2 of `agents/tasks/audit-no-intercept/tasks-1.md`, from the full audit run
`/Users/simonb/src/metafopy/.audit/runs/20261115-160207`: each is a finding of that run that `src/audit/data/known.toml` does not accept. This list is
row 6 of `agents/tasks/order-todo-items-1-10.md` and runs after rows 2 (`rma-mv-reml-stop`, the REML stop of `rma_mv`) and 3
(`selmodel-fixed-delta0`, a fixed first `delta` other than 1 on a step function), so its one full audit run sees their output changes. The plan was
approved on 2026-11-17 (item 1 of `agents/reviews/order-todo-items-1-10-specs.md`).

What each item's fix direction asks, and what this list does:

- **Exact permutest of `uni_mods`:** run the case by hand in metafopy; if it refuses with R's "Number of iterations requested too large." or
  finishes, add the case to the `r-killed` entry of `options/permutest/ls_bcg/exact=true` in `src/audit/data/known.toml` (Task 2).
- **selmodel logistic Hessian element:** check both cases in R and Python, then add an entry for them (Task 2). The existing `$hessian` entry for
  `options/selmodel/*` accepts detail `*R *e-0[5-9], Python *e-0[5-9]*` only, and Python's 2.69e-15 does not match it.
- **Printed flat location-scale fits:** compare R's and metafopy's log-likelihoods per case; where they agree to 1e-6, add the case to a `known.toml`
  entry with the values and date; a case where they do not is filed as a new `TODO.md` item, not fixed and not a stop (Task 3).
- **`ls_na` with `mods = ~ ablat * alloc`:** decided, D2 (a) of `agents/reviews/order-todo-items-1-10-specs.md`, answered 2026-11-17: accept the
  difference with a `known.toml` entry and a `docs/differences.md` entry; metafopy keeps returning the fit with its warning (Task 4).

<!-- doc-counts: accept-start probed on 2026-11-17 and measured on 2026-11-15 - dated records, not claims about the present -->
D2's probe, 2026-11-17 (R 4.6.2, metafor 5.2-1): R drops the redundant predictor ("Redundant predictors dropped from the model."), then stops with
"No missing values allowed in 'alpha.init'."; metafopy returns beta `[0.282, -0.024, 0.213, -0.134, -0.013]` and warns "Error when trying to invert
the Hessian.". The full run `20261115-160207` took about 37 minutes (its directory was made at 16:02 and `findings.json` written at 16:38).
<!-- doc-counts: accept-end -->

How every audit command of this list runs (as in `agents/tasks/audit-no-intercept/tasks-1.md`):

- From the worktree, with `main=$(git worktree list --porcelain | sed -n '1s/^worktree //p')` (the main checkout is the first entry of
  `git worktree list --porcelain`; it is `/Users/simonb/src/metafopy`), and with both `METAFOPY_AUDIT_RLIB="$main/.audit/Rlib"` (a worktree has no
  `.audit/Rlib`) and `METAFOPY_AUDIT_WORK="$main/.audit"` (so the `scripts` step reuses the metafor source in `$main/.audit/source` instead of
  downloading it, `--recall` reuses the checkouts in `$main/.audit/worktrees`, and runs land in `$main/.audit/runs/`).
- A full run or a recall run is long: start it with the Bash tool's `run_in_background` and wait for the notice that it exited; do not poll it.
- If `check-r` reports that the R library is missing or differs from `src/audit/data/r-versions.json`, stop as `blocked` and say so. Never pass
  `--allow-r-drift`, and never change `src/audit/data/r-versions.json`.
- An R call made to check a reason runs as
  `export METAFOPY_AUDIT_RLIB=/Users/simonb/src/metafopy/.audit/Rlib; LC_ALL=C.UTF-8 R_LIBS=$METAFOPY_AUDIT_RLIB Rscript <script>`, with the script in
  the session's scratch directory, outside the repository; so do the Python probe scripts.
- A fresh worktree has no `.venv`; `uv sync --frozen` (it does not touch `uv.lock`) makes one before the audit can import numpy.

Rules for every `known.toml` entry this list adds (`src/audit/README.md`, "What audit 7 could not see"): it names the fixture and the path it is about
(`match` and `path` as lists where needed, never a glob wider than the reason); its `reason` says what R does as reproduced in R, with the values of both
programs, the date and "R 4.6.2, metafor 5.2-1"; it has `ref = "docs/differences.md"`; a literal `[` or `]` in a glob is written `[[]` and `[]]`. No
entry is added for a case the full run of Task 1 does not report.

Stops: no task changes `src/metafopy/`. One stop for the operator, in Task 1: **stops for the operator: a finding of the full run that none of the four
items covers and that shows metafopy's output wrong** (Task 1 defines the test). Every other task stops as `blocked` only on the guards it names. No
version bump (the project is at 0.0.1). The project has no affected-tests tool: each task names the test files of what it changes
(`known.toml` is read by `src/tests/test_harness_accounts.py` and `src/tests/test_harness_compare.py`; `docs/` by `src/tests/test_docs_examples.py`),
and the last task runs the full suite, as the run order requires.

---

## Task 1: The full audit and sorting its findings

**Status:** todo
**Priority:** high
**Complexity:** medium
**Source Reference:** the approved plan (one full audit run after rows 2 and 3); each item's "Found on" paragraph

**Technical Goal:** one full audit run of the current code (rows 2 and 3 merged) against R metafor 5.2-1, with every unaccepted finding and every
stale entry sorted into this list's four items or the "Other" list, and any finding that shows metafopy's output wrong stopped for the operator.

**Subtasks:**

- [ ] Re-check the starting point, and stop as `blocked`, naming what differs, if any of these is not so:
      - `agents/tasks/rma-mv-reml-stop/tasks-1.md` and `agents/tasks/selmodel-fixed-delta0/tasks-1.md` exist on `HEAD` and neither has a line
        `**Status:** todo` or `**Status:** in-progress` (rows 2 and 3 are merged)
      - each of the four items above is in `TODO.md` with a `tasks:` line naming `agents/tasks/audit-ledger-gaps/tasks-1.md`
      - `grep -n -F` of each case id finds no `match` in `src/audit/data/known.toml`: `options/permutest/uni_mods/exact=true`,
        `options/selmodel/uni_factor/type=\"logistic\"`, `options/vif/ls_na/table=true`, `options/permutest/ls_bangert/digits=`,
        `construct/uni_mods_year/scale=`, `construct/ls_na/mods={\"$formula\":\"~ ablat * alloc\"}` (a wider glob that happens to cover one is
        not a clash; only a literal id is)
- [ ] Run the full audit in the background, as the header says:
      `METAFOPY_AUDIT_RLIB="$main/.audit/Rlib" METAFOPY_AUDIT_WORK="$main/.audit" bin/metafor-audit.sh`, and record the run directory it prints
      (`$main/.audit/runs/<timestamp>`) in this task's Notes, with its start and end time
- [ ] Read the run's `wrong-or-missing-full.md` and the `new` list of its `findings.json`, and sort every finding, and every entry the report lists
      as stale, into lists in this task's Notes, each line with its case id, `kind`, `path` and one line of detail:
      - **Item 1 to item 4:** the cases each item names (above and in its `TODO.md` body). A case an item names that the run no longer reports is
        recorded as "not reported" under its item
      - **Other:** everything else, each marked with its likely source: row 2 (an `rma_mv` fit or its methods, `agents/specs/rma-mv-reml-stop/spec.md`),
        row 3 (`selmodel` with a fixed `delta`, `agents/specs/selmodel-fixed-delta0/spec.md`), a stale entry, or unknown. Compare with
        `/Users/simonb/src/metafopy/.audit/runs/20261115-160207` (the last full run) to say which are new since it
- [ ] For each finding in "Other", decide whether it shows metafopy's output wrong. It does **not** when one of these holds, recorded beside it:
      (a) the spec of row 2 or 3 lists the change as intended; (b) R gives no value to compare (`r-killed`, an R timeout, an R error that metafopy
      reports as an error too); (c) it is a stop point of an optimiser on a flat likelihood, shown by an R call and a Python call whose
      log-likelihoods agree to 1e-6 or where metafopy's is the higher; (d) it is a stale entry (a ledger entry matching nothing). Otherwise, or if the
      task cannot tell, it does. **Stops for the operator: a finding of the full run that none of the four items covers and that shows metafopy's output
      wrong** - stop as `blocked`, naming the case, R's and metafopy's values and why none of (a) to (d) holds
- [ ] Every finding in "Other" that does not stop is filed in Task 5, not here

**Acceptance Criteria:**

- [ ] This task's Notes hold the run directory and the sorted lists, and every finding and stale entry of the run is in exactly one list
- [ ] Every finding in "Other" carries (a), (b), (c) or (d), or the task stopped for the operator

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)

**Notes:**

- Only this task list file changes
- Key files: `src/audit/README.md` (commands, kinds, the `known.toml` rules), `src/audit/data/known.toml`, `src/audit/data/fixtures.json`
- Prerequisite outside this repository: R 4.6.2 with metafor 5.2-1 and the pinned packages in `/Users/simonb/src/metafopy/.audit/Rlib`, and the
  metafor source in `/Users/simonb/src/metafopy/.audit/source`; `.audit/` is ignored by git and exists only in the main checkout

---

## Task 2: Entries for the exact permutest of uni_mods and the selmodel logistic Hessian

**Status:** todo
**Priority:** high
**Complexity:** small
**Source Reference:** items 1 and 2, "Fix direction"

**Technical Goal:** the R kill of `options/permutest/uni_mods/exact=true` and the zero Hessian element of `options/selmodel/uni_factor/type="logistic"`
and `type="logistic1"` are accepted in `src/audit/data/known.toml`, each with what R and metafopy were measured to do.

**Subtasks:**

- [ ] If Task 1 recorded a case of item 1 or 2 as "not reported", add no entry for it and record that in this task's Notes
- [ ] **Item 1:** build the `uni_mods` fixture as `src/audit/data/fixtures.json` defines it and run `metafopy.permutest(fit, exact=True)` in a scratch
      Python script under `/usr/bin/time -l` (macOS: it prints the maximum resident set size), with a wall-clock limit of 30 minutes
      (`timeout 1800`). Record the outcome, its time and its peak memory in this task's Notes, dated
      - If it raises `InputError` "Number of iterations requested too large." or finishes: add `options/permutest/uni_mods/exact=true` to the `match`
        of the `r-killed` entry of `options/permutest/ls_bcg/exact=true` (making `match` a list), and extend its `reason` with metafopy's outcome on
        `uni_mods`, its time and memory, and the date
      - If it crashes, passes 4 GiB or the time limit: add no entry; file it in Task 5 as a new item (metafopy cannot finish what R cannot either;
        not a stop)
- [ ] **Item 2:** for both cases, print the whole `$hessian` of R's `selmodel(type = "logistic")` and `type = "logistic1"` fit of `uni_factor` and of
      metafopy's (the call and arguments from the case in Task 1's run). Record element 11 and the largest relative difference of the other elements
      in this task's Notes, dated
      - If element 11 is below 1e-6 in absolute value in both and every other element agrees to relative 1e-3: add a `[[known]]` entry with `match` the
        two case ids exactly (escape the brackets as the header says if any), `kind = "value-diff"`, `path = "$hessian"`, a `detail` glob that matches
        this finding's detail string and not a larger value (copy its shape from the run's `findings.json`), and a `reason` with both values, the date
        and "R 4.6.2, metafor 5.2-1"
      - Otherwise add no entry and file it in Task 5 (a difference the item did not expect)
- [ ] Re-render the report with the new ledger (seconds, no R):
      `METAFOPY_AUDIT_WORK="$main/.audit" bin/metafor-audit.sh --rerender "$main/.audit/runs/<timestamp>"` (Task 1's run), and check that it no
      longer lists the cases accepted here and does not list either changed entry as stale

**Acceptance Criteria:**

- [ ] Each case of items 1 and 2 is accepted by an entry that matches it and no other finding of Task 1's run, or is recorded for Task 5 with the reason
- [ ] This task's Notes hold metafopy's outcome on the exact permutest and the Hessian values

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `pytest src/tests/test_harness_accounts.py src/tests/test_harness_compare.py` passes

**Notes:**

- Depends on Task 1
- Key files: `src/audit/data/known.toml` (the `r-killed` entry of `options/permutest/ls_bcg/exact=true`; the `$hessian` entries matching
  `options/selmodel/*`), `src/metafopy/methods/permutest.py` (`MAX_PERMUTATION_BYTES`, the refusal), read only
- R's side of item 1 needs no R call: R passes the harness's 4 GiB limit (Task 1's run shows it); the reason quotes that finding

---

## Task 3: The printed flat location-scale fits

**Status:** todo
**Priority:** high
**Complexity:** medium
**Source Reference:** item 3, "Fix direction"

**Technical Goal:** each `print-diff` of item 3 is accepted where R's and metafopy's log-likelihoods agree to 1e-6, and filed as a new `TODO.md` item
where they do not.

**Subtasks:**

- [ ] For each case of item 3 that Task 1's run still reports - `options/vif/ls_na/table=true`, `options/permutest/ls_bangert/digits=2`,
      `options/permutest/ls_bangert/digits=[2,{"$na":true},1]`, `construct/uni_mods_year/scale={"$formula":"~ 0 + I(1|ablat)"}` - fit the model the
      case prints in R and in metafopy (fixtures from `src/audit/data/fixtures.json`) and print the log-likelihood with 10 significant digits and the
      value on the differing line. For the `permutest` cases, also print both permutation p-values with the number of permutations behind each.
      Record both programs' values in this task's Notes, dated
- [ ] **Log-likelihoods agree to 1e-6 (relative):** add the case to the existing entry whose `reason` explains it (the `print-diff` entries of
      `construct/ls_*` and `options/permutest/ls_*`, near "The printed tests and coefficient rows of the flat location-scale fits"), or add one entry
      naming exactly the cases of one cause; extend the `reason` with the values and the date. A `match` list grows by the literal case ids, not by a
      wider glob
- [ ] **Log-likelihoods do not agree to 1e-6:** file the case as a new `TODO.md` item following the `todo-add` skill's conventions (verified, dated,
      the run directory and both values named, inserted in implementation order), and add no entry. This is not a stop. Record the new item's title in
      this task's Notes
- [ ] Re-render the report as in Task 2 and check that it lists none of the cases accepted here and no changed entry as stale

**Acceptance Criteria:**

- [ ] Every case of item 3 that Task 1's run reports is accepted with its log-likelihoods in the `reason`, or is a new `TODO.md` item
- [ ] No entry changed here matches a finding of Task 1's "Other" list

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `pytest src/tests/test_harness_accounts.py src/tests/test_harness_compare.py` passes

**Notes:**

- Depends on Task 1
- The item's cause is suspected, not confirmed: the stop points of R's `nlminb` and metafopy's optimiser on a flat scale likelihood
  (`docs/differences.md`, "Refits of a location-scale model that R's optimiser leaves short" and "Standard errors of a location-scale model on a flat
  scale"). The `permutest` p-values (`0.90` against `0.88`) differ by more than the last digit; the log-likelihood test still decides, but the
  `reason` says what moved the p-value if the probe shows it (a refit counted in one program and not the other, as the `ls_bangert` entry records)
- An item filed here changes `TODO.md` by an addition only, which the closing body check allows

---

## Task 4: Accept ls_na with an interaction formula that R refuses

**Status:** todo
**Priority:** high
**Complexity:** small
**Source Reference:** item 4, decided by D2 (a) of `agents/reviews/order-todo-items-1-10-specs.md` (2026-11-17)

**Technical Goal:** `construct/ls_na/mods={"$formula":"~ ablat * alloc"}` is accepted in `src/audit/data/known.toml` and described in
`docs/differences.md`; metafopy keeps returning the fit with its warning.

**Subtasks:**

- [ ] If Task 1 recorded the case as "not reported", stop as `blocked`: the decision was made on a case the run no longer shows, so something
      changed metafopy's or R's behaviour since 2026-11-17
- [ ] Reproduce both sides: in R, the `ls_na` fixture's call with `mods = ~ ablat * alloc` (from `src/audit/data/fixtures.json`), printing its
      messages and error; in metafopy, the same call, printing `beta` and the warnings. Record both in this task's Notes, dated. Stop as `blocked`,
      naming the difference, if R's error is not "No missing values allowed in 'alpha.init'.", if metafopy raises, or if metafopy's `beta` differs
      from D2's `[0.282, -0.024, 0.213, -0.134, -0.013]` by more than 5e-4 in any element, or it no longer warns "Error when trying to invert the
      Hessian.": D2 covers only the behaviour it measured
- [ ] Add a `[[known]]` entry: `match` the case id exactly (escape any bracket), `kind = "accepts-where-R-refuses"`, a `reason` saying that R drops
      the redundant predictor of the interaction and then stops in its own starting-value step because `alpha.init` carries an NA from the dropped
      column, that metafopy fits the model without the redundant column (its `beta`, and its Hessian warning), the decision (D2 (a),
      `agents/reviews/order-todo-items-1-10-specs.md`, 2026-11-17), the date and "R 4.6.2, metafor 5.2-1", and `ref = "docs/differences.md"`
- [ ] Add a subsection to `docs/differences.md` under "## metafopy gives a value where R does not", in the shape of its neighbours (a short
      description, a `*Why:*` paragraph - R's refusal comes from its starting values after it has already dropped the column, not from a model that
      cannot be fitted - and the R and Python compare block with the output as printed, dated). If the Python example takes more than a few seconds,
      start its block with `# docs: skip`
- [ ] Re-render the report as in Task 2 and check that it no longer lists the case and does not list the new entry as stale

**Acceptance Criteria:**

- [ ] The re-rendered report does not list the case, and the new entry matches only it
- [ ] `docs/differences.md` has the subsection, and no file under `src/metafopy/` changes

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `pytest src/tests/test_harness_accounts.py src/tests/test_harness_compare.py src/tests/test_docs_examples.py` passes

**Notes:**

- Depends on Task 1
- Key files: `src/audit/data/known.toml`, `docs/differences.md`, `src/metafopy/models/rma_ls_fit.py` (`start_alpha`, read only)
- The precedent D2 cites: "metafopy does not reproduce a failure of R's optimiser" (the `known.toml` entry for `*/permutest/ls_na`, 2026-11-04)

---

## Task 5: File the other findings and the recall check

**Status:** todo
**Priority:** high
**Complexity:** small
**Source Reference:** the approved plan (`--recall`; a new finding that does not show metafopy wrong is filed in `TODO.md`)

**Technical Goal:** every finding of the full run that this list did not accept has a `TODO.md` item, and the harness still reproduces what the
earlier audits found.

**Subtasks:**

- [ ] File each cause in Task 1's "Other" list, and each case Tasks 2 and 3 sent here, with the `todo-add` skill (verified against the run's
      `findings.json`, dated, the run directory named), never fixed here. Where `todo-add` merges a cause into an existing item, record that item's
      title in this task's Notes: its body changes on purpose, and the closing check of Task 6 must allow for it. If there is nothing to file, write
      "none" in the Notes
- [ ] Re-render the report as in Task 2 and check that every finding it lists is in a `TODO.md` item (one filed here or in Task 3, or one already
      there), and that no entry this list added or changed is listed as stale
- [ ] Run the recall check in the background, as the header says:
      `METAFOPY_AUDIT_RLIB="$main/.audit/Rlib" METAFOPY_AUDIT_WORK="$main/.audit" bin/metafor-audit.sh --recall`, and record the recall report it
      prints in this task's Notes. If it exits non-zero, stop as `blocked`, naming each item it missed: a miss is a hole in the harness, fixed in
      `src/audit/data/` (`AGENTS.md`), which is outside these items; the recall run passes `--no-ledger`, so no entry of this list can cause it

**Acceptance Criteria:**

- [ ] Every finding of the re-rendered report is in a `TODO.md` item
- [ ] `bin/metafor-audit.sh --recall` exits 0

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `pytest src/tests/test_harness_accounts.py src/tests/test_harness_compare.py` passes

**Notes:**

- Depends on Tasks 2, 3 and 4
- Key files: `TODO.md`, `src/audit/data/recall.toml`

---

## Task 6: The full test run and closing the items

**Status:** todo
**Priority:** high
**Complexity:** small
**Source Reference:** the four items as a whole

**Technical Goal:** the whole branch is checked once, and the four items move to `DONE.md`.

**Subtasks:**

- [ ] Move each of the four items from `TODO.md` to `DONE.md`, in the order of the Source line: append its heading and body unchanged at the end of
      `DONE.md`, then a line `Done: 2026-MM-DD, agents/tasks/audit-ledger-gaps/tasks-1.md; <outcome>` with the date of the commit, where `<outcome>`
      says in one line what the item ended as (for example "added to the r-killed entry of ls_bcg; metafopy refuses in <time>", "new $hessian entry
      for the two logistic cases", "accepted <case ids>, <title> filed for the rest", "accepted by D2 (a);
      known.toml and docs/differences.md entries")
- [ ] Run the final tests from the worktree and save their whole output for the merge gate (the main checkout is the first entry of
      `git worktree list --porcelain`, `/Users/simonb/src/metafopy`):
      `mkdir -p /Users/simonb/src/metafopy/ralpher-logs/audit-ledger-gaps/tasks-1 && source bin/activate.sh && pytest src/tests 2>&1 | tee /Users/simonb/src/metafopy/ralpher-logs/audit-ledger-gaps/tasks-1/final-tests.log`

**Acceptance Criteria:**

- [ ] The four items are in `DONE.md` with their bodies unchanged and a `Done:` line each naming this list and the outcome, and are no longer in
      `TODO.md`
- [ ] The items filed by Tasks 3 and 5 are in `TODO.md`

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `pre-commit run --all-files` passes
- [ ] `pre-commit run auditme-all --hook-stage manual --all-files` passes
- [ ] `mypy` passes
- [ ] The final test run above passes
- [ ] `git show $(git log --diff-filter=A --format=%H -- agents/tasks/audit-ledger-gaps/tasks-1.md):TODO.md > <scratch>/todo-before.md` and
      `python3 ~/.claude/skills/todo-review/verify_bodies.py <scratch>/todo-before.md TODO.md DONE.md` exits 0 (`<scratch>` is the session's scratch
      directory, outside the repository); a body changed on purpose by a `todo-add` merge in Task 5 is named in its Notes

**Notes:**

- Depends on Task 5
- No version bump: no task changes library output
- The full `pytest src/tests` run is the run order's required final check (its checkpoint command)

---
