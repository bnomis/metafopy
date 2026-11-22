# Tasks: Direct tests for the audit harness

**Spec:** `agents/specs/direct-tests-for-the-audit-harness/spec.md`
**Created:** 2026-11-22
**Version:** 1

The spec's decisions were answered on 2026-11-22 (`agents/reviews/direct-tests-for-the-audit-harness-spec.md`):
- D1: groups A (R-library shims), B (transpiler) and C (pure helpers) get direct tests; group D (drivers) is dismissed in the ledger;
- D2: the oracle for groups A and B is a fixture recorded from R and committed;
- D3: each `src/audit` module gets an `__all__` (Task 1);
- D4: `bin/metafor-audit.py` and `bin/platform-probe.py` get tests; `bin/logo.py` is dismissed.

`agents/specs/direct-tests-for-the-audit-harness/measurements.md` gives the group rules and the candidates measured on 2026-11-22. It is a
dated record, so never edit it. The open list after the re-judge (Task 2) is the work.

How to run things, in every task (from the worktree root):

- **Tests:** `export PATH=/Users/simonb/src/metafopy/.venv/bin:$PATH PYTHONPATH=$PWD/src && pytest <files>`
- **Type check:** `mypy <files>`, with the same `PATH`.
- **R:** `LC_ALL=C.UTF-8 Rscript`, with the R library at `/Users/simonb/src/metafopy/.audit/Rlib` (`R_LIBS=` that path); the pin is
  `src/audit/data/r-versions.json`.
- **T1 rule:** a public candidate is covered by a test that calls it; a private one also by a test of a caller that reaches it
  (`~/.claude/skills/auditme/phaseb/briefs/T1.md`). **T2 rule:** a hypothesis `given` test calls it directly (`briefs/T2.md`). An `xfail`
  test covers nothing.

How to write each test:

- **File:** the file the spec's Module location names for the candidate's module.
- **Docstring:** name the candidate id and the branch or case it checks.
- **Imports:** full module paths (`import audit.valuetree`), no leading underscores, no `from __future__` (`claude/rules/`).
- **Hypothesis:** `@hypothesis.settings(max_examples=100, deadline=None)`; strategies from `src/tests/rstrategies.py` where one fits.
- **Slow:** a test over 10 seconds gets `@pytest.mark.slow`; one needing `SIGALRM` or `resource` gets `refutil.posix_only`.

When a test finds a defect (spec, Error handling): stop the task as `blocked`, naming the function and the input. File the defect with
`/todo-add`, add a strict `xfail` test with `reason=` the item's title, and never fix the harness. Give the branches that agree with R or the
docstring their own passing test, so the candidate is still covered.

---

## Task 1: Declare each src/audit module's public names with `__all__`

**Status:** done
**Priority:** high
**Complexity:** medium
**Spec Reference:** Implementation step 1; decision D3

**Technical Goal:** helpers used only inside their module stop being public, so tests through the entry points cover them.

**Subtasks:**

- [x] For each module in `src/audit` except `__init__.py` and `rlib.py`, collect the names that another file under `src/audit`, `bin` or
      `src/tests` reads as `audit.<module>.<name>` (grep `audit\.<module>\.` and `getattr(audit.<module>`), and add an `__all__` listing
      exactly those, sorted, after the imports
- [x] A module no other file reads by attribute gets an `__all__` naming the entry points that `bin/` or a subprocess runs (such as `main`),
      found by grepping `bin/` and `src/audit` for the module name; with none, `__all__ = []`. Note each such module in the commit body
- [x] Run `python3 -c "import audit.cli"`, then `mypy src/audit`, then the harness tests `src/tests/test_harness_*.py`. If a name is missing
      from an `__all__` and a test or mypy fails, add it

**Acceptance Criteria:**

- [x] Every `src/audit` module another file reads has an `__all__`, and no harness test fails

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy src/audit` passes
- [x] `pytest src/tests/test_harness_*.py src/tests/test_platform_probe.py` passes

**Notes:**

- No behaviour change: `__all__` lists only. Check that no `src/audit` module does `from audit.<x> import *` before starting

---

## Task 2: Re-judge T1 and T2 and record what is open

**Status:** done
**Priority:** high
**Complexity:** large
**Spec Reference:** Implementation step 2

**Technical Goal:** every ledger verdict predates the current briefs (T1 version 6, T2 version 3), and Task 1 changed what is public, so a
fresh verdict for each candidate tells the later tasks exactly what to test.

**Subtasks:**

- [x] Run `python3 ~/.claude/skills/auditme/checks/phaseb.py plan . --question T1 --out <scratch>/t1`, `<scratch>` a directory outside the
      repository. Judge every candidate in every batch by `~/.claude/skills/auditme/phaseb/briefs/T1.md`, reading the candidate's source and
      the tests it lists (do not run them), and write `result-NN.jsonl` beside each `batch-NN.jsonl` in the brief's format
- [x] `phaseb.py check <scratch>/t1` must report every batch complete; then `phaseb.py merge . --question T1 <scratch>/t1`
- [x] The same three steps for T2, with `briefs/T2.md` and `<scratch>/t2`
- [x] Write `agents/specs/direct-tests-for-the-audit-harness/open-after-rejudge.md`, a dated record ("measured on <date> at <commit>") of
      every open `src/audit` and `bin` candidate in `phaseb.py report . --question T1` and `--question T2`: its id, question, group (by the
      rules in `measurements.md`, first match wins) and the task that handles it (A: Tasks 4-5, B: 6, C: 7-9 by the module lists there,
      D: 10, E: 9)
- [x] Stop as `blocked`, naming the candidate, if the brief does not settle one
- [x] Note any open candidate under `src/metafopy` in the record as outside this list; do not test it here

**Acceptance Criteria:**

- [x] Every planned candidate has a fresh ledger verdict, and the open list with its group and task columns is committed

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `phaseb.py check <scratch>/t1` and `phaseb.py check <scratch>/t2` report every batch complete

**Notes:**

- Depends on Task 1
- `plan` re-plans every candidate of both questions (1230 for T1 and 193 for T2 at `73ffe43`, measured 2026-11-22), so this is the longest
  task; the `src/metafopy` candidates are judged the same way, since a merge records them too

---

## Task 3: The R snippet fixture and its helpers

**Status:** done
**Priority:** high
**Complexity:** large
**Spec Reference:** Module location; API; Implementation step 3; Settings; decision D2

**Technical Goal:** a committed record of R's parse tree and value for each R snippet, and the helpers that run a snippet through the
transpiler and shims and compare it with R, so groups A and B test against R without R installed.

**Subtasks:**

- [x] Move R's `ast()` unchanged from `src/audit/r/runscripts.R` into `src/audit/r/common.R`, and make `runscripts.R` get it from there
      (it already sources `common.R`; check). Diff the function text before and after: it must be identical
- [x] Write `src/tests/r/generate_audit_snippets.R` as Implementation step 3 says: a named list of R snippets; for each, `parse(text = )`,
      `ast()` of every expression, evaluation in a fresh environment after `set.seed(1234)` with `metafor` and `metadat` attached,
      `node_of()` of the last value (or the error message), `deparse()` for formula snippets; a `meta` block with `R.version.string`,
      `R.version$major`/`minor` and `packageVersion("metafor")`. It stops with an error naming the installed and the pinned version when
      they differ from `src/audit/data/r-versions.json`. It writes `src/tests/reference/audit/snippets.json`
- [x] Start the snippet list with one snippet per case Tasks 4-6 need, read from each open group A and B candidate's ledger `reason`
      in `open-after-rejudge.md`; Tasks 4-6 may add snippets and rerun the generator
- [x] Run it with `LC_ALL=C.UTF-8 R_LIBS=/Users/simonb/src/metafopy/.audit/Rlib Rscript src/tests/r/generate_audit_snippets.R`
- [x] Write `src/tests/auditsnippets.py` with `Snippet`, `load_snippet`, `run_in_python`, `assert_same_as_r` and `SNIPPET_RTOL = 1e-12`,
      as the spec's API section gives them
- [x] Write `src/tests/test_auditsnippets.py`: a snippet loads; `load_snippet` fails naming both versions on a fixture whose `meta` differs
      from the pin (a temporary copy); `assert_same_as_r` passes on a match and fails on a changed cell

**Acceptance Criteria:**

- [x] `ast()` is defined once, in `common.R`, with its text unchanged
- [x] The fixture is committed, its `meta` matches `r-versions.json`, and `test_auditsnippets.py` passes without R on the path

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy src/tests/auditsnippets.py src/tests/test_auditsnippets.py` passes
- [x] `pytest src/tests/test_auditsnippets.py src/tests/test_harness_scripts.py` passes

**Notes:**

- Depends on Task 2
- Do not add the generator to `.github/workflows/r-fixtures.yml`, which installs the latest R (spec, Error handling)
- The `ast()` move means `bin/metafor-audit.sh --recall` is due; Task 10 reports it, and the operator starts it. Do not run it here

---

## Task 4: Group A, R-library shims (first half)

**Status:** todo
**Priority:** high
**Complexity:** large
**Spec Reference:** Implementation step 4; decision D1

**Technical Goal:** the first half of the open `src/audit/rlib.py` candidates, in the order `open-after-rejudge.md` lists them, are checked
against R through the snippet fixture.

**Subtasks:**

- [ ] For each of the first half (rounded up) of the group A rows in `open-after-rejudge.md`: add one snippet per case its ledger `reason`
      names to the generator, rerun it, and write a test in `src/tests/test_harness_rlib.py` that runs each through
      `auditsnippets.run_in_python` and checks it with `auditsnippets.assert_same_as_r`
- [ ] Take inputs from R text on bundled datasets, on fits whose values the parity fixtures already match (spec step 4)
- [ ] A private helper (`accepts`, `as_array`, `format_cell`, `Named.__array_wrap__`, `recycled`, `truth`) is covered by the snippet that
      reaches it, or by a direct call where none does; a branch R text cannot reach gets a hand-built expected value and a comment saying why

**Acceptance Criteria:**

- [ ] Each candidate in this half has a passing test per case its finding names

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy src/tests/test_harness_rlib.py` passes
- [ ] `pytest src/tests/test_harness_rlib.py src/tests/test_auditsnippets.py` passes

**Notes:**

- Depends on Task 3

---

## Task 5: Group A, R-library shims (second half)

**Status:** todo
**Priority:** high
**Complexity:** large
**Spec Reference:** Implementation step 4; decision D1

**Technical Goal:** the remaining open `src/audit/rlib.py` candidates are checked against R as in Task 4.

**Subtasks:**

- [ ] The rest of the group A rows in `open-after-rejudge.md`, as Task 4's subtasks say

**Acceptance Criteria:**

- [ ] Each group A candidate has a passing test per case its finding names

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy src/tests/test_harness_rlib.py` passes
- [ ] `pytest src/tests/test_harness_rlib.py src/tests/test_auditsnippets.py` passes

**Notes:**

- Depends on Task 4

---

## Task 6: Group B, the transpiler

**Status:** todo
**Priority:** high
**Complexity:** large
**Spec Reference:** Implementation step 5; decisions D1, D3

**Technical Goal:** every open `src/audit/transpile.py` candidate is checked on each case its finding names: a snippet that runs code against
R's value, `unparse` against R's `deparse()`, and a refused construct against `audit.transpile.UntranslatableError`.

**Subtasks:**

- [ ] For each group B row in `open-after-rejudge.md`, add the snippets its ledger `reason` names (node types, `$<-`, `[<-` and `f<-`
      replacements, plain, nested and superassignment targets, operators, `if`, `for`, `function` with and without defaults, lazy and NSE
      calls, reserved names, `pairmat()` inside a model call), rerun the generator, and add tests to `src/tests/test_harness_transpile.py`
- [ ] Test through the entry point `audit.transpile.translate_unit`; with D3 answered yes, a private helper is covered by the entry-point
      test that reaches it

**Acceptance Criteria:**

- [ ] Each group B candidate has a passing test per case its finding names

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy src/tests/test_harness_transpile.py` passes
- [ ] `pytest src/tests/test_harness_transpile.py src/tests/test_auditsnippets.py` passes

**Notes:**

- Depends on Task 3

---

## Task 7: Group C, comparison and report helpers

**Status:** todo
**Priority:** high
**Complexity:** large
**Spec Reference:** Implementation step 6; decision D1

**Technical Goal:** the open T1 group C candidates in `printdiff`, `valuetree`, `report`, `steps` and `apidiff`, which decide what an audit
counts as a difference and how it reports it, each have a direct test.

**Subtasks:**

- [ ] For each group C T1 row in these modules: a direct test on hand-built inputs asserting the value its docstring and
      `src/audit/README.md` state (for example: two printed numbers differing in the last digit are `close_numbers`; `0` against `0.0000` is
      not), in `src/tests/test_harness_compare.py`, or `src/tests/test_harness_apidiff.py` (new) for `apidiff`
- [ ] `apidiff` tests use small fits of bundled datasets

**Acceptance Criteria:**

- [ ] Each group C T1 candidate in these modules has a passing direct test

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy src/tests/test_harness_compare.py src/tests/test_harness_apidiff.py` passes
- [ ] `pytest src/tests/test_harness_compare.py src/tests/test_harness_apidiff.py` passes

**Notes:**

- Depends on Task 2

---

## Task 8: Group C, the other harness helpers

**Status:** todo
**Priority:** medium
**Complexity:** large
**Spec Reference:** Implementation step 6; decision D1

**Technical Goal:** every other open T1 group C candidate (in `pycall`, `generate`, `domains`, `ledger`, `universe`, `engine`, `pyrun` and
any module not in Task 7) has a direct test.

**Subtasks:**

- [ ] For each remaining group C T1 row: a direct test on hand-built inputs in the file the spec's Module location names
      (`test_harness_pycall.py`, `test_harness_accounts.py` for `generate`, `domains`, `ledger` and `universe`, `test_harness_engine.py`,
      `test_harness_errors.py` for `pyrun`); a module the spec does not name goes in the harness test file that already imports it
- [ ] `universe` tests build a `Universe` from a small dict in the shape of `src/audit/data/universe.json`; `pycall` tests use small fits of
      bundled datasets

**Acceptance Criteria:**

- [ ] Each group C T1 candidate has a passing direct test

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy` passes on each test file the task changed
- [ ] `pytest src/tests/test_harness_pycall.py src/tests/test_harness_accounts.py src/tests/test_harness_engine.py src/tests/test_harness_errors.py`
      passes, with any other file the task changed

**Notes:**

- Depends on Task 7 (both edit harness test files; run one after the other)

---

## Task 9: Group C property tests and the bin scripts

**Status:** todo
**Priority:** medium
**Complexity:** medium
**Spec Reference:** Implementation steps 6 and 7; decision D4

**Technical Goal:** each open T2 group C candidate has a hypothesis test of the property its ledger `reason` states, and the
`bin/metafor-audit.py` and `bin/platform-probe.py` candidates are tested.

**Subtasks:**

- [ ] For each group C T2 row: a hypothesis test of the property in its ledger `reason`
      (`grep -F '"id": "<candidate id>"' agents/audit-reviews/ledger/T2.jsonl`), in the file Tasks 7 and 8 use for its module
- [ ] `bin/metafor-audit.py`: load it by path with `importlib.util.spec_from_file_location`, as `src/tests/test_platform_probe.py` does, and
      give each open candidate (`metafopy_source` among them) a direct test and, for its T2 row, a property test, in
      `src/tests/test_metafor_audit_script.py` (new)
- [ ] `bin/platform-probe.py`: a direct test per open candidate in `src/tests/test_platform_probe.py`, and a `main()` run on the groups that
      reach `ls_runs` and `measure_selmodel_decreasing`
- [ ] Do not test `bin/logo.py` (D4: dismissed in Task 10)

**Acceptance Criteria:**

- [ ] Each group C T2 candidate has a passing property test, and each open `metafor-audit.py` and `platform-probe.py` candidate a passing test

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy` passes on each test file the task changed
- [ ] `pytest src/tests/test_metafor_audit_script.py src/tests/test_platform_probe.py` passes, with every other test file the task changed

**Notes:**

- Depends on Task 8

---

## Task 10: Re-judge, dismiss, the full check and closing

**Status:** todo
**Priority:** high
**Complexity:** medium
**Spec Reference:** Implementation step 8; decisions D1, D4

**Technical Goal:** no `src/audit` or `bin` candidate is open in T1 or T2 (each tested or dismissed), the branch passes every check once, the
spec is marked implemented and the `TODO.md` item moves to `DONE.md`.

**Subtasks:**

- [ ] Repeat Task 2's plan, judge, check and merge for T1 and T2. A candidate still open, other than a group D or `bin/logo.py` one, goes back to the task
      its row in `open-after-rejudge.md` names
- [ ] After the last merge, dismiss each open group D candidate on each of T1 and T2 it is open on:
      `python3 ~/.claude/skills/auditme/checks/phaseb.py dismiss . --question <Q> <id> --reason "Group D of agents/specs/direct-tests-for-the-audit-harness/spec.md (decision D1): drives R, git, subprocesses, sockets, signals or the network; covered end to end by bin/metafor-audit.sh --recall, which runs the harness steps against R on the commits of earlier audits."`
- [ ] Dismiss each open `bin/logo.py` candidate the same way, with the reason
      `"bin/logo.py (decision D4 of agents/specs/direct-tests-for-the-audit-harness/spec.md): a PEP 723 script whose dependencies (uharfbuzz among them) are outside the project environment; its output is the committed logo, checked by eye."`
- [ ] `phaseb.py report . --question T1` and `--question T2` list no open `src/audit` or `bin` candidate
- [ ] Hoist anything the new test modules duplicate from the existing harness tests or `refutil.py` to a single owner: grep the tree for each
      literal collection, constant and one-line helper they define, and import rather than copy
- [ ] Move the `TODO.md` item "Audit-harness and bin helpers with branches have no direct test" to the end of `DONE.md`: append its heading
      and body unchanged, then the line `Done: <date> - agents/specs/direct-tests-for-the-audit-harness/spec.md,
      agents/tasks/direct-tests-for-the-audit-harness/tasks-1.md; group D and bin/logo.py dismissed in the ledger (D1, D4).`, and remove it
      from `TODO.md`
- [ ] Set the spec's `**Status:**` to `implemented`, and its row in `agents/specs/index.md` to say so
- [ ] Run the final tests from the worktree and save their whole output for the merge gate. The main checkout is the first entry of
      `git worktree list --porcelain`, `/Users/simonb/src/metafopy`:
      `mkdir -p /Users/simonb/src/metafopy/ralpher-logs/direct-tests-for-the-audit-harness/tasks-1 && export PATH=/Users/simonb/src/metafopy/.venv/bin:$PATH PYTHONPATH=$PWD/src && pytest src/tests 2>&1 | tee /Users/simonb/src/metafopy/ralpher-logs/direct-tests-for-the-audit-harness/tasks-1/final-tests.log`
- [ ] In the task's Notes, record that `bin/metafor-audit.sh --recall` is due after the `ast()` move of Task 3, with that command; the
      operator starts it

**Acceptance Criteria:**

- [ ] No `src/audit` or `bin` candidate is open in the T1 or T2 report, the spec says `implemented`, and the item is in `DONE.md` with its
      body unchanged

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `pre-commit run --all-files` passes
- [ ] `pre-commit run auditme-all --hook-stage manual --all-files` passes
- [ ] `mypy` passes
- [ ] The final test run above passes
- [ ] `git show $(git log --diff-filter=A --format=%H -- agents/tasks/direct-tests-for-the-audit-harness/tasks-1.md):TODO.md > <scratch>/todo-before.md`
      and `python3 ~/.claude/skills/todo-review/verify_bodies.py <scratch>/todo-before.md TODO.md DONE.md` exits 0

**Notes:**

- Depends on Tasks 4-9
- Dismiss only after the last merge: a merge keeps a dismissal only while the candidate's hash and findings are unchanged (spec step 8)
- The full run can outlast a ten-minute command limit; run it in the background with the longest limit the session allows, and wait for its
  exit status before going on
