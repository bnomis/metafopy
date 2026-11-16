# Tasks: A harness test runs Rscript with no skip where R is missing

**Source:** TODO.md, "A harness test runs Rscript with no skip where R is missing"
**Created:** 2026-11-16
**Version:** 1

Row 1 of the run order `agents/tasks/order-ci-platform-failures.md` (plan approved on 2026-11-16 in
`agents/reviews/order-ci-platform-failures-specs.md`). `test_r_takes_the_cluster_port_it_is_given` (`src/tests/test_harness_engine.py`) runs
`audit.rdriver.rscript_command(...)` and fails in CI on Linux and macOS with `FileNotFoundError: [Errno 2] No such file or directory: 'Rscript'`
(runs 43832306599 and 43883708179, 2026-11-16): the CI test job installs no R. On the development Mac R is installed, so the test runs there and must
keep running. The project has no affected-tests tool, so each task names its test files.

---

## Task 1: Skip the cluster-port test where Rscript is missing

**Status:** done
**Priority:** high
**Complexity:** small
**Source Reference:** the item's fix direction (skip when `shutil.which('Rscript')` is `None`, with the reason)

**Technical Goal:** the test runs wherever `Rscript` is on the path and is skipped, with its reason, where it is not.

**Subtasks:**

- [x] Check that no other test in `src/tests/` already has a marker for a missing `Rscript` (`grep -rnw 'Rscript' src/tests/*.py src/tests/refutil.py`);
      reuse one if it exists, otherwise add to `src/tests/refutil.py`, beside `linux_only` and `macos_only`:
      `needs_rscript = pytest.mark.skipif(shutil.which('Rscript') is None, reason='Rscript is not on the path (CI installs no R)')`, with `import shutil`
- [x] Mark `test_r_takes_the_cluster_port_it_is_given` with it (import `refutil` the way the other test modules do)
- [x] Check every other test in `src/tests/test_harness_engine.py` and `src/tests/test_harness_scripts.py` that calls `Rscript` for real (not a stand-in
      Python process): `grep -n 'rscript_command\|Rscript' src/tests/test_harness_*.py`. Mark each such test too; list them in this task's Notes
- [x] Check the skip by hand once: `PATH=/usr/bin:/bin pytest src/tests/test_harness_engine.py -k cluster_port -rs` (with the venv's `python` called by
      its full path if the shortened `PATH` hides it) reports the test skipped with the reason, and the plain run reports it passed

**Acceptance Criteria:**

- [x] With `Rscript` on the path the test passes; without it the test is skipped with the reason above
- [x] No other test changes behaviour

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy` passes
- [x] `pytest src/tests/test_harness_engine.py src/tests/test_harness_scripts.py` passes

**Notes:**

- Key files: `src/tests/test_harness_engine.py`, `src/tests/refutil.py`
- No other test marked: the only real `Rscript` call is in `test_r_takes_the_cluster_port_it_is_given`; `test_harness_scripts.py` replaces `rscript_command` with a Python stand-in
- The next list in the order, `agents/tasks/audit-windows-tests/tasks-1.md`, also edits `src/tests/test_harness_engine.py` and waits for this one

---

## Task 2: The full test run and closing the item

**Status:** done
**Priority:** high
**Complexity:** small
**Source Reference:** the item as a whole

**Technical Goal:** the branch is checked once, and the item moves to `DONE.md`.

**Subtasks:**

- [x] Move the item from `TODO.md` to `DONE.md`: append its heading and body unchanged at the end of `DONE.md`, then a line
      `Done: 2026-MM-DD, agents/tasks/rscript-skip-guard/tasks-1.md` with the date of the commit
- [x] Run the final tests from the worktree and save their whole output for the merge gate (the main checkout is the first entry of
      `git worktree list --porcelain`, `/Users/simonb/src/metafopy`):
      `mkdir -p /Users/simonb/src/metafopy/ralpher-logs/rscript-skip-guard/tasks-1 && source bin/activate.sh && pytest src/tests 2>&1 | tee /Users/simonb/src/metafopy/ralpher-logs/rscript-skip-guard/tasks-1/final-tests.log`

**Acceptance Criteria:**

- [x] The item is in `DONE.md` with its body unchanged and a `Done:` line naming this list, and no longer in `TODO.md`

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `pre-commit run --all-files` passes
- [x] `pre-commit run auditme-all --hook-stage manual --all-files` passes
- [x] `mypy` passes
- [x] The final test run above passes
- [x] `git show $(git log --diff-filter=A --format=%H -- agents/tasks/rscript-skip-guard/tasks-1.md):TODO.md > <scratch>/todo-before.md` and
      `python3 ~/.claude/skills/todo-review/verify_bodies.py <scratch>/todo-before.md TODO.md DONE.md` exits 0 (`<scratch>` is the session's scratch
      directory, outside the repository)

**Notes:**

- Depends on Task 1
- No version bump: no library output changes
- CI shows the skip only after the merge (or in the draft pull request of `agents/tasks/cross-platform-numerics/tasks-1.md`); this list cannot run CI

---
