# Tasks: Audit harness tests assume POSIX and fail on Windows

**Source:** TODO.md, "Audit harness tests assume POSIX and fail on Windows"
**Created:** 2026-11-16
**Version:** 1

Row 2 of the run order `agents/tasks/order-ci-platform-failures.md`. The plan approved on 2026-11-16 in
`agents/reviews/order-ci-platform-failures-specs.md` settles the item's "decide per test" for every test it names: the audit harness (`src/audit/`)
does not support Windows, so each of the eight tests below is skipped on Windows with its reason, and `src/audit/README.md` says the audit runs on
macOS and Linux. Porting the harness is out of scope.

The eight tests, from CI runs 43832306599 and 43883708179 (Windows job, 2026-11-16):

| File | Test | Windows failure |
|---|---|---|
| `src/tests/test_harness_engine.py` | `test_an_r_shard_process_starts_on_this_platform` | "preexec_fn is not supported on Windows platforms" |
| `src/tests/test_harness_engine.py` | `test_a_process_past_the_memory_limit_is_killed` | `assert None == 'memory'` |
| `src/tests/test_harness_engine.py` | `test_resident_bytes_of_this_process_is_positive_and_of_a_finished_one_none` | `assert None is not None` |
| `src/tests/test_harness_engine.py` | `test_a_python_shard_process_streams_a_start_line_before_each_result` | `audit.pyrun` exits with status 1 |
| `src/tests/test_harness_engine.py` | `test_run_cases_returns_a_result_for_every_case_when_a_python_process_dies` | exit status 1, not 3 |
| `src/tests/test_harness_scripts.py` | `test_lost_script_gives_r_killed_at_the_dead_unit_and_one_not_run` | "module 'signal' has no attribute 'SIGALRM'" |
| `src/tests/test_harness_scripts.py` | `test_stream_scripts_lines_read_back_and_sum_to_run_script` | `AssertionError` (traceback in the log) |
| `src/tests/test_harness_scripts.py` | `test_run_returns_every_script_when_the_python_process_dies` | lost-process results differ |

Windows exists only in CI; this list cannot run CI, so the skips are checked on Windows after the merge, or in the draft pull request of
`agents/tasks/cross-platform-numerics/tasks-1.md`. The project has no affected-tests tool, so each task names its test files.

---

## Task 1: Skip the audit harness tests on Windows

**Status:** done
**Priority:** high
**Complexity:** small
**Source Reference:** the item's fix direction, as settled by the approved plan (skip on Windows with the reason; say so in `src/audit/README.md`)

**Technical Goal:** the eight tests are skipped on Windows, each with a reason naming the POSIX facility it needs, and still run on macOS and Linux.

**Subtasks:**

- [x] Read the Windows job's log of run 43883708179 for the two tests whose cause the table does not name (`gh run view 43883708179 --log-failed`, grep
      for each test name and read its traceback; read-only): `test_a_python_shard_process_streams_a_start_line_before_each_result` (the `py-0.log` error
      of `audit.pyrun`) and `test_stream_scripts_lines_read_back_and_sum_to_run_script`. Record each cause in this task's Notes
- [x] If either cause is not a POSIX facility of the harness (`preexec_fn`, `resource` limits, resident-memory reads, `SIGALRM`, signal exit codes) -
      for example a path or encoding bug that would also matter on POSIX - do not skip that test: stop as `blocked` and name the cause, since the
      approved plan covers only tests that need POSIX
- [x] Add to `src/tests/refutil.py`, beside `linux_only` and `macos_only`, a marker
      `posix_only = pytest.mark.skipif(sys.platform == 'win32', reason=...)` taking the reason per use, or a function `posix_only(reason: str)` that
      returns the marker, so each test names its own facility
- [x] Mark each of the eight tests with it, the reason naming the facility (for example "the audit harness caps memory with preexec_fn and resource,
      which Windows lacks; the audit runs on macOS and Linux"). If `agents/tasks/rscript-skip-guard/tasks-1.md` added `needs_rscript` to the same file,
      keep both markers where both apply
- [x] Add one line to `src/audit/README.md`, near the top: the audit runs on macOS and Linux; it uses `preexec_fn`, resource limits, resident-memory
      reads and `SIGALRM`, which Windows lacks, and its harness tests are skipped there

**Acceptance Criteria:**

- [x] The eight tests carry the Windows skip with a reason each; on the development Mac they still run and pass
- [x] `src/audit/README.md` says which platforms the audit runs on and why not Windows
- [x] No test outside the eight changes, and no file under `src/audit/` other than `README.md` changes

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy` passes
- [x] `pytest src/tests/test_harness_engine.py src/tests/test_harness_scripts.py -rs` passes and reports no test skipped on the development Mac other
      than any skipped before this task

**Notes:**

- Depends on `agents/tasks/rscript-skip-guard/tasks-1.md` (merged): both edit `src/tests/test_harness_engine.py` and `src/tests/refutil.py`
- Causes read from run 43883708179: both `test_a_python_shard_process_streams_a_start_line_before_each_result` and
  `test_stream_scripts_lines_read_back_and_sum_to_run_script` fail with `AttributeError: module 'signal' has no attribute 'SIGALRM'` at
  `src/audit/pyrun.py:124`, a POSIX facility, so both are skipped
- Key files: `src/tests/test_harness_engine.py`, `src/tests/test_harness_scripts.py`, `src/tests/refutil.py`, `src/audit/README.md`

---

## Task 2: The full test run and closing the item

**Status:** done
**Priority:** high
**Complexity:** small
**Source Reference:** the item as a whole

**Technical Goal:** the branch is checked once, and the item moves to `DONE.md`.

**Subtasks:**

- [x] Move the item from `TODO.md` to `DONE.md`: append its heading and body unchanged at the end of `DONE.md`, then a line
      `Done: 2026-MM-DD, agents/tasks/audit-windows-tests/tasks-1.md; the harness tests are skipped on Windows, which the audit does not support` with the
      date of the commit
- [x] Run the final tests from the worktree and save their whole output for the merge gate (the main checkout is the first entry of
      `git worktree list --porcelain`, `/Users/simonb/src/metafopy`):
      `mkdir -p /Users/simonb/src/metafopy/ralpher-logs/audit-windows-tests/tasks-1 && source bin/activate.sh && pytest src/tests 2>&1 | tee /Users/simonb/src/metafopy/ralpher-logs/audit-windows-tests/tasks-1/final-tests.log`

**Acceptance Criteria:**

- [x] The item is in `DONE.md` with its body unchanged and a `Done:` line naming this list, and no longer in `TODO.md`

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `pre-commit run --all-files` passes
- [x] `pre-commit run auditme-all --hook-stage manual --all-files` passes
- [x] `mypy` passes
- [x] The final test run above passes
- [x] `git show $(git log --diff-filter=A --format=%H -- agents/tasks/audit-windows-tests/tasks-1.md):TODO.md > <scratch>/todo-before.md` and
      `python3 ~/.claude/skills/todo-review/verify_bodies.py <scratch>/todo-before.md TODO.md DONE.md` exits 0 (`<scratch>` is the session's scratch
      directory, outside the repository)

**Notes:**

- Depends on Task 1
- No version bump: no library output changes

---
