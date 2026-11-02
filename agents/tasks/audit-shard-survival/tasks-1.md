# Tasks: Audit shard survival

**Spec:** `agents/specs/audit-shard-survival/spec.md`
**Created:** 2026-11-02
**Version:** 1

The review of 2026-11-02 (`agents/specs/audit-shard-survival/review.md`)
approved every part of the spec and kept the `scripts` step out of it. Work on
the branch `feat/audit-shard-survival`. The project has no affected-tests
tool, so each task names its test files; the last task runs every test file
that imports a changed `audit` module.

---

## Task 1: Watch a shard's output and memory

**Status:** done
**Priority:** high
**Complexity:** small
**Spec Reference:** API (`Stop`, `resident_bytes`, `watch`), Implementation - The memory watch, Settings

**Technical Goal:** Replace `wait_or_kill` with `watch`, which kills a process that stalls or passes `MEMORY_LIMIT` resident bytes and tells a process that died by itself from one that finished, so both languages can share it.

**Subtasks:**

- [x] Add `POLL_SECONDS = 2.0` to `src/audit/config.py`, and say in `MEMORY_LIMIT`'s comment and `limit_memory`'s docstring that the limit is also the resident size `audit.engine.watch` enforces
- [x] Add `Stop` and `resident_bytes(pid)` to `src/audit/engine.py`: `ps -o rss= -p <pid>`, KiB times 1024; None when `ps` fails, prints nothing, or is missing (Windows)
- [x] Add `watch(proc, out, stall, limit)` per the spec, polling every `config.POLL_SECONDS`; it returns `'stalled'`, `'memory'`, `'died'` (non-zero exit by itself) or None
- [x] Switch `run_r_shard` to `watch(..., audit.config.MEMORY_LIMIT)`, keeping today's reason texts for now (Task 2 changes them), and remove `wait_or_kill`
- [x] Update the `wait_or_kill` tests in `src/tests/test_harness_engine.py` to `watch`, and add: `resident_bytes` of `os.getpid()` is positive and of a finished process is None; a `python -c` process that allocates past a small limit gives `'memory'`; one that exits 1 gives `'died'`

**Acceptance Criteria:**

- [x] `watch` gives each of `'stalled'`, `'memory'`, `'died'` and None for the matching stand-in process
- [x] No reference to `wait_or_kill` remains in `src/`

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy` passes
- [x] `pytest src/tests/test_harness_engine.py` passes

**Notes:**

- Key files: `src/audit/engine.py`, `src/audit/config.py`, `src/tests/test_harness_engine.py`
- Use a limit of a few hundred MB in the memory test, with a stand-in that allocates well past it, so the test stays fast and never strains the machine

---

## Task 2: One restart loop for both languages

**Status:** done
**Priority:** high
**Complexity:** medium
**Spec Reference:** API (`run_shard`, `run_r_shard`), Implementation - One shard loop for both languages, Error handling

**Technical Goal:** Lift `run_r_shard`'s restart loop into `run_shard(work, name, shard, timeout, start, dead)`, so the Python side can reuse it, and give R's dead cases reasons that end in `(killed)`.

**Subtasks:**

- [x] Write `run_shard` per the spec's steps 1 to 4, using `watch` and `read_outcomes`; the stall limit is `2 * timeout + 30`
- [x] When a process stops with no case running, give every remaining case `dead(case_id, stop)` and end the loop; the detail names the exit status and the log path
- [x] Rewrite `run_r_shard` on `run_shard`, writing `r-{i}.json` as today; its `dead` gives `R did not return within <s>s (timeout)` for `stalled`, `R passed the memory limit of <n> GiB (killed)` for `memory`, `R process died on this case (killed)` for `died`
- [x] Tests in `src/tests/test_harness_engine.py` with a `python -c` stand-in for `start` that writes start and result lines: one that dies on its second case (that case gets `dead`, the rest finish, `<name>-attempt1.jsonl` is kept); one that dies before its first case (every case gets `dead`, one attempt)

**Acceptance Criteria:**

- [x] A shard whose process dies mid-way returns a result for every case
- [x] A process that dies before its first case does not restart
- [x] `pyrun.r_timed_out` still recognises the `stalled` reason

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy` passes
- [x] `pytest src/tests/test_harness_engine.py` passes

**Notes:**

- Depends on Task 1
- Key files: `src/audit/engine.py`, `src/tests/test_harness_engine.py`

---

## Task 3: Stream a Python shard's results

**Status:** done
**Priority:** high
**Complexity:** small
**Spec Reference:** API (`run_stream`, `python -m audit.pyrun`), Implementation - The Python shard process

**Technical Goal:** Give `pyrun` an entry point that runs a shard as its own process and writes a start line before each case and its result line after, flushed each time, as `runcases.R` does.

**Subtasks:**

- [x] Add `run_stream(fixtures_path, cases_path, r_path, out_path, timeout)` to `src/audit/pyrun.py`, built from today's `run_shard` body: `limit_memory`, fixtures, seed 1234 before each case, one result record per case in today's shape, written as a JSON line after a `{"start": id}` line
- [x] Add `python -m audit.pyrun <job.json>`, where the job file holds `run_stream`'s arguments; read `sys.argv` directly (an internal entry point, not a CLI)
- [x] Keep `run_shard(job)` working until Task 4 removes it
- [x] Test in `src/tests/test_harness_engine.py`: run the module as a subprocess on two tiny cases with a stub R result file, and check `read_outcomes` gives both results and no running case

**Acceptance Criteria:**

- [x] A process killed between two cases leaves the first case's result and the second case's start line in its output

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy` passes
- [x] `pytest src/tests/test_harness_engine.py src/tests/test_harness_compare.py` passes

**Notes:**

- Depends on Task 2 for `read_outcomes`' use in the test only
- Key files: `src/audit/pyrun.py`

---

## Task 4: Run the Python shards through the restart loop

**Status:** done
**Priority:** high
**Complexity:** medium
**Spec Reference:** API (`run_py_shard`), Implementation - The Python shard process, Error handling

**Technical Goal:** Replace the `ProcessPoolExecutor` in `run_cases` with one `run_py_shard` thread per shard, so a dead Python process costs one case, not the step.

**Subtasks:**

- [x] Add `run_py_shard`: writes the job file, starts `sys.executable -m audit.pyrun <job.json>` with the inherited environment and output to `py-{i}.log`, runs it through `run_shard`, and writes `py-{i}.json` as `{"results": [...]}` in shard order
- [x] Its `dead` gives the spec's result record with one finding: `memory` (detail names the limit), `process-died` (detail names the exit status) or `timeout` for `stalled`
- [x] Run the Python shards in a `ThreadPoolExecutor` in `run_cases`; remove the process pool and `pyrun.run_shard`
- [x] End-to-end test: `run_cases` on two tiny cases with R stubbed and the Python call of one case monkeypatched (through the job's environment or a test hook) to exit the process; both cases come back, one with `process-died`

**Acceptance Criteria:**

- [x] `run_cases` returns a result for every case when a Python shard process dies
- [x] `--metafopy-src` still reaches the Python shards (the child inherits `PYTHONPATH`)
- [x] No `ProcessPoolExecutor` remains in `src/audit/engine.py`

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy` passes
- [x] `pytest src/tests/test_harness_engine.py src/tests/test_harness_compare.py src/tests/test_harness_errors.py` passes

**Notes:**

- Depends on Task 2 and Task 3
- Key files: `src/audit/engine.py`, `src/audit/pyrun.py`
- `audit.scripts` keeps its own pool; it is the separate `TODO.md` item "The audit's scripts step stops when one of its processes dies"

---

## Task 5: Report killed cases

**Status:** done
**Priority:** medium
**Complexity:** small
**Spec Reference:** Implementation - Classification and report

**Technical Goal:** Stop blaming metafopy for R's killed processes, and give the new kinds a place in the report and the README.

**Subtasks:**

- [x] Add `r_killed(error)` to `src/audit/pyrun.py` (R's error ends in `(killed)`), and in `classify` give `r-killed` with R's error as detail before the `accepts-where-R-refuses` branch, whatever Python did
- [x] In `src/audit/report.py`, add `memory` and `process-died` to *Crashes*, and `('R processes killed', ('r-killed',))` after *Timeouts*
- [x] In `src/audit/README.md`, add a section beside "R calls slower than the case limit" on how a shard that dies or passes the memory limit is reported
- [x] Tests: `classify` gives `r-killed` for an R error ending `(killed)` when Python succeeded and when it failed, and still `timeout` for `(timeout)`; a report built from findings of the new kinds lists them in the expected sections

**Acceptance Criteria:**

- [x] No R error ending `(killed)` is classified `accepts-where-R-refuses`

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy` passes
- [x] `pytest src/tests/test_harness_compare.py src/tests/test_harness_errors.py src/tests/test_audit10_fixes.py` passes

**Notes:**

- Depends on Task 2 (the reason texts) and Task 4 (the Python kinds)
- Key files: `src/audit/pyrun.py`, `src/audit/report.py`, `src/audit/README.md`

---

## Task 6: Check the whole change and close the spec

**Status:** done
**Priority:** medium
**Complexity:** small
**Spec Reference:** whole spec

**Technical Goal:** Prove the change on the real audit, run the whole-tree checks once, and record the work as done.

**Subtasks:**

- [x] Run `bin/metafor-audit.sh --steps options` in the background and confirm it completes, with every case given a result; compare its cause list with `.audit/runs/20261102-061046/wrong-or-missing.md` and explain any cause that appeared or vanished
- [x] Set the spec's `**Status:**` to `implemented`, and its row in `agents/specs/index.md` to "Implemented: ..."
- [x] Move the `TODO.md` item "A Python audit worker that dies stops the whole audit step" to the end of `DONE.md`: append its heading and body unchanged, then `Done: <date> - agents/specs/audit-shard-survival/spec.md, agents/tasks/audit-shard-survival/tasks-1.md.`, and remove it from `TODO.md`
- [x] Run `python3 ~/.claude/skills/todo-review/verify_bodies.py <baseline> TODO.md DONE.md`, where the baseline is `TODO.md` at the commit that added this task list (`git show $(git log --diff-filter=A --format=%H -- agents/tasks/audit-shard-survival/tasks-1.md):TODO.md`)

**Acceptance Criteria:**

- [x] The real `options` step completes, and the comparison with the earlier run is explained
- [x] `verify_bodies.py` exits 0

**Definition of Done:**

- [x] `pre-commit run --all-files` passes
- [x] `pre-commit run auditme-all --hook-stage manual --all-files` passes
- [x] `mypy` passes
- [x] `pytest src/tests/test_harness_engine.py src/tests/test_harness_compare.py src/tests/test_harness_errors.py src/tests/test_harness_transpile.py src/tests/test_audit10_fixes.py` passes (every test file that imports a changed `audit` module)

**Notes:**

- Depends on Tasks 1 to 5
- The `options` step takes a long time; do not change files it reads while it runs
- Real run on 2026-11-02 (`.audit/runs/20261102-143634`, metafopy `72d295d`): the `options` step completed, 20029 cases, every one with a result. Against `.audit/runs/20261102-061046`: no cause vanished. Two appeared. "R processes killed: R passed the memory limit of 4 GiB" (2 `permutest` cases, `exact=true` on `uni_mods` and `ls_bcg`) is the section Task 5 added; the earlier run had no finding for those cases, as R failed on them and Python refused them, so the two agreed. "R: creation of server socket failed: port 11628 cannot be opened" (1 `gosh` case, `ncpus=2`) is a port clash in R's `gosh` cluster during this run, not a result of this change; it is why the `gosh` counts are one lower. Total findings 16751 against 16758.
- The worktree had no `.venv` or `.audit/Rlib`: the run used `uv sync --frozen --all-extras` and `METAFOPY_AUDIT_RLIB` pointing at the main checkout's library.
