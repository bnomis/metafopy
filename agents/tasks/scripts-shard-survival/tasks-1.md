# Tasks: Scripts shard survival

**Spec:** `agents/specs/scripts-shard-survival/spec.md`
**Created:** 2026-11-02
**Version:** 1

The review of 2026-11-02 (`agents/specs/scripts-shard-survival/review.md`)
approved every part of the spec. The project has no affected-tests tool, so
each task names its test files; the last task runs every test file that
imports a changed `audit` module. This work follows the cases path built by
`agents/tasks/audit-shard-survival/tasks-1.md` (done 2026-11-02): reuse its
`engine.run_shard`, `watch` and `pyrun.run_stream` patterns, not copies of
them.

---

## Task 1: Run the scripts step once before the change

**Status:** done
**Priority:** high
**Complexity:** small
**Spec Reference:** Tests (the real run before the change)

**Technical Goal:** Record what the `scripts` step finds today, so the run after the change can be compared with it. No scripts run exists in this
checkout (checked 2026-11-02).

**Subtasks:**

- [x] Before changing any file, run `bin/metafor-audit.sh --steps scripts` in the background and wait for it; it reads `src/audit/`, so change nothing
      while it runs
- [x] Record in this task's Notes the run directory under `.audit/runs/`, the commit it ran at, its exit status, and the scripts coverage tally and
      cause lines from its `wrong-or-missing.md`, dated
- [x] If the step stops on a dead process (the defect this list fixes), record that and what it left, instead of a tally

**Acceptance Criteria:**

- [x] This task's Notes name the run and hold its tally and causes, or the failure it hit

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)

**Notes:**

- A worktree has no `.venv` or `.audit/Rlib`: build the environment with `uv sync --frozen --all-extras` (never edit `uv.lock`) and point
  `METAFOPY_AUDIT_RLIB` at the main checkout's `.audit/Rlib`, as `agents/tasks/audit-shard-survival/tasks-1.md` Task 6 did
- The run directory lives in the worktree and is lost when the worktree is removed, so the Notes must carry the figures, not only the path
- Baseline run, 2026-11-02, `bin/metafor-audit.sh --steps scripts` at commit `d811e57`, exit status 0, run directory
  `.audit/runs/20261102-202532/` (worktree; lost with it). The step did not stop on a dead process.
- Scripts coverage tally: 8064 findings; 6446 expressions, 4906 compared (261s). Of the findings, 7301 matched `data/known.toml` and 760 numeric
  differences were below the reporting threshold; 4 findings make 3 causes
- Cause lines in `wrong-or-missing.md`: two wrong values (a `plot(sav, ...)` `$ys` element 941, relative 0.00019, 2 cases; an
  `expect_equivalent(c(residuals(res)), ...)` element 5, relative 0.00017, 1 case) and one other (the R universe differs from
  `data/universe.json`; refresh with `--update-universe`)

---

## Task 2: Part lines and any item in the shared loop

**Status:** todo
**Priority:** high
**Complexity:** small
**Spec Reference:** API (`Item`, `read_outcomes`, `run_shard`), Implementation - Shared loop

**Technical Goal:** Let `engine.run_shard` drive items that are not cases and keep the part lines an item wrote before its process stopped, without
changing any result of the cases path.

**Subtasks:**

- [ ] Add the `Item` protocol to `src/audit/engine.py`
- [ ] Make `read_outcomes` return `(done, running, running_parts)`: a `{"part": id, ...}` line is appended to that item's parts; a finished item's
      record gets `"parts"` only when it has any; a line with none of `start`, `part`, `id` is skipped
- [ ] Give `run_shard` the type parameter `T: Item`, the `write` argument in place of `audit.cases.write_cases`, and pass the running item's parts as
      `dead`'s third argument
- [ ] Update `run_r_shard` and `run_py_shard` (the cases path) to pass `audit.cases.write_cases` and to take and ignore the parts
- [ ] Tests in `src/tests/test_harness_engine.py`: part lines gathered for a finished item and returned for the running one; `run_shard` over plain
      dataclass items with its own `write`, where a `python -c` stand-in writes two parts of its second item and exits 1, passes those two parts to
      `dead` and finishes the other items

**Acceptance Criteria:**

- [ ] Every existing test in `src/tests/test_harness_engine.py` passes unchanged, apart from call sites of `read_outcomes` updated for its third value
- [ ] A finished case's record has no `parts` key, so `r-N.json` and `py-N.json` of the cases path keep their shape

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy` passes
- [ ] `pytest src/tests/test_harness_engine.py` passes

**Notes:**

- Depends on Task 1 (the run before the change must not see this code)
- Key files: `src/audit/engine.py`, `src/tests/test_harness_engine.py`

---

## Task 3: Stream the R side of the scripts step

**Status:** todo
**Priority:** high
**Complexity:** medium
**Spec Reference:** API (`Script`, `list_scripts`, `run_r_scripts`), Implementation - The R side

**Technical Goal:** Give `runscripts.R` its `list` and `run` modes and drive each R shard through `engine.run_shard`, so a dead R process costs the
rest of one script, not the shard.

**Subtasks:**

- [ ] In `src/audit/r/runscripts.R`, add `list <source> <out.json>` (today's job discovery and `AUDIT_ONLY` filter, writing `[{id, kind, topic,
      file}]`) and `run <source> <scripts.json> <out.jsonl> <timeout>` (start, one `{"part": id, "unit": ...}` per unit, closing line, each flushed,
      `Rd2ex` at script start); remove the shard arguments
- [ ] Add `Script` and `list_scripts` to `src/audit/scripts.py`; `list_scripts` raises `RuntimeError` naming its log when `list` fails or writes
      nothing readable
- [ ] Add `run_r_scripts`: `start` uses `audit.rdriver.start` with `r-{i}.log`; a finished script's record is its closing line with `units` from its
      parts; `dead` gives the spec's lost record with the dead unit and the cases path's reasons; writes `r-{i}.json` as `{"scripts": [...]}`
- [ ] New `src/tests/test_harness_scripts.py`: `run_r_scripts` with `audit.rdriver.start` monkeypatched to a `python -c` stand-in that writes the
      R lines, once finishing and once exiting 1 after one unit of its first script; check `r-0.json` holds every script, the lost one with its
      finished unit and a dead unit ending `(killed)`
- [ ] By hand, in the worktree: `Rscript src/audit/r/runscripts.R list` on the fetched source, then `run` on two short scripts, and check the lines
      with `audit.engine.read_outcomes`; note the commands and result in this task's Notes

**Acceptance Criteria:**

- [ ] `list` gives the same script ids, in the same order, as the scripts in Task 1's run
- [ ] A stand-in R that dies partway through a script leaves a record for every script in `r-0.json`

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy` passes
- [ ] `pytest src/tests/test_harness_scripts.py src/tests/test_harness_transpile.py` passes

**Notes:**

- Depends on Task 2
- Key files: `src/audit/r/runscripts.R`, `src/audit/scripts.py`, `src/tests/test_harness_scripts.py`
- `audit.scripts.run` still calls the old R entry until Task 5; keep the step runnable at this commit by leaving the old Python side as it is, and
  switching `run`'s R half to `list_scripts` and `run_r_scripts` here
- The fetched metafor source is under the audit work directory's `source/metafor` (`audit.scripts.fetch_source`)

---

## Task 4: Stream the Python side of a scripts shard

**Status:** todo
**Priority:** high
**Complexity:** medium
**Spec Reference:** API (`stream_scripts`, `python -m audit.scripts`), Implementation - The Python side

**Technical Goal:** Run a scripts shard as its own process that writes one part line per unit, and report R's killed units without blaming
metafopy.

**Subtasks:**

- [ ] Turn `run_script` into a generator of per-unit `(findings, tally)`, keeping every finding and tally entry it gives today
- [ ] In it, give a unit whose R error satisfies `audit.pyrun.r_killed` an `r-killed` finding with R's error as detail, without translating it;
      count a dead unit with the `(timeout)` stall reason as `R failed (skipped)`; count `not run: R process died partway through the script` once
      for a script with `"lost": true`
- [ ] Add `stream_scripts(r_path, scripts_path, out_path, nse, timeout)`: `limit_memory`, the metafor function names, seed 1234 before each unit,
      start / part / closing lines flushed each time
- [ ] Add `python -m audit.scripts <job.json>`, reading `sys.argv` directly (an internal entry point, not a CLI)
- [ ] Tests in `src/tests/test_harness_scripts.py`: a lost script record gives `r-killed` at the dead unit and one `not run` tally; `stream_scripts`
      run as a subprocess on a stub `r-0.json` of two tiny scripts writes lines that `read_outcomes` reads back with no running item

**Acceptance Criteria:**

- [ ] Summing the part lines of a script gives the same findings and tally as today's `run_script` on the same R records (test it on a stub)

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy` passes
- [ ] `pytest src/tests/test_harness_scripts.py src/tests/test_harness_transpile.py` passes

**Notes:**

- Depends on Task 2 (`read_outcomes` in the test) and Task 3 (the R record shape)
- Key files: `src/audit/scripts.py`, `src/audit/pyrun.py` (`r_killed`, `run_guarded`)
- Model it on `audit.pyrun.run_stream` and its `__main__` block

---

## Task 5: Run the Python scripts shards through the restart loop

**Status:** todo
**Priority:** high
**Complexity:** medium
**Spec Reference:** API (`run_py_scripts`), Implementation - The Python side, Error handling

**Technical Goal:** Replace the scripts step's process pool with one `run_py_scripts` thread per shard, so a dead Python process costs the rest of one
script, not the step.

**Subtasks:**

- [ ] Add `run_py_scripts`: writes the job file, starts `sys.executable -m audit.scripts <job.json>` with the inherited environment and output to
      `py-{i}.log`, runs it through `engine.run_shard`, and writes `py-{i}.json` as `{"findings": [...], "tally": {...}}` from the parts
- [ ] Its `dead` keeps the parts' findings and tallies, adds the `memory`, `process-died` or `timeout` finding at unit `len(parts)` with id
      `scripts/<topic>/<kind>/<n>`, and counts the units after it as `not run: Python process died partway through the script`
- [ ] Rewrite `run`: `list_scripts`, R shards in a `ThreadPoolExecutor`, then Python shards in another; remove the process pool and `run_shard(job)`
- [ ] End-to-end test in `src/tests/test_harness_scripts.py`: `run` with `list_scripts` and `run_r_scripts` stubbed and one unit made to exit the
      Python process; every script comes back, one `process-died` finding, and `not run` for the units after it
- [ ] In `src/audit/README.md`, add the scripts step to "Shards that die or pass the memory limit"

**Acceptance Criteria:**

- [ ] `run` returns findings for every script when a Python shard process dies
- [ ] No `ProcessPoolExecutor` remains in `src/audit/scripts.py`, and `audit.steps.scripts_step` is unchanged

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy` passes
- [ ] `pytest src/tests/test_harness_scripts.py src/tests/test_harness_transpile.py src/tests/test_harness_engine.py` passes

**Notes:**

- Depends on Task 3 and Task 4
- Key files: `src/audit/scripts.py`, `src/audit/README.md`
- Model `run_py_scripts` on `audit.engine.run_py_shard`; hoist anything the two would copy (the job-file write, the stop-to-kind mapping, the detail
  texts) to one owner in `audit.engine` and import it, rather than keeping two copies

---

## Task 6: Check the whole change and close the spec

**Status:** todo
**Priority:** medium
**Complexity:** small
**Spec Reference:** whole spec

**Technical Goal:** Prove the change on the real audit, run the whole-tree checks once, and record the work as done.

**Subtasks:**

- [ ] Run `bin/metafor-audit.sh --steps scripts` in the background and confirm it completes; compare its tally and causes with Task 1's Notes and
      explain every difference in this task's Notes, dated
- [ ] Hoist anything the scripts drivers still duplicate from the cases path to a single owner: grep the tree for each literal collection,
      constant and one-line helper the changed modules define, and import rather than copy. Choose the owner from the existing import graph; prefer
      a module the consumer already imports over adding a new edge
- [ ] Set the spec's `**Status:**` to `implemented`, and its row in `agents/specs/index.md` to "Implemented: ..."
- [ ] Move the `TODO.md` item "The audit's scripts step stops when one of its processes dies" to the end of `DONE.md`: append its heading and body
      unchanged, then `Done: <date> - agents/specs/scripts-shard-survival/spec.md, agents/tasks/scripts-shard-survival/tasks-1.md.`, and remove it
      from `TODO.md`
- [ ] Run `python3 ~/.claude/skills/todo-review/verify_bodies.py <baseline> TODO.md DONE.md`, where the baseline is `TODO.md` at the commit that
      added this task list (`git show $(git log --diff-filter=A --format=%H -- agents/tasks/scripts-shard-survival/tasks-1.md):TODO.md`)

**Acceptance Criteria:**

- [ ] The real `scripts` step completes, and every difference from Task 1's run is explained
- [ ] `verify_bodies.py` exits 0

**Definition of Done:**

- [ ] `pre-commit run --all-files` passes
- [ ] `pre-commit run auditme-all --hook-stage manual --all-files` passes
- [ ] `mypy` passes
- [ ] `pytest src/tests/test_harness_engine.py src/tests/test_harness_scripts.py src/tests/test_harness_transpile.py src/tests/test_harness_compare.py src/tests/test_harness_errors.py src/tests/test_harness_accounts.py src/tests/test_audit10_fixes.py`
      passes (every test file that imports a changed `audit` module)

**Notes:**

- Depends on Tasks 1 to 5
- The `scripts` step takes a long time; do not change files it reads while it runs
- The worktree setup of Task 1's Notes applies here too
