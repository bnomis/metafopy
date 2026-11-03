# Tasks: R oracle pin

**Spec:** `agents/specs/r-oracle-pin/spec.md`
**Created:** 2026-11-03
**Version:** 1

The review of 2026-11-03 (`agents/specs/r-oracle-pin/review.md`) approved
every part of the spec. The project has no affected-tests tool, so each task
names its test files; the last task runs every test file that imports a
changed `audit` module. The tests go in `src/tests/test_harness_compare.py`,
which already tests `audit.recall` and `audit.steps` (the spec's Module
location names `test_harness_errors.py` and `test_harness_accounts.py`;
`test_harness_compare.py` is where these modules' tests live today).

---

## Task 1: Pin the R library and refuse a different one

**Status:** done
**Priority:** high
**Complexity:** small
**Spec Reference:** API (`Context.allow_r_drift`, `library_drift`, `check_r`), Implementation - The pin, Error handling, Settings

**Technical Goal:** Commit the pin and make `check-r` compare the whole library with it, refusing a missing pin or any difference unless
`--allow-r-drift` is given, so the oracle can no longer move unseen.

**Subtasks:**

- [x] Write `src/audit/data/r-versions.json` from the 2026-11-02 run's library: run `Rscript src/audit/r/versions.R <out>` against the installed
      library (`METAFOPY_AUDIT_RLIB`) and check it reports R 4.6.2 and lme4 2.1-0; if the installed library differs from those, stop the task as
      `blocked` and name the differences
- [x] Add `library_drift(lock, installed)` to `src/audit/steps.py`
- [x] Rewrite `check_r`: a missing pin raises `RuntimeError` naming the re-pin step (also with the flag); drift raises listing every package as
      `name pinned -> installed` and the three ways on (reinstall with `setup.R`, re-pin after review, `--allow-r-drift`); with
      `ctx.allow_r_drift` the same list becomes the note `R library differs from the pin (--allow-r-drift): ...`; the metafor check stays first
- [x] Add `Context.allow_r_drift` and `--allow-r-drift` to `src/audit/cli.py`, and pass the flag on from `--recall` to the runs
      `audit.recall.run_audit` starts
- [x] Tests in `src/tests/test_harness_compare.py`: `library_drift` on equal, changed, missing and extra packages; `check_r` with
      `audit.rdriver.run` stubbed (writing a given versions file) and `audit.config.DATA_DIR` pointed at a temporary pin: no pin raises, drift
      raises naming the package and the three ways, drift with `allow_r_drift` returns the note

**Acceptance Criteria:**

- [x] `bin/metafor-audit.sh --steps check-r` passes on this machine's library
- [x] The same command with one pinned version edited in a scratch copy of the pin (restored afterwards) fails naming that package

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy` passes
- [x] `pytest src/tests/test_harness_compare.py` passes

**Notes:**

- Key files: `src/audit/steps.py`, `src/audit/cli.py`, `src/audit/recall.py`, `src/audit/data/r-versions.json`
- A worktree has no `.audit/Rlib`: point `METAFOPY_AUDIT_RLIB` at the main checkout's `.audit/Rlib`, and build `.venv` with
  `uv sync --frozen --all-extras` (never edit `uv.lock`)

---

## Task 2: Install the pinned versions in setup

**Status:** done
**Priority:** high
**Complexity:** small
**Spec Reference:** Implementation - Setup, Error handling

**Technical Goal:** Make `setup.R` install every package at its pinned version, so a new machine gets the same oracle.

**Subtasks:**

- [x] In `src/audit/r/setup.R`, read `src/audit/data/r-versions.json`; install metafor, metadat and gsl first as today, then, in the pin file's
      order, every package whose installed version differs, with `remotes::install_version(p, version, upgrade = "never", dependencies = FALSE)`;
      skip `R`; print each failure under `NOT INSTALLED` with its pinned version
- [x] Update the header comment of `setup.R` and the "Setting up R" section of `src/audit/README.md`: every package is pinned in
      `data/r-versions.json`, `check-r` refuses a different library, and how to re-pin
- [x] Run `Rscript src/audit/r/setup.R <library> <out.json>` against the installed library and check it installs nothing (every version already
      matches) and that `out.json` equals the pin

**Acceptance Criteria:**

- [x] On the installed library, `setup.R` reports no package to install and writes versions equal to the pin

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)

**Notes:**

- Depends on Task 1 (the pin file)
- Key files: `src/audit/r/setup.R`, `src/audit/README.md`
- Do not install into the main checkout's library from a worktree with anything but matching versions; the check above must be a no-op

---

## Task 3: Count recall items the oracle moved apart

**Status:** todo
**Priority:** high
**Complexity:** small
**Spec Reference:** API (`Item.oracle_moved`), Implementation - Recall items the oracle moved, Error handling

**Technical Goal:** Keep a recall item whose difference the pinned R no longer shows as a record, counted neither as reproduced nor as missed.

**Subtasks:**

- [ ] Add `oracle_moved` to `audit.recall.Item` and to `Corpus.load` (default `''`)
- [ ] In `recall.write`, leave such items out of each audit's counts and the total, add `; <n> not shown by the pinned R` to the total line,
      and list them under `## Oracle moved (<n>)` with their text and the field; an `oracle_moved` item that matches a finding is listed as
      `reproduced again` and does not make the run fail
- [ ] Describe the field in the header comment of `src/audit/data/recall.toml`, and give `a6-glmm-um-rs`
      `oracle_moved = "2026-11-02, R 4.6.2 and lme4 2.1-0: R gives se 0.182140, Python 0.182139"`
- [ ] Update the existing `Item(...)` calls in `src/tests/test_harness_compare.py` for the new field, and add tests: an `oracle_moved` item is
      in neither count and is listed in its section; one that matches a finding is marked `reproduced again`

**Acceptance Criteria:**

- [ ] `recall.write` on audit 6's results without the `a6-glmm-um-rs` hit reports every other item and one item not shown by the pinned R

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy` passes
- [ ] `pytest src/tests/test_harness_compare.py` passes

**Notes:**

- Key files: `src/audit/recall.py`, `src/audit/data/recall.toml`, `src/tests/test_harness_compare.py`

---

## Task 4: Bring the ledger, universe and docs up to the pin

**Status:** todo
**Priority:** high
**Complexity:** medium
**Spec Reference:** Implementation - Bringing the data up to the pin

**Technical Goal:** Make the known differences, the universe and `docs/differences.md` match the pinned oracle, and confirm with real audit and
recall runs.

**Subtasks:**

- [ ] Run `bin/metafor-audit.sh` in the background and wait; change no file it reads while it runs
- [ ] In `src/audit/data/known.toml`: remove the `*/glmm_rs*` value-diff entry, and its print-diff companion if the run lists it as stale; remove
      the `options/selmodel/*` `delta.1` print-diff entry and the `construct/glmm_cm/*` refused-where-R-works entry; move the `detail` of the
      `matrix/dfbetas/mv_wide` and `matrix/rstudent/mv_wide` entries to where the same cause now shows (element 53, line 3 in the 2026-11-02
      run), keeping their reasons; turn the `glmm_cm` `warning-missing` entry for "Cannot invert the Hessian" into a `warning-extra` entry with a
      reason saying metafopy warns where R now inverts
- [ ] If the run lists a stale ledger entry, or a new cause, that the step above does not name, stop the task as `operator` and list them: the
      decision covers only the entries named
- [ ] Review the universe difference (compare the run's `universe.json` with `src/audit/data/universe.json`) and write it with
      `bin/metafor-audit.sh --steps check-r,universe --update-universe`; record in this task's Notes what changed, dated
- [ ] Re-render the run with `bin/metafor-audit.sh --rerender <run>` and check that no stale ledger entry and no universe finding remain
- [ ] Run `bin/metafor-audit.sh --recall` and check that every item is reproduced or `oracle_moved`
- [ ] In `docs/differences.md`, state R 4.6.2 and lme4 2.1-0 where it records R 4.3.3 and lme4 2.0.6, and change each entry whose ledger entry
      this task changed (the `UM.RS` section no longer shows a difference)
- [ ] Compare the run's causes with `TODO.md` items 2 to 8 (filed from `.audit/runs/20261102-223017`); add a dated note to each item where a
      cause vanished or appeared, and record the comparison in this task's Notes

**Acceptance Criteria:**

- [ ] The re-rendered report has no stale ledger entries and no universe finding
- [ ] The recall report counts every item as reproduced or not shown by the pinned R

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `pytest src/tests/test_harness_compare.py` passes

**Notes:**

- Depends on Tasks 1 and 3
- Key files: `src/audit/data/known.toml`, `src/audit/data/universe.json`, `docs/differences.md`, `TODO.md`
- The run directory lives in the worktree and is lost when it is removed, so the Notes carry the figures, not only the path
- The whole audit took about 35 minutes on 2026-11-02, and `--recall` several hours

---

## Task 5: Check the whole change and close the spec

**Status:** todo
**Priority:** medium
**Complexity:** small
**Spec Reference:** whole spec

**Technical Goal:** Run the whole-tree checks once and record the work as done.

**Subtasks:**

- [ ] Set the spec's `**Status:**` to `implemented`, and its row in `agents/specs/index.md` to "Implemented: ..."
- [ ] Move the `TODO.md` item "The audit harness misses one recall item and carries stale ledger entries" to the end of `DONE.md`: append its
      heading and body unchanged, then `Done: <date> - agents/specs/r-oracle-pin/spec.md, agents/tasks/r-oracle-pin/tasks-1.md.`, and remove it
      from `TODO.md`
- [ ] Run `python3 ~/.claude/skills/todo-review/verify_bodies.py <baseline> TODO.md DONE.md`, where the baseline is `TODO.md` at the commit that
      added this task list (`git show $(git log --diff-filter=A --format=%H -- agents/tasks/r-oracle-pin/tasks-1.md):TODO.md`); the notes Task 4
      added to items 2 to 8 are additions, so their bodies still contain the baseline text

**Acceptance Criteria:**

- [ ] `verify_bodies.py` exits 0

**Definition of Done:**

- [ ] `pre-commit run --all-files` passes
- [ ] `pre-commit run auditme-all --hook-stage manual --all-files` passes
- [ ] `mypy` passes
- [ ] `pytest src/tests/test_harness_compare.py src/tests/test_harness_accounts.py src/tests/test_harness_engine.py src/tests/test_harness_errors.py src/tests/test_harness_scripts.py src/tests/test_harness_transpile.py src/tests/test_audit10_fixes.py`
      passes (every test file that imports a changed `audit` module)

**Notes:**

- Depends on Tasks 1 to 4
