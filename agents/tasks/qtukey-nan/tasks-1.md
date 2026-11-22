# Tasks: qtukey returns NaN where R does

**Source:** `TODO.md`, "`qtukey` raises ZeroDivisionError where R returns NaN"
**Created:** 2026-11-22
**Version:** 1

How to run things, in every task (from the worktree root):

- **Tests:** `export PATH=/Users/simonb/src/metafopy/.venv/bin:$PATH PYTHONPATH=$PWD/src && pytest <files>`
- **Type check:** `mypy <files>`, with the same `PATH`.
- **R:** `LC_ALL=C.UTF-8 R_LIBS=/Users/simonb/src/metafopy/.audit/Rlib Rscript -e '<code>'`. Worktrees have no `.audit/Rlib`, so use the main
  checkout's.

The fix and its output change were approved on 2026-11-22 (item 1 of `agents/reviews/order-r-parity-fixes-from-property-tests-specs.md`, Answers section). That approval covers only the change the
item describes. Any other change to metafopy's output stops the task as `blocked`, naming the function, the input and both values
(`claude/rules/metafopy.md`).

---

## Task 1: Make `qtukey` follow R's C code on a flat secant step and on non-convergence

**Status:** in-progress
**Priority:** high
**Complexity:** medium
**Source Reference:** `TODO.md`, "`qtukey` raises ZeroDivisionError where R returns NaN"; review item 1

**Technical Goal:** `qtukey` returns R's NaN, with R's warnings, where it raises `ZeroDivisionError` today, and every other value stays bit for
bit the same.

**Subtasks:**

- [ ] Before changing anything, save `qtukey(p, nmeans, df)` over the item's grid to `<scratch>/before.json`, recording which points raise.
      The grid is p in {0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99}, nmeans in {2, 3, 5, 10, 17, 30, 50} and df in {2, 3, 5, 10, 30, 100,
      1000, 1e6, inf}. `<scratch>` is a directory outside the repository
- [ ] Get R's values and warnings on the same grid (`qtukey(p, nmeans, df)` in base R), saved to `<scratch>/r.json`
- [ ] In `src/metafopy/core/tukey.py::qtukey`, do the secant division with C's rules: a zero denominator gives `+-inf`, or NaN for `0/0`, as
      `numpy.float64` division does with errors silenced. Carry the iteration on as C does. After the last iteration without convergence,
      warn "convergence failed in 'qtukey'" as R's `ML_WARNING(ME_NOCONV)` does. Warn "NaNs produced" when the result is NaN from finite,
      in-domain input. Use `warnings.warn(..., stacklevel=2)` as `src/metafopy/methods/vcalc.py` does
- [ ] Compare the new grid with both files. Every point that did not raise gives the same value bit for bit. Every point that raised gives
      what R gives: NaN with R's warnings. A point that is neither stops the task as `blocked`, naming it and the three values
- [ ] In `src/tests/test_tukey.py`, widen the property test's `nmeans` from 2-10 to 2-50 and drop the comment that narrowed it. Let the
      property allow NaN where R returns NaN. Add a direct test of `qtukey(0.25, 17, math.inf)`: it gives NaN with both warnings
      (`pytest.warns`)
- [ ] Find a Tukey family of 17 or more means that reaches `src/metafopy/methods/emm_adjust.py:231` (or `:125`) with a small level. Compare
      its result with R's emmeans on the same input (`emmeans` is in the Rlib). Add a test of that case to `src/tests/test_emmeans_inference.py`
      that asserts R's output. If metafopy then differs from R, stop as `blocked` with both outputs

**Acceptance Criteria:**

- [ ] No grid point raises, and the non-raising points are unchanged bit for bit
- [ ] The raising points give R's NaN and warnings, and the emmeans case matches R

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `mypy src/metafopy/core/tukey.py src/tests/test_tukey.py src/tests/test_emmeans_inference.py` passes
- [ ] `pytest src/tests/test_tukey.py src/tests/test_emmeans_inference.py` passes

**Notes:**

- R's source is `src/nmath/qtukey.c` (R 4.x). Port its loop line for line; the only change is that the division no longer raises
- The item notes that the port's `return ans` after 50 iterations was never compared with R; the grid comparison covers it

---

## Task 2: Full check and closing

**Status:** todo
**Priority:** high
**Complexity:** small
**Source Reference:** `TODO.md`, "`qtukey` raises ZeroDivisionError where R returns NaN"

**Technical Goal:** the branch passes every check once, and the `TODO.md` item moves to `DONE.md`.

**Subtasks:**

- [ ] Move the `TODO.md` item "`qtukey` raises ZeroDivisionError where R returns NaN" to the end of `DONE.md`: append its heading and body unchanged, then the line
      `Done: <date> - agents/tasks/qtukey-nan/tasks-1.md Task 1 (<commit>); approved in agents/reviews/order-r-parity-fixes-from-property-tests-specs.md item 1.`, and remove it from `TODO.md`
- [ ] Run the final tests from the worktree and save their whole output for the merge gate. The main checkout is the first entry of
      `git worktree list --porcelain`, `/Users/simonb/src/metafopy`:
      `mkdir -p /Users/simonb/src/metafopy/ralpher-logs/qtukey-nan/tasks-1 && export PATH=/Users/simonb/src/metafopy/.venv/bin:$PATH PYTHONPATH=$PWD/src && pytest src/tests 2>&1 | tee /Users/simonb/src/metafopy/ralpher-logs/qtukey-nan/tasks-1/final-tests.log`

**Acceptance Criteria:**

- [ ] The item is in `DONE.md` with its body unchanged, and the final run passes

**Definition of Done:**

- [ ] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [ ] `pre-commit run --all-files` passes
- [ ] `pre-commit run auditme-all --hook-stage manual --all-files` passes
- [ ] `mypy` passes
- [ ] The final test run above passes
- [ ] `git show $(git log --diff-filter=A --format=%H -- agents/tasks/qtukey-nan/tasks-1.md):TODO.md > <scratch>/todo-before.md` and
      `python3 ~/.claude/skills/todo-review/verify_bodies.py <scratch>/todo-before.md TODO.md DONE.md` exits 0

**Notes:**

- Depends on Task 1
- The full run can outlast a ten-minute command limit; run it in the background with the longest limit the session allows, and wait for its
  exit status before going on
