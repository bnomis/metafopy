# Tasks: rma_uni zero QE

**Source:** TODO.md, "rma_uni raises ZeroDivisionError for an equal-effects fit when every effect is identical"
**Created:** 2026-11-14
**Version:** 1

Checked against the code on 2026-11-14 (metafopy at `1bd2712`) and against R metafor 5.2-1 in `.audit/Rlib`. With `yi = [0, 0, 0, 0]` and
`vi = [0.1, 0.2, 0.3, 0.4]`:

| Call | metafopy | R metafor 5.2-1 |
|------|----------|-----------------|
| `rma_uni(..., method='EE')` (also `FE`, `CE`) | `ZeroDivisionError` at `I2 = max(0.0, 100 * (QE - (k - p)) / QE)`, `src/metafopy/models/rma_uni.py` | `I2 = 0`, `H2 = 0`, `QE = 0` |
| `rma_uni(..., method='REML')` | `I2 = 0`, `H2 = 1` (the random-effects branch divides by `vt + tau2`, not by `QE`) | same |
| `rma_mh(ai=[1,2,3], n1i=[10,20,30], ci=[1,2,3], n2i=[10,20,30], measure='OR')` | `I2 = nan` (`if qe != 0 else NAN`, `src/metafopy/models/rma_mh.py`) | `I2 = 0` |
| `rma_peto(...)` on the same tables | `I2 = nan` (`if qe != 0 else math.nan`, `src/metafopy/models/rma_peto.py`) | `I2 = 0` |
| `gosh()` of an `EE` fit to `yi = [0, 0, 0, 1]` | `ZeroDivisionError` in `closed_form`, `src/metafopy/methods/gosh.py` (`I2 = max(0.0, 100 * (Q - (k - 1)) / Q)` on a subset whose effects are equal) | not run; R evaluates the same line in floating point, where it is 0 |

R evaluates `max(0, 100 * (QE - df) / QE)` in floating point: with `QE = 0` and `df > 0` the quotient is `-Inf`, so `I2` is 0. That is also the
reference: Higgins and Thompson (2002) truncate `I2 = (Q - df) / Q` at 0 whenever `Q < df`. So the fix matches both oracles and needs no review.
`H2 = QE / df` stays (0 here, as in R). The project has no affected-tests tool, so each task names its test files, and the last task runs every test
file that imports a changed module, and `src/tests/correctness`.

---

## Task 1: I2 is 0 when QE is 0, in every heterogeneity block

**Status:** done
**Priority:** high
**Complexity:** small
**Source Reference:** the item's fix direction

**Technical Goal:** no fit raises or returns `nan` for `I2` because its Q statistic is 0. Instead it gives 0, as R metafor and Higgins and Thompson
(2002) do.

**Subtasks:**

- [x] Re-check the starting point: run the calls in the header table and stop as `blocked`, naming the call, if any of them no longer behaves as
      the table says
- [x] Add one helper to `src/metafopy/models/tau2.py`, which `rma_uni` and `gosh` already import and which `rma_mh` and `rma_peto` load through
      `metafopy.models.rma_uni` (so importing it there adds no cycle): `i2_from_q(q, df)` returns `max(0.0, 100 * (q - df) / q)`, and 0 when
      `q == 0`. Its docstring cites Higgins and Thompson (2002) and the `I2` lines of metafor's `rma.uni`, `rma.mh` and `rma.peto`
- [x] Use it in the equal-effects branch of the heterogeneity block of `src/metafopy/models/rma_uni.py`, in `src/metafopy/models/rma_mh.py`, in
      `src/metafopy/models/rma_peto.py` and in `closed_form` of `src/metafopy/methods/gosh.py`. Keep `H2 = Q / df` unchanged
- [x] Check that nothing else divides by a Q statistic: `grep -rnE '/ *(QE|qe|Q)\b' src/metafopy`. Fix any further hit the same way, and name it
      in the commit body
- [x] Regression tests in `src/tests/test_rma_uni_extras.py`: identical effects for each of `EE`, `FE` and `CE` give `I2 == 0`, `H2 == 0`,
      `QE == 0`, and `REML` gives `I2 == 0`, `H2 == 1`
- [x] Regression tests in `src/tests/test_rma_mh_peto.py`: the tables in the header give `I2 == 0` for `rma_mh` (`OR`, `RR`, `RD`) and for
      `rma_peto`
- [x] Regression test for `gosh` in `src/tests/test_diagnostics.py`: an `EE` fit to
      `yi = [0, 0, 0, 1]`, `vi = [0.1, 0.2, 0.3, 0.4]` runs, and every subset whose effects are equal has `I2 == 0`

**Acceptance Criteria:**

- [x] Every call in the header table returns `I2 = 0`, matching R metafor 5.2-1
- [x] No fitting function under `src/metafopy` divides by a Q statistic without the helper

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `mypy` passes
- [x] `pytest src/tests/test_rma_uni_extras.py src/tests/test_rma_uni_help.py src/tests/test_rma_uni_parity.py src/tests/test_rma_mh_peto.py src/tests/test_diagnostics.py` passes

**Notes:**

- Key files: `src/metafopy/models/rma_uni.py`, `src/metafopy/models/rma_mh.py`, `src/metafopy/models/rma_peto.py`,
  `src/metafopy/methods/gosh.py`, `src/metafopy/models/tau2.py`
- `rma_mh` and `rma_peto` returned `nan` where R returns 0: that is a parity fix as well as the crash fix, and no R parity fixture held the `nan`
  (`grep -rn 'I2.*nan' src/tests/*.py` found none on 2026-11-14)
- No difference from metafor is introduced, so `docs/differences.md` and `src/audit/data/known.toml` do not change

---

## Task 2: A correctness case, the full test run and closing the item

**Status:** done
**Priority:** high
**Complexity:** small
**Source Reference:** the item as a whole

**Technical Goal:** the correctness harness holds the zero-Q case, the whole branch is checked once, and the item moves to `DONE.md`.

**Subtasks:**

- [x] Add a `property` case to `src/tests/correctness/cases/properties.toml`, beside the `I2 = 100 (H2 - 1) / H2` case: for each estimator in `metafopy.models.tau2.ESTIMATORS` that runs on identical effects, `I2` is 0. Read the estimators from
      `ESTIMATORS` at test time, as the other property cases do
- [x] Move the item from `TODO.md` to `DONE.md`: append its heading and body unchanged at the end of `DONE.md`, then a line
      `Done: 2026-MM-DD, agents/tasks/rma-uni-zero-q/tasks-1.md` with the date of the commit
- [x] Run the final tests from the worktree and save their output for the merge gate:
      `main=$(git worktree list --porcelain | sed -n '1s/^worktree //p'); mkdir -p "$main/ralpher-logs/rma-uni-zero-q/tasks-1" && pytest src/tests/test_rma_uni_extras.py src/tests/test_rma_uni_help.py src/tests/test_rma_uni_parity.py src/tests/test_rma_mh_peto.py src/tests/test_audit13_fixes.py src/tests/test_audit4_predict.py src/tests/test_audit8_values_a.py src/tests/test_audit8_refusals.py src/tests/test_audit9_fixes.py src/tests/test_fits.py src/tests/test_hettest.py src/tests/test_diagnostics.py src/tests/test_audit3_misc.py src/tests/test_audit8_crashes.py src/tests/test_audit8_args.py src/tests/test_audit8_fields.py src/tests/correctness 2>&1 | tee "$main/ralpher-logs/rma-uni-zero-q/tasks-1/final-tests.log"`

**Acceptance Criteria:**

- [x] The property case passes for every estimator it runs over
- [x] The item is in `DONE.md` with its body unchanged, and no longer in `TODO.md`

**Definition of Done:**

- [x] The task's commit passes its pre-commit hooks (never `--no-verify`)
- [x] `pre-commit run --all-files` passes
- [x] `pre-commit run auditme-all --hook-stage manual --all-files` passes
- [x] `mypy` passes
- [x] The final test run above passes, `slow` tests included
- [x] `git show $(git log --diff-filter=A --format=%H -- agents/tasks/rma-uni-zero-q/tasks-1.md):TODO.md > <scratch>/todo-before.md` and
      `python3 ~/.claude/skills/todo-review/verify_bodies.py <scratch>/todo-before.md TODO.md DONE.md` exits 0 (`<scratch>` is the session's scratch
      directory, outside the repository)

**Notes:**

- Depends on Task 1
- The test files in the final run are, for each module Task 1 changed, its own tests and the test files that import it directly
  (`grep -l 'metafopy\.models\.rma_uni\b' src/tests/*.py`), the files that test `gosh` without its plot (`grep -ln 'gosh' src/tests/test_*.py`,
  less the plot and image tests, which draw and do not reach `closed_form`'s values), and `src/tests/correctness`. Checked on 2026-11-14; if a
  file has been added or renamed since, run the greps again and extend the command
- `src/metafopy/models/tau2.py` has no test file of its own; its estimators are tested through `rma_uni` in the files above

---
