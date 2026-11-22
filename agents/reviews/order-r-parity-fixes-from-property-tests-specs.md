# Review: the R parity fixes from property tests run order

**Review page:** https://claude.ai/artifact/Bg2GyR614GDTA8uqdpLEtc
**Date:** 2026-11-22

**What is being reviewed and how.** `/todo-run-order 1-3` takes the three defects that the property tests found on 2026-11-22 (`TODO.md`
items 1 to 3) to a run order of three task lists. Each item is small, one function and its test, so it gets a task list and no spec. Each
fix changes an output, and `claude/rules/metafopy.md` changes metafopy's output only after the user confirms it. Approving a fix confirms
that output change before any list is written. Flag a fix to stop it.

## Automatic checks

<!-- doc-counts: accept-start measured on 2026-11-22 - a dated record, not a claim about the present -->
- Pass: `main` is clean and level with `origin/main` at `ea53581`.
- Pass: each item is still open at `ea53581`:
  - the secant step in `src/metafopy/core/tukey.py` still divides by `valx1 - valx0` unguarded;
  - `weibull_shape_start` still minimises with `lbfgsb`;
  - `make_unique` in `src/metafopy/metadata.py` still checks only the labels it has emitted.
- Pass: no two items share a cause or a file, so the lists can run in any order. `/rro` merges each list before the next starts, so their
  `TODO.md` and `DONE.md` edits do not conflict.
- Pass: R metafor 5.2-1 and emmeans are installed in `/Users/simonb/src/metafopy/.audit/Rlib`, the only prerequisite.
<!-- doc-counts: accept-end -->

## The items

### 1. `qtukey` returns NaN where R does, instead of raising

- **What changes:** the secant step divides with C's floating-point rules: a zero denominator gives an infinity or NaN instead of
  `ZeroDivisionError`. Running out of iterations warns as R does ("convergence failed in 'qtukey'"), and a NaN result from finite input
  warns "NaNs produced". The property test of `qtukey` widens from 2 to 10 means to 2 to 50.
- **Output change:** `qtukey` returns NaN, with R's warnings, for the combinations that raise today. On 2026-11-22 that was 53 of 504
  grid points, each with 17 or more means. Every other value stays the same: the task checks that bit for bit over the grid, and compares
  the NaN cases with R. The Tukey adjustment in `methods/emm_adjust.py` then gets R's NaN instead of an error, and the task compares one such
  family with R's emmeans.
- **Recommendation:** approve. It is a line-for-line port of R's C code.

### 2. `weibull_shape_start` reaches the shape R's `nlminb` reaches

- **What changes:** the one-parameter L-BFGS-B fit is replaced by a bracketed root find (`scipy.optimize.brentq`) of the Weibull
  coefficient of variation on `[1e-3, upper]`, where `upper` doubles until the root is bracketed. The squared-error objective that R's
  `nlminb` minimises has its minimum, 0, at that root, and a positive coefficient of variation always has one. The strict `xfail` comes off.
- **Output change:** start shapes above about 4 move to R's value, for example 8 where metafopy gives 7.9419 today. Below 4 they move by
  less than 1e-5. The caller (`fivenum_estimators.py:348`) uses the shape only as a start for a later fit. The task runs every fivenum parity
  test, and a fixture that moves past its tolerance stops the task.
- **Recommendation:** approve. The item's Fix direction proposed the root find, and it gives R's root to ten digits.

### 3. `make_unique` keeps every first occurrence, as R's `make.unique` does

- **What changes:** the set of labels to avoid starts with every input label, so a generated suffix never takes a label the caller gave.
  This is R's own algorithm in `do_makeunique`. The strict `xfail` comes off.
- **Output change:** labels change only when an input label equals a suffix that `make_unique` generates. For `a, a, a.1`, metafopy gives
  `a, a.2, a.1` (R's result) instead of `a, a.1, a.1.1`. That reaches the study labels (`slab`) of a fit and
  `effects.datatools.make_names(unique=True)`. The task compares both with R's `make.unique` and `make.names(unique = TRUE)` on the same
  inputs.
- **Recommendation:** approve. It is R's documented behaviour (`?make.unique`).

## Decisions

None: each fix follows R's code, which settles how it works.

## Answers

Answered 2026-11-22 on the review page (https://claude.ai/artifact/Bg2GyR614GDTA8uqdpLEtc), each with no comment:

- **1. `qtukey` returns NaN where R does:** approved. This confirms its output change.
- **2. `weibull_shape_start` reaches R's shape:** approved. This confirms its output change.
- **3. `make_unique` keeps every first occurrence:** approved. This confirms its output change.
