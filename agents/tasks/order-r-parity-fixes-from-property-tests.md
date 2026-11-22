# Order: R parity fixes from property tests

Written on 2026-11-22 by `/todo-run-order 1-3`. Run it from the main checkout, on `main`, clean and level with `origin/main`:

```text
/rro agents/tasks/order-r-parity-fixes-from-property-tests.md
```

It covers these `TODO.md` items, found by the property tests on 2026-11-22:

- "`qtukey` raises ZeroDivisionError where R returns NaN" (row 1);
- "`weibull_shape_start` stops at its start value for large shapes" (row 2);
- "`make_unique` renames a label that occurs once when it collides with a generated one" (row 3).

Each is a small item with one task list and no spec. Each fix changes metafopy's output. You approved each fix and its output change on
2026-11-22 in `agents/reviews/order-r-parity-fixes-from-property-tests-specs.md` (page https://claude.ai/artifact/Bg2GyR614GDTA8uqdpLEtc).
The rows edit different source files.

## Prerequisites

- **R metafor 5.2-1 and emmeans in `/Users/simonb/src/metafopy/.audit/Rlib`** (already there): every row compares with R. Worktrees have no
  `.audit/Rlib`, so the lists point `R_LIBS` at the main checkout's.

## Stops for you

Any change to metafopy's output beyond what the approved review lists waits for you (`claude/rules/metafopy.md`). No stop is expected: each
row's cell names the stop that fires only if the run finds such a change.

## Settings

| Setting | Value |
|---|---|
| checkpoint command | `source bin/activate.sh && pytest src/tests` |
| checkpoint every | 3 |
| test log | `ralpher-logs/{spec}/tasks-{index}/final-tests.log` |
| test pass line | `passed` |
| test fail pattern | `FAILED\|ERROR` |
| always wait for | `claude/rules/metafopy.md` |
| progress page | https://claude.ai/artifact/SJ48yWkK7GKEicbTZBc8av |

| # | Task list | Waits for | Stops for you | Status |
|---|---|---|---|---|
| 1 | `qtukey-nan/tasks-1.md` | - | a grid point or the emmeans case differing from R (Task 1), only if one does | merged 9a30721 |
| 2 | `weibull-shape-start/tasks-1.md` | - | a fivenum parity test changing outcome (Task 1), only if one does | running |
| 3 | `make-unique-first-occurrences/tasks-1.md` | - | a label list differing from R's (Task 1), only if one does | todo |
