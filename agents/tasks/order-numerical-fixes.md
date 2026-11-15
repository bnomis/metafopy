# Order: numerical fixes

Written on 2026-11-15 by `/todo-run-order 1-8`. Run it from the main checkout, on `main`, clean and level with `origin/main`:

```text
/rro agents/tasks/order-numerical-fixes.md
```

It covers these `TODO.md` items, in this order:

- "No audit entries record the no-intercept QE and tau2 difference" (row 1);
- "The fail-safe N root search rounds up one too far on a step function" (row 2);
- "The truncest estimate stops at a local maximum of a step-function likelihood" (row 3);
- "The CM.EL information is taken by finite differences of the exact likelihood" (row 4), through `agents/specs/cm-el-information/spec.md`;
- "The location-scale identity-link fit stops about 1e-5 short of the maximum" (row 5);
- "The location-scale profile-likelihood intervals of the scale coefficients are not crossings where the profile is flat" (row 6);
- "A precision measure on a step-function selection model makes the likelihood not a density" (row 7), through
  `agents/specs/selmodel-prec-stepfun/spec.md`;
- "Whether R metafor's SJIT converges on the scaled scale-invariance data was never checked" (row 8).

The two specs were reviewed in `agents/reviews/order-numerical-fixes-specs.md` and answered on 2026-11-15 (every recommendation accepted); each is
at version 1.1. Items 2, 3, 5 and 6 were answered *follow the reference* on 2026-11-15, so their output changes need no further confirmation.

## Prerequisites

- **R libraries:** the audit's pinned library `/Users/simonb/src/metafopy/.audit/Rlib` (metafor 5.2-1; checked loading on 2026-11-15). Worktrees
  have no `.audit/Rlib`, so every list sets `METAFOPY_AUDIT_RLIB` to that path for audit runs and R calls. `src/audit/data/r-versions.json` must not
  change.
- Nothing else: no paper, account or network source is needed.

## Stops for you

Only row 7 can stop: its Task 1, which adds `halfnorm` cases with each precision measure, stops if one of them fails, because a failing new
case is reported before it is set to `mismatch`. A probe of those cases passed on 2026-11-15. Row 3's Task 1 stops for the decision on how
`truncest` reports a truncation point whose likelihood has no maximum; it was answered on 2026-11-15 (option B, in
`agents/tasks/truncest-piecewise-search/review-1.md`), so that task passes straight through. Every other row runs straight through.

| Setting | Value |
|---|---|
| checkpoint command | `source bin/activate.sh && pytest src/tests` |
| checkpoint every | 3 |
| test log | `ralpher-logs/{spec}/tasks-{index}/final-tests.log` |
| test pass line | `passed` |
| test fail pattern | `FAILED\|ERROR` |
| always wait for | `claude/rules/metafopy.md` |
| progress page | |

| # | Task list | Waits for | Stops for you | Status |
|---|---|---|---|---|
| 1 | `audit-no-intercept/tasks-1.md` | - | - | todo |
| 2 | `fsn-integer-search/tasks-1.md` | - | - | todo |
| 3 | `truncest-piecewise-search/tasks-1.md` | - | the truncation-point decision, answered on 2026-11-15, so a no-op (Task 1) | todo |
| 4 | `cm-el-information/tasks-1.md` | - | - | todo |
| 5 | `ls-identity-polish/tasks-1.md` | - | - | todo |
| 6 | `ls-profile-crossings/tasks-1.md` | 5 | - | todo |
| 7 | `selmodel-prec-stepfun/tasks-1.md` | 3 | a new halfnorm precision case fails, only if one does (Task 1) | todo |
| 8 | `sjit-scaled-convergence/tasks-1.md` | - | - | todo |
