# Order: TODO items 1-10

Written on 2026-11-17 by `/todo-run-order 1-10`. Run it from the main checkout, on `main`, clean and level with `origin/main`:

```text
/rro agents/tasks/order-todo-items-1-10.md
```

It covers these `TODO.md` items:

- "The Docs workflow fails on three docstring links to docs/differences.md" (row 1);
- "The rma_mv REML fit stops short of the optimum on Linux and Windows" (row 2), through `agents/specs/rma-mv-reml-stop/spec.md`;
- "A fixed delta[0] other than 1 on a step-function selection model makes the likelihood a constant multiple" (row 3), through
  `agents/specs/selmodel-fixed-delta0/spec.md`;
- "Simmonds and Higgins (2016) Fig 1 one-stage random effects differs from metafopy and R metafor" (row 4);
- "Stijnen et al. (2010) Table VII treated arm, PN, differs from metafopy and R metafor" (row 5);
- "The audit lists the exact permutest of uni_mods as an R kill and has no entry for it", "The audit lists a selmodel logistic Hessian element that
  is zero in both R and metafopy", "The audit lists printed fits of flat location-scale models that differ in the last digit" and "The audit has no
  entry for ls_na with an interaction formula that R refuses" (row 6);
- "ls.confint.scale.ML.bcg stays an expected failure with no decision" (row 7).

The plan and both specs were reviewed in `agents/reviews/order-todo-items-1-10-specs.md` and answered on 2026-11-17 (every recommendation accepted):
both specs are at version 1.1, a fixed first step weight other than 1 is rejected with `InputError` (D1), and `ls_na` with `~ ablat * alloc` is
accepted as a difference (D2). After the review, the location-scale interval list moved from fourth to last: its probe (2026-11-17) found that
metafopy's own profile leaves the highest branch, so it is expected to stop for you, and placed last it holds up no other row.

## Prerequisites

- **R metafor 5.2-1 in `/Users/simonb/src/metafopy/.audit/Rlib`**: every row but row 1 compares with R; worktrees have no `.audit/Rlib`, so the lists
  point `METAFOPY_AUDIT_RLIB` at the main checkout's.
- **Docker running** (Docker Desktop): row 2 runs the test suite in the Linux container (`bin/linux-tests.sh`), because this Mac's numpy and scipy use
  Apple's Accelerate while CI's Linux wheels use OpenBLAS; its Task 5 stops as blocked if `docker info` fails.
- **Network access**: row 1 installs the docs requirements and builds the site in strict mode, which fetches the Python, NumPy and pandas inventories.

## Stops for you

Any change to metafopy's output beyond what an approved spec lists waits for you (`claude/rules/metafopy.md`).

- **Row 7, Task 2, expected.** The list fixes the reference refit of the test, then stops: metafopy's profile of `alpha.2` jumps to the `tau2 = 0`
  plateau between a = 1.36 and 1.38, so its upper bound 1.3598 is that jump rather than the crossing (about 1.381). Fixing it changes the output of
  `confint` and `profile`. You decide whether that fix joins the list or becomes its own `TODO.md` item.
- **Rows 2 to 6, only if the run finds metafopy wrong**, as each row's cell says; the probes made while writing the lists (2026-11-17) expect none of
  them to fire.

## Settings

| Setting | Value |
|---|---|
| checkpoint command | `source bin/activate.sh && pytest src/tests` |
| checkpoint every | 3 |
| test log | `ralpher-logs/{spec}/tasks-{index}/final-tests.log` |
| test pass line | `passed` |
| test fail pattern | `FAILED\|ERROR` |
| always wait for | `claude/rules/metafopy.md` |
| progress page | https://claude.ai/artifact/YG6Q6y25VT3wZmEcxzsQUo |

| # | Task list | Waits for | Stops for you | Status |
|---|---|---|---|---|
| 1 | `docs-strict-links/tasks-1.md` | - | - | merged 0b55252 |
| 2 | `rma-mv-reml-stop/tasks-1.md` | - | an output change beyond the spec's measured bounds or a failing parity test (Task 2; Task 5 for the Linux container), or a new audit finding on the `rma_mv` fixtures (Task 5), only if one occurs | merged 4ac74d4 |
| 3 | `selmodel-fixed-delta0/tasks-1.md` | 1 | an existing test or case that reaches the new error (Task 1), only if one does | merged 89709b4 |
| 4 | `simmonds2016-fig1/tasks-1.md` | - | the marginal likelihood not higher at tau2 0.213 than at 0 (Task 1), only if it is not | merged 3a6f89e |
| 5 | `stijnen2010-pn/tasks-1.md` | 4 | metafopy differing from R or the reference formulas on the PN fit (Task 1), only if it does | merged 2df6689 |
| 6 | `audit-ledger-gaps/tasks-1.md` | 2, 3 | a finding of the full audit that no item covers and that shows metafopy's output wrong (Task 1), only if one does | merged 5e16b4d |
| 7 | `ls-confint-scale-branch/tasks-1.md` | - | the fix of metafopy's own profile of `alpha.2`, expected (Task 2) | running |

Checkpoint after 3: passed 8594 passed, 37 skipped, 14 xfailed
