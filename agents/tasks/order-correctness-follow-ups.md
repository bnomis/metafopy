# Order: correctness follow-ups

Written on 2026-11-14 by `/todo-run-order 1-4`. Run it from the main checkout, on `main`, clean and level with `origin/main`:

```text
/rro agents/tasks/order-correctness-follow-ups.md
```

It covers these `TODO.md` items, in this order:

- "rma_uni raises ZeroDivisionError for an equal-effects fit when every effect is identical" (row 1);
- "confint type "ht" of a REML fit centres on the fit's H2, not on H from Q" (row 2);
- "The correctness harness cannot check exp(beta), prediction intervals or test-based tau2 and I2 intervals" (row 3);
- "QE and the moment tau2 estimators centre yi in models without an intercept" (row 4), filed by the user's answer to D1 of the specs review and
  widened by the user on 2026-11-14 to the ML, REML and EB estimators;
- "Extend the correctness tests beyond the core models" (rows 5 to 8), one spec per area.

The four specs were reviewed in `agents/reviews/order-correctness-follow-ups-specs.md` and answered on 2026-11-13; each is at version 1.1.

## Prerequisites

- **Papers** in `/Users/simonb/src/metafopy papers`, outside the repository, read with `pdftotext`. `download_report.md` there lists them. As of
  2026-11-14 every source is present except Citkowicz and Vevea (2017), *Psychological Methods* 22:28-41, doi:10.1037/met0000119, which row 8 Task 8
  stops for. Saving its PDF there before row 8 starts lets that task pass straight through.
- **R libraries:** the audit's pinned library `/Users/simonb/src/metafopy/.audit/Rlib` (metafor 5.2-1, mvmeta, robumeta, sandwich, glmmTMB; nlme from
  R's own library), and `METAFOPY_CORRECTNESS_RLIB=/Users/simonb/.local/share/metafopy/r-correctness-lib` set in the queue's session, holding R
  `meta` 8.5-0, `glmmML` 1.1.7 and `weightr` 2.0.2. `src/audit/data/r-versions.json` must not change.
- **Network** for the open-access sources rows 5, 7 and 8 fetch (PMC articles, software documentation).

## Stops for you

Rows 1, 2 and 4 have none. The source stops (row 5 Task 10, row 6 Task 10, row 7 Task 2, row 8 Task 8) pass through when every source is present,
and row 8 Task 11 passes through when R `weightr` and `meta` load. Each mismatch review (row 3 Task 4, row 5 Task 12, row 6 Task 13, row 7 Task 11,
row 8 Task 13) stops only when its list recorded a mismatch, a convention or a source still needed, because `claude/rules/metafopy.md` changes
metafopy's output only after you confirm each disagreement. An operator session copies a page's answers into its review document before the list
restarts.

| Setting | Value |
|---|---|
| checkpoint command | `source bin/activate.sh && pytest src/tests` |
| checkpoint every | 3 |
| test log | `ralpher-logs/{spec}/tasks-{index}/final-tests.log` |
| test pass line | `passed` |
| test fail pattern | `FAILED\|ERROR` |
| always wait for | `claude/rules/metafopy.md` |
| progress page | https://claude.ai/artifact/JXkwFEMSuvsbQkhv1trWaS |

| # | Task list | Waits for | Stops for you | Status |
|---|---|---|---|---|
| 1 | `rma-uni-zero-q/tasks-1.md` | - | - | merged d58e1a3 |
| 2 | `confint-ht-centre/tasks-1.md` | - | - | merged 04e6ef5 |
| 3 | `correctness-harness-checks/tasks-1.md` | 2 | the mismatch review, only if there is a mismatch (Task 4) | waiting: the mismatch review (Task 4) |
| 4 | `qe-no-intercept/tasks-1.md` | 2 | - | todo |
| 5 | `correctness-mv-robust/tasks-1.md` | 3, 4 | sources still missing (Task 10); the mismatch review (Task 12) | todo |
| 6 | `correctness-glmm/tasks-1.md` | 3, 5 | sources still missing (Task 10); the mismatch and conventions review (Task 13) | todo |
| 7 | `correctness-location-scale/tasks-1.md` | 3, 4 | sources still missing (Task 2); the mismatch review (Task 11) | todo |
| 8 | `correctness-publication-bias/tasks-1.md` | 3, 5, 6, 7 | the Citkowicz and Vevea (2017) PDF (Task 8); R `weightr` and `meta` if they do not load (Task 11); the review of mismatches, `negexppow` and `stepcon` (Task 13) | todo |
