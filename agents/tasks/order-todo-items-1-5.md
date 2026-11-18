# Order: TODO items 1-5

Written on 2026-11-18 by `/todo-run-order 1-5`. Run it from the main checkout, on `main`, clean and level with `origin/main`:

```text
/rro agents/tasks/order-todo-items-1-5.md
```

It covers these `TODO.md` items:

- "metafopy's profile of the ls scale coefficient leaves the highest branch before the crossing" (row 1), through
  `agents/specs/ls-profile-starts/spec.md`;
- "The exact permutest of uni_mods neither refuses nor finishes in metafopy" (row 2), through `agents/specs/permutest-exact-limit/spec.md`;
- "The rma_mv funnel test still skips when rstandard is missing" (row 3);
- "Find the data behind Table 5 of Bagos and Nikolopoulos (2009)" (row 4);
- "Six ledger entries of known.toml match nothing in the audit" (row 5).

The plan and both specs were reviewed in `agents/reviews/order-todo-items-1-5-specs.md` and answered on 2026-11-18 (every recommendation accepted):
both specs are at version 1.1 and approved, with the output changes `ls-profile-starts` lists; an exact `permutest` counts R's permutation matrix
in its 16 GiB guard (D1); the Bagos item is narrowed to Hooper, which becomes its own `TODO.md` item (D2). The stale-entry list runs last, so its
full audit sees the output changes of rows 1 and 2.

## Prerequisites

- **R metafor 5.2-1 in `/Users/simonb/src/metafopy/.audit/Rlib`**: rows 1, 2, 4 and 5 compare with R; worktrees have no `.audit/Rlib`, so the lists
  point `METAFOPY_AUDIT_RLIB` at the main checkout's.
- **The Bagos and Nikolopoulos (2009) PDF in `/Users/simonb/src/metafopy papers/`** (already there): row 4 checks its Appendix III and Table 5.
- **Network access**: row 4 fetches Guevara, Berlin and Wolf (2004), PMC481068, from Europe PMC to check the school-absence counts.

## Stops for you

Any change to metafopy's output beyond what an approved spec lists waits for you (`claude/rules/metafopy.md`). No stop is expected; each row's cell
names the one that fires only if the run finds it.

## Settings

| Setting | Value |
|---|---|
| checkpoint command | `source bin/activate.sh && pytest src/tests` |
| checkpoint every | 3 |
| test log | `ralpher-logs/{spec}/tasks-{index}/final-tests.log` |
| test pass line | `passed` |
| test fail pattern | `FAILED\|ERROR` |
| always wait for | `claude/rules/metafopy.md` |
| progress page | https://claude.ai/artifact/9DgK34ysrgjKaas9xT4ztj |

| # | Task list | Waits for | Stops for you | Status |
|---|---|---|---|---|
| 1 | `ls-profile-starts/tasks-1.md` | - | an output that moves beyond the spec's list (Tasks 1 and 3), only if one does | merged 45bbfc3 |
| 2 | `permutest-exact-limit/tasks-1.md` | - | a test or audit case besides `uni_mods` and `ls_bcg` with `exact=True` refused (Tasks 1 and 3), only if one is | merged da2aa13 |
| 3 | `mv-funnel-skip/tasks-1.md` | - | the `funnel_mvm` case failing once the skip is removed (Task 1), only if it does | merged 87cd4df |
| 4 | `bagos2009-table5/tasks-1.md` | - | a printed row of Table 5 not reproducing as the probe found (Task 1), only if one does not | merged 7253d10 |
| 5 | `audit-stale-entries/tasks-1.md` | 1, 2 | an entry whose case shows metafopy's output wrong (Task 1), or a new full-audit finding no row caused or expects (Task 3), only if one occurs | merged 22518aa |

Checkpoint after 5: passed 8608 passed, 37 skipped, 14 xfailed
