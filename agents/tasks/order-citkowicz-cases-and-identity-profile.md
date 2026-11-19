# Order: Citkowicz cases and identity profile

Written on 2026-11-19 by `/todo-run-order 1,4`. Run it from the main checkout, on `main`, clean and level with `origin/main`:

```text
/rro agents/tasks/order-citkowicz-cases-and-identity-profile.md
```

It covers these `TODO.md` items:

- "Add the printed Baskerville estimates of Citkowicz and Vevea (2017) as published cases" (row 1);
- "The reference ls_profile_ll cannot fit the identity link at some feasible values" (row 2).

Both are small items with a task list each and no spec, so there was no specs review. The user chose on 2026-11-19 how row 1 records the article's
values that metafopy does not reproduce: as strict expected failures marked "not yet reviewed", for the user to judge later. Neither row changes
metafopy's output; the two rows edit different files.

## Prerequisites

- **R metafor 5.2-1 in `/Users/simonb/src/metafopy/.audit/Rlib`**: row 1 compares with R; worktrees have no `.audit/Rlib`, so the list points
  `METAFOPY_AUDIT_RLIB` at the main checkout's.
- **The Citkowicz and Vevea (2017) PDF in `/Users/simonb/src/metafopy papers/`** (already there, added by the user on 2026-11-19): row 1 checks the
  printed values of its "Example" section.

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
| progress page | https://claude.ai/artifact/Rka9m1cDRsTA4NtKhjUxx7 |

| # | Task list | Waits for | Stops for you | Status |
|---|---|---|---|---|
| 1 | `citkowicz2017-cases/tasks-1.md` | - | a printed value not reproducing as the 2026-11-19 probe found (Task 1), only if one does not | merged 253d45e |
| 2 | `ls-profile-identity-start/tasks-1.md` | - | an existing reference value moving (Task 1), or metafopy's identity-link bounds differing from the reference (Task 2), only if one does | waiting: metafopy's identity-link bounds differing from the reference (Task 2) |
