# Review: queue decisions of the TODO items 1-10 order, after the first checkpoint

**Date:** 2026-11-18
**Order:** `agents/tasks/order-todo-items-1-10.md`
**Review page:** https://claude.ai/artifact/3TECWBBoshWKAiRCsxVZGX

## What is reviewed and how

While `/rro agents/tasks/order-todo-items-1-10.md` ran rows 4 to 7 (2026-11-18), one task-list stop that was not reserved for the operator was
decided by Claude and committed with a `Queue-decision:` trailer (`git log --grep '^Queue-decision:'`). Approve or flag it. A flag means the
decision is revisited in a follow-up; the list is already merged. The stop reserved for the operator (row 7 Task 2, metafopy's own profile of
`alpha.2`) was answered by the operator on 2026-11-18 (its own `TODO.md` item) and is not reviewed here.

## Automatic checks

- `bin/rro.sh decisions agents/tasks/order-todo-items-1-10.md --since 3fef5c7` (the first checkpoint) lists exactly this decision: pass.
- Row 6 passed its merge gate (test log: 8596 passed, 37 skipped, 14 xfailed) and is merged as `5e16b4d`; the final checkpoint passed on `main`
  (8596 passed, 37 skipped, 14 xfailed, 2026-11-18): pass.

## The items

### 1. Clear the starting-point stop, the ledger entries are stale (`agents/tasks/audit-ledger-gaps/tasks-1.md` Task 1, commit `247bbb2`)

- **The stop:** before running the audit, Task 1 checked that none of the item's case ids was already in `src/audit/data/known.toml`, and two were:
  `options/vif/ls_na/table=true` and `construct/uni_mods_year/scale={"$formula":"~ 0 + I(1|ablat)"}`, both print-diff cases of the item "The audit
  lists printed fits of flat location-scale models that differ in the last digit".
- **The decision:** the item stands and the list goes on. Both entries were added on 2026-11-05 (`342e0b8`, `e23afc5`) with an exact `detail`, and
  metafopy's printed last digits have moved since (`-21.4102` is now `-21.4103`; `-142.8013 ... -4.1436` is now `-142.8012 ... -4.1435`), so the
  entries match nothing and the audits of 2026-11-15 and 2026-11-17 list both cases as new. Task 3 updated the `detail` of those two entries after its
  log-likelihood check, instead of adding second entries.
- **Recommendation:** approve - the entries could not have been meant to hide the current print, and updating them is what the item asks for.

## Decisions

None beyond the items: each item's approve or flag is the decision.

## Answers

Answered on the review page on 2026-11-18 (07:49 UTC), read back from its `answers` store; no comments:

| Item | Answer |
|---|---|
| 1. Clear the starting-point stop, the ledger entries are stale (Task 1, `247bbb2`) | approve |

The decision stands; no follow-up is needed.
