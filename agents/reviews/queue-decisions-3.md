# Review: queue decisions of the TODO items 1-10 order

**Date:** 2026-11-18
**Order:** `agents/tasks/order-todo-items-1-10.md`
**Review page:** https://claude.ai/artifact/8HQYRppXe7kpTsJyVnAr4E

## What is reviewed and how

While `/rro agents/tasks/order-todo-items-1-10.md` ran rows 1 to 3 (2026-11-17 to 2026-11-18), one task-list stop that was not reserved for the
operator was decided by Claude and committed with a `Queue-decision:` trailer (`git log --grep '^Queue-decision:'`). Approve or flag it. A flag means
the decision is revisited in a follow-up; the list is already merged. The stops reserved for the operator (row 2 Task 2, the two log-likelihood
changes; row 2 Task 5, Docker not running) were answered by the operator on 2026-11-17 and are not reviewed here.

## Automatic checks

- `bin/rro.sh decisions agents/tasks/order-todo-items-1-10.md --since 4fc49fd` lists exactly this decision: pass.
- Row 3 passed its merge gate (test log: 8594 passed, 37 skipped, 14 xfailed) and is merged as `89709b4`; the checkpoint after row 3 passed on
  `main` (8594 passed, 37 skipped, 14 xfailed, 2026-11-18): pass.

## The items

### 1. Clear the audit stop, the findings predate the list (`agents/tasks/selmodel-fixed-delta0/tasks-1.md` Task 3, commit `93a96a4`)

- **The stop:** the whole audit run for the fixed first selmodel step weight (`.audit/runs/20261117-233604`) flagged no `selmodel` case with a first
  `delta` other than 1, so no `known.toml` entry was needed. It reported seven findings that the newest earlier report (`20261117-215038`) did not
  hold, and the task said any such finding stops it.
- **The decision:** clear the stop and go on with the recall check. All seven are in the `new` list of the full audit of 2026-11-15
  (`.audit/runs/20261115-160207/findings.json`): the selmodel logistic Hessian element, the flat location-scale print differences and `ls_na` with
  `~ ablat * alloc`, filed as `TODO.md` items that row 6 (`audit-ledger-gaps`) covers. `20261117-215038` was a partial run, so it could not hold them.
  The recall check then ran in `d553eb9`.
- **Recommendation:** approve - the findings are older than the list and already filed; the stop came from comparing against a partial run.

## Decisions

None beyond the items: each item's approve or flag is the decision.

## Answers

(Empty until the review page is answered.)
