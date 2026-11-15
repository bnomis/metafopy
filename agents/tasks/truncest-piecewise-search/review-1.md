# Review: how truncest reports a truncation point whose likelihood has no maximum

**Date:** 2026-11-15
**Task list:** `agents/tasks/truncest-piecewise-search/tasks-1.md` (Task 1)
**Review page:** none; the operator answered the decision in the session that wrote the run order, before the run started (see Answers)

## What is reviewed and how

While writing the task list for the `TODO.md` item "The truncest estimate stops at a local maximum of a step-function likelihood", a search of
every stretch between observed effects found that the item's reference values are a local maximum too, and that the best stretch has no
maximum, only a supremum at an observed effect. The operator decides how `selmodel(type='truncest')` reports such a truncation point.

## Automatic checks

- Measurement reproduced on 2026-11-15 at `89ec047` (metafopy, `rma_uni(method='ML')` on `dat.bangertdrowns2004`): today's fit beta 0.2649,
  tau2 0.05422, delta (1.4223, 0.2707), ll -17.7909; cut fixed at 0.48999999: beta 0.0961, tau2 0.00967, delta (0.1275, 0.49), ll -10.6519; cut
  fixed at 0.49: beta 0.1058, tau2 0.01132, delta (0.1599, 0.49), ll -12.5993: pass, as the task list's header table records.

## The items

The measurement table in the header of `agents/tasks/truncest-piecewise-search/tasks-1.md`: the likelihood rises towards the right-hand end of
the stretch (0.42, 0.49) and drops at 0.49, because a study counts as selected only when `y > cut` (metafor's `.selmodel.ll.trunc` and
`formulas.truncation_loglik`).

## Decisions

### How to report a truncation point whose supremum is not attained

- **Background:** the user answered *follow the reference* on 2026-11-15, on the item's reference values (beta 0.276, ll -17.741). Those are a
  local maximum; the supremum gives beta 0.096, ll -10.652, and is not attained.
- **What it affects:** `selmodel(type='truncest')` estimates, the case `selmodel.truncest.bangertdrowns2004`, the audit's `type="truncest"`
  findings, `docs/differences.md`.
- **Options:**
  - **B (recommended): the supremum, one step inside the stretch.** Refit with the cut fixed at `end - 1e-8 * max(1, |end|)`, report that value,
    warn that the truncation point sits at an observed effect and its likelihood has no maximum there, and leave the cut out of the Hessian.
  - **A: count a study at the cut as selected (`y >= cut`).** The supremum becomes a maximum; departs from metafor and from the reference formula.
  - **C: keep metafor's search.** No output change; the case stays a documented mismatch and the item closes as not fixed.

## Answers

- 2026-11-15: **B**, the supremum one step inside the stretch, with the warning and the cut left out of the Hessian (answered by the operator in
  the session that wrote `agents/tasks/order-numerical-fixes.md`, before the run).
