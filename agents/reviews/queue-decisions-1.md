# Review: queue decisions of the numerical fixes order

**Date:** 2026-11-16
**Order:** `agents/tasks/order-numerical-fixes.md`
**Review page:** https://claude.ai/artifact/3wUAuWQPpKydwpstUpqA9j

## What is reviewed and how

While `/rro agents/tasks/order-numerical-fixes.md` ran (2026-11-15 to 2026-11-16), two task-list stops that were not reserved for the operator were
decided by Claude, each committed with a `Queue-decision:` trailer (`git log --grep '^Queue-decision:'`). Approve or flag each. A flag means the
decision is revisited in a follow-up; both lists are already merged.

## Automatic checks

- `bin/rro.sh decisions agents/tasks/order-numerical-fixes.md --since 6bb7108` lists exactly these two decisions: pass.
- Both lists passed their merge gate (saved test log, pre-push hooks) and are merged: row 4 as `93f1856`, row 7 as `1e10abc`: pass.

## The items

### 1. The CM.EL ML standard error check (row 4, `agents/tasks/cm-el-information/tasks-1.md` Task 2, commit `e958fde`)

- **The stop:** the new analytic ML information gave a standard error of 0.35104049278 on `dat.nielweise2007` at the default-control fit, against
  the spec's 0.35104032933, a relative difference of 4.7e-7 where the task allowed 1e-8.
- **The evidence:** at the true optimum (tight integrals, a converged search) the same code gives 0.35104032970, 1e-9 from the spec; it agrees
  with a numerical second difference and with quadrature. The default search stops about 1.5e-7 from the optimum.
- **The decision:** check the formula against the spec to 1e-8 at the tight optimum, and the default-control fit to 1e-5, the tolerance the ML
  correctness cases (`glmm.fit.or.cm.el.ml.*.se`) already use. The default search control is out of the spec's scope.
- **Recommendation:** approve - the gap comes from where the search stops, not from the formula, and both checks are as strict as the spec's
  measurement allows.

### 2. The audit findings unrelated to row 7 (`agents/tasks/selmodel-prec-stepfun/tasks-1.md` Task 3, commit `5afc6c3`)

- **The stop:** the whole audit reported 10 findings that the newest earlier report (a slice, 0 findings) did not hold; none named `stepfun` or
  `prec`.
- **The evidence:** checked against the run's `findings.json` on 2026-11-16. Eight are the four `TODO.md` items row 1 filed on 2026-11-15 (the
  exact permutest of `uni_mods` as an R kill; the `selmodel` logistic Hessian element; the last-digit prints of flat location-scale fits; `ls_na`
  with an interaction formula). Two, `scripts/misc_rma_uni_ls/test/18` and `/21`, are `coef(res)$alpha` of identity-link ML fits that moved by 3e-4
  relative with row 5's identity-link polish, whose `known.toml` entry matched only `scripts/misc_rma_ls/*`.
- **The decision:** leave the eight under their open `TODO.md` items, add one `known.toml` entry for the two identity-link tests with row 5's
  reason, and go on with the recall check.
- **Recommendation:** approve - every finding has a confirmed cause, and the new entry only completes row 5's accepted difference.

## Decisions

None beyond the items: each item's approve or flag is the decision.

## Answers

(Empty until the review page is answered.)
