# Mismatches, conventions and sources needed: correctness tests for rma_glmm

**Spec:** `agents/specs/correctness-glmm/spec.md`
**Created:** 2026-11-14 (started by Task 4; Task 13 of `agents/tasks/correctness-glmm/tasks-1.md` completes and publishes it)

This file collects every disagreement the `rma_glmm` correctness cases found, each convention difference that changes a printed digit, the sources
still needed, and the limits of the harness. Items are appended by the tasks that write cases.

## Mismatches

Each item gives the case (in `src/tests/correctness/cases/glmm.toml`), the reference, metafopy's value and R metafor 5.2-1's value (computed with the
audit's pinned library).

### CM.EL standard error of the log odds ratio, `FE` and `ML`

The reference is the observed information of the exact conditional likelihood of the formulas in `src/tests/correctness/glmm_formulas.py` (van
Houwelingen et al. 1993; `FE`: analytic sum of the conditional variances, `ML`: finite-difference Hessian of the exact marginal likelihood). The
coefficients and `tau2` agree with the formulas to the case tolerance; only the standard error disagrees.

| Case | Tolerance | metafopy | formulas | R metafor |
|---|---|---|---|---|
| `glmm.fit.or.cm.el.fe.nielweise2007.se` | 1e-6 | 0.22403812206 | 0.22403481655 | 0.2240349774 |
| `glmm.fit.or.cm.el.fe.collins1985a.se` | 1e-6 | 0.09940438923 | 0.09955789399 | 0.09952916761 |
| `glmm.fit.or.cm.el.fe.jackson2018.pneumonia.se` | 1e-6 | 0.25175750395 | 0.25175350296 | 0.2517466148 |
| `glmm.fit.or.cm.el.ml.nielweise2007.se` | 1e-5 | 0.35126655007 | 0.35104032913 | 0.3511061238 |
| `glmm.fit.or.cm.el.ml.collins1985a.se` | 1e-5 | 0.13635526517 | 0.13638547729 | 0.1364815073 |
| `glmm.fit.or.cm.el.ml.jackson2018.pneumonia.se` | 1e-5 | 0.88813349 | 0.88810230 | 0.8881240252 |

R metafor itself differs from the formulas (its `clogit` and `numDeriv` results are numerical); for `FE` on `dat.collins1985a` metafopy is further from
both than they are from each other. The `ML` rows of metafor use `control = list(intCtrl = list(rel.tol = 1e-10))`, as the cases do (`int_rel_tol`).
Jackson et al. (2018) Table 3, outcome 1 (pneumonia), is the data of the Jackson rows.

## Conventions

### `ll` of the `lme4` mixed fits

`fit_stats_ml.ll` of `UM.FS` and `CM.AL` with `ML`, and of `UM.RS` with `FE` and `ML`, is metafor's saturated-model log-likelihood minus half the
conditional deviance (`ml_loglik` in `src/metafopy/models/glmm_fit.py`), not the marginal log-likelihood of the model. The derived cases therefore
compare it with nothing, and check that the formulas' marginal log-likelihood at metafopy's estimates equals the formulas' maximum and has a small
score. It changes no printed digit of a published coefficient, standard error or `tau2`.

## Sources needed

None found by Task 4.

## Harness limits

- `UM.RS` with `ML` on `dat.nielweise2007` has no case: the fit is nearly unidentifiable (R metafor warns of it) and the formulas' search stops at a
  score of 1.3e-6, above the 1e-6 it requires.

## Answers

Left empty for the user's verdicts.
