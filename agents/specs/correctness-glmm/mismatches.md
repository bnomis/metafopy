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

### `ll` of the `lme4` fits of `IRR` and the one-group measures

`fit_stats_ml.ll` of `IRR` (every model with `ML`) and of the one-group measures with `ML` is the same saturated-model quantity as for the `OR`
mixed fits above, so the derived cases do not compare it with the formulas' marginal log-likelihood; they check that the formulas' log-likelihood at
metafopy's estimates equals the formulas' maximum and has a small score. `IRR` with `CM.AL` is the `CM.EL` fit: a check compares the two calls' `beta`,
`se`, `tau2` and `ll`, which are identical.

### Standard errors of fits without random effects, non-canonical links

`RR` (log link) and `RD` (identity link) fits whose variances are zero (`FE` with `UM.FS`, and the `ML` fits of `dat.hine1989`) report the standard
errors of `glm()`, from the expected information; the formulas' observed information differs (`dat.nielweise2007`, `RR`, `UM.FS`, `FE`: 0.2174 and
0.2172, R metafor 5.2-1 0.2174). Those cases read `rma_glmm_se_expected`, the expected information coded from McCullagh and Nelder (1989). For the
one-group `FE` fits the closed-form and numerical standard errors agree with R metafor to about 4e-6, the stopping rule of the `glm()` iterations
(`IRLN`, `FE`: metafopy and R 0.144337, the closed form `1/sqrt(sum xi)` 0.1443376), so those cases use a tolerance of 1e-5.

## Sources needed

None found by Task 4.

## Harness limits

- `UM.RS` with `ML` on `dat.nielweise2007` has no case: the fit is nearly unidentifiable (R metafor warns of it) and the formulas' search stops at a
  score of 1.3e-6, above the 1e-6 it requires.
- `RD` cannot be fitted to `dat.nielweise2007` (R metafor 5.2-1 stops with "Cannot fit FE model", as metafopy does), so its cases use `dat.hine1989`,
  where the fitted variances are zero.
- `UM.RS` with `ML` and `RR` on `dat.nielweise2007`: the formulas' search stops at a score of 1.07e-6, and the formulas' likelihood has a score of 0.027
  at the estimates that metafopy and R metafor 5.2-1 agree on (-1.2348, tau2 0.6475). The formulas treat a quadrature node with a mean above one as
  impossible, which `glmer` does not (the same holds for `RR` on `dat.damico2009` with `ML`). The case uses `dat.hine1989`.
- `PR` and `PLN` with `ML` on `dat.pritz1997`: R metafor 5.2-1 cannot fit them, and the formulas disagree with metafopy for the same reason. The cases use
  `dat.hine1989` (`PR`; zero variance) and the control arm of `dat.nielweise2007` (`PLN`).

## Answers

Left empty for the user's verdicts.
