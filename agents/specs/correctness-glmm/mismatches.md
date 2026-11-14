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

### CM.EL Wald statistic of the saturated model, `QE_Wld`

The reference is the Wald statistic of the study coefficients of the saturated exact-likelihood fit of the formulas (analytic information of the
non-central hypergeometric likelihood; `saturated_fits` in `src/tests/correctness/glmm_formulas.py`). The same statistic from the formulas agrees with
`CM.AL` and `UM.FS` to the case tolerance, and `QE_LRT` of `CM.EL` agrees to 1e-14, so only the Wald statistic of `CM.EL` disagrees. R metafor 5.2-1
differs from both (the metafopy documentation says the statistic is ill-conditioned for `CM.EL` and differs between optimisers in R as well).

| Case | Tolerance | metafopy | formulas | R metafor |
|---|---|---|---|---|
| `glmm.heterogeneity.qe-wld.cm.el.fe.bcg` (`dat.bcg`, `OR`, `CM.EL`, `FE`) | 1e-5 | 191.71241948 | 163.08947856 | 268.5989693 |

For comparison the same data give `QE_Wld` 147.906 with `CM.AL` and 163.165 with `UM.FS` in both metafopy and the formulas, and `QE_LRT` 176.8738393
with `CM.EL` in all three.

### Published values (Task 9, `src/tests/correctness/cases/glmm_published.toml`)

Each row is a `.mismatch` case beside the case of the values that agree. R metafor 5.2-1 was run with the audit's pinned library and gives the same
value as metafopy in every row, so none is a metafopy-only difference: the sources used other software (Jackson et al. 2018: metafor 1.9-9 and lme4
1.1-12; Stijnen et al. 2010: SAS Proc NLMIXED; Simmonds and Higgins 2016: `glmer`).

| Case | Value | Printed | metafopy | R metafor |
|---|---|---|---|---|
| Jackson et al. (2018) Table 4, Model 4 (`UM.FS`), outcome 1 | tau2 | 2.664 | 2.6628 | 2.66292 |
| Jackson et al. Table 4, Model 5 (`UM.RS`), outcome 2 | SE; tau2 | 0.413; 0.119 | 0.4214; 0.1196 | 0.42136; 0.11956 |
| Jackson et al. Table 4, Model 5, outcome 4 | SE | 0.396 | 0.3972 | 0.39719 |
| Jackson et al. Table 4, Model 5, outcome 4b | SE | 0.396 | 0.3970 | 0.39703 |
| Jackson et al. Table 4, Model 7 (`CM.EL`), outcome 1 | tau2 | 4.341 | 4.3416 | 4.34164 |
| Jackson et al. Table 4, Model 7, outcomes 4 and 4b (starred: Nelder-Mead, `maxit` 20000, `reltol` 1e-4) | log OR; tau2 | -0.793; 0.001 | -0.8015; 0.0000 | -0.80149; 0.00003 |
| Stijnen et al. (2010) Table III, HN (`CM.EL`) | median OR | 0.25 | 0.2584 | exp of the printed -1.353 is 0.258, so the printed 0.25 is inconsistent within the paper |
| Stijnen et al. Table III, BN (`CM.AL`) | SD of tau | 0.77 | 0.77504 | 0.77504 |
| Stijnen et al. Table VII, control, PN (`IRLN`) | tau2 | 0.316 | 0.31660 | 0.31659 |
| Stijnen et al. Table VII, treated, PN | log rate; SE; tau2 | 0.850; 0.330; 0.654 | 0.81997; 0.31962; 0.59899 | 0.81997; 0.31962; 0.59898 |
| Simmonds and Higgins (2016) Fig 1, one-stage random (`UM.FS`, `ML`) | OR; CI | 1.74 (0.59, 5.17) | 1.79068 (0.54155, 5.92107) | 1.7907 (0.5415, 5.9211) |
| Turner et al. (2000) Table IV, fixed effect, pre-eclampsia (`UM.FS`, `FE`) | CI | (-0.59, -0.23) | (-0.58398, -0.23688) | (-0.58398, -0.23688) |

Several rows differ by one unit of the last printed digit and are rounding or optimiser differences (the Jackson `tau2` rows, Stijnen BN and control
rows); the Stijnen treated row (Table VII) and the Simmonds and Higgins random-effects row are larger. In the Simmonds and Higgins case the paper's
random-effects result equals its fixed-effect result (1.74), which metafopy and R metafor reproduce for `FE` (1.74126) but not for `ML`
(`tau2` 0.21307).

## Conventions

### The saturated-model Wald statistic of a moderator model

`QE_Wld` of a model with a moderator (`glmm.heterogeneity.qe-wld.cm.al.fe.mods.bcg`) is compared at 1e-3, not 1e-5: the statistic comes from an
information matrix over one coefficient for each study beyond the moderators, and metafopy and the formulas each build that matrix by finite
differences (29.8946 and 29.9013). An intercept-only model agrees to better than 1e-6 (147.90608 and 147.90607 for `CM.AL`).

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

Found by Task 9 (read 2026-11-14): Bagos and Nikolopoulos (2009) prints pooled totals and back-transformed rate ratios only, never per-study counts and
person-times, so no Poisson case could be written from it. Its data come from two reviews that are not in `/Users/simonb/src/metafopy papers`:

- Hooper L, Summerbell CD, Higgins JP, Thompson RL, Capps NE, Smith GD, Riemersma RA, Ebrahim S (2001). Dietary fat intake and prevention of
  cardiovascular disease: systematic review. BMJ 322(7289):757-763. Wanted: the per-trial combined cardiovascular events and person-years of the
  intervention and control groups (10 trials). File name: `Hooper_2001_Dietary_fat_intake_and_prevention_of_cardiovascular_disease.pdf`.
- Wolf FM, Guevara JP, Grum CM, Clark NM, Cates CJ (2003). Educational interventions for asthma in children. Cochrane Database Syst Rev (1):CD000326.
  Wanted: the per-study school absences and ER visits with person-months for each group. File name:
  `Wolf_2003_Educational_interventions_for_asthma_in_children.pdf`.

With them the wanted values are Table 5 of Bagos and Nikolopoulos (random study effects, `z` 0/1 and bivariate rows: Hooper 0.746 (0.592, 0.940), school
absences 0.721 (0.579, 0.897), ER visits 0.525 (0.361, 0.764), as printed IRRs).

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
