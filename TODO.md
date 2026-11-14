# TODO

Open work only, in implementation order, nearest first. Finished items move to
`DONE.md` unchanged. See the shared `todo-add` and `todo-review` skills for
the shape of an item.

## The CM.EL standard error of the log odds ratio differs from the observed information of the exact likelihood

<!-- doc-counts: accept-start found on 2026-11-14 - a dated record, not a claim about the present -->
Found on 2026-11-14 by the `rma_glmm` correctness cases (`agents/specs/correctness-glmm/mismatches.md`, decision D1); the user answered *follow the
reference* for the `FE` fits and *investigate* for the `ML` fits on 2026-11-14. The coefficients and `tau2` of `rma_glmm(model='CM.EL')` agree with the
formulas of `src/tests/correctness/glmm_formulas.py` (van Houwelingen et al. 1993); only the standard error disagrees.

### FE fits: follow the reference

Cases `glmm.fit.or.cm.el.fe.{nielweise2007,collins1985a,jackson2018.pneumonia}.se` in `src/tests/correctness/cases/glmm.toml`, tolerance 1e-6, each a
strict expected failure. metafopy / formulas / R metafor 5.2-1: `dat.nielweise2007` 0.22403812 / 0.22403482 / 0.22403498; `dat.collins1985a`
0.09940439 / 0.09955789 / 0.09952917; the pneumonia outcome of Jackson et al. (2018) 0.25175750 / 0.25175350 / 0.25174661. The `FE` information of the
exact conditional likelihood is analytic (the sum of the conditional variances), and metafopy is furthest from it, 1.5e-3 relative on
`dat.collins1985a`.

### ML fits: investigate

Cases `glmm.fit.or.cm.el.ml.{nielweise2007,collins1985a,jackson2018.pneumonia}.se`, tolerance 1e-5, `int_rel_tol` 1e-10, each a strict expected
failure. metafopy / formulas / R metafor: 0.35126655 / 0.35104033 / 0.35110612; 0.13635527 / 0.13638548 / 0.13648151; 0.88813349 / 0.88810230 /
0.88812403. The formulas use a finite-difference Hessian of the exact marginal likelihood, as does R metafor (`numDeriv`), and the three values disagree
with one another, so which is nearest the true observed information is not known.
<!-- doc-counts: accept-end -->

Fix direction: find the standard error code for `CM.EL` in `src/metafopy/models/glmm_fit.py`; make the `FE` standard error the analytic information of
the exact conditional likelihood, then set the three `FE` cases to `pass`. For `ML`, compare the Hessian with a step-size sweep and a higher-order
difference, decide which value is nearest, then either change metafopy or accept the difference with its reason. Mark the R parity fixtures that hold a
`CM.EL` standard error as differing, add the matching `src/audit/data/known.toml` entries, and add the difference to `docs/differences.md` under "Where
metafor departs from its reference". A change to `src/metafopy/` goes through a spec first, or a small task list if the fix is local.

## The CM.EL saturated-model Wald statistic QE_Wld is ill-conditioned

<!-- doc-counts: accept-start found on 2026-11-14 - a dated record, not a claim about the present -->
Found on 2026-11-14 by the case `glmm.heterogeneity.qe-wld.cm.el.fe.bcg` (`agents/specs/correctness-glmm/mismatches.md`, decision D2); the user
answered *follow the reference* on 2026-11-14. For `dat.bcg`, `OR`, `CM.EL`, `FE`, `QE_Wld` is 191.71241948 in metafopy, 163.08947856 in the formulas
(Wald statistic of the study coefficients of the saturated exact-likelihood fit with the analytic information of the non-central hypergeometric
likelihood; `saturated_fits` in `src/tests/correctness/glmm_formulas.py`) and 268.5989693 in R metafor 5.2-1. The same statistic agrees between the
formulas, metafopy and R metafor for `CM.AL` (147.906) and `UM.FS` (163.165), and `QE_LRT` of `CM.EL` is 176.8738393 in all three.
<!-- doc-counts: accept-end -->

Metafopy takes the Hessian of the saturated `CM.EL` model by finite differences of a noisy objective, as R does; the `src/audit/data/known.toml` entries
for `*/glmm_cm*` record it as a difference from R.

Fix direction: compute the information of the saturated `CM.EL` fit analytically, as the formulas do, and set the case to `pass`. Mark the R parity
fixtures that hold `QE_Wld` of `CM.EL` as differing, update the `known.toml` entries for `*/glmm_cm*` and add the difference to `docs/differences.md`
under "Where metafor departs from its reference".

## Find the data behind Table 5 of Bagos and Nikolopoulos (2009)

<!-- doc-counts: accept-start found on 2026-11-14 - a dated record, not a claim about the present -->
Found on 2026-11-14 by Tasks 9 and 11 of `agents/tasks/correctness-glmm/tasks-1.md` (`agents/specs/correctness-glmm/mismatches.md`, decision D3); the
user answered *investigate* on 2026-11-14. Bagos and Nikolopoulos (2009) Table 5 prints random-study-effects incidence rate ratios (Hooper 0.746 (0.592,
0.940), school absences 0.721 (0.579, 0.897), ER visits 0.525 (0.361, 0.764)) but no per-study counts or person-times, so no Poisson case could be
written. The Table 1 trial rows of Hooper et al. (2001) give an IRR of 0.859 (`UM.FS`) against the printed 0.819, and Wolf et al. (2003) prints means
and standard deviations, not counts.
<!-- doc-counts: accept-end -->

Fix direction: find the original per-trial counts and person-times that reproduce Table 5 (the authors' supplementary data, or the primary trial
reports), then write the published cases in `src/tests/correctness/cases/glmm_published.toml`.

## Stijnen et al. (2010) Table VII treated arm, PN, differs from metafopy and R metafor

<!-- doc-counts: accept-start found on 2026-11-14 - a dated record, not a claim about the present -->
Found on 2026-11-14 by the case `glmm.published.stijnen2010.table7.pn.treated.mismatch` (`agents/specs/correctness-glmm/mismatches.md`); the user
answered *investigate* on 2026-11-14. The printed values are log rate 0.850, standard error 0.330 and `tau2` 0.654; metafopy and R metafor 5.2-1 both
give 0.81997, 0.31962 and 0.59899 (0.59898 in R) for `measure='IRLN'`, `method='ML'` on the treated data of Table VI. The control arm agrees to one unit
of the last digit, so the difference is larger than rounding.
<!-- doc-counts: accept-end -->

Fix direction: check the transcription of the treated rows of Table VI against the paper, then the likelihood (SAS Proc NLMIXED against a Poisson
log-normal model with an offset) with the formulas of `src/tests/correctness/glmm_formulas.py`; either correct the data, record the source difference
with its reason, or fix metafopy.

## Simmonds and Higgins (2016) Fig 1 one-stage random effects differs from metafopy and R metafor

<!-- doc-counts: accept-start found on 2026-11-14 - a dated record, not a claim about the present -->
Found on 2026-11-14 by the case `glmm.published.simmonds2016.hrt.random.mismatch` (`agents/specs/correctness-glmm/mismatches.md`); the user answered
*investigate* on 2026-11-14. The paper prints OR 1.74 (0.59, 5.17) for the one-stage random-effects model (`glmer`); metafopy and R metafor 5.2-1 give
1.79068 (0.54155, 5.92107) for `UM.FS`, `ML` on the same data, with `tau2` 0.21307. The paper's random-effects result equals its fixed-effect result
(1.74), which metafopy and R reproduce for `FE` (1.74126), so the paper's fit may have a `tau2` of zero.
<!-- doc-counts: accept-end -->

Fix direction: compare the marginal likelihood at `tau2 = 0` and at 0.213 with the formulas of `src/tests/correctness/glmm_formulas.py`; if the
likelihood is higher at 0.213, record the printed value as a source difference (a `glmer` that stopped at `tau2 = 0`); otherwise fix metafopy.

## Extend the correctness tests beyond the core models

Found on 2026-11-12 while reviewing `TODO.md`: the correctness-tests spec (`agents/specs/correctness-tests/spec.md`, introduction) covers `escalc`,
`rma_uni`, `rma_mh` and `rma_peto`, and says "Later specs extend the same harness to `rma_mv`, `rma_glmm`, `rma_ls`, the bias methods, `selmodel` and
`robust`." No such spec exists yet, so those functions are checked only against R metafor, not against their defining references.

Fix direction: one spec per area with `/specdev`, each listing its defining references, the sources to fetch or ask the user for, and its coverage items in
`src/tests/correctness/test_coverage.py`; do the harness item above first, since several of these areas report back-transformed or interval values.

Spec: `agents/specs/correctness-mv-robust/spec.md`; tasks: `agents/tasks/correctness-mv-robust/tasks-1.md`. Spec:
`agents/specs/correctness-glmm/spec.md`; tasks: `agents/tasks/correctness-glmm/tasks-1.md`. Spec: `agents/specs/correctness-location-scale/spec.md`;
tasks: `agents/tasks/correctness-location-scale/tasks-1.md`. Spec: `agents/specs/correctness-publication-bias/spec.md`; tasks:
`agents/tasks/correctness-publication-bias/tasks-1.md`.
