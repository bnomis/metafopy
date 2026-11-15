# TODO

Open work only, in implementation order, nearest first. Finished items move to
`DONE.md` unchanged. See the shared `todo-add` and `todo-review` skills for
the shape of an item.

## The audit lists the exact permutest of uni_mods as an R kill and has no entry for it

<!-- doc-counts: accept-start found on 2026-11-15 - a dated record, not a claim about the present -->
Found on 2026-11-15 by the full audit `.audit/runs/20261115-160207` (`findings.json`, `new` list): the case `options/permutest/uni_mods/exact=true` has kind
`r-killed`, detail "R passed the memory limit of 4 GiB (killed)". `src/audit/data/known.toml` accepts the same kind for
`options/permutest/ls_bcg/exact=true` only.
<!-- doc-counts: accept-end -->

Cause: `permutest(exact = TRUE)` enumerates every permutation of the covariate in R, which passes the harness's 4 GiB limit, so R gives nothing to compare
with (the same cause as the `ls_bcg` entry). Not yet checked that metafopy refuses or finishes on this fixture.

Fix direction: run the case by hand in metafopy; if it refuses with R's "Number of iterations requested too large." or finishes, add the case to the
`r-killed` entry of `options/permutest/ls_bcg/exact=true` in `src/audit/data/known.toml` with the measurement and date.

## The audit lists a selmodel logistic Hessian element that is zero in both R and metafopy

<!-- doc-counts: accept-start found on 2026-11-15 - a dated record, not a claim about the present -->
Found on 2026-11-15 by the full audit `.audit/runs/20261115-160207` (`findings.json`, `new` list): `options/selmodel/uni_factor/type="logistic"` and
`type="logistic1"` have a `value-diff` on `$hessian`, element 11: R -1.15258004719678e-08, Python 2.69121241741294e-15 (relative 1). Every other element
of the matrix agrees.
<!-- doc-counts: accept-end -->

Cause: both values are zero to rounding, so the relative difference is 1. The ledger entry for `options/selmodel/*` `$hessian` covers differences with detail
`*(relative 0.00*` only. Not yet checked that the other elements agree to that level on both cases.

Fix direction: check the two cases in R and Python, then widen the `detail` of that entry or add an entry for these two cases, with the values and date.

## The audit lists printed fits of flat location-scale models that differ in the last digit

<!-- doc-counts: accept-start found on 2026-11-15 - a dated record, not a claim about the present -->
Found on 2026-11-15 by the full audit `.audit/runs/20261115-160207` (`findings.json`, `new` list), four `print-diff` findings: `options/vif/ls_na/table=true`
line 7 (`5.5982` against `5.5980`); `options/permutest/ls_bangert/digits=2` and `digits=[2,{"$na":true},1]` line 12 (permutation p-value `0.90` against
`0.88`); `construct/uni_mods_year/scale={"$formula":"~ 0 + I(1|ablat)"}` line 13 (`-73.4722` against `-73.4724`).
<!-- doc-counts: accept-end -->

Cause: suspected, not confirmed: the stop points of R's `nlminb` and metafopy's optimiser on a flat scale likelihood, which `known.toml` already accepts for
other printed fits of `ls_na`, `ls_bangert` and `uni_*` (`docs/differences.md`, "Refits of a location-scale model that R's optimiser leaves short"); the
entries do not match these rows or options.

Fix direction: compare the log-likelihoods of R and metafopy for each case; where they agree to 1e-6 add the case to the existing entry with the values
and date, and file any that does not as a defect.

## The audit has no entry for ls_na with an interaction formula that R refuses

<!-- doc-counts: accept-start found on 2026-11-15 - a dated record, not a claim about the present -->
Found on 2026-11-15 by the full audit `.audit/runs/20261115-160207` (`findings.json`, `new` list): `construct/ls_na/mods={"$formula":"~ ablat * alloc"}` has
kind `accepts-where-R-refuses`; R stops with "No missing values allowed in 'alpha.init'." and metafopy returns a fit.
<!-- doc-counts: accept-end -->

Cause: suspected, not confirmed: the starting values of the scale model are computed from a location fit that has NA coefficients (`alloc` is collinear
with `ablat` on the rows of `ls_na` that remain), which R refuses and metafopy handles (`start_alpha` in `src/metafopy/models/rma_ls_fit.py`).

Fix direction: reproduce the R message with `rma(..., mods = ~ ablat * alloc, scale = ...)` on the `ls_na` data, read why `alpha.init` has NA, then either
make metafopy refuse with the same error or accept the difference in `src/audit/data/known.toml` and `docs/differences.md`. A change to `src/metafopy/`
goes through a spec first, or a small task list if the fix is local.

## The truncest estimate stops at a local maximum of a step-function likelihood

<!-- doc-counts: accept-start found on 2026-11-15 - a dated record, not a claim about the present -->
Found on 2026-11-15 by the case `selmodel.truncest.bangertdrowns2004` in `src/tests/correctness/cases/selmodel.toml`
(`agents/specs/correctness-publication-bias/mismatches.md`); the user answered *follow the reference* on 2026-11-15. On `dat.bangertdrowns2004`
metafopy gives beta 0.26491, tau2 0.054216, delta 1.42227, truncation point 0.27073 and ll -17.7909; the maximum of the likelihood
(`formulas.truncation_loglik`) gives beta 0.27598, tau2 0.054737, delta 1.48969, truncation point 0.20001 and ll -17.7406. R metafor 5.2-1 gives
metafopy's values (operator session, 2026-11-15), so it stops at the same local maximum.
<!-- doc-counts: accept-end -->

Cause: the likelihood is a step function of the truncation point with a jump at each observed effect, so a gradient search stops inside one piece
(`type == 'truncest'` in `src/metafopy/models/selmodel.py`).

Fix direction: search each stretch between observed effects and keep the best, then set the case to `pass`. Mark the R parity fixture that holds a
`truncest` fit as differing, add the `src/audit/data/known.toml` entry, and add the difference to `docs/differences.md` under "Where metafor departs from
its reference". A change to `src/metafopy/` goes through a spec first, or a small task list if the fix is local.

Spec: none (small item); tasks: `agents/tasks/truncest-piecewise-search/tasks-1.md`.

## The CM.EL information is taken by finite differences of the exact likelihood

Both findings below share one cause: for `rma_glmm(model='CM.EL')`, metafopy takes the Hessian of the exact conditional (non-central hypergeometric)
likelihood by finite differences, as R metafor does (`src/metafopy/models/glmm_fit.py`, `invert_hessian` and the `hesspack` options), while the
correctness formulas (`src/tests/correctness/glmm_formulas.py`) use its analytic information. The standard error of the pooled log odds ratio (first
finding) and the Wald statistic of the saturated model, `QE_Wld` (second finding), both come from that numerical Hessian. Consolidated on 2026-11-15 by a
TODO review.

Fix direction: compute the analytic information of the exact conditional likelihood once and use it for both the `FE` standard error and the saturated
model's `QE_Wld`; then settle the `ML` standard error as the first finding says. Each finding's own fix direction lists the cases, the R parity fixtures,
the `known.toml` entries and the `docs/differences.md` entry it needs. A change to `src/metafopy/` goes through a spec first.

### The CM.EL standard error of the log odds ratio differs from the observed information of the exact likelihood

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

### The CM.EL saturated-model Wald statistic QE_Wld is ill-conditioned

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

Spec: `agents/specs/cm-el-information/spec.md`; tasks: `agents/tasks/cm-el-information/tasks-1.md`.

## The location-scale identity-link fit stops about 1e-5 short of the maximum

<!-- doc-counts: accept-start found on 2026-11-15 - a dated record, not a claim about the present -->
Found on 2026-11-15 by the location-scale correctness cases (`agents/specs/correctness-location-scale/mismatches.md`, item `ls_identity_opt`); the user
answered *follow the reference* on 2026-11-15. The strict expected failures `ls.fit.{ML,REML}.identity.{bd,bcg}` and
`property.ls.fit.{ML,REML}.identity.{bd,bcg}` in `src/tests/correctness/cases/location_scale.toml`, and `property.ls.links.{ML,REML}.bd` (the
identity-link and log-link fits of one two-group scale model span the same variances), show that the identity-link `alpha` differs from the independent
reference (`formulas.ls_reference_fit`) by about 1e-5 relative (tolerance 1e-6) and the log-likelihood by 1e-8 to 2e-7 (tolerance 1e-9). On
`dat.bangertdrowns2004` with `imag` in both parts the ML log-likelihood is -17.058054743801 in metafopy against -17.058054672888 in the reference; on
`dat.bcg` with `ablat` in both parts, -7.139288984088 against -7.139288817730. The reference ends higher in all four fits and satisfies
`Z alpha >= 0`. The log-link fits agree to better than 1e-9. metafor's own identity-link values are not on record: no parity fixture in
`src/tests/reference/phase5/location_scale.json` fits these models.
<!-- doc-counts: accept-end -->

Suspected cause (not confirmed): `constr_optim` in `src/metafopy/models/rma_ls_opt.py` is a port of R's `constrOptim` (adaptive log barrier around
Nelder-Mead), which stops at its own convergence tolerance, and metafopy's identity-link path in `src/metafopy/models/rma_ls_fit.py` takes the result
without a final polish.

Fix direction: polish the identity-link optimum, for example by a final Newton or BFGS step on the analytic score inside the feasible region, then set
the ten cases to `pass`. Mark the R parity fixtures that hold an identity-link fit as differing at the 1e-4 level `src/tests/test_rma_ls.py` accepts,
add the matching `src/audit/data/known.toml` entries, and add the difference to `docs/differences.md` under "Where metafor departs from its
reference". A change to `src/metafopy/` goes through a spec first, or a small task list if the fix is local.

Spec: none (small item); tasks: `agents/tasks/ls-identity-polish/tasks-1.md`.

## The location-scale profile-likelihood intervals of the scale coefficients are not crossings where the profile is flat

<!-- doc-counts: accept-start found on 2026-11-15 - a dated record, not a claim about the present -->
Found on 2026-11-15 by the location-scale correctness cases (`agents/specs/correctness-location-scale/mismatches.md`, items `ls_plci_bcg` and
`ls_plci_vl2022`); the user answered *follow the reference* for both on 2026-11-15. metafopy's profile-likelihood bounds for a scale coefficient
(`confint` of `rma_uni(..., scale=)`, `src/metafopy/methods/ls_profile.py` and `src/metafopy/methods/ls_confint.py`) differ from the independent
reference `formulas.ls_profile_bounds` (a root of `2 (ll_max - ll_j(a)) = chi2_{1, 0.95}`, the other alpha re-maximised at each `a`) and from the
printed values of Viechtbauer and Lopez-Lopez (2022), which were computed with metafor.
<!-- doc-counts: accept-end -->

### dat.bcg scale intercept, ML and REML

<!-- doc-counts: accept-start found on 2026-11-15 - a dated record, not a claim about the present -->
Cases `ls.confint.scale.ML.bcg` and `ls.confint.scale.REML.bcg` in `src/tests/correctness/cases/location_scale.toml`, strict expected failures (the
same check passes on `dat.bangertdrowns2004`). Log link, `ablat` in both parts, alpha.1 the intercept. ML lower bound: reference `nan` (not crossed
within 20 standard errors), metafopy -23.309450; ML upper 1.803009 against 1.022354; REML upper 3.402673 against 3.286219 (REML lower agrees,
-16.010862). At a = 1.0 the reference profile gives 3.04, below the 3.84 cut-off, so metafopy's ML upper bound lies where the profile has not crossed.
<!-- doc-counts: accept-end -->

### Viechtbauer and Lopez-Lopez (2022) Tables 1 and 2

<!-- doc-counts: accept-start found on 2026-11-15 - a dated record, not a claim about the present -->
Cases `published.vl2022.table1.plci.size` and `published.vl2022.table2.plci.soc`, strict expected failures; the bounds metafopy does reproduce are in
the passing `published.vl2022.table1.plci` and `published.vl2022.table2.plci`. Table 1 sample size lower bound: paper -7.159, metafopy -5.93003,
reference -7.159061 (so a metafopy defect, not a misprint). Table 1 social sciences vs. math: paper `< -10` and `> 10`, metafopy -3.61002 and 4.41224,
reference no crossing. Table 2 social sciences vs. math: upper bound paper 7.700, metafopy 5.72067; lower paper `< -10`, metafopy -4.68066. The
Table 2 reference fit does not converge, so no independent value was computed there.
<!-- doc-counts: accept-end -->

Suspected cause (not confirmed): where the profile is flat, the refit of the other coefficients at a fixed `a` ends at a lower log-likelihood than the
reference's (which starts from two points, one keeping the fitted line of `Z alpha`), so the root search stops before the true crossing. Read
`src/metafopy/methods/ls_profile.py` for the starting values and the bracket.

Fix direction: refit the profile as metafor and the reference do, starting from the fitted line and re-maximising from more than one start, then set the
four cases to `pass`. Check metafor's `confint` on the Table 2 model to read the bound the paper prints. Mark the R parity fixtures that hold a
location-scale profile interval as differing where they do, add the matching `src/audit/data/known.toml` entries, and add the difference to
`docs/differences.md` under "Where metafor departs from its reference". A change to `src/metafopy/` goes through a spec first, or a small task list
if the fix is local.

Spec: none (small item); tasks: `agents/tasks/ls-profile-crossings/tasks-1.md`.

## A precision measure on a step-function selection model makes the likelihood not a density

<!-- doc-counts: accept-start found on 2026-11-15 - a dated record, not a claim about the present -->
Found on 2026-11-15 by the cases `selmodel.prec.{sei,vi,ninv,sqrtninv}.bangertdrowns2004` in `src/tests/correctness/cases/selmodel.toml`
(`agents/specs/correctness-publication-bias/mismatches.md`); the user answered *investigate* on 2026-11-15. With `type='stepfun'` and `prec` set,
metafopy gives beta 0.13780 (`sei`), 0.16908 (`vi`), 0.16769 (`ninv`) and 0.13711 (`sqrtninv`); the maximum of the density with weights `delta[j] /
preci` gives beta 0.10569 and ll -17.4217 for each. R metafor 5.2-1 gives metafopy's values (operator session, 2026-11-15), and warns that the
combination is undocumented and experimental.
<!-- doc-counts: accept-end -->

Cause: metafopy, like metafor, normalises the first interval with weight 1 but weights the observed p-value by `delta[0] / preci`, so the likelihood is
not a density and its maximum moves with the precision measure. The consistent likelihood makes `prec` a no-op, so neither is the intended model.

Fix direction: decide whether `selmodel` rejects the combination with an `InputError`, or defines it from a source (none found for step functions). Then
set the four cases to `pass` or exempt them with the decision, mark the `hack_stepfun_sei` R parity fixture as differing if the behaviour changes, and
name the decision in `docs/differences.md` under "Where metafor departs from its reference". A change to `src/metafopy/` goes through a spec first.

Spec: `agents/specs/selmodel-prec-stepfun/spec.md`; tasks: `agents/tasks/selmodel-prec-stepfun/tasks-1.md`.

## Whether R metafor's SJIT converges on the scaled scale-invariance data was never checked

<!-- doc-counts: accept-start found on 2026-11-15 - a dated record, not a claim about the present -->
Found on 2026-11-15 by a TODO review. Task 13 of `agents/tasks/correctness-location-scale/tasks-1.md` asks to check whether metafor's
`rma(..., method="SJIT")` converges on the data that makes `test_scale_invariance[SJIT]` fail (`yi = [0, -0.5625, 0, 0.875, -0.7734375, 0]`,
`vi = [0.5, 0.625, 0.5, 0.5, 0.125, 1]`, scale 2.0, `threshold` 1e-13, `maxiter` 1000), and to file an item if it does. R metafor could not be run,
so it was not checked; `rma_uni` raises `EstimationError` there, and the property test now rejects scaled copies that do not converge.
<!-- doc-counts: accept-end -->

Fix direction: run the scaled data through R metafor 5.2-1 with the same control. If R converges, find why metafopy's iteration (the absolute stopping
rule on `tau2` in `src/metafopy/models/rma_uni.py`) does not, and fix it; if R does not converge either, close this item with the result.

Spec: none (small item); tasks: `agents/tasks/sjit-scaled-convergence/tasks-1.md`.

## Simmonds and Higgins (2016) Fig 1 one-stage random effects differs from metafopy and R metafor

<!-- doc-counts: accept-start found on 2026-11-14 - a dated record, not a claim about the present -->
Found on 2026-11-14 by the case `glmm.published.simmonds2016.hrt.random.mismatch` (`agents/specs/correctness-glmm/mismatches.md`); the user answered
*investigate* on 2026-11-14. The paper prints OR 1.74 (0.59, 5.17) for the one-stage random-effects model (`glmer`); metafopy and R metafor 5.2-1 give
1.79068 (0.54155, 5.92107) for `UM.FS`, `ML` on the same data, with `tau2` 0.21307. The paper's random-effects result equals its fixed-effect result
(1.74), which metafopy and R reproduce for `FE` (1.74126), so the paper's fit may have a `tau2` of zero.
<!-- doc-counts: accept-end -->

Fix direction: compare the marginal likelihood at `tau2 = 0` and at 0.213 with the formulas of `src/tests/correctness/glmm_formulas.py`; if the
likelihood is higher at 0.213, record the printed value as a source difference (a `glmer` that stopped at `tau2 = 0`); otherwise fix metafopy.

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

## The rma_mv funnel test still skips when rstandard is missing

<!-- doc-counts: accept-start found on 2026-11-15 - a dated record, not a claim about the present -->
Found on 2026-11-15 by a TODO review. `test_funnel_matches_metafor` in `src/tests/test_fits.py` skips the `mvm` case when `has_mv_residuals` finds no
`rstandard` for `rma_mv` fits ("rstandard() of rma_mv fits is not available yet"), but `rstandard_rma_mv` now exists in
`src/metafopy/methods/mv_diagnostics.py`, so the skip guards a gap that has closed.
<!-- doc-counts: accept-end -->

Fix direction: confirm the `mvm` case runs, then remove `has_mv_residuals` and the skip.

## Correctness coverage the extension specs left for later

<!-- doc-counts: accept-start found on 2026-11-15 - a dated record, not a claim about the present -->
Found on 2026-11-15 by a TODO review. `agents/specs/correctness-location-scale/spec.md` (out of scope) leaves for a later spec `profile` beyond the
property, `vif`, `blup`, `ranef`, `rstandard`, `weighted=False`, fixed `alpha` and `beta`, and `alpha_min` and `alpha_max`;
`agents/specs/correctness-location-scale/mismatches.md` adds REML with `optbeta=True`, the deviance and `vb` of `optbeta` fits.
`agents/specs/correctness-mv-robust/spec.md` says a later spec can add the `rma_mv` diagnostics `cooks_distance`, `dfbetas`, `rstudent` and `ranef`.
No spec, task list or item covers them; they are checked against R metafor only.
<!-- doc-counts: accept-end -->

Fix direction: deferred. When taken up, one spec per area with `/specdev`, listing the defining references and the coverage items in
`src/tests/correctness/test_coverage.py`.
