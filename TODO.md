# TODO

Open work only, in implementation order, nearest first. Finished items move to
`DONE.md` unchanged. See the shared `todo-add` and `todo-review` skills for
the shape of an item.

## The rma_mv REML fit stops short of the optimum on Linux and Windows

<!-- doc-counts: accept-start found on 2026-11-17 - a dated record, not a claim about the present -->
Found on 2026-11-17 by Task 8 of `agents/tasks/cross-platform-numerics/tasks-1.md` (probe run 44002318304 of commit `f57ab45`; figures in
`agents/specs/cross-platform-numerics/measurements.md`, "The rma_mv REML score"). The REML score of `sigma2` in `property.mv.reml-score.kon`, relative to
its scale `tr(P Z Z')`, is 1e-11 on the development Mac and the containers, 1.3e-07 and 3.8e-08 on CI Linux, and up to 1.1e-06 on CI Windows; the BFGS
gradient is 1.3e-10 (Mac), 6.0e-07 (Linux) and 1.8e-05 (Windows). The stop also varies from run to run on one platform.
<!-- doc-counts: accept-end -->

Cause: `scipy.optimize.minimize` BFGS ends on "precision loss" with `success` false on every platform, and `mv_opt.maximise` accepts the stop because the
gradient is under `SLACK_GRADIENT` (1e-4). Where the line search loses precision early, the fit is accepted up to 4 to 7 orders of magnitude from the
optimum. The test asserts the measured range (2e-6 of the scale) meanwhile.

Fix direction: tighten the stop in `src/metafopy` through a spec (for example a Newton polish on the analytic score after BFGS), then narrow the test's
`tolerance` in `src/tests/correctness/cases/properties.toml` and the entry in `docs/differences-notes.md` ("Optimisers and flat likelihoods").

## Numerical tests whose results or tolerances hold only on the development machine

<!-- doc-counts: accept-start found on 2026-11-16 - a dated record, not a claim about the present -->
Found on 2026-11-16 by a TODO review in CI run 43832306599 on `ff7eb2b` (Python 3.14.8; Linux `ubuntu-26.04`, macOS `macos-26`, Windows
`windows-2025-vs2026`), while the final checkpoint of `agents/tasks/order-numerical-fixes.md` passed on the development Mac the same day. Failures
beyond the two harness items above:

- `ls.*.bcg` correctness cases: on Windows the ML reference fit ends at another optimum (`ls.fitstats.ML.bcg` ll -7.5393 against -9.4736;
  `ls.predict.fitted.ML.z.bcg` "expected a nonnegative input, got -1.5653430049987447e+22"); on Linux every REML case says "the reference fit did not
  converge" (`ls_reference_fit`: "no start converged (log link, REML): Desired error not necessarily achieved due to precision loss.").
- `ls.fit.ML.identity.bcg` (Windows, Linux) and `ls.fit.REML.identity.bcg` (Linux): `alpha` off by 4e-8 to 3e-6 at tolerance 1e-6.
- `selmodel.decreasing.bangertdrowns2004` on all three: the reference beta is -4.537 (Linux), -2248.16 (macOS) or the optimiser stops with "Positive
  directional derivative for linesearch" (Windows); metafopy gives 0.10569.
- `glmm.fit.or.um.rs.ml.jackson2018.pneumonia` and `glmm.cor.um.rs.ml.bcg` (Windows, Linux): "UM.RS: the score at the optimum is 1.05e-06 to 3.02e-06,
  not below 1e-06".
- `test_nchg.py::test_ml_se_matches_spec_measurement_at_tight_optimum`: `pneumonia` (Windows, 0.8881022865 against 0.88810229648 at 8.9e-09) and
  `collins1985a` (Linux, 0.1363854789 against 0.13638547727 at 1.4e-09).
- `property.mv.reml-score.kon` (Linux): REML score 1.3354e-05, theory 0.
- `test_stepcon.py::test_inference[rau_stepcon_mods]` (Windows, Linux); `test_ls_vif.py::test_sim_permutes_the_data_variables_as_r` (all three);
  `test_audit4_numerics.py::test_permutest_ls_boundary_fits_match_r` (all three); `test_rma_ls.py::test_identity_fit_is_the_maximum[bcg_year_identity]`
  and `test_selmodel.py::test_print[hack_halfnorm_less]` (macOS).
<!-- doc-counts: accept-end -->

Cause: not confirmed. Suspected: the optimisers' stopping points and the floating-point results of the platforms' BLAS and libm builds differ, and the
tests compare against a reference fit or an absolute tolerance measured on one machine. Some tests were added in the numerical fixes order on
2026-11-15 and 2026-11-16 (the CM.EL tight-optimum check, the identity-link polish).

Fix direction: reproduce one failure of each group on Linux (a container, or CI's artifacts), then decide per group whether metafopy, the reference fit
or the tolerance is wrong: fix the code, make the reference fit robust (more starts, a tighter control), or set the tolerance from the measured spread
across platforms with the date. A change to `src/metafopy/` goes through a spec first.

Spec: `agents/specs/cross-platform-numerics/spec.md`; tasks: `agents/tasks/cross-platform-numerics/tasks-1.md`.

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

## A fixed delta[0] other than 1 on a step-function selection model makes the likelihood a constant multiple

<!-- doc-counts: accept-start found on 2026-11-16 - a dated record, not a claim about the present -->
Found on 2026-11-16 while closing `agents/specs/selmodel-prec-stepfun/spec.md`. `selmodel(type='stepfun', steps=[0.025, 0.5, 1], delta=[0.5, nan,
nan])` on an ML fit of `dat.bangertdrowns2004` gives the estimates of `delta=[1, nan, nan]` (beta 0.10569, delta 0.40141 and 0.27643) and an ll of
-27.1257 against -17.4217, in metafopy and in R metafor 5.2-1 alike (`R_LIBS=/Users/simonb/src/metafopy/.audit/Rlib LC_ALL=C.UTF-8 Rscript`). The
ll differs by `n1 * log(0.5)`, with `n1` the number of observed p-values in the first interval.
<!-- doc-counts: accept-end -->

Cause: the first interval's weight is hard-coded to 1 in the normalising constant, but the numerator uses the user's `delta[0]`, so a fixed `delta[0]`
only adds `n1 * log(delta[0])` to the ll and never moves the estimates. See `.selmodel.ll.stepfun` (`R/misc.func.hidden.selmodel.r`) and
`ll_stepfun` (`src/metafopy/models/sel_lik.py`).

Fix direction: decide whether `selmodel` rejects a `delta[0]` other than 1 on a step-function model, or normalises with the user's `delta[0]` (which
departs from metafor). Name the decision in `docs/differences.md` under "Where metafor departs from its reference". A change to `src/metafopy/` goes
through a spec first.

## ls.confint.scale.ML.bcg stays an expected failure with no decision

<!-- doc-counts: accept-start found on 2026-11-16 - a dated record, not a claim about the present -->
Found on 2026-11-16 by a TODO review. Task list `agents/tasks/ls-profile-crossings/tasks-1.md` (its closing notes) left the case
`ls.confint.scale.ML.bcg` in `src/tests/correctness/cases/location_scale.toml` as `status = "mismatch"`: for the upper bound of `alpha.2`,
`formulas.ls_profile_ll` refits from two starts and ends on the `tau2 = 0` branch at a = 1.3169166758 (2 (ll_max - ll) 3.8686, above the cut-off
3.8415), where metafopy keeps the continuous branch (3.6850) and returns the interval end 1.3598 with a `>` sign. Every other mismatch case is accepted as
a source difference or has an item; this one has neither.
<!-- doc-counts: accept-end -->

Fix direction: find which branch is the profile likelihood's maximum at a = 1.3169 (refit over a grid of starts, compare both log-likelihoods). If the
`tau2 = 0` branch is higher, metafopy's profile misses it and is a defect to fix; if the continuous branch is, the reference refit is wrong, so fix
`formulas.ls_profile_ll` and set the case to `pass`.

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
