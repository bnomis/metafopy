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

Spec: none (small item); tasks: `agents/tasks/audit-ledger-gaps/tasks-1.md`.

## The audit lists a selmodel logistic Hessian element that is zero in both R and metafopy

<!-- doc-counts: accept-start found on 2026-11-15 - a dated record, not a claim about the present -->
Found on 2026-11-15 by the full audit `.audit/runs/20261115-160207` (`findings.json`, `new` list): `options/selmodel/uni_factor/type="logistic"` and
`type="logistic1"` have a `value-diff` on `$hessian`, element 11: R -1.15258004719678e-08, Python 2.69121241741294e-15 (relative 1). Every other element
of the matrix agrees.
<!-- doc-counts: accept-end -->

Cause: both values are zero to rounding, so the relative difference is 1. The ledger entry for `options/selmodel/*` `$hessian` covers differences with detail
`*(relative 0.00*` only. Not yet checked that the other elements agree to that level on both cases.

Fix direction: check the two cases in R and Python, then widen the `detail` of that entry or add an entry for these two cases, with the values and date.

Spec: none (small item); tasks: `agents/tasks/audit-ledger-gaps/tasks-1.md`.

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

Spec: none (small item); tasks: `agents/tasks/audit-ledger-gaps/tasks-1.md`.

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

Spec: none (small item); tasks: `agents/tasks/audit-ledger-gaps/tasks-1.md`.

## The exact permutest of uni_mods neither refuses nor finishes in metafopy

<!-- doc-counts: accept-start measured on 2026-11-18 - a dated record, not a claim about the present -->
Measured on 2026-11-18 by Task 2 of `agents/tasks/audit-ledger-gaps/tasks-1.md` (metafopy `dd58553`; full audit run
`.audit/runs/20261118-042436`, case `options/permutest/uni_mods/exact=true`, kind `r-killed`, "R passed the memory limit of 4 GiB (killed)"): for
`rma(yi, vi, mods = ~ ablat, data = es_bcg)`, `permutest(fit, exact='i')` reports 389188800 iterations and 15567552000 bytes of arrays, just under
`MAX_PERMUTATION_BYTES` (17179869184) in `src/metafopy/methods/permutest.py`, so metafopy does not refuse. `permutest(fit, exact=True)` was still running
after about 29 minutes (cut off by the background limit) at about 275 MB resident at 20 minutes.
<!-- doc-counts: accept-end -->

Cause: the guard bounds the size of the arrays, not the run time, so a case R cannot hold in 4 GiB passes the guard and runs for a very long time. The
case stays listed as an R kill in the audit report because no entry can say what metafopy does.

Fix direction: decide whether metafopy should refuse an exact permutest whose iteration count R cannot hold (a limit on iterations, as R's "Number of
iterations requested too large." is), or finish it faster; then add the case to the `r-killed` entry of `options/permutest/ls_bcg/exact=true` in
`src/audit/data/known.toml`. Needs a spec if the guard changes.

Spec: none yet.

## Six ledger entries of known.toml match nothing in the audit

<!-- doc-counts: accept-start found on 2026-11-18 - a dated record, not a claim about the present -->
Found on 2026-11-18 by the full audit `.audit/runs/20261118-042436` (`wrong-or-missing.md`, "Not covered", "Stale ledger entries"), and already listed by
`.audit/runs/20261115-160207`: the entries of `src/audit/data/known.toml` matching `scripts/misc_selmodel/*` (`*sav <- selmodel(res, type = "beta")*`),
`scripts/misc_coef_se/*` (`*sel <- selmodel(res, type = "beta")*`), `scripts/misc_selmodel/test/41` (`*[[]out <- capture.output(print(sav))[]]*`),
`options/predict/ls_na/newscale=*`, the `ls_year`/`permutest_ls*` print-diff `*QM(df = 2) = 15.610[45]*` and the `print/sel_bcg` print entry
`98.1051`. None shows metafopy's output wrong (a ledger entry that matches nothing hides nothing).
<!-- doc-counts: accept-end -->

Cause: the difference each entry accepted no longer occurs in the pinned R 4.6.2 with metafor 5.2-1 (the same kind of cause as the stale `delta.1`
entry removed on 2026-11-03), or metafopy's output moved; not traced entry by entry.

Fix direction: for each entry, run its case in R and metafopy; remove the entry where the two now agree, narrow its `detail` where they differ in
another way, and rerun `bin/metafor-audit.sh` to see the report list no stale entry.

Spec: none (small item).

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

Spec: none (small item); tasks: `agents/tasks/ls-confint-scale-branch/tasks-1.md`.

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
