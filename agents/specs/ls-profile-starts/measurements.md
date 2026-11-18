# Measurements for the profile starts of location-scale models

<!-- doc-counts: accept-start measured on 2026-11-18 - a dated record, not a claim about the present -->
Measured on 2026-11-18 at metafopy `dcfb11c` on the development Mac (R 4.6.2, metafor 5.2-1 from `/Users/simonb/src/metafopy/.audit/Rlib`), in a copy of
the tree outside the repository (`git archive HEAD`) whose `profile_starts` and `profile_fit` were switched by an environment variable. Probes and raw
output: `/private/tmp/claude-501/-Users-simonb-src-metafopy/e0e74c58-f9bf-4c48-9388-c492bb10b331/scratchpad/lsstarts/` (`why.py`, `basin.py`, `bcg.py`,
`fine.py`, `sweep.py`, `cmp.py`, `brute.py`, `brute3.py`, `sweep-*.json`, `pytest-*.log`). Values are `2 (ll_max - ll)` unless named otherwise; the
cut-off is 3.841458820694124.

## Why the refit falls onto the plateau

ML fit of `dat.bcg` log risk ratios, `mods = scale = ~ ablat`, log link: `alpha` = (-4.9575, 0.0386), `ll_max` -7.539290. Refits of `profile_fit` at
`alpha.2 = a` from each start of `profile_starts` (`why.py`; the start's `alpha.1` at a = 1.36 in the header, the `alpha.1` reached in brackets, P the
plateau 3.868641):

| a | default | own (-4.96) | line 0.25 (-16.0) | line 0.5 (-27.1) | line 0.75 (-38.1) | line 1.0 (-49.2) |
|---|---|---|---|---|---|---|
| 1.30 | 3.644455 (-71.1) | 3.644455 (-71.1) | 3.644455 (-71.1) | 3.644455 (-71.1) | 3.644455 (-71.1) | 3.644455 (-71.1) |
| 1.317 | P (-88.4) | P (-88.4) | 3.685207 (-72.0) | P (-88.3) | 3.685207 (-72.0) | P (-88.4) |
| 1.35 | P (-93.1) | P (-93.0) | 3.765070 (-73.8) | P (-93.0) | P (-93.3) | P (-93.0) |
| 1.36 | P (-94.1) | P (-93.8) | P (-94.1) | P (-94.1) | P (-94.0) | P (-94.0) |
| 1.38 | P (-95.4) | P (-95.7) | P (-95.4) | P (-95.4) | P (-95.7) | P (-95.4) |

The continuous branch has `alpha.1` near -74 to -75 there (Task 1 of `agents/tasks/ls-confint-scale-branch/tasks-1.md`: -74.314 at 1.36, -75.382 at
1.38), beyond the far end of the line (about -49). Refits from a fixed `alpha.1` start (`basin.py`, `refit_ls` with `alpha_init = (a0, a)`) reach the
continuous branch only from -75 and -70 at every `a` of 1.36, 1.37, 1.38 and 1.385, and from a few scattered starts (-60, -35, -25, -20) at some of
them; every other start from -120 to 0 in steps of 5 ends on the plateau 3.868641. So the starts end on the plateau because none lies near the branch.

## Candidate starts on `ls.confint.scale.ML.bcg`

`profile_fit(res, 1, a)` and `confint(res, alpha=1, tol=1e-12)` (`bcg.py`):

| variant | 1.317 | 1.36 | 1.37 | 1.38 | 1.3812 | 1.39 | 1.40 | upper bound | time of `confint` |
|---|---|---|---|---|---|---|---|---|---|
| now (line weights 0.25 to 1.0) | 3.685207 | 3.868641 | 3.868641 | 3.868641 | 3.868641 | 3.868641 | 3.868641 | 1.3598266 | 2.36 s |
| line weights also 1.5, 2.0 | 3.685207 | 3.789459 | 3.813933 | 3.838488 | 3.841440 | 3.863123 | 3.868641 | 1.3812075 | 1.64 s |
| reference shifts (-130 to 10 by 10, log link) | same as the row above | | | | | | | 1.3812075 | 2.91 s |
| continuation, 4 steps from the estimate | same as the row above | | | | | | | 1.3812075 | 1.75 s |
| continuation, 8 steps from the estimate | 3.685207 | 3.868641 | 3.868641 | 3.868641 | 3.868641 | 3.868641 | 3.868641 | 1.3598266 | 4.08 s |

The reference (`formulas.ls_profile_bounds`) gives 1.38120753377 and the crossing of Task 1's grid of starts is 1.38120753375. Over `a` from 1.2 to 1.6 in
steps of 0.005 (`fine.py`, ML and REML fits) the profile is below the reference's `ls_profile_ll` by more than 1e-6 at no point with weights (1.5, 2.0),
with (1.25, 1.5, 1.75, 2.0, 2.5, 3.0) or with the shifts. Weight 1.5 alone also passes; 1.25 alone and 2.0 alone do not (the ML fit at 1.375, 0.0424
below; the REML fit at 1.40 and 1.475, 0.2312 and 0.0363 below), so the start at 1.5 does the work: its `alpha.1` is about -71, inside the basin.

## Every output that moves (line weights also 1.5, 2.0)

`sweep.py` ran `confint(fit, alpha=j, tol=1e-12)` and `profile(fit, alpha=j)` (20 steps) for every free scale coefficient of every location-scale fit of
`src/tests/correctness/cases/location_scale.toml` (cases `ls.*`, `property.ls.*`, `published.vl2022.*`, deduplicated by data and arguments) and of
`FITS` in `src/tests/test_rma_ls.py` (no `optbeta`). A move is a change above 1e-9.

| fit, coefficient | output | now | with the new starts | checked against |
|---|---|---|---|---|
| ML `dat.bcg`, log link, `alpha.2` (`ls.fit.ML.log.bcg`, the four `ls.test.*.ML.bcg`, `FITS.bcg_ablat_ml`, `ls.confint.scale.ML.bcg`) | `ci.ub` | 1.3598266 (a jump) | 1.3812075 | reference 1.38120753377 |
| ML `dat.bcg`, identity link, `alpha.2` (`ls.fit.ML.identity.bcg`) | `ci.lb` | -0.0082027 `<` | -0.0135631 | 1-D brute force: crossing to 3.6e-11 |
| same model, `scale = ~ I(ablat/10)` (`FITS.bcg_year_identity`), `alpha.2` | `ci.lb` | -0.0820267 `<` | -0.1356308 | 1-D brute force: 3.0e-12 |
| REML `dat.bcg`, identity link, `alpha.1` (`ls.fit.REML.identity.bcg`) | `ci.lb`; profile | -5.6e-16 `<`; 2 points NaN | -0.2686286; the 2 points fitted | 1-D brute force: 2.2e-15; the points to 1e-6 |
| same fit, `alpha.2` | `ci.lb`; profile | NaN (not found); 9 of 20 points NaN | -0.0288299; the 9 points fitted | 1-D brute force: 2.0e-12; the points to 1e-6 |
| `FITS.bd_identity_ni` (`dat.bangertdrowns2004`, identity link, `scale = ~ meta + imag`), `alpha.2` | `ci.lb`; profile | -1.1e-16 `<`; 4 points NaN | -0.0048896; the 4 points fitted | multi-start SLSQP: 4.2e-12; the points to 1e-6 |
| `published.vl2022.table1.wald`, `alpha.1` | profile ll at -4.849 | -12.885975 | -12.731085 (+0.155) | higher refit of the same likelihood |
| `published.vl2022.table3.ML.mixed`, `alpha.1` | profile ll at -2.882 | -14.308188 | -14.288765 (+0.019) | higher refit of the same likelihood |
| `FITS.bd_3_unweighted`, `FITS.bd_3_weights` | bounds; profile ll | | at most 2.6e-08 (bounds), 4.4e-07 (ll), every move upwards | optimiser noise |

On the identity-link rows the bound shown now with `<` is not a crossing: the brute-force profile at it is 1.53 (ML), 3.65 (REML) and 0.51 (`bd_identity_ni`)
below the cut-off. The profile there was NaN because every start of `profile_starts` left a `tau2_i` negative and was dropped, so the search moved its end
towards the estimate (`eptries`) and stopped. R metafor 5.2-1 gives for the ML identity fit `alpha.2` `<-0.0082 >0.0082` and for the REML identity fit `NA`
for both lower bounds ("Cannot obtain lower bound"). Every other output of the sweep is unchanged.

## Tests and run time

The tests that import `metafopy.methods.ls_profile` or call `confint`/`profile` on a location-scale fit (`test_rma_ls.py`, `correctness/test_derived.py`,
`test_formulas_ls.py`, `test_properties.py`, `test_published.py`, `test_audit5_numerics.py`, `test_audit8_args.py`, `test_profile_parallel.py`,
`test_profile_plots.py`, `test_audit13_fixes.py`, `test_audit3_misc.py`, `test_audit5_plots.py`, `test_audit5_printing.py`, `test_audit8_crashes.py`,
`test_audit8_fields.py`, `test_audit8_values_a.py`, `test_emmeans.py`): now 1559 passed, 14 xfailed in 825 s; with the new starts 1559 passed, 13 xfailed
and one failure, `test_derived[ls.confint.scale.ML.bcg]` as `XPASS(strict)`, in 839 s. No R parity fixture fails.

| run | now | weights (1.5, 2.0) | weights (1.25 to 3.0) | shifts | continuation, 4 steps |
|---|---|---|---|---|---|
| `pytest src/tests/test_rma_ls.py` (twice each) | 34.5 s | 38.9 s, 38.7 s (40.2 s in a second copy) | 45.5 s | | |
| `pytest src/tests/correctness/test_derived.py -k ls.confint.scale` | 44.5 s, 45.0 s | 44.9 s, 45.3 s | | | |
| the sweep (four sweeps run side by side) | 348 s | 449 s | | 841 s for the first two thirds (217 s now) | 576 s |

The continuation of 4 steps moves the log-link bound but none of the identity-link outputs; the shifts are for the log link only and also leave them.
<!-- doc-counts: accept-end -->
