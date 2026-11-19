# Measurements: a feasible start for identity-link profile refits

<!-- doc-counts: accept-start measured on 2026-11-19 - a dated record, not a claim about the present -->
Measured on 2026-11-19 at `f1fdc9b`, on the ML and REML identity-link fits of `dat.bcg` (`escalc` RR, `mods = scale = ~ ablat`). The probes patched
`profile_starts` from a pytest plugin outside the repository; no file in it was changed.

## Starts and refits below the alpha.1 estimate (ML)

The estimate is `alpha = (-0.02666, 0.00205)`; the lower search limit is -0.29325. At `alpha.1 = v` the feasible `alpha.2` is `>= -v / min(ablat)`.

| alpha.1 | starts kept by `profile_starts` | `profile_fit` | feasible alpha.2 from |
|---|---|---|---|
| -0.2932 | default only | None | 0.022554 |
| -0.2600 | default only | None | 0.020000 |
| -0.2200 | default only | None | 0.016923 |
| -0.2143 | default only | None | 0.016485 |
| -0.1500 | default only | None | 0.011538 |
| -0.0500 | default only | None | 0.003846 |
| -0.0300 | default only | None | 0.002308 |

With the feasible start, `profile_fit` at the reference bound -0.214322988803 gives the log-likelihood -9.060018, the crossing level.

## confint bounds and run time

`confint(x, tol=1e-12)`, both coefficients; time on the development Mac.

| fit | variant | alpha.1 | alpha.2 | time |
|---|---|---|---|---|
| ML | today | (NaN, >0.239928) | (-0.013563, 0.016486) | 0.39 s |
| ML | start added to every refit | (-0.214323, >0.239928) | (-0.013563, 0.016486) | 1.14 s |
| ML | start added when only the default is left | (-0.214323, >0.239928) | (-0.013563, 0.016486) | 0.49 s |
| REML | today | (-0.268629, 1.585647) | (-0.028830, 0.021181) | 1.92 s |
| REML | start added to every refit | (-0.268629, 1.585647) | (-0.028830, 0.021181) | 4.50 s |
| REML | start added when only the default is left | (-0.268629, 1.585647) | (-0.028830, 0.021181) | 1.89 s |

The ML lower bound of alpha.1 with the start is -0.21432298821997195; the reference `ls_profile_bounds` gives -0.214322988803.

## Test suite

`pytest src/tests/test_rma_ls.py src/tests/correctness`, each variant run once:

| variant | result |
|---|---|
| today | 1501 passed, 1 skipped, 19 xfailed |
| start added to every refit | 1 failed, 1501 passed, 1 skipped, 18 xfailed |
| start added when only the default is left | 1 failed, 1501 passed, 1 skipped, 18 xfailed |

The one failure in both variants is `ls.confint.scale.ML.identity.bcg.alpha1_lb`, a strict expected failure that now passes.
<!-- doc-counts: accept-end -->
