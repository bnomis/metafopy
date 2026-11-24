# Measurements: location-scale fits stable across platforms

Dated records for [the spec](spec.md). Never regenerate them: each figure belongs to its date and commit.

<!-- doc-counts: accept-start measured on 2026-11-24 - a dated record, not a claim about the present -->

## CI

Python 3.14.8, numpy 2.5.4, scipy 1.18.1 on every runner, `uv sync --locked --all-extras`, `uv run pytest --mpl --cov --cov-report=term-missing`.

| Test | Run 45270718992 (`93fea02`) | Run 45335133105 (`00f7fda`) |
|---|---|---|
| `test_page_examples_run[differences.md]` | fails on ubuntu-26.04, macos-26, windows-2025-vs2026 | fails on all three |
| `test_derived[ls.vif.reestimate.ML.bcg2]` | `LinAlgError: Singular matrix` on all three | on ubuntu and macos; passes on windows |
| `test_derived[ls.vif.REML.bcg2]` | ubuntu: location vif `4.723135238933293`, reference `4.723144269851787`; passes on macos and windows | ubuntu: `Singular matrix` in the reference fit; windows: vif `4.7231618202910175`, reference `4.723144270692351`; passes on macos |
| `test_derived[ls.permutest.location.exact.ML.z.bcglat]` | passes on all three | windows: pval `0.61693862` (794/1287), reference bounds `[0.61072261, 0.61616162]` (786/1287 to 793/1287, 7 permutations without a reference fit) |

`00f7fda` changes only `TODO.md` from `93fea02`, so the same runner gives different outcomes from one run to the next. The `Singular matrix` errors
are raised by `numpy.linalg.inv` with `err = 'invalid value', flag = 8`, at `formulas.py:3111` (`'va': numpy.linalg.inv(ls_information(...))`).

## Development Mac (arm64), `main` at `00f7fda`

R is 4.6.2 with metafor 5.2-1 from `.audit/Rlib`, OpenBLAS 0.3.34. The probes are scratch scripts outside the repository, not kept.

### The `differences.md` example (`~ ablat * alloc`, `scale = ~ ablat`, three NA studies)

- After the NA studies are dropped, k = 10 and the location model has five columns; study 4 (row 4 of the ten) has hat value 1 (`1 - h` is
  `1.11e-15`; the next largest hat value is 0.921, `1 - h` = 0.079).
- Raw HE residual of study 4: metafopy `2.309e-14`, R `-7.438e-15` (`yi.f - X.f %*% beta` of R's own HE fit). R's `rstandard.rma.uni` and
  metafopy's `marginal_residuals` set a residual under `100 * .Machine$double.eps` = `2.220e-14` to 0. R's is under it, so `log(0)` gives
  `alpha.init` NaN and R stops with "No missing values allowed in 'alpha.init'."; metafopy's is just over it, so it fits.
- With `yi` multiplied by `1 + 1e-15 * N(0, 1)` (numpy `default_rng(1)`), metafopy raises R's error on 9 of 9 perturbed inputs and fits only the
  unperturbed one. R with `1e-15` noise (`set.seed(1:5)`): residual of study 4 between `-1.3e-14` and `0`, every one under the snap.
- With the residual of study 4 forced to 0 (monkeypatched `marginal_residuals`), metafopy raises "No missing values allowed in 'alpha_init'.":
  the CI failure, reproduced.
- The end point depends on the start: with the residual left at `2.3e-14` (start `[-38.24, -37.80]`) the fit ends at alpha `[-6.19, -1.26]`,
  log-likelihood `0.0858`, "Error when trying to invert the Hessian." (beta `0.282, -0.024, 0.213, -0.134, -0.013`, as `differences.md` prints);
  with study 4 left out of the start regression (start `[-1.33, 1.03]`) or its residual at `1e-15`, at alpha `[2.47, -0.135]`, log-likelihood
  `-4.1456`, beta `2.924, -0.084, -2.262, -0.536, 0.044`.

### The reference fit of the `bcg2` cases (`mods` and `scale` `~ ablat + year`, `year` not centred)

`ls_reference_fit` runs BFGS from five starts (intercepts `-1.40`, `-3.70`, `0.90`, `-6.01`, `3.20`, slopes 0). Per start, after `ls_newton_polish`:

| Case | Starts ending at the interior optimum (score) | Starts ending at a degenerate point |
|---|---|---|
| `ls.vif.REML.bcg2` | 1st (`1.03e-05`, rejected), 2nd (`8.19e-06`, rejected), 4th (`9.30e-07`, kept) | 3rd and 5th: one BFGS step to `year` slope `-1.00986`, `tau2 = exp(-2000) = 0`, score exactly 0, kept |
| `ls.vif.reestimate.ML.bcg2` | 2nd (`7.59e-07`, kept), 4th (`3.28e-06`, rejected) | 1st, 3rd, 5th, as above, kept |

`LS_SCORE_TOLERANCE` is `1e-6`. With it set to `5e-7`, both cases raise `LinAlgError: Singular matrix` locally: the CI failure, reproduced (only
degenerate points are kept, and their information is zero). The interior optimum: REML alpha `[-151.3288, 0.10020, 0.073875]`, ML
`[-129.3312, 0.17954, 0.060688]`; information condition number `1.3e11` (REML) and `1.1e11` (ML) in the raw `Z`, `8.5` and `6.7` with the
non-constant columns of `Z` centred and scaled.

A prototype of the reference fit in centred and scaled coordinates (pytest plugin outside the repository: fit on the standardised `Z`, keep a start
only when the information is positive definite, no final Nelder-Mead for the log link, map `alpha` and `va` back): both cases pass with
`LS_SCORE_TOLERANCE` at `1e-6`, `5e-7` or `1e-9` (location vif REML `4.7231443089` at `1e-6`, `4.7231442702` at `1e-9`; ML `5.4154230352`,
`5.4154230210`), and `pytest src/tests/correctness/test_derived.py -k "ls." -m "not slow"` passes with it as without it.

### metafopy's location vif of `ls.vif.REML.bcg2`

| Fit | Location vif | Log-likelihood | Largest score (raw `Z`) |
|---|---|---|---|
| reference (committed) | `4.72314426735735` | `-7.602255954644` | `9.3e-07` |
| metafopy, unperturbed | `4.723148585534548` (9.1e-07 relative from the reference: passes at 1e-6) | `-7.602255955362` | `2.0e-04` |
| metafopy, `yi * (1 + 1e-15 N(0, 1))`, five draws | `4.723136993` to `4.723154747` (up to 2.2e-06 relative) | `-7.6022559549` to `...5527` | up to `5.5e-03` |
| R metafor 5.2-1, unperturbed | `4.7231321376` (2.6e-06 relative) | `-7.602255955097` | |
| R, `1e-15` noise, `set.seed(1:4)` | `4.7231328` to `4.7231492` | | |

So the CI values (`4.723135`, `4.723162`) lie inside the spread that rounding-size changes of the input give. The ML case is not sensitive: R
`5.4154228197`, with noise `5.4154228172` to `5.4154228226`. Newton steps on the analytic score and information in the standardised coordinates,
started from metafopy's end point, bring every one of the six REML fits to location vif `4.7231442705` to `4.7231442712` (score at most
`1.6e-10`, last step at most `6e-10`) and every ML fit to `5.4154230208` to `5.4154230219`.

### The exact permutation test of `ls.permutest.location.exact.ML.z.bcglat`

- metafopy (local): pval `[0.61693862, 0.23931624]` (794 and 308 of 1287), the same as on CI windows. Its tie handling is metafor's:
  `mean(abs(zval.perm) >= abs(zval) - comptol, na.rm = TRUE)` with `comptol = sqrt(.Machine$double.eps)` (`permutest.rma.ls.R`), in
  `ls_permutest.py` and in `formulas.ls_exact_permutation_p` alike.
- Reference (local): intercept hits 786, 12 permutations without a reference fit, bounds 786 to 798; CI windows had 7 without a fit, bounds 786
  to 793. The difference is not a tie: the nearest permutation to the observed `|z|` other than the observed one is 7.4e-05 away.
- 77 of the 1287 permuted models have no finite maximum (`tau2` tends to 0 in one `lat` group, alpha near `[-37, 37]`): the information at the
  reference's end point has eigenvalue ratio min/max at most `1.1e-13`, against at least `4.3e-03` for the other 1210. The reference keeps 65 of
  them as fits (their `vb`, and so the location `z`, are finite) and loses 12 to `LinAlgError` in `numpy.linalg.inv`; which ones raise depends on
  the platform. metafopy's `z` on some of the 77 differs from the reference's (804, 839, 1028, 1042 and 1115 among them: both stop somewhere on the ridge), and 19 of metafopy's 794 hits are
  among the 77.
- With all 77 counted as having no reference fit, the bounds are 774 to 851 (0.6014 to 0.6612), and metafopy's 794 lies inside.
- On 8 of the 1210 regular permutations (indices 68, 264, 433, 651, 805, 979, 1110, 1278 in `distinct_label_orders` order) metafopy's `z` differs
  from the reference's by more than 1e-4: metafopy and R stop at the same lower local maximum (permutation 68: log-likelihood `-12.58248229`, alpha
  `[-0.778, -0.951]`, in both), the reference at a higher one (`-12.57323737`, alpha `[-3.042, 2.526]`); 264 and 433 likewise. Out of scope here;
  see the spec's Motivation.

<!-- doc-counts: accept-end -->

## Task 4: the log-link Newton polish (`polish_log`), development Mac (arm64), 2026-11-24

<!-- doc-counts: accept-start measured on 2026-11-24 - a dated record, not a claim about the present -->

Largest relative change of each log-link fit of `src/tests/test_rma_ls.py` (`FITS`) between the unpolished BFGS stop and the polished fit, over `alpha`,
`beta` and `se_alpha`; every fixture still passes at its committed tolerance and none is regenerated.

| Fit | `alpha` | `beta` | `se_alpha` |
|---|---|---|---|
| `bd_poly` | 1.3e-07 (largest `alpha`) | 4.3e-10 | 3.4e-08 |
| `mc_factors` | 1.5e-08 | 1.2e-07 (largest `beta`) | 2.1e-08 |
| `mo_quality` | 4.1e-08 | 2.9e-09 | 2.5e-07 (largest `se_alpha`) |
| `bcg_ablat`, `bcg_ablat_t` | 2.3e-08 | 1.7e-11 | 3.6e-08 |
| `bd_3_ml` | 1.0e-08 | 2.0e-09 | 1.7e-08 |
| every other fit with a free scale | at most 3e-09 | at most 6e-10 | at most 4e-09 |
| `bcg_int_scale`, the three `optbeta` fits | 0 (not polished) | 0 | 0 |

The exact p of `ls.permutest.location.exact.ML.z.bcglat` after the polish: `[0.61693862, 0.23931624]` (794 and 308 of 1287), as before (794/1287).

<!-- doc-counts: accept-end -->

## Task 5: the exact permutation bounds with Tasks 3 and 4 in, development Mac (arm64), 2026-11-24

<!-- doc-counts: accept-start measured on 2026-11-24 - a dated record, not a claim about the present -->

`pytest src/tests/correctness/test_derived.py -k "permutest and exact"`: the scale case `ls.permutest.scale.exact.ML.z.bcglat` passes; the location
case `ls.permutest.location.exact.ML.z.bcglat` fails.

- Permutations without a reference fit: 68 of 1287, not the 77 expected. The task stops as blocked on the difference (spec, Error handling).
- Reference bounds on the location p-values: `[0.60761461, 0.18570319]` to `[0.66045066, 0.23853924]`. metafopy gives `[0.61693862, 0.23931624]`: the
  first coefficient lies inside, the second is `7.7e-4` above the upper bound (308 of 1287 against at most 307).
- Reason, not yet traced: Task 3's reference keeps a start only where `ls_is_maximum` holds, which keeps 9 of the 77 permutations that had no fit
  in the earlier classification, so those now count as regular and their `z` may differ from metafopy's (the ridge fits of the earlier
  classification). Which of the 9 add a hit to the reference or move metafopy's is to be found before any change.

<!-- doc-counts: accept-end -->
