# Correctness tests for the rma_uni diagnostics: measurements

Measured for the [spec](spec.md) before it was written. The probe scripts lived in the session's scratch directory, outside the repository; each
figure is the largest relative difference `max |a - b| / max(1, |b|)` over every value the function returns, unless it is a printed value.

<!-- doc-counts: accept-start measured on 2026-11-20 - a dated record, not a claim about the present -->
## Printed values of Viechtbauer and Cheung (2010) against metafopy

Measured on 2026-11-20 at commit `1c93ae3`, macOS (development machine). The paper's first example (section 5.1, Table I, which is `dat.bcg`): log
risk ratios, `mods = ~ ablat_c + year_c` with latitude centred at 33 and year at 1966, `tau2` by the generalised DerSimonian-Laird estimator of its
equation 9 (`method = "DL"`). Study numbers are the paper's (row 4 is Hart and Sutherland).

| Quantity | Paper (section 5.1) | metafopy | R metafor 5.2-1 |
|---|---|---|---|
| `tau2`, `QE` | 0.0790, 28.33 | 0.07904, 28.325 | - |
| `beta` (CI) | -0.71 (-0.93, -0.49), -0.03 (-0.05, -0.01), 0.00 (-0.03, 0.03) | -0.711 (-0.930, -0.493), -0.029 (-0.046, -0.011), 0.001 (-0.025, 0.026) | - |
| study 4: `h`, `t`, `D` | 0.82, -1.51, 9.64 | 0.8226, -1.5056, 9.6362 | 0.8226, -1.5056, 9.6362 |
| study 7: `h`, `t`, `D` | 0.07, -2.65, 0.41 | 0.0654, -2.6470, 0.4117 | 0.0654, -2.6470, 0.4117 |
| study 13: `h`, `t`, `D` | 0.15, 2.06, 0.59 | 0.1507, 2.0617, 0.5930 | 0.1507, 2.0617, 0.5930 |
| `R_i` (eq 27) for 4, 7, 13 | 14.50, 43.24, 26.18 | 14.504, 43.236, 26.177 | from `tau2.del` 0.0676, 0.0449, 0.0583 |
| DFBETAS of study 4 | -0.87, -2.86, -2.47 | -0.8677, -2.8575, -2.4662 | -0.8677, -2.8575, -2.4662 |
| mean-shift dummy for study 4: estimate, SE, z | -1.08, 0.716, -1.51 | -1.0783, 0.7162, -1.5056 | - |

Every printed value is reproduced within half its last digit. With `method = "REML"` the same model gives `tau2` 0.1108 and `D4` 7.32, so the
paper's figures are the DL fit. The second example (section 5.2, Table II, the 26 rows of `dat.bangertdrowns2004` with grade 3 or 4, after 1926 and
with `length` and `meta`, identical to Table II) reproduces `tau2` 0.0472 and `QE` 44.14 (DL, `~ length + meta + college`), names studies 7 and 25
as the outliers (metafopy: `t` = -2.363 and 2.762, the two largest), and after deleting both gives `tau2` 0.0144, `QE` 26.07 and the
metacognition coefficient 0.31 (0.03, 0.59), all reproduced; it prints no per-study diagnostic value.

## Printed BLUPs of Raudenbush (2009) against metafopy

The Handbook of Research Synthesis chapter 16, Table 16.2 and section 16.5.6.1, `dat.raudenbush1985`, REML, no moderators. The extracted text drops
minus signs, as it does for the coefficient -0.157 that the existing published case reads with its sign.

- `theta*_4` 0.21, range of `theta*` (-0.03, 0.25), reliability `lambda_4` 0.12, mean reliability 0.37: metafopy's `blup` 0.2144, (-0.0293,
  0.2485), and `tau2 / (tau2 + v)` 0.1192 and 0.3715. R metafor 5.2-1: `blup` 0.2143968163, `ranef` 0.130688581 for study 4, as metafopy.
- The mixed-effects model (`x` = weeks capped at 3): the book prints the range (-0.07, 0.41); metafopy gives (-0.0646, 0.4072), and the book's own
  rounded coefficients give 0.407 - 3 x 0.157 = -0.064. The lower end is not reproduced from the book's own numbers, so it is not used.

## Diagnostics against brute-force refits

Same date and commit. The reference refits each deletion with `correctness.formulas.rma_tau2` and weighted least squares, and computes every
quantity from the paper's equations: Cook's distance (eq 20), DFBETAS (eq 23, the variance on all rows at `tau2(-i)`), the deleted residual and
`rstudent` (eqs 15, 16), DFFITS (eq 19), COVRATIO (eq 25), `tau2(-i)`, `QE(-i)`, the hat values (eq 13), `rstandard` (eq 14); `ranef` and `blup` by
Henderson's mixed model equations with one random effect per study. `tight` is `control = {threshold: 1e-13, tol: 1e-13, maxiter: 1000, tau2_max:
1e6}`, as the existing `inference.toml` REML cases use.

| Fit (`dat.bcg` unless named) | control | Cook | DFBETAS | rstudent `z` | DFFITS | cov.r | tau2.del | hat | rstandard | ranef | ranef `se` | blup |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| REML, no mods | default | 2.1e-07 | 3.5e-07 | 4.1e-07 | 3.8e-07 | 4.2e-07 | 2.1e-07 | 4.0e-09 | 1.1e-07 | 6.1e-08 | 3.2e-08 | 5.5e-08 |
| REML, `~ ablat` | default | 3.7e-05 | 3.9e-05 | 6.2e-05 | 4.8e-05 | 9.7e-05 | 1.1e-05 | 6.1e-06 | 3.8e-05 | 2.1e-05 | 1.0e-05 | 2.4e-05 |
| ML, `~ ablat` | default | 1.2e-03 | 6.2e-04 | 7.4e-04 | 1.0e-03 | 1.7e-03 | 4.0e-05 | 2.0e-05 | 9.3e-05 | 3.4e-05 | 1.8e-05 | 3.9e-05 |
| `dat.raudenbush1985` REML, no mods | default | 7.5e-04 | 1.1e-03 | 7.6e-04 | 1.1e-03 | 7.8e-04 | 3.4e-05 | 9.4e-06 | 1.4e-04 | 5.3e-05 | 2.4e-05 | 6.0e-05 |
| REML, no mods | tight | 4.7e-15 | 7.0e-15 | 4.4e-15 | 7.2e-15 | 6.3e-15 | 2.5e-15 | 8.3e-17 | 1.1e-15 | 2.0e-15 | 1.1e-16 | 1.4e-15 |
| REML, `~ ablat` | tight | 3.5e-13 | 4.3e-13 | 6.7e-13 | 5.1e-13 | 9.3e-13 | 1.2e-13 | 5.5e-14 | 3.4e-13 | 1.9e-13 | 8.9e-14 | 2.1e-13 |
| ML, `~ ablat` | tight | 1.2e-11 | 6.1e-12 | 7.2e-12 | 1.0e-11 | 1.6e-11 | 3.9e-13 | 2.4e-13 | 1.1e-12 | 4.1e-13 | 2.2e-13 | 4.7e-13 |
| REML, `~ ablat_c + year_c` | tight | 4.0e-13 | 2.9e-13 | 3.7e-13 | 5.1e-13 | 1.2e-12 | 8.8e-14 | 3.1e-14 | 1.0e-13 | 6.8e-14 | 3.4e-14 | 7.4e-14 |
| `dat.raudenbush1985` REML, no mods | tight | 8.2e-12 | 1.2e-11 | 8.2e-12 | 1.2e-11 | 8.5e-12 | 3.6e-13 | 9.8e-14 | 1.4e-12 | 5.5e-13 | 2.5e-13 | 6.2e-13 |
| DL, `~ ablat` | default | 1.5e-14 | 1.5e-14 | 7.9e-15 | 1.2e-14 | 2.5e-14 | 1.9e-15 | 6.7e-16 | 4.5e-15 | 9.7e-15 | 1.0e-15 | 7.9e-15 |
| DL, `~ ablat_c + year_c` (the paper's fit) | default | 7.5e-15 | 7.5e-15 | 4.4e-15 | 7.0e-15 | 3.2e-14 | 1.7e-15 | 1.3e-15 | 5.6e-15 | 2.4e-15 | 8.3e-16 | 2.9e-15 |
| Table II, DL, `~ length + meta + college` | default | 5.8e-15 | 6.8e-15 | 1.3e-15 | 4.7e-15 | 3.8e-15 | 4.5e-17 | 2.2e-16 | 1.6e-15 | 2.7e-16 | 5.6e-17 | 3.3e-16 |
| EE, `~ ablat` | default | 3.5e-14 | 3.7e-14 | 2.0e-14 | 2.0e-14 | 3.7e-15 | 0 | 4.4e-16 | 2.7e-14 | 0 | 0 | 1.3e-15 |
| EE, no mods | default | 5.9e-15 | 8.2e-15 | 1.2e-15 | 8.2e-15 | 1.1e-15 | 0 | 0 | 4.4e-16 | 0 | 0 | 5.6e-17 |

`QE.del` agrees to 1.8e-14 or better in every row (it does not depend on `tau2`). With the default `threshold` (1e-5) metafopy's REML and ML refits
stop before the root, so the iterative cases use `tight`; with it every difference is below the harness's `DERIVED_TOLERANCE` (1e-10). The closed
form fits (DL, EE) agree to rounding with the defaults. Each fit with its reference took under 0.2 s.

Further cases, same date, all against the same kind of reference:

- `tau2 = 0.05` fixed, `~ ablat`: every refit keeps `tau2`, so `tau2.del` is 0.05 throughout; Cook 1.1e-14, DFBETAS 1.1e-14, `rstudent` 6.9e-15.
- `cooks_distance(btt=[1, 2])` on the paper's fit (the two moderators): 3.3e-15.
- `ranef` of REML `~ ablat` with `test = "t"` and `tight` against `formulas.mv_blups` with one `~ 1 | study` term (`V = diag(v)`): BLUPs 8.4e-15,
  `se` 6.1e-16, `pi.lb` and `pi.ub` with the t quantile on `k - p` degrees of freedom 9.8e-15 and 7.5e-15, `beta` from the equations 1.5e-14.
- `ranef` of an EE fit: every BLUP and `se` is exactly 0 (the help page: "The BLUPs for these models will therefore be 0").

## The mean-shift outlier model

Viechtbauer and Cheung (section 3, eqs 17 and 18): adding a dummy for study `i` gives a coefficient whose z statistic is `t_i`. On `dat.bcg`,
`~ ablat`, every study in turn, the largest relative difference between `rstudent` `z` and the dummy's `zval`:

| Method | DL | REML (`tight`) | EE | ML (`tight`) |
|---|---|---|---|---|
| Difference | 1.5e-14 | 3.9e-15 | 1.1e-14 | 0.39 |

The identity holds for DL, REML and EE and not for ML, whose likelihood keeps the term `log(v_i + tau2)` of the study the dummy absorbs, so its
`tau2` differs from `tau2(-i)`; the paper's example is DL.

## Knapp and Hartung: a scaling the paper does not define

With `test = "knha"` metafopy (as metafor's `influence.rma.uni` code does) multiplies the variances in DFBETAS and the deleted residual's standard
error by `s2w`, and Cook's distance uses the fit's scaled `vb`. On `dat.bcg`, DL, `~ ablat`, against the same fit with `test = "z"`: Cook's
distance differs by up to 0.106 relative, DFBETAS by 0.080, `rstudent` `z` by 0.20. The paper does not consider the adjustment, and the
`influence.rma.uni` and `residuals.rma` help pages of metafor 5.2-1 (read 2026-11-20) do not mention it.
<!-- doc-counts: accept-end -->
