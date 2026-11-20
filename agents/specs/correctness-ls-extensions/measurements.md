# Correctness tests for the location-scale extensions: measurements

Measured for the [spec](spec.md) before it was written. The probe scripts lived in the session's scratch directory, outside the repository. Unless a
row says otherwise, a figure is the largest relative difference `max |a - b| / max(1, max |b|)` between metafopy and an independent reference coded
from the source the spec names; log-likelihoods are absolute differences.

<!-- doc-counts: accept-start measured on 2026-11-20 - a dated record, not a claim about the present -->
## Setting

Measured on 2026-11-20 at commit `936021a`, macOS (development machine), R 4.6.2 with metafor 5.2-1 from the audit's pinned library
(`.audit/Rlib`). Datasets: `bd` is `dat.bangertdrowns2004` with `imag` in both parts; `bcg` is `dat.bcg` (log risk ratios) with `ablat` in both
parts; `bd2` adds `ni / 100` (`ni100`) to both parts of `bd`, `bcg2` adds `year` to both parts of `bcg`. Log link unless named. The reference fit is
`correctness.formulas.ls_reference_fit`, the reference profile `correctness.formulas.ls_profile_ll`. Every fit with its reference took under 0.1 s,
except the profile checks (2.0 to 7.0 s for both coefficients of one model, reference included).

## optbeta (joint maximisation over beta and alpha)

| Fit | alpha | beta | tau2 | log-likelihood | vb: metafopy vs inverse joint observed information | vb: equation 10 vs the same | va: inverse joint information vs profile `va` |
|---|---|---|---|---|---|---|---|
| `bd` ML | 7.2e-09 | 1.4e-09 | 5.5e-09 | 3.6e-15 | 7.7e-08 | 3.7e-03 | 5.5e-16 |
| `bd` REML | 8.7e-09 | 5.5e-10 | 3.8e-09 | 7.1e-15 | 7.7e-10 | 2.2e-03 | 2.5e-16 |
| `bcg` ML | 3.9e-09 | 3.2e-10 | 3.5e-10 | 3.4e-14 | 1.3e-10 | 4.4e-03 | 3.3e-16 |
| `bcg` REML | 3.6e-08 | 1.8e-09 | 6.0e-09 | 2.5e-14 | 3.2e-09 | 2.0e-03 | 6.1e-15 |

The estimates are compared with the profile fit (equations 6, 7 and 9): the REML objective metafor maximises with `optbeta=True` is equation 9 with
`y'Py` replaced by the weighted residual sum of squares at a free `beta`, whose maximum over `beta` at fixed `alpha` is equation 6, so both routes
reach the same estimates. The joint observed information is coded analytically (ML: minus the Hessian of equation 5; REML: of that objective) and
agrees with central finite differences of the coded objective to 6.4e-08. metafopy's `va` from the joint fit agrees with the inverse joint
information to 4.2e-07 or better, and the alpha block of that inverse equals the inverse information of the profile likelihood (the last column).
metafopy's `vb` is the beta block of the inverse joint information; equation 10 (`(X'WX)^-1`, also Harvey 1976 equation 12) differs from it by
2.0e-03 to 4.4e-03. R metafor 5.2-1 on `bd` ML: `vb[2,2]` 0.1156763 (metafopy 0.1156658, relative difference 9.1e-05); its estimates and
log-likelihood agree with metafopy's to the printed digits under both methods.

The deviance metafopy reports equals `-2 (ll - sum(-1/2 ln(2 pi v_i)))` under ML and `-2 ll` under REML at the reference optimum to 7.1e-14, for
`optbeta` and ordinary fits alike. No publication or help page defines it.

## Fixed alpha, fixed beta, alpha_min and alpha_max

| Case | free coefficients | beta | log-likelihood | `va` of the free coefficient |
|---|---|---|---|---|
| `bd` ML, `alpha = [nan, 2.7]` | 6.2e-09 | 1.4e-09 | 0 | 7.2e-09 |
| `bd` REML, `alpha = [nan, 3.2]` | 5.0e-09 | 8.7e-10 | 3.6e-15 | 5.4e-09 |
| `bcg` REML, `alpha = [-3.5, nan]` | 1.4e-09 | 2.0e-09 | 1.4e-14 | 7.9e-12 |
| `bcg` ML, `alpha = [-6.0, nan]` | - | - | below 1e-8 (`ls_profile_ll` -7.5837572040, metafopy -7.58375720) | - |
| `bd` ML, `optbeta`, `beta = [nan, 0.05]` | 6.0e-09 (alpha), 1.3e-09 (beta[0]) | - | 1.4e-14 | se of beta[0] 1.3e-10 |
| `bcg` ML, `optbeta`, `beta = [nan, 0.02]` | 1.7e-07 (alpha), 4.7e-08 (beta[0]) | - | 7.1e-15 | se of beta[0] 9.1e-08 |

The reference holds the fixed coefficients and maximises equation 7 or 9 (fixed alpha, the constrained likelihood of equation 17) or equation 5
(fixed beta) over the rest. The fixed coefficient's standard error is NaN and `parms` drops by one. For `bcg` ML with the intercept fixed at -6.0, a
one-dimensional search started from the unconstrained slope falls onto the `tau2 = 0` plateau (-9.47); started from the fitted line, as
`ls_profile_ll` does, it gives -7.5837572040, which metafopy and R metafor (-7.58375720) reach.

Bounds (`control = {alpha_min, alpha_max}`), reference `scipy.optimize.minimize(method='L-BFGS-B')` on equation 7 or 9 within the box:

| Case | box (min; max) | reference alpha, ll | metafopy alpha, ll | alpha | R metafor 5.2-1 |
|---|---|---|---|---|---|
| `bd` ML, not binding | (-10, -10; 10, 10) | -3.3807912, 2.0502625; -17.0580546729 | the same | 6.8e-09 | - |
| `bd` ML, slope binding | (-20, -10; 20, 1.4) | -3.25688837, 1.4; -17.1856765437 | the same; -17.1856765438 | 1.9e-10 | -3.25688837, 1.4; -17.1856765453 |
| `bd` REML, slope binding | (-20, -10; 20, 1.9) | -3.18368412, 1.9; -17.0105349983 | the same | 1.2e-10 | - |
| `bcg` ML, slope binding | (-20, -1; 20, 0) | -3.37111127, 0; -7.6856655284 | the same | 3.4e-11 | - |
| `bcg` REML, slope binding | (-20, -1; 20, -0.02) | -1.76215454, -0.02; -8.1161267830 | the same | 1.1e-10 | - |
| `bcg` ML, not binding | (-10, -1.2; 0, 1.2) | -4.95749055, 0.03859326; -7.5392902501 | 0, -0.0795806; -8.6761968866 | 1.0 | stops: "Optimizer (nlminb) did not achieve convergence" |
| `bcg` REML, not binding | (-10, -1.2; 0, 1.2) | -2.54250107, -0.00073014; -8.0872779252 | 0, -0.0795806; -8.7022248643 | 1.0 | stops, the same error |

In the last two rows metafopy warns only that an estimate is "(almost) equal to" a bound. The intercept sits on its upper bound and the slope is
-0.0795806 in every such run, whatever the box; with `control = {alpha_init: [-5, 0]}` (a start inside the box) metafopy reaches the reference
maximum under both methods. With the box (-10, -1.2; 20, 1.2) or (-10, -1.2; 1e6, 1.2) metafopy ends at alpha -9.9998, -1.14 (ll -9.4736109) and R
at -9.95, -1.09 (ll -9.473611): the `tau2 = 0` plateau, 1.93 below the maximum. With bounds, neither program rescales the scale model matrix.

## weighted=False

| Fit | alpha vs reference | beta vs OLS | vb vs `(X'X)^-1 X' M X (X'X)^-1` at metafopy's `tau2_i` |
|---|---|---|---|
| `bd` ML | 6.9e-09 | 1.1e-16 | 2.8e-17 |
| `bd` REML | 8.7e-09 | 1.1e-16 | 4.3e-18 |
| `bcg` ML | 2.8e-10 | 2.7e-15 | 3.5e-17 |
| `bcg` REML | 4.5e-08 | 2.7e-15 | 2.8e-17 |

`alpha` is the weighted fit's: the likelihood does not depend on the weighting of the location estimate. The log-likelihood metafopy reports is
equation 5 at the OLS `beta` (ML; -18.14129496 on `bd`, R -18.14129527) and the restricted form with the same residual sum of squares (REML), not the
maximised likelihood (-17.05805467); the deviance follows it. R metafor 5.2-1 on `bd`: `alpha` -3.38079138, 2.05026242 (ML), `beta` 0.26777778,
-0.12777778, the same as metafopy.

## blup, ranef, rstandard

At metafopy's `tau2_i`, against Henderson's mixed model equations with `R = diag(v)`, `Z = I`, `G = diag(tau2_i)` (BLUPs, prediction error
variances from the inverse coefficient matrix, intervals with the normal or the t quantile on `k - p` degrees of freedom), and against section 2.8 of
Viechtbauer and Lopez-Lopez (2022) for `rstandard` (`Var[e_i] = (1 - h_i)(v_i + tau2_i)`; conditional: `(1 - lambda_i) e_i` over its standard error):

| Fit | `blup` pred, se, PI | `ranef` pred, se, PI | `beta` from the equations | `rstandard` marginal z, conditional z |
|---|---|---|---|---|
| `bd` ML, `test` z and t | 1.2e-15, 1.1e-16, 1.3e-15 | 2.2e-16, 1.1e-16, 3.3e-16 | 8.9e-16 | 1.9e-16, 4.5e-16 |
| `bd` REML | 8.9e-16, 1.1e-16, 9.3e-16 | 2.6e-15, 1.1e-16, 2.0e-15 | 1.7e-15 | 1.3e-16, 6.2e-16 |
| `bcg` ML | 3.1e-15, 1.7e-16, 2.3e-15 | 3.2e-15, 1.7e-16, 3.6e-15 | 7.1e-15 | 1.9e-15, 2.1e-15 |
| `bcg` REML | 4.6e-15, 3.3e-16, 3.6e-15 | 3.9e-15, 8.6e-16, 4.1e-15 | 1.1e-14 | 4.0e-16, 2.0e-15 |

At the reference `tau2_i` instead the differences are 6.5e-11 to 2.0e-08, the size of the difference between the two fits. The marginal and
conditional standardised residuals are equal (`(1 - lambda_i)` cancels): metafopy's two agree to 2.0e-15.

## vif

Fox and Monette's GVIF `det(R_11) det(R_22) / det(R)` on the correlation matrix of the reference `vb` and `va` (intercept dropped), and with
`reestimate=True` the ratio `Var[b_j] / Var[b'_j]` of the metafor `vif` help page, `b'_j` from a reference fit with only that moderator in the part:

| Fit | location | scale | `reestimate`, location | `reestimate`, scale |
|---|---|---|---|---|
| `bd2` ML | 4.6e-12 | 2.5e-11 | 3.7e-10 | 4.2e-08 |
| `bd2` REML | 3.0e-11 | 1.9e-09 | 3.2e-08 | 5.5e-08 |
| `bcg2` ML | 5.5e-10 | 1.7e-10 | 4.6e-06 | 4.4e-08 |
| `bcg2` REML | 9.1e-07 | 2.0e-08 | 4.8e-07 | 1.8e-06 (the `year`-only scale refit has no reference: its information is singular) |

The scale figures carry metafopy's numerical Hessian (the existing `ls.scale.vcov` cases allow 1e-4 for it).

## profile

`profile(fit, alpha=j, steps=7)`: metafopy's log-likelihood at each grid value against `ls_profile_ll` at the same value, and `maxll` against the
reference fit:

| Fit | alpha[0] | alpha[1] | `maxll` |
|---|---|---|---|
| `bd` ML | 7.1e-15 | 1.4e-14 | 1.1e-14 |
| `bd` REML | 1.1e-14 | 1.1e-14 | 3.6e-15 |
| `bcg` ML | 2.2e-11 | 6.4e-10 | 3.6e-14 |
| `bcg` REML | 3.9e-13 | 1.2e-12 | 2.8e-14 |
| `bd` ML, identity | 3.6e-15 | 5.2e-10 | 7.1e-15 |
| `bd` REML, identity | 7.1e-15 | 7.1e-15 | 0 |
| `bcg` ML, identity | 1.06e-08 | 9.5e-09 | 5.7e-09 |
| `bcg` REML, identity | 7.2e-10 | 8.9e-10 | 1.8e-10 |

Grid values where the identity-link refit leaves no feasible `tau2_i` are NaN in metafopy and -1e10 in the reference (`bd`: two values of alpha[0]
under ML, one under REML). On `bcg` ML identity every metafopy value is at or below the reference's: -1.06e-08 at the first grid value of alpha[0]
(-0.13329335) and 1e-14 or less where the refit is interior. The harness tolerance of `ls_fit` (1e-9 relative to the largest absolute log-likelihood)
allows 8.3e-09 here. R metafor 5.2-1 on the same grid: every alpha[0] refit fails ("All model fits failed"), and alpha[1] gives values 1.3e-08 to
1.2e-07 below the reference with the last grid value NA. The fit itself ends 5.7e-09 below the reference (the existing `ls.fit.ML.identity.bcg`
passes at its relative tolerance).
<!-- doc-counts: accept-end -->
