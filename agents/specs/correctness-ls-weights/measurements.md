# Correctness tests for user weights: measurements

**Spec:** `agents/specs/correctness-ls-weights/spec.md`

Measured on 2026-11-20 at `0cc8edc`, metafopy against the references the spec names, and R metafor 5.2-1 (R 4.6.2) where stated. The figures are a
dated record, not a claim about the present.

## Location-scale fits with `weights`

`rma_uni(yi, vi, weights=w, mods=~ m, scale=~ m)`; `alpha` against `formulas.ls_reference_fit` (equation 7 or 9, no weights), `beta` and `vb`
against `formulas.gls(yi, diag(vi + tau2_i), X, diag(w))` (Hedges, Tipton and Johnson 2010, equations 3 and 4) at metafopy's `tau2_i`. Largest
relative difference (`|a - b| / max(1, |b|)`):

| Data, weights, model | Method | Link | alpha | beta | vb |
|---|---|---|---|---|---|
| `dat.bcg` RR, `tpos`, `~ ablat` | ML | log | 2.8e-10 | 3.9e-16 | 3.1e-17 |
| `dat.bcg` RR, `tpos`, `~ ablat` | ML | identity | 2.3e-10 | 3.9e-16 | 1.7e-17 |
| `dat.bcg` RR, `tpos`, `~ ablat` | REML | log | 4.4e-08 | 3.9e-16 | 1.2e-16 |
| `dat.bcg` RR, `tpos`, `~ ablat` | REML | identity | 4.1e-10 | 3.9e-16 | 5.0e-16 |
| `dat.bangertdrowns2004`, `ni`, `~ imag` | ML | log | 6.8e-09 | 1.1e-16 | 5.6e-17 |
| `dat.bangertdrowns2004`, `ni`, `~ imag` | ML | identity | 4.7e-09 | 1.1e-16 | 2.8e-17 |
| `dat.bangertdrowns2004`, `ni`, `~ imag` | REML | log | 1.1e-08 | 1.1e-16 | 1.1e-16 |
| `dat.bangertdrowns2004`, `ni`, `~ imag` | REML | identity | 9.7e-12 | 1.1e-16 | 5.6e-17 |

The identity-link `dat.bcg` fits warn "Error when trying to invert the Hessian." (and numpy "invalid value encountered in scalar subtract"); `va`
is then NaN, which the checks do not compare.

On `dat.bcg` with weights `tpos + cpos + 1`, metafopy's `alpha` is the same with and without weights (difference 0), and R's with weights, ML log
link, is (-4.9574906909, 0.0385932586), the maximum without weights; both give `beta` (0.3689422735, -0.0309237534). Both report a different log-likelihood with weights than without (ML log link:
-7.5392902501 without, -7.6930272498 in R and -7.6930272593 in metafopy with): the log-likelihood is equation 5 (ML) or 9 with `log|X'X|`
(REML) evaluated at the user-weighted `beta` (metafopy's value equals that formula to 1e-15, and R's to 1e-8). `optbeta=True` with `weights` stops in R with "Cannot use custom
weights when 'optbeta=TRUE'."; metafopy fits it without an error (`dat.bcg`, `mods = scale = ~ ablat`, weights `tpos`: `beta` (0.33244, -0.02878),
`alpha` (-2.54250, -0.00073)).

## Plain `rma_uni` fits with `weights`

`tau2` against `formulas.rma_tau2` (no weights), `beta` and `vb` against `formulas.gls` with `diag(w)` at that `tau2`; default control settings:

| Data, weights | Method | mods | tau2 | beta | vb |
|---|---|---|---|---|---|
| `dat.bcg` RR, `tpos` | DL | none | 0 | 5.6e-17 | 0 |
| `dat.bcg` RR, `tpos` | DL | `~ ablat` | 5.8e-16 | 3.9e-16 | 8.5e-16 |
| `dat.bcg` RR, `tpos` | REML | none | 6.8e-08 | 5.6e-17 | 2.0e-08 |
| `dat.bcg` RR, `tpos` | REML | `~ ablat` | 6.7e-06 | 3.9e-16 | 8.6e-06 |
| `dat.bangertdrowns2004`, `ni` | DL | none | 0 | 2.8e-17 | 8.7e-19 |
| `dat.bangertdrowns2004`, `ni` | DL | `~ imag` | 0 | 1.1e-16 | 6.9e-18 |
| `dat.bangertdrowns2004`, `ni` | REML | none | 2.4e-06 | 2.8e-17 | 1.0e-07 |
| `dat.bangertdrowns2004`, `ni` | REML | `~ imag` | 2.2e-06 | 1.1e-16 | 1.4e-06 |

The REML differences are the Fisher scoring stop at the default `threshold = 1e-5`; the existing REML `tau2` cases pass `control = { threshold =
1e-13, tol = 1e-13, maxiter = 1000, tau2_max = 1e6 }`, and the REML cases here do the same. With `weights`, metafopy's `tau2` equals the fit
without weights (`dat.bcg`, REML, `~ ablat`: 0.07635469470565989 both), and its log-likelihood differs (REML -8.0873 without, -8.2804 with).
