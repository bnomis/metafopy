# Coming from R

metafopy is built so that metafor code translates line by line. This page lists the rules of the translation. Where metafopy
does not behave like metafor, see [Differences from metafor](differences.md), which is kept up to date with every known
deviation.

## The same analysis in both languages

R:

```r
library(metafor)
dat <- escalc(measure="RR", ai=tpos, bi=tneg, ci=cpos, di=cneg, data=dat.bcg)
res <- rma(yi, vi, mods = ~ ablat, data=dat)
print(res)
predict(res, newmods=c(10, 30, 50), transf=exp)
confint(res)
forest(res, atransf=exp)
```

Python:

```python
import numpy

from metafopy import confint, datasets, escalc, forest, predict, rma

dat = escalc(measure='RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=datasets.load('dat.bcg'))
res = rma('yi', 'vi', mods='~ ablat', data=dat)
print(res)
print(predict(res, newmods=[10, 30, 50], transf=numpy.exp))
print(confint(res))
forest(res, atransf=numpy.exp)
```

```text
Mixed-Effects Model (k = 13; tau^2 estimator: REML)

tau^2 (estimated amount of residual heterogeneity):     0.0764 (SE = 0.0591)
tau (square root of estimated tau^2 value):             0.2763
I^2 (residual heterogeneity / unaccounted variability): 68.39%
H^2 (unaccounted variability / sampling variability):   3.16
R^2 (amount of heterogeneity accounted for):            75.62%

Test for Residual Heterogeneity:
QE(df = 11) = 30.7331, p-val = 0.0012

Test of Moderators (coefficient 2):
QM(df = 1) = 16.3571, p-val < .0001

Model Results:

         estimate      se     zval    pval    ci.lb    ci.ub
intrcpt    0.2515  0.2491   1.0095  0.3127  -0.2368   0.7397
ablat     -0.0291  0.0072  -4.0444  <.0001  -0.0432  -0.0150  ***

---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

     pred   ci.lb   ci.ub   pi.lb   pi.ub
0  0.9612  0.6668  1.3857  0.5000  1.8478
1  0.5371  0.4355  0.6623  0.3005  0.9600
2  0.3001  0.2144  0.4201  0.1586  0.5678
        estimate    ci.lb    ci.ub
tau^2     0.0764   0.0167   0.7849
tau       0.2763   0.1292   0.8859
I^2(%)   68.3931  32.1080  95.6976
H^2       3.1639   1.4729  23.2428
```

The printed output is the same as metafor's.

## Rules

| In R | In metafopy |
|------|-------------|
| `rma.uni`, `rma.mv`, `leave1out` | `rma_uni`, `rma_mv`, `leave1out`: `.` becomes `_` |
| `ci.lb`, `tau2`, `QE` | `ci_lb`, `tau2`, `QE`: fields of the result, `.` becomes `_` |
| `res$beta`, `res$se`, `coef(res)` | `res.beta`, `res.se` |
| `dat.bcg` (lazy-loaded) | `datasets.load('dat.bcg')` |
| `tpos`, `tneg` (unquoted columns, `data=dat`) | `'tpos'`, `'tneg'` (strings naming columns of `data=`) or arrays |
| `mods = ~ ablat + year` | `mods='~ ablat + year'` (a string; [formulaic](https://matthewwardrop.github.io/formulaic/) syntax) |
| `random = ~ 1 \| study/esid` | `random='~ 1 \| study/esid'` |
| `random = list(~ 1 \| a, ~ 1 \| b)` | `random=['~ 1 \| a', '~ 1 \| b']` |
| `btt = 2:3` (1-based) | `btt=[1, 2]` (0-based) |
| `subset = year > 1950` | `subset=dat['year'] > 1950` (a boolean array) |
| `transf = exp` | `transf=numpy.exp` |
| `TRUE` / `FALSE`, `NULL`, `NA` | `True` / `False`, `None`, `numpy.nan` |
| `control = list(tau2.max = 10)` | `control={'tau2_max': 10}`, or a `Control` / `MVControl` instance |
| `set.seed(1234)` | `set_seed(1234)` (R's generator, so permutation tests reproduce R) |
| `summary(res)` | `print(res)`: the fit is its own summary |
| `plot(res)` family, `forest(res)` | functions on matplotlib axes: `forest`, `funnel`, `plot_infl`... |

Names that metafor writes with `.` in the middle of a word also lose it in prefixed functions: `plot.gosh` is `plot_gosh`,
`plot.infl.rma.uni` is `plot_infl`, `plot.profile.rma` is `plot_profile`, `plot.selmodel` is `plot_selmodel`, and
`transf.ilogit` is `transf_ilogit`.

## Classes and generic functions

R's S3 classes (`rma.uni`, `rma.mv`, `escalc`...) become frozen dataclasses (`RMAResult`, `RMAMVResult`...) and `escalc`
returns a plain `pandas.DataFrame` whose `attrs` carry the attributes metafor stores on the data frame. R's generic functions
(`predict`, `confint`, `anova`, `profile`, `influence`...) are single-dispatch functions that choose the implementation from
the class of the result. `predict(res, ...)` works as in R. The implementations for each class are documented in the
[reference](reference/methods.md).

## Numbers

metafopy is validated against metafor 5.2-1. The test suite compares the results of the models with R fixtures: closed-form
quantities agree to rounding error, and quantities found by numerical optimisation agree to the optimiser's tolerance. The
exact tolerances and the cases that differ more are in [Differences from metafor](differences.md).

## What is not ported

`reporter()`, `emmprep`, `matreg`, `verbose` for `rma_mv`, `forest_cumul` and `addpoly` of `predict` results are not ported. See
[Differences from metafor](differences.md) for details and for the places where a metafor feature behaves differently.
