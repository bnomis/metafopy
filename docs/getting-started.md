# Getting started

This page takes one dataset from raw 2x2 tables to a pooled estimate, a meta-regression and two plots. Every code block on
the page runs as shown.

The data are the BCG vaccine trials of Colditz et al. (1994), <https://doi.org/10.1001/jama.1994.03510330076038>: for each
trial the number of tuberculosis cases and non-cases among vaccinated (`tpos`, `tneg`) and control (`cpos`, `cneg`)
subjects, with the absolute latitude of the trial site (`ablat`).

## Conventions

- Function and argument names are metafor's, with `.` replaced by `_`. `rma.uni` is `rma_uni`, and `rma` is an alias.
- Columns of `data` are given by name (`'yi'`, `'vi'`) or as arrays. Moderators are a formula string
  (`mods='~ ablat'`) or a matrix.
- Indices are 0-based in code (`btt=[1]`), while printed output counts from 1, as metafor does.
- A fitted model is a frozen dataclass. `print(res)` gives metafor's summary, and the fields hold the numbers
  (`res.beta`, `res.se`, `res.tau2`, `res.I2`).
- The functions you call on a fit are the same as in R: `predict`, `confint`, `anova`, `profile`, `influence` and so on.

## Effect sizes

`escalc` turns the raw counts into log risk ratios (`yi`) and their sampling variances (`vi`):

```python
from metafopy import datasets, escalc

bcg = datasets.load('dat.bcg')
dat = escalc(measure='RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=bcg)
print(dat[['trial', 'author', 'year', 'yi', 'vi']].head())
```

```text
   trial                author  year        yi        vi
0      1               Aronson  1948 -0.889311  0.325585
1      2      Ferguson & Simes  1949 -1.585389  0.194581
2      3       Rosenthal et al  1960 -1.348073  0.415368
3      4     Hart & Sutherland  1977 -1.441551  0.020010
4      5  Frimodt-Moller et al  1973 -0.217547  0.051210
```

## A random-effects model

```python
from metafopy import rma

res = rma('yi', 'vi', data=dat)
print(res)
```

```text
Random-Effects Model (k = 13; tau^2 estimator: REML)

tau^2 (estimated amount of total heterogeneity): 0.3132 (SE = 0.1664)
tau (square root of estimated tau^2 value):      0.5597
I^2 (total heterogeneity / total variability):   92.22%
H^2 (total variability / sampling variability):  12.86

Test for Heterogeneity:
Q(df = 12) = 152.2330, p-val < .0001

Model Results:

estimate      se     zval    pval    ci.lb    ci.ub
 -0.7145  0.1798  -3.9744  <.0001  -1.0669  -0.3622  ***

---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

The default heterogeneity estimator is REML, as in metafor. Choose another with `method`, for example
`rma('yi', 'vi', data=dat, method='DL')`. Pick the tests and intervals with `test='knha'` (Knapp-Hartung) or `level=90`.

The numbers are fields of the result:

```python
print(res.beta, res.se, res.tau2, res.I2)
```

```text
[-0.71453235] [0.17978153] 0.3132433259808949 92.22138607496555
```

## Back-transform and predict

The pooled log risk ratio is hard to read. `predict` takes a transformation:

```python
import numpy

from metafopy import predict

print(predict(res, transf=numpy.exp))
```

```text
     pred   ci.lb   ci.ub   pi.lb  pi.ub
0  0.4894  0.3441  0.6962  0.1546  1.549
```

The columns `pi.lb` and `pi.ub` are the prediction interval: where the true effect of a new trial would fall.

## Meta-regression

Add a moderator with a formula. The effect of the vaccine depends strongly on latitude:

```python
res_lat = rma('yi', 'vi', mods='~ ablat', data=dat)
print(res_lat)
print(predict(res_lat, newmods=[10, 30, 50], transf=numpy.exp))
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
```

`newmods` lists the moderator values to predict at (the intercept is added for you).

## Confidence intervals for heterogeneity

```python
from metafopy import confint

print(confint(res))
```

```text
        estimate    ci.lb    ci.ub
tau^2     0.3132   0.1197   1.1115
tau       0.5597   0.3460   1.0543
I^2(%)   92.2214  81.9177  97.6781
H^2      12.8558   5.5303  43.0680
```

## Plots

```python
import matplotlib.pyplot

from metafopy import forest, funnel

forest(res, atransf=numpy.exp)
matplotlib.pyplot.savefig('forest.png')

matplotlib.pyplot.figure()
funnel(res)
matplotlib.pyplot.savefig('funnel.png')
```

Plots draw on the current matplotlib figure and return a result object with the plotted values. See [Plots](guide/plots.md).

## Next steps

- [Effect sizes](guide/effect-sizes.md) lists the measures `escalc` supports and how to pool precomputed effects.
- [Fitting models](guide/models.md) covers estimators, tests, moderators and the methods you call on a fit.
- [Analyses](analyses.md) works through larger examples.
