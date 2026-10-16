# Diagnostics

Diagnostics for `rma_uni` fits follow metafor's functions of the same names. The examples use the BCG data:

```python
from metafopy import datasets, escalc, rma

dat = escalc(measure='RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=datasets.load('dat.bcg'))
res = rma('yi', 'vi', data=dat)
```

## Residuals

```python
from metafopy import rstandard, rstudent

print(rstandard(res))
print(rstudent(res))
```

```text
     resid     se       z
1  -0.1748 0.7788 -0.2244
2  -0.8709 0.6896 -1.2629
3  -0.6335 0.8344 -0.7592
4  -0.7270 0.5486 -1.3253
5   0.4970 0.5763  0.8624
6  -0.0716 0.5365 -0.1334
7  -0.9064 0.7099 -1.2768
8   0.7265 0.5337  1.3611
9   0.2451 0.5808  0.4220
10 -0.6568 0.5949 -1.1040
11  0.3752 0.5416  0.6927
12  1.1604 0.9019  1.2867
13  0.6972 0.5936  1.1746


     resid     se       z
1  -0.1822 0.8354 -0.2181
2  -0.9314 0.7210 -1.2918
3  -0.6625 0.8778 -0.7547
4  -0.8131 0.5603 -1.4512
5   0.5466 0.6448  0.8477
6  -0.0752 0.6376 -0.1180
7  -0.9657 0.7407 -1.3037
8   0.8067 0.5563  1.4501
9   0.2718 0.6668  0.4076
10 -0.7183 0.6369 -1.1277
11  0.4185 0.6255  0.6691
12  1.2057 0.9348  1.2898
13  0.7602 0.6400  1.1879
```

`rstandard` gives internally standardized residuals and `rstudent` externally studentized residuals (the residual of each
study from a model fitted without it). Studies with absolute values above about 2 or 3 deserve a look.

`residuals` takes a `type`: `'response'` (the default, the raw residuals), `'rstandard'`, `'rstudent'`, `'pearson'` (the raw residuals
divided by the square root of the sampling variance plus tau^2) and `'cholesky'` (the residuals decorrelated with the Cholesky factor of
the marginal variance-covariance matrix). Which types a fit class allows follows metafor. `na_action` stands in for R's
`options(na.action=)`. `weights_(type='matrix')` returns the weight matrix.

```python
from metafopy import residuals

print(residuals(res, 'pearson'))
print(res.residuals('rstudent', na_action='exclude'))
print(res.weights_('matrix').shape)
```

## Influence

`influence` computes the case deletion diagnostics of Viechtbauer and Cheung (2010),
<https://doi.org/10.1002/jrsm.11>: externally studentized residuals, DFFITS, Cook's distances, covariance ratios, hat values,
leave-one-out estimates of tau^2 and Q, and DFBETAS.

```python
from metafopy import influence

inf = influence(res)
print(inf)
```

```text
   rstudent  dffits cook.d  cov.r tau2.del   QE.del    hat  weight    dfbs inf
1   -0.2181 -0.0407 0.0017 1.1164   0.3362 151.5826 0.0506  5.0595 -0.0403
2   -1.2918 -0.3436 0.1132 1.0102   0.2926 145.3176 0.0636  6.3647 -0.3459
3   -0.7547 -0.1604 0.0260 1.0670   0.3207 150.1970 0.0444  4.4360 -0.1597
4   -1.4512 -0.5201 0.2295 0.9651   0.2628  96.5626 0.0970  9.6987 -0.5138
5    0.8477  0.2707 0.0762 1.1380   0.3278 151.3200 0.0887  8.8685  0.2711
6   -0.1180 -0.0190 0.0004 1.2407   0.3596 128.1867 0.1010 10.0957 -0.0192
7   -1.3037 -0.3362 0.1087 1.0077   0.2930 145.8296 0.0603  6.0272 -0.3387
8    1.4501  0.4775 0.1992 1.0011   0.2732  67.9858 0.1019 10.1894  0.4714
9    0.4076  0.1415 0.0220 1.1968   0.3495 152.2051 0.0874  8.7431  0.1419
10  -1.1277 -0.3487 0.1170 1.0505   0.2987 139.8271 0.0837  8.3676 -0.3487
11   0.6691  0.2315 0.0581 1.1860   0.3405 151.4655 0.0993  9.9250  0.2330
12   1.2898  0.2526 0.0634 1.0259   0.3082 150.7868 0.0382  3.8216  0.2535
13   1.1879  0.3549 0.1228 1.0650   0.3037 149.7884 0.0840  8.4029  0.3548
```

The last column, `inf`, is empty unless a study is flagged as influential (as in metafor, by the usual cutoffs). The individual diagnostics are also available as
`cooks_distance` and `dfbetas`; `influence(res, measure='cooks.distance')`, `measure='dfbetas'` and `measure='rstudent'` return the same
objects as those functions and `rstudent`, as in metafor. `plot_infl` plots them all:

```python
import matplotlib.pyplot

from metafopy import plot_infl

plot_infl(inf)
matplotlib.pyplot.savefig('influence.png')
```

## Leave-one-out and cumulative analyses

`leave1out` refits the model leaving out each study in turn and `cumul` adds the studies one at a time, in their order in
the data (sort the data first to cumulate by year or precision).

```python
from metafopy import cumul, leave1out

print(leave1out(res).estimate[:3])
print(cumul(res).estimate[:3])
```

```text
[-0.70708379 -0.6540312  -0.68555442]
[-0.88931133 -1.32500344 -1.33023415]
```

Both return a result whose fields (`estimate`, `se`, `ci_lb`, `ci_ub`, `tau2`...) are arrays with one entry per fit, and print
as a table. `leave1out` and `cumul` also work for `rma_mh` and `rma_peto` fits.

## Heterogeneity sources: Baujat and GOSH

A Baujat plot (Baujat et al., 2002, <https://doi.org/10.1002/sim.1221>) shows which studies contribute most to the
heterogeneity and to the overall result. The GOSH plot (Olkin et al., 2012, <https://doi.org/10.1002/jrsm.1053>) fits the
model to many subsets of the studies and shows the distribution of the results. With many studies it fits a random sample of
the subsets, controlled by `subsets` and `seed`.

```python
from metafopy import baujat, gosh, plot_gosh, set_seed

baujat(res)
set_seed(1234)
plot_gosh(gosh(res, subsets=200))
```

## Heteroscedastic heterogeneity

The random-effects model assumes the same amount of heterogeneity, tau^2, for every study. `hettest` (new in metafor 5.2-1) tests
this assumption, H0: tau^2_1 = ... = tau^2_k, for a model without moderators or for a vector of estimates with their sampling
variances (`vi=`) or standard errors (`sei=`). The default is a score test; `test='lrt'` and `test='wald'` give a likelihood
ratio and a Wald-type test, and `'ksn'`, `'ksx2'`, `'adn'` and `'adx2'` check the Pearson residuals with Kolmogorov-Smirnov and
Anderson-Darling goodness-of-fit tests. The p-values come from a parametric bootstrap (`boot=1000` samples unless you set
`boot=` to another count), which draws from R's random number generator, so `seed=` gives the results of `set.seed()` in R.
`boot=False` uses the chi-squared reference distribution of the three model-based tests instead.

```python
from metafopy import hettest

print(hettest(res, seed=1234))
print(hettest(res, test=['lrt', 'ksn'], boot=200, seed=1234))
```

```text

Test for Heteroscedastic Heterogeneity

Estimation method: Restricted maximum likelihood
Test type:         Score test
Bootstrapping:     Yes (1000/1000 iterations)

X^2(df = 12) = 2.8884, p = 0.9920

```

The result holds the statistics (`statistic`), p-values (`pval`), the estimated study-specific variances (`tau2i`) and the
bootstrapped statistics (`statistic_boot`). Fitted models of the classes `rma_mv`, `rma_glmm`, `selmodel` and location-scale
models are not accepted, as in metafor.

## Variance inflation factors

For meta-regression, `vif` reports collinearity among the moderators:

```python
from metafopy import vif

print(vif(rma('yi', 'vi', mods='~ ablat + year', data=dat)))
```

```text
 ablat    year
1.5391  1.5391
```

`vif` also takes `rma_mv`, location-scale and `selmodel` fits with moderators. The options are those of metafor: `btt` (and `att`
for the scale part), `table=True` for the coefficient table, `intercept=True` to include the intercept, `fixed=True` to match a
string `btt` literally, and, except for `selmodel` fits, `reestimate=True` and `sim`.

## Best linear unbiased predictions

`blup` gives the shrunken estimate of the true effect of each study, which combines its observed effect and the model
estimate, and `ranef` the estimated random effects:

```python
from metafopy import blup

print(blup(res))
```

```text
      pred     se   pi.lb   pi.ub
1  -0.8002 0.4099 -1.6037  0.0032
2  -1.2517 0.3532 -1.9440 -0.5594
3  -0.9869 0.4348 -1.8392 -0.1346
4  -1.3979 0.1376 -1.6675 -1.1283
5  -0.2874 0.2113 -0.7015  0.1268
6  -0.7846 0.0823 -0.9459 -0.6233
7  -1.2440 0.3686 -1.9664 -0.5215
8   0.0029 0.0626 -0.1198  0.1255
9  -0.5068 0.2204 -0.9388 -0.0749
10 -1.2472 0.2457 -1.7288 -0.7656
11 -0.3537 0.1095 -0.5682 -0.1391
12 -0.2847 0.4583 -1.1830  0.6135
13 -0.1467 0.2434 -0.6239  0.3304
```
