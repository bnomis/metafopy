# Fitting models

`rma_uni` (alias `rma`) fits equal-effects, fixed-effects, random-effects and mixed-effects (meta-regression) models. It
mirrors metafor's [`rma.uni`](https://wviechtb.github.io/metafor/reference/rma.uni.html).

```python
import numpy

from metafopy import datasets, escalc, rma

dat = escalc(measure='RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=datasets.load('dat.bcg'))
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

## Ways to give the data

All of these fit the same model:

```python
a = rma('yi', 'vi', data=dat)
b = rma(yi=dat['yi'], vi=dat['vi'])
c = rma(yi=dat['yi'], sei=numpy.sqrt(dat['vi']))
d = rma(measure='RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=datasets.load('dat.bcg'))
print(a.beta, b.beta, c.beta, d.beta)
```

```text
[-0.71453235] [-0.71453235] [-0.71453235] [-0.71453235]
```

The last form lets `rma` call `escalc` for you. It takes the same `add`, `to`, `drop00` and `vtype` arguments.

## The model and the estimator

The model is chosen by `method`. `'EE'` (aliases `'FE'` and `'CE'`) is the equal-effects model. Every other value is a random-effects
model with that estimator of the amount of heterogeneity tau^2:

| `method` | Estimator |
|----------|-----------|
| `'REML'` (default) | Restricted maximum likelihood |
| `'ML'` | Maximum likelihood |
| `'DL'`, `'DLIT'` | DerSimonian-Laird, and its iterative version |
| `'HE'` | Hedges |
| `'HS'`, `'HSk'` | Hunter-Schmidt, and its small-sample variant |
| `'SJ'`, `'SJIT'` | Sidik-Jonkman, and its iterative version |
| `'EB'` | Empirical Bayes |
| `'PM'` (alias `'MP'`), `'PMM'` | Paule-Mandel, and its median-unbiased version |
| `'GENQ'`, `'GENQM'` | Generalized Q-statistic estimators (Jackson, 2013) |

Viechtbauer (2005), <https://doi.org/10.3102/10769986030003261>, and Veroniki et al. (2016),
<https://doi.org/10.1002/jrsm.1164>, compare them. `metafopy.models.tau2.ESTIMATORS` lists all of them. A list such as
`method=['REML', 'DL']` tries each in turn until one converges. Optimiser settings go in `control=`, a `Control` instance or a
dict with metafor's names (`tau2_init`, `tau2_min`, `tau2_max`, `threshold`, `maxiter`, `stepadj`...).

## Tests and intervals

By default the pooled estimate has a z-test and Wald-type confidence interval. Use `test='knha'` for the Knapp-Hartung
adjustment (a t-test with a rescaled variance; Knapp and Hartung, 2003, <https://doi.org/10.1002/sim.1482>), which has better
coverage with few studies:

```python
print(rma('yi', 'vi', data=dat, test='knha'))
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

estimate      se     tval  df    pval    ci.lb    ci.ub
 -0.7145  0.1808  -3.9522  12  0.0019  -1.1084  -0.3206  **

---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

`level` is the confidence level in percent (default 95), and `digits` controls how much is printed.

## Meta-regression

Moderators are given as a formula in `mods`, evaluated on `data` with [formulaic](https://matthewwardrop.github.io/formulaic/)
(R-style syntax: `'~ ablat + year'`, `'~ C(alloc)'`, `'~ ablat * year'`).
As in R, `factor()`, `relevel(x, ref)` and `scale()` (which skips missing values) can be used in the formula, as can functions defined in the
calling code (they are looked up where `rma` is called). The same goes for the `scale` formula of location-scale models.

```python
res = rma('yi', 'vi', mods='~ ablat + year', data=dat)
print(res)
```

```text
Mixed-Effects Model (k = 13; tau^2 estimator: REML)

tau^2 (estimated amount of residual heterogeneity):     0.1108 (SE = 0.0845)
tau (square root of estimated tau^2 value):             0.3328
I^2 (residual heterogeneity / unaccounted variability): 71.98%
H^2 (unaccounted variability / sampling variability):   3.57
R^2 (amount of heterogeneity accounted for):            64.63%

Test for Residual Heterogeneity:
QE(df = 10) = 28.3251, p-val = 0.0016

Test of Moderators (coefficients 2:3):
QM(df = 2) = 12.2043, p-val = 0.0022

Model Results:

         estimate       se     zval    pval     ci.lb    ci.ub
intrcpt   -3.5455  29.0959  -0.1219  0.9030  -60.5724  53.4814
ablat     -0.0280   0.0102  -2.7371  0.0062   -0.0481  -0.0080  **
year       0.0019   0.0147   0.1299  0.8966   -0.0269   0.0307

---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

Categorical moderators are coded with treatment contrasts, so the intercept is the first level. Use `'~ alloc - 1'` for one
coefficient per level. `rma` also accepts `yi='yi ~ ablat'`, and a model matrix as `mods`.

The coefficients are `res.beta`, in the order of `res.coef_names`. `btt` selects the coefficients that the omnibus test
tests. It takes 0-based positions (or a regular expression matched against the names):

```python
res = rma('yi', 'vi', mods='~ alloc + ablat', data=dat)
print(res.coef_names)
print(rma('yi', 'vi', mods='~ alloc + ablat', data=dat, btt=[1, 2]))
```

```text
('intrcpt', 'allocrandom', 'allocsystematic', 'ablat')

Mixed-Effects Model (k = 13; tau^2 estimator: REML)

tau^2 (estimated amount of residual heterogeneity):     0.1446 (SE = 0.1124)
tau (square root of estimated tau^2 value):             0.3803
I^2 (residual heterogeneity / unaccounted variability): 70.11%
H^2 (unaccounted variability / sampling variability):   3.35
R^2 (amount of heterogeneity accounted for):            53.84%

Test for Residual Heterogeneity:
QE(df = 9) = 26.2034, p-val = 0.0019

Test of Moderators (coefficients 2:3):
QM(df = 2) = 1.2850, p-val = 0.5260

Model Results:

                 estimate      se     zval    pval    ci.lb    ci.ub
intrcpt            0.2932  0.4050   0.7239  0.4691  -0.5006   1.0870
allocrandom       -0.2675  0.3504  -0.7633  0.4453  -0.9543   0.4193
allocsystematic    0.0585  0.3795   0.1540  0.8776  -0.6854   0.8023
ablat             -0.0273  0.0092  -2.9650  0.0030  -0.0453  -0.0092  **

---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

## Methods you call on a fit

These work on a fitted model the same way as in metafor. The result object is passed as the first argument.

```python
res = rma('yi', 'vi', mods='~ ablat', data=dat)
```

### predict

```python
from metafopy import predict

print(predict(res, newmods=[10, 30, 50]))
print(predict(rma('yi', 'vi', data=dat), transf=numpy.exp))
```

```text
     pred      se   ci.lb   ci.ub   pi.lb   pi.ub
0 -0.0396  0.1866 -0.4053  0.3262 -0.6931  0.6140
1 -0.6216  0.1069 -0.8312 -0.4120 -1.2023 -0.0409
2 -1.2036  0.1716 -1.5400 -0.8672 -1.8412 -0.5660
     pred   ci.lb   ci.ub   pi.lb  pi.ub
0  0.4894  0.3441  0.6962  0.1546  1.549
```

`transf=` applies a transformation to the predictions and their bounds. Prediction intervals (`pi.lb`, `pi.ub`) include the
heterogeneity. `predtype=` (`pi_type=` is its old spelling) chooses how the interval is built: `'default'`, `'simple'` (no uncertainty
of the predicted value, normal distribution), `'riley'` (a t distribution with `k - parms` degrees of freedom, Riley et al., 2011,
<https://doi.org/10.1111/j.1467-985X.2010.00696.x>) or `'t'`; `newvi=` adds the sampling variances of new studies. Both work for every
fit class. `na_action=` (`'omit'`, `'exclude'`, `'pass'` or `'fail'`, standing for R's `options(na.action=)`) decides whether the fitted
values of `fitted()`, `predict()` and `hatvalues()` cover the studies used in fitting or one value for each study of the data.

### confint

For a random-effects model, `confint` gives intervals for tau^2, tau, I^2 and H^2. The default is the Q-profile method of
Viechtbauer (2007), <https://doi.org/10.1002/sim.2514>. `type='pl'` gives profile-likelihood intervals.

```python
from metafopy import confint

print(confint(res))
print(confint(res, type='pl'))
```

```text
        estimate    ci.lb    ci.ub
tau^2     0.0764   0.0167   0.7849
tau       0.2763   0.1292   0.8859
I^2(%)   68.3931  32.1080  95.6976
H^2       3.1639   1.4729  23.2428
        estimate    ci.lb    ci.ub
tau^2     0.0764   0.0058   0.4432
tau       0.2763   0.0763   0.6657
I^2(%)   68.3931  14.1564  92.6258
H^2       3.1639   1.1649  13.5607
```

### anova

`anova` runs Wald-type tests of coefficients and likelihood ratio tests that compare two nested models. Fit both with
`method='ML'` for the likelihood ratio test of fixed effects.

```python
from metafopy import anova

full = rma('yi', 'vi', mods='~ ablat', data=dat, method='ML')
reduced = rma('yi', 'vi', data=dat, method='ML')
print(anova(full, reduced))
print(anova(res, btt=[1]))
```

```text
         df      AIC      BIC     AICc   logLik     LRT    pval        QE   tau^2      R^2
Full      3  21.3713  23.0662  24.0380  -7.6857     NaN     NaN   30.7331  0.0344      NaN
Reduced   2  29.3302  30.4601  30.5302 -12.6651  9.9588  0.0016  152.2330  0.2800  87.7302
Test of Moderators (coefficient 2):
QM(df = 1) = 16.3571, p-val < .0001
```

For a fit made robust with `robust(res, cluster, club_sandwich=True)`, `anova` runs the cluster-robust Wald test with the small-sample
approximation of the fit's `wald_test` (`HTZ` by default), which gives an F test with an estimated denominator degrees of freedom
(`F(df1 = 1, df2 = 1.52) = 41.6109, p-val = 0.0440` for `btt` on the bcg data with `cluster='alloc'`); `anova(res, X=...)` uses the
degrees of freedom of the fit's `coef_test` for each linear combination. `L` is R's old spelling of `X`, `fixed=True` matches a
character `btt` literally, and `anova(full, reduced, test='Wald')` tests the extra coefficients of the full model with its Wald test.

### Permutation test

`permutest` gives permutation-based p-values for the coefficients, with an exact test for small numbers of studies. Set the
seed with `set_seed` first: metafopy reproduces R's random number generator, so seeds give the same permutations as
`set.seed` in R.

```python
from metafopy import permutest, set_seed

set_seed(1234)
print(permutest(res, iter=500))
```

```text
Test of Moderators (coefficient 2):¹
QM(df = 1) = 16.3571, p-val = 0.0100

Model Results:

         estimate      se     zval    pval¹    ci.lb    ci.ub
intrcpt    0.2515  0.2491   1.0095  0.6980   -0.2368   0.7397
ablat     -0.0291  0.0072  -4.0444  0.0100   -0.0432  -0.0150  **

---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

1) p-values based on permutation testing
```

### profile

`profile` evaluates the likelihood over a grid of tau^2 values, which is useful to see whether the likelihood is flat:

```python
from metafopy import profile

print(profile(rma('yi', 'vi', data=dat)).tau2[:3])
```

```text
[0.11969531 0.17189484 0.22409436]
```

### Residuals and BLUPs

`rstandard`, `rstudent`, `blup` and `ranef` give residuals and the best linear unbiased predictions of the study-specific
effects. See [Diagnostics](diagnostics.md).

## Equal-effects and fixed-effects models

`method='EE'` fits the equal-effects model: every study estimates the same true effect. metafor also accepts `'FE'` and
`'CE'` as synonyms. The results are identical and only the interpretation differs. A random-effects
model estimates the mean of a distribution of true effects instead. See Hedges and Vevea (1998),
<https://doi.org/10.1037/1082-989X.3.4.486>, for the distinction.

## Weights

`weights` replaces the inverse-variance weights with your own, and `weighted=False` fits by ordinary least squares.
The standard errors of the coefficients then account for the weights not being inverse-variance weights. For inference that
is robust to a misspecified model, see [`robust`](multilevel.md#cluster-robust-inference).
