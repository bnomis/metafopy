# Selection models and location-scale models

## Selection models

Selection models (Vevea and Hedges, 1995, <https://doi.org/10.1007/BF02294384>) model how the probability that a study is
published depends on its p-value, and estimate the effect corrected for that selection. `selmodel` takes a fitted `rma_uni`
model and a selection function `type`. It mirrors metafor's
[`selmodel`](https://wviechtb.github.io/metafor/reference/selmodel.html); see Citkowicz and Vevea (2017),
<https://doi.org/10.1037/met0000119>, for the parametric functions.

```python
from metafopy import datasets, rma, selmodel

dat = datasets.load('dat.hackshaw1998')
res = rma(data=dat, method='ML')
print(selmodel(res, type='halfnorm'))
```

```text
Random-Effects Model (k = 37; tau^2 estimator: ML)

tau^2 (estimated amount of total heterogeneity): 0.0321 (SE = 0.0262)
tau (square root of estimated tau^2 value):      0.1792

Test for Heterogeneity:
LRT(df = 1) = 5.0652, p-val = 0.0244

Model Results:

estimate      se    zval    pval    ci.lb   ci.ub
  0.1011  0.1175  0.8608  0.3894  -0.1291  0.3313

Test for Selection Model Parameters:
LRT(df = 1) = 2.5337, p-val = 0.1114

Selection Model Results:

estimate      se    zval    pval   ci.lb   ci.ub
  2.0287  1.4126  1.4361  0.1510  0.0000  4.7974

---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

The first part of the output is the corrected random-effects model. The test of the selection model parameters asks whether
there is evidence for selection at all.

The `type` can be a parametric function (`'beta'`, `'halfnorm'`, `'negexp'`, `'logistic'`, `'power'`, `'negexppow'`) or a
step function. Step functions need the cut points of the p-value, given as `steps`:

```python
print(selmodel(res, type='stepfun', steps=[0.025, 0.10, 0.50, 1]))
```

```text
Random-Effects Model (k = 37; tau^2 estimator: ML)

tau^2 (estimated amount of total heterogeneity): 0.0096 (SE = 0.0128)
tau (square root of estimated tau^2 value):      0.0981

Test for Heterogeneity:
LRT(df = 1) = 1.1374, p-val = 0.2862

Model Results:

estimate      se    zval    pval    ci.lb   ci.ub
  0.0020  0.0889  0.0225  0.9821  -0.1723  0.1763

Test for Selection Model Parameters:
LRT(df = 3) = 6.4020, p-val = 0.0936

Selection Model Results:

                     k  estimate      se      zval    pval   ci.lb   ci.ub
0     < p <= 0.025   7    1.0000     ---       ---     ---     ---     ---
0.025 < p <= 0.1     8    0.4946  0.3152   -1.6032  0.1089  0.0000  1.1125
0.1   < p <= 0.5    16    0.2136  0.1687   -4.6612  <.0001  0.0000  0.5443  ***
0.5   < p <= 1       6    0.0618  0.0689  -13.6171  <.0001  0.0000  0.1969  ***

---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

Each row of the table is the relative probability of publication of a study with a p-value in that interval, relative to
the first interval. `alternative` sets the direction of the one-sided test that selection is based on (default
`'greater'`).

Results can be plotted with `plot_selmodel`, which draws the estimated selection function. `confint`, `profile`, `anova`
and `predict` work on a selection model fit, and the `profile` function takes the parallel options of metafor.

Selection models are sensitive to the choice of function and need many studies to be estimated reliably. Use them as a
sensitivity analysis. See also [Publication bias](bias.md).

## Location-scale models

Location-scale models (Viechtbauer and Lopez-Lopez, 2022, <https://doi.org/10.1002/jrsm.1562>) let the amount of
heterogeneity depend on moderators, so tau^2 is modelled alongside the average effect. Pass a `scale` formula to `rma`:

```python
from metafopy import rma

dat = datasets.load('dat.bangertdrowns2004')
res = rma('yi', 'vi', mods='~ length', scale='~ length', data=dat)
print(res)
```

```text
Location-Scale Model (k = 46; tau^2 estimator: REML)

Test for Residual Heterogeneity:
QE(df = 44) = 96.2810, p-val < .0001

Test of Location Coefficients (coefficient 2):
QM(df = 1) = 4.2107, p-val = 0.0402

Model Results (Location):

         estimate      se    zval    pval    ci.lb   ci.ub
intrcpt    0.0719  0.0766  0.9392  0.3476  -0.0782  0.2220
length     0.0149  0.0073  2.0520  0.0402   0.0007  0.0292  *

Test of Scale Coefficients (coefficient 2):
QM(df = 1) = 0.4194, p-val = 0.5172

Model Results (Scale):

         estimate      se     zval    pval    ci.lb    ci.ub
intrcpt   -3.6011  0.9112  -3.9520  <.0001  -5.3870  -1.8151  ***
length     0.0472  0.0728   0.6476  0.5172  -0.0956   0.1899

---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

The location part (the average effect) and the scale part (the log of tau^2, by default) are reported separately. Use
`link='identity'` to model tau^2 itself. The result is an `RMALSResult`, and `predict`, `confint`, `anova` (Wald and
likelihood ratio tests), `profile`, `vif`, `blup`, `ranef`, `rstandard` and `permutest` work on it.

`control={'optimizer': ...}` accepts the optimisers of `rma.uni` (`'optim'` with `optmethod`, `'nlm'`, `'bobyqa'`, `'hjk'`, ..., and with
`link='identity'` also `'constrOptim'`, `'solnp'`, `'nloptr'`, `'constrOptim.nl'`/`'alabama'` and `'auglag'`; any other optimiser is reset to
`'constrOptim'` with a warning there, and the barrier optimisers cannot be used with the log link). They are mapped as for `selmodel`; `ncpus`
above 1 selects `'optimParallel'`, but the likelihood is evaluated serially. `confint(res, control={'vc.min': ..., 'vc.max': ...})` sets the
interval searched for the bounds (`vc_min` and `vc_max` work too).

```python
from metafopy import predict

print(predict(res, newmods=[5, 25], newscale=[5, 25]))
```

```text
     pred      se   ci.lb   ci.ub   pi.lb   pi.ub
1  0.1466  0.0516  0.0455  0.2477 -0.2315  0.5247
2  0.4454  0.1278  0.1949  0.6960 -0.1899  1.0807
```
