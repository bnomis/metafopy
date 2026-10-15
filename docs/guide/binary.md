# Binary and count data

For 2x2 tables (events and non-events in a treatment and a control group) and for event counts with person-time, the
data can be analysed in several ways. All of them take the raw counts, so you do not call `escalc` yourself.

| Function | Method | Measures |
|----------|--------|----------|
| `rma_uni` | Inverse-variance, on `escalc` effect sizes | any |
| `rma_mh` | Mantel-Haenszel | `OR`, `RR`, `RD`, `IRR`, `IRD` |
| `rma_peto` | Peto's one-step method | `OR` |
| `rma_glmm` | Generalized linear mixed-effects models | `OR`, `IRR`, `PLO`, `IRLN` and the experimental measures |

The arguments are the same throughout: `ai`, `bi`, `ci`, `di` for the cells of the table (events and non-events in the
treatment and control group), or `ai`, `n1i`, `ci`, `n2i` with the group sizes, and `data=`.

## Mantel-Haenszel

Mantel-Haenszel pooling (Mantel and Haenszel, 1959, <https://doi.org/10.1093/jnci/22.4.719>) behaves well with sparse data, and
needs no continuity correction for studies with a zero in a single cell. It is an equal-effects method.

```python
from metafopy import datasets, rma_mh

bcg = datasets.load('dat.bcg')
print(rma_mh(ai='tpos', bi='tneg', ci='cpos', di='cneg', data=bcg, measure='RR'))
```

```text
Equal-Effects Model (k = 13)

I^2 (total heterogeneity / total variability):  92.13%
H^2 (total variability / sampling variability): 12.71

Test for Heterogeneity:
Q(df = 12) = 152.5676, p-val < .0001

Model Results (log scale):

estimate      se      zval    pval    ci.lb    ci.ub
 -0.4537  0.0393  -11.5338  <.0001  -0.5308  -0.3766

Model Results (RR scale):

estimate   ci.lb   ci.ub
  0.6353  0.5881  0.6862
```

## Peto's method

Peto's method (Yusuf et al., 1985, <https://doi.org/10.1016/S0033-0620(85)80003-7>) works on the observed minus expected
number of events and is accurate for rare events and balanced groups:

```python
from metafopy import rma_peto

print(rma_peto(ai='tpos', bi='tneg', ci='cpos', di='cneg', data=bcg))
```

```text
Equal-Effects Model (k = 13)

I^2 (total heterogeneity / total variability):  92.85%
H^2 (total variability / sampling variability): 13.98

Test for Heterogeneity:
Q(df = 12) = 167.7302, p-val < .0001

Model Results (log scale):

estimate      se      zval    pval    ci.lb    ci.ub
 -0.4744  0.0407  -11.6689  <.0001  -0.5541  -0.3948

Model Results (OR scale):

estimate   ci.lb   ci.ub
  0.6222  0.5746  0.6738
```

## Generalized linear mixed-effects models

`rma_glmm` fits the exact binomial or Poisson likelihood with random effects, which avoids the normal approximation and the
continuity corrections of the inverse-variance method. This is the best choice for sparse data, and it is slower. Choose the
model with `model`: `'UM.FS'` and `'UM.RS'` are unconditional models with fixed or random study effects, `'CM.AL'` and
`'CM.EL'` are conditional models (approximate and exact likelihood). See Stijnen et al. (2010),
<https://doi.org/10.1002/sim.4040>.

```python
from metafopy import rma_glmm

print(rma_glmm(ai='tpos', bi='tneg', ci='cpos', di='cneg', data=bcg, measure='OR', model='UM.FS'))
```

```text
Random-Effects Model (k = 13; tau^2 estimator: ML)
Model Type: Unconditional Model with Fixed Study Effects

tau^2 (estimated amount of total heterogeneity): 0.2949
tau (square root of estimated tau^2 value):      0.5430
I^2 (total heterogeneity / total variability):   91.02%
H^2 (total variability / sampling variability):  11.14

Tests for Heterogeneity:
Wld(df = 12) = 163.1649, p-val < .0001
LRT(df = 12) = 176.9544, p-val < .0001

Model Results:

estimate      se     zval    pval    ci.lb    ci.ub
 -0.7450  0.1756  -4.2436  <.0001  -1.0891  -0.4009  ***

---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

`rma_glmm` accepts moderators (`mods='~ ablat'`). `predict`, `blup` and `ranef` work on its result.
Mixed-model fits can differ slightly from R, because lme4 stops at a looser optimiser tolerance; see
[Differences from metafor](../differences.md).

As in metafor, `control={'package': 'GLMMadaptive'}` fits the mixed models with adaptive Gauss-Hermite quadrature
([Pinheiro and Bates, 1995](https://doi.org/10.1007/BF00140873); [GLMMadaptive](https://drizopoulos.github.io/GLMMadaptive/)) and
`control={'package': 'glmmTMB'}` with the Laplace approximation ([Brooks et al., 2017](https://doi.org/10.32614/RJ-2017-066)); `nAGQ` is ignored by glmmTMB.

For the conditional model `CM.EL` with `method='FE'`, `control={'optimizer': 'clogit'}` or `'clogistic'` fit it with the partial likelihood of
`survival::clogit` or `Epi::clogistic`. `link=` and `family=` change the link function or the family of the unconditional models.

The other optimisers of `rma.glmm` are available for `CM.EL` too (`control={'optimizer': 'BFGS'}`, `'Nelder-Mead'`, or `'optim'` with `optmethod=`, `optCtrl={'maxit': ...}`, and `'CG'`, `'L-BFGS-B'`, `'nlm'`, `'bobyqa'`, ...). `optim`'s BFGS and Nelder-Mead are exact ports of R's routines; the rest use the closest SciPy method.
See [differences from metafor](../differences.md) for what can differ.

## Rates

For incidence rates, pass the event counts `x1i`, `x2i` and the person-time `t1i`, `t2i`, with `measure='IRR'`:

```python
from metafopy import escalc

print(escalc(measure='IRR', x1i=[3, 5], t1i=[100, 120], x2i=[7, 4], t2i=[110, 90])[['yi', 'vi']])
```

```text
         yi       vi
0 -0.751988  0.47619
1 -0.064539  0.45000
```

## Zero cells

Studies with a zero cell need a continuity correction for the inverse-variance method. `add`, `to` and `drop00` control it
and have the same meaning in `escalc`, `rma_uni`, `rma_mh`, `rma_peto` and `rma_glmm`. Studies with no events in both groups
carry no information about the relative effect and are dropped by `rma_mh` and `rma_peto` (`drop00=True`). Sweeting et al.
(2004), <https://doi.org/10.1002/sim.1761>, discuss the alternatives.
