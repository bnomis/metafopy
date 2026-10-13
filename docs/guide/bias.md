# Publication bias

Small-study effects and publication bias distort the pooled estimate when studies with small or non-significant effects
are less likely to be published. No method can prove that there is no bias; use several as sensitivity analyses, together with a
[funnel plot](plots.md#funnel-plot). Sterne et al. (2011), <https://doi.org/10.1136/bmj.d4002>, give recommendations.

The examples use the passive smoking data of Hackshaw et al.:

```python
from metafopy import datasets, rma

dat = datasets.load('dat.hackshaw1998')
res = rma(data=dat)
```

The data already hold `yi` and `vi`, so `rma(data=dat)` needs no column names.

## Regression test

The regression test of Egger et al. (1997), <https://doi.org/10.1136/bmj.315.7109.629>, regresses the effects on their
standard errors. A slope that differs from zero indicates funnel plot asymmetry. The default model is metafor's
`model='rma'` (a weighted regression with a random-effects component). `predictor` chooses the predictor: `'sei'`, `'vi'`,
`'ni'` or `'ninv'`, and `model='lm'` gives the classical test.

```python
from metafopy import regtest

print(regtest(res))
print(regtest(res, model='lm'))
```

```text
Regression Test for Funnel Plot Asymmetry

Model:     mixed-effects meta-regression model
Predictor: standard error

Test for Funnel Plot Asymmetry: z = 2.1055, p = 0.0352
Limit Estimate (as sei -> 0):   b = 0.0216 (CI: -0.1774, 0.2207)


Regression Test for Funnel Plot Asymmetry

Model:     weighted regression with multiplicative dispersion
Predictor: standard error

Test for Funnel Plot Asymmetry: t = 2.3824, df = 35, p = 0.0228
Limit Estimate (as sei -> 0):   b = 0.0050 (CI: -0.1694, 0.1794)
```

## Rank correlation test

Begg and Mazumdar's (1994), <https://doi.org/10.2307/2533446>, test is the rank correlation between standardized effects and
their variances:

```python
from metafopy import ranktest

print(ranktest(res))
```

```text
Rank Correlation Test for Funnel Plot Asymmetry

Kendall's tau = 0.1441, p = 0.2157
```

## Trim and fill

Trim and fill (Duval and Tweedie, 2000, <https://doi.org/10.1111/j.0006-341X.2000.00455.x>) estimates the number of missing
studies on one side of the funnel plot, adds them, and re-estimates the pooled effect. It applies to models without
moderators.

```python
from metafopy import trimfill

tf = trimfill(res)
print(tf)
```

```text
Estimated number of missing studies on the left side: 7 (SE = 4.0399)

Random-Effects Model (k = 44; tau^2 estimator: REML)

tau^2 (estimated amount of total heterogeneity): 0.0245 (SE = 0.0182)
tau (square root of estimated tau^2 value):      0.1564
I^2 (total heterogeneity / total variability):   28.80%
H^2 (total variability / sampling variability):  1.40

Test for Heterogeneity:
Q(df = 43) = 60.4755, p-val = 0.0404

Model Results:

estimate      se    zval    pval   ci.lb   ci.ub
  0.1745  0.0484  3.6024  0.0003  0.0795  0.2694  ***

---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

Choose `side='left'` or `'right'` to fix the side with the missing studies and `estimator` (`'L0'`, `'R0'`, `'Q0'`). The
filled data show in a funnel plot with `funnel(tf)`.

## Fail-safe N

The fail-safe N is the number of unpublished null studies that would be needed to make the pooled result non-significant.
`fsn` offers the `type` values `'Rosenthal'`, `'Orwin'`, `'Rosenberg'`, `'General'` (the default for model fits) and `'Binomial'`. The method has been criticised, see
Becker (2005), <https://doi.org/10.1002/0470870168.ch7>, and `trimfill` and selection models are usually preferable.

```python
from metafopy import fsn

print(fsn(res))
```

```text
Fail-safe N Calculation Using the General Approach

Average Effect Size:         0.2189 (with file drawer: 0.0473)
Amount of Heterogeneity:     0.0224 (with file drawer: 0.0312)
Observed Significance Level: <.0001 (with file drawer: 0.0504)
Target Significance Level:   0.05

Fail-safe N: 112
```

## Test of excess significance

`tes` tests whether more studies are significant than their power would suggest (Ioannidis and Trikalinos, 2007,
<https://doi.org/10.1177/1740774507079441>):

```python
from metafopy import tes

print(tes(res))
```

```text
Test of Excess Significance

Observed Number of Significant Findings: 7 (out of 37)
Expected Number of Significant Findings: 7.0244
Observed Number / Expected Number:       0.9965

Estimated Power of Tests (based on theta = 0.2189)

   min      q1  median      q3     max
0.0652  0.0933  0.1476  0.2685  0.5605

Test of Excess Significance: p = 0.5041 (X^2 = 0.0001, df = 1)
Limit Estimate (theta_lim):  0.1028 (where p = 0.1)
```

## Henmi-Copas interval

`hc` gives a confidence interval for a random-effects model that is robust to small-study effects and has better coverage than
the Wald interval when the number of studies is small (Henmi and Copas, 2010, <https://doi.org/10.1002/sim.4029>):

```python
from metafopy import hc

print(hc(res))
```

```text
    method   tau2 estimate     se  ci.lb  ci.ub
rma   REML 0.0224   0.2189 0.0494 0.1221 0.3157
hc      DL 0.0170   0.1858 0.0493 0.0830 0.2885
```

## Selection models

For a model-based correction use [`selmodel`](advanced.md#selection-models).
