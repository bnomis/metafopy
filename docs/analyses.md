# Analyses

Worked examples that mirror the [metafor analyses gallery](https://www.metafor-project.org/doku.php/analyses). The data are
the metadat datasets bundled with metafopy, so every block runs as shown. Compare the output with the metafor pages to see
that the numbers agree.

## BCG vaccine against tuberculosis

Colditz et al. (1994), <https://doi.org/10.1001/jama.1994.03510330076038>, and the analysis of Berkey et al. (1995),
<https://doi.org/10.1002/sim.4780140406>. Trials compared vaccinated and unvaccinated subjects. The effect size
is the log risk ratio.

```python
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot
import numpy

from metafopy import confint, datasets, escalc, forest, funnel, predict, rma

dat = escalc(measure='RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=datasets.load('dat.bcg'))
res = rma('yi', 'vi', data=dat)
print(res)
print(predict(res, transf=numpy.exp))
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

     pred   ci.lb   ci.ub   pi.lb  pi.ub
0  0.4894  0.3441  0.6962  0.1546  1.549
```

The average risk ratio is about 0.5, with a prediction interval so wide that the vaccine could have no effect, or a
strong one, in a new setting. The I^2 value says that nearly all of the variability is heterogeneity, so look for
moderators. Trials were run at different latitudes:

```python
res_lat = rma('yi', 'vi', mods='~ ablat', data=dat)
print(res_lat)
print(predict(res_lat, newmods=[10, 20, 30, 40, 50], transf=numpy.exp))
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
1  0.7185  0.5526  0.9343  0.3936  1.3117
2  0.5371  0.4355  0.6623  0.3005  0.9600
3  0.4015  0.3151  0.5115  0.2218  0.7266
4  0.3001  0.2144  0.4201  0.1586  0.5678
```

The estimated risk ratio falls steadily with absolute latitude: the vaccine does little near the equator and cuts the risk by
well over half in the north and south. The moderator accounts for most of the heterogeneity. Allocation method is another
candidate moderator. Comparing nested models with a likelihood ratio test needs `method='ML'`; the larger model comes first:

```python
from metafopy import anova

ml_lat = rma('yi', 'vi', mods='~ ablat', data=dat, method='ML')
ml_both = rma('yi', 'vi', mods='~ ablat + alloc', data=dat, method='ML')
print(anova(ml_both, ml_lat))
```

```text
         df      AIC      BIC     AICc  logLik     LRT    pval       QE   tau^2      R^2
Full      5  24.0269  26.8517  32.5984 -7.0135     NaN     NaN  26.2034  0.0294      NaN
Reduced   3  21.3713  23.0662  24.0380 -7.6857  1.3444  0.5106  30.7331  0.0344  14.3976
```

Allocation adds nothing once latitude is in the model. Forest and funnel plots of the random-effects model:

```python
forest(res, atransf=numpy.exp, xlab='Risk ratio')
matplotlib.pyplot.savefig('bcg-forest.png')

matplotlib.pyplot.figure()
funnel(res)
matplotlib.pyplot.savefig('bcg-funnel.png')
```

## Beta blockers after myocardial infarction (Yusuf, 1985)

Yusuf et al. (1985), <https://doi.org/10.1016/S0033-0620(85)80003-7>, described Peto's one-step method for pooling 2x2
tables of rare events. The dataset holds several tables of the article; this one is total mortality in short-term trials of
oral beta blockers:

```python
from metafopy import rma_mh, rma_peto

dat = datasets.load('dat.yusuf1985')
print(rma_peto(ai='ai', n1i='n1i', ci='ci', n2i='n2i', data=dat, subset=dat['table'] == '6'))
```

```text
Equal-Effects Model (k = 21)

I^2 (total heterogeneity / total variability):  0.00%
H^2 (total variability / sampling variability): 0.64

Test for Heterogeneity:
Q(df = 17) = 10.8275, p-val = 0.8654

Model Results (log scale):

estimate      se     zval    pval    ci.lb   ci.ub
 -0.0692  0.1194  -0.5794  0.5623  -0.3031  0.1648

Model Results (OR scale):

estimate   ci.lb   ci.ub
  0.9332  0.7385  1.1792
```

One trial has missing counts, which metafopy reports with a warning, as metafor does, and trials without events in either group are dropped by the Peto and Mantel-Haenszel methods. The Mantel-Haenszel estimate of the same trials is close:

```python
print(rma_mh(ai='ai', n1i='n1i', ci='ci', n2i='n2i', data=dat, subset=dat['table'] == '6', measure='OR'))
```

```text
Equal-Effects Model (k = 21)

I^2 (total heterogeneity / total variability):  0.00%
H^2 (total variability / sampling variability): 0.55

Test for Heterogeneity:
Q(df = 17) = 9.4233, p-val = 0.9260

Model Results (log scale):

estimate      se     zval    pval    ci.lb   ci.ub
 -0.0690  0.1192  -0.5786  0.5629  -0.3026  0.1646

Model Results (OR scale):

estimate   ci.lb   ci.ub
  0.9334  0.7389  1.1790

Cochran-Mantel-Haenszel Test:    CMH =  0.2701, df = 1,  p-val = 0.6033
Tarone's Test for Heterogeneity: X^2 = 10.9288, df = 17, p-val = 0.8602
```

Both treat the trials as sharing a single effect. When the trials are heterogeneous, as in the BCG data, fit a random-effects
model instead, see the [model guide](guide/models.md).

## Hospital stay of stroke patients

Normand (1999), <https://doi.org/10.1002/(SICI)1097-0258(19990215)18:3<321::AID-SIM28>3.0.CO;2-P>, compared the length of
hospital stay of stroke patients under specialized and under routine care. The raw mean difference is meaningful, so use `measure='MD'`:

```python
dat = escalc(measure='MD', n1i='n1i', n2i='n2i', m1i='m1i', m2i='m2i', sd1i='sd1i', sd2i='sd2i', data=datasets.load('dat.normand1999'))
res = rma('yi', 'vi', data=dat, slab=dat['source'])
print(res)
print(confint(res))
```

```text
Random-Effects Model (k = 9; tau^2 estimator: REML)

tau^2 (estimated amount of total heterogeneity): 684.6462 (SE = 359.7541)
tau (square root of estimated tau^2 value):      26.1657
I^2 (total heterogeneity / total variability):   98.97%
H^2 (total variability / sampling variability):  97.21

Test for Heterogeneity:
Q(df = 8) = 238.9158, p-val < .0001

Model Results:

estimate      se     zval    pval     ci.lb   ci.ub
-15.1060  8.9466  -1.6885  0.0913  -32.6409  2.4289  .

---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

        estimate     ci.lb      ci.ub
tau^2   684.6462  292.9181  2889.4513
tau      26.1657   17.1149    53.7536
I^2(%)   98.9713   97.6282    99.7543
H^2      97.2076   42.1614   407.0306
```

The huge heterogeneity makes the pooled mean difference hard to interpret. The prediction interval shows how far a new study
may deviate:

```python
print(predict(res))
```

```text
     pred      se    ci.lb   ci.ub    pi.lb    pi.ub
0 -15.106  8.9466 -32.6409  2.4289 -69.3049  39.0928
```

## Teacher expectancy and IQ

Raudenbush (1984), <https://doi.org/10.1037/0022-0663.76.1.85>, synthesised experiments on whether teachers' expectations affect pupils' IQ. The effect sizes
are standardized mean differences. The studies differ in the number of weeks of contact between teacher and pupils before the
expectancy was induced, which may moderate the effect:

```python
dat = datasets.load('dat.raudenbush1985')
print(rma('yi', 'vi', data=dat))
print(rma('yi', 'vi', mods='~ weeks', data=dat))
```

```text
Random-Effects Model (k = 19; tau^2 estimator: REML)

tau^2 (estimated amount of total heterogeneity): 0.0188 (SE = 0.0155)
tau (square root of estimated tau^2 value):      0.1372
I^2 (total heterogeneity / total variability):   41.86%
H^2 (total variability / sampling variability):  1.72

Test for Heterogeneity:
Q(df = 18) = 35.8295, p-val = 0.0074

Model Results:

estimate      se    zval    pval    ci.lb   ci.ub
  0.0837  0.0516  1.6208  0.1051  -0.0175  0.1849

---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1


Mixed-Effects Model (k = 19; tau^2 estimator: REML)

tau^2 (estimated amount of residual heterogeneity):     0.0112 (SE = 0.0128)
tau (square root of estimated tau^2 value):             0.1057
I^2 (residual heterogeneity / unaccounted variability): 29.40%
H^2 (unaccounted variability / sampling variability):   1.42
R^2 (amount of heterogeneity accounted for):            40.63%

Test for Residual Heterogeneity:
QE(df = 17) = 27.6730, p-val = 0.0489

Test of Moderators (coefficient 2):
QM(df = 1) = 7.5094, p-val = 0.0061

Model Results:

         estimate      se     zval    pval    ci.lb    ci.ub
intrcpt    0.1942  0.0633   3.0649  0.0022   0.0700   0.3183  **
weeks     -0.0157  0.0057  -2.7403  0.0061  -0.0270  -0.0045  **

---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

The effect is smaller the longer teachers had known their pupils. The Knapp-Hartung adjustment is more conservative with few
studies, and still finds the moderator significant:

```python
print(rma('yi', 'vi', mods='~ weeks', data=dat, test='knha'))
```

```text
Mixed-Effects Model (k = 19; tau^2 estimator: REML)

tau^2 (estimated amount of residual heterogeneity):     0.0112 (SE = 0.0128)
tau (square root of estimated tau^2 value):             0.1057
I^2 (residual heterogeneity / unaccounted variability): 29.40%
H^2 (unaccounted variability / sampling variability):   1.42
R^2 (amount of heterogeneity accounted for):            40.63%

Test for Residual Heterogeneity:
QE(df = 17) = 27.6730, p-val = 0.0489

Test of Moderators (coefficient 2):
F(df1 = 1, df2 = 17) = 6.0270, p-val = 0.0252

Model Results:

         estimate      se     tval  df    pval    ci.lb    ci.ub
intrcpt    0.1942  0.0707   2.7457  17  0.0138   0.0450   0.3434  *
weeks     -0.0157  0.0064  -2.4550  17  0.0252  -0.0293  -0.0022  *

---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

## Publication bias in passive smoking studies

```python
from metafopy import fsn, ranktest, regtest, selmodel, tes, trimfill

dat = datasets.load('dat.hackshaw1998')
res = rma(data=dat, method='ML')
print(regtest(res))
print(ranktest(res))
print(trimfill(res))
print(fsn(res))
print(selmodel(res, type='halfnorm'))
```

```text
Regression Test for Funnel Plot Asymmetry

Model:     mixed-effects meta-regression model
Predictor: standard error

Test for Funnel Plot Asymmetry: z = 2.1713, p = 0.0299
Limit Estimate (as sei -> 0):   b = 0.0189 (CI: -0.1736, 0.2113)


Rank Correlation Test for Funnel Plot Asymmetry

Kendall's tau = 0.1441, p = 0.2157


Estimated number of missing studies on the left side: 7 (SE = 4.0402)

Random-Effects Model (k = 44; tau^2 estimator: ML)

tau^2 (estimated amount of total heterogeneity): 0.0222 (SE = 0.0170)
tau (square root of estimated tau^2 value):      0.1490
I^2 (total heterogeneity / total variability):   26.86%
H^2 (total variability / sampling variability):  1.37

Test for Heterogeneity:
Q(df = 43) = 60.5603, p-val = 0.0397

Model Results:

estimate      se    zval    pval   ci.lb   ci.ub
  0.1736  0.0476  3.6472  0.0003  0.0803  0.2669  ***

---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1


Fail-safe N Calculation Using the General Approach

Average Effect Size:         0.2171 (with file drawer: 0.0458)
Amount of Heterogeneity:     0.0204 (with file drawer: 0.0289)
Observed Significance Level: <.0001 (with file drawer: 0.0511)
Target Significance Level:   0.05

Fail-safe N: 116


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

The data are from Hackshaw et al. (1997), <https://doi.org/10.1136/bmj.315.7114.980>, on the risk of lung cancer in
women exposed to tobacco smoke by a smoking spouse. The regression test is significant but the rank correlation test is not,
trim and fill imputes studies on the left of the funnel and lowers the pooled estimate, and the selection model does not
reject the absence of selection. Such mixed answers are typical. See [Publication bias](guide/bias.md).

## Multilevel meta-analysis of recidivism and mental health

Assink and Wibbelink (2016), <https://doi.org/10.20982/tqmp.12.3.p154>, use studies on the association between recidivism and
mental health in delinquent juveniles. Studies contribute several effect sizes each, so a three-level model has random effects
for studies and for effect sizes within studies. The sampling errors of the effect sizes of one study are correlated. The
correlation is not reported, so the sampling variance-covariance matrix assumes 0.6, and cluster-robust inference protects
against a wrong assumption:

```python
from metafopy import rma_mv, robust, vcalc

dat = datasets.load('dat.assink2016')
V = vcalc(vi=dat['vi'], cluster=dat['study'], obs=dat['esid'], rho=0.6)
res = rma_mv(yi='yi', V=V, random='~ 1 | study/esid', data=dat)
print(res)
print(robust(res, cluster=dat['study']))
```

```text
Multivariate Meta-Analysis Model (k = 100; method: REML)

Variance Components:

            estim    sqrt  nlvls  fixed      factor
sigma^2.1  0.0807  0.2841     17     no       study
sigma^2.2  0.1545  0.3931    100     no  study/esid

Test for Heterogeneity:
Q(df = 99) = 745.4385, p-val < .0001

Model Results:

estimate      se    zval    pval   ci.lb   ci.ub
  0.3678  0.0965  3.8097  0.0001  0.1786  0.5570  ***

---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1


Multivariate Meta-Analysis Model (k = 100; method: REML)

Variance Components:

            estim    sqrt  nlvls  fixed      factor
sigma^2.1  0.0807  0.2841     17     no       study
sigma^2.2  0.1545  0.3931    100     no  study/esid

Test for Heterogeneity:
Q(df = 99) = 745.4385, p-val < .0001

Number of estimates:   100
Number of clusters:    17
Estimates per cluster: 1-22 (mean: 5.88, median: 5)

Model Results:

estimate      se¹    tval¹  df¹    pval¹   ci.lb¹   ci.ub¹
  0.3678  0.0962   3.8210   16   0.0015   0.1637   0.5718   **

---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

1) results based on cluster-robust inference (var-cov estimator: CR1,
   approx t-test and confidence interval, df: residual method)
```

Moderators such as publication year are added with `mods`, and models with different `rho` show the sensitivity:

```python
for rho in (0.3, 0.6, 0.9):
    V = vcalc(vi=dat['vi'], cluster=dat['study'], obs=dat['esid'], rho=rho)
    fit = rma_mv(yi='yi', V=V, random='~ 1 | study/esid', data=dat)
    print(rho, float(fit.beta[0]), float(fit.se[0]))
```

```text
0.3 0.394264622380792 0.10553049844741037
0.6 0.36775488559263914 0.0965318193954781
0.9 0.3487289923578839 0.09146206349192922
```

## Schools within districts

Konstantopoulos (2011), <https://doi.org/10.1002/jrsm.35>, analysed studies of the effect of modified school calendars on
student achievement. Each study is a school, and schools are nested in districts:

```python
dat = datasets.load('dat.konstantopoulos2011')
res = rma_mv(yi='yi', V='vi', random='~ 1 | district/school', data=dat)
print(res)
print(confint(res, sigma2=1))
```

```text
Multivariate Meta-Analysis Model (k = 56; method: REML)

Variance Components:

            estim    sqrt  nlvls  fixed           factor
sigma^2.1  0.0651  0.2551     11     no         district
sigma^2.2  0.0327  0.1809     56     no  district/school

Test for Heterogeneity:
Q(df = 55) = 578.8640, p-val < .0001

Model Results:

estimate      se    zval    pval   ci.lb   ci.ub
  0.1847  0.0846  2.1845  0.0289  0.0190  0.3504  *

---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

           estimate   ci.lb   ci.ub
sigma^2.2    0.0327  0.0163  0.0628
sigma.2      0.1809  0.1276  0.2507
```

## Writing-to-learn interventions: a location-scale model

Bangert-Drowns et al. (2004), <https://doi.org/10.3102/00346543074001029>. The studies examine school-based
writing-to-learn interventions and academic achievement. The effect may depend on the length of the intervention, and so may
the amount of heterogeneity:

```python
dat = datasets.load('dat.bangertdrowns2004')
print(rma('yi', 'vi', mods='~ length', data=dat))
print(rma('yi', 'vi', mods='~ length', scale='~ length', data=dat))
```

```text
Mixed-Effects Model (k = 46; tau^2 estimator: REML)

tau^2 (estimated amount of residual heterogeneity):     0.0441 (SE = 0.0188)
tau (square root of estimated tau^2 value):             0.2100
I^2 (residual heterogeneity / unaccounted variability): 55.26%
H^2 (unaccounted variability / sampling variability):   2.24
R^2 (amount of heterogeneity accounted for):            5.08%

Test for Residual Heterogeneity:
QE(df = 44) = 96.2810, p-val < .0001

Test of Moderators (coefficient 2):
QM(df = 1) = 4.2266, p-val = 0.0398

Model Results:

         estimate      se    zval    pval    ci.lb   ci.ub
intrcpt    0.0692  0.0825  0.8384  0.4018  -0.0925  0.2309
length     0.0149  0.0073  2.0559  0.0398   0.0007  0.0292  *

---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1


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

Viechtbauer and Lopez-Lopez (2022), <https://doi.org/10.1002/jrsm.1562>, use these data. The length of the intervention
does not predict the amount of heterogeneity here, so the extra scale model adds little.
