# Effect sizes

Most analyses start from raw study data (2x2 tables, means and standard deviations, correlations, proportions) that
`escalc` converts to an effect size `yi` and its sampling variance `vi`. If the studies already report effect sizes and
standard errors, skip `escalc` and pass them to `rma` directly.

## escalc

`escalc` takes a `measure` and the arguments that measure needs, and returns a pandas `DataFrame`. With `data=` the new
columns `yi` and `vi` are appended to your data.

```python
from metafopy import datasets, escalc

dat = datasets.load('dat.curtis1998')
dat = escalc(measure='ROM', m1i='m1i', sd1i='sd1i', n1i='n1i', m2i='m2i', sd2i='sd2i', n2i='n2i', data=dat)
print(dat[['id', 'genus', 'species', 'yi', 'vi']].head())
```

```text
   id    genus    species        yi        vi
0  21    ALNUS      RUBRA  0.550192  0.038472
1  22    ALNUS      RUBRA  0.147040  0.017455
2  27     ACER     RUBRUM  0.437769  0.032753
3  32  QUERCUS     PRINUS -0.110832  0.029516
4  35    MALUS  DOMESTICA  0.117135  0.046786
```

Here `ROM` is the log ratio of means, comparing woody plant mass under elevated and ambient CO2, the example of Hedges et
al. (1999), <https://doi.org/10.1890/0012-9658(1999)080[1150:TMAORR]2.0.CO;2>. A ratio of means suits outcomes that are
positive and measured on a ratio scale.

The measures are metafor's, with the same abbreviations, and fall into groups by the data they need:

| Data | Arguments | Measures, for example |
|------|-----------|-----------------------|
| 2x2 tables | `ai`, `bi`, `ci`, `di` (or `n1i`, `n2i` with `ai`, `ci`) | `RR`, `OR`, `RD`, `PETO`, `AS`, `PHI`, `YUQ`, `RTET` |
| Person-time | `x1i`, `x2i`, `t1i`, `t2i` | `IRR`, `IRD`, `IRSD` |
| Two group means | `m1i`, `m2i`, `sd1i`, `sd2i`, `n1i`, `n2i` | `MD`, `SMD`, `SMDH`, `ROM`, `VR`, `CVR` |
| Correlations | `ri`, `ni` | `COR`, `ZCOR` |
| Proportions | `xi`, `ni` | `PR`, `PLO`, `PAS`, `PFT` |
| Incidence rates | `xi`, `ti` | `IR`, `IRLN`, `IRS`, `IRFT` |
| Single means | `mi`, `sdi`, `ni` | `MN`, `MNLN`, `SMN`, `SDLN`, `CVLN` |
| Change scores | `m1i`, `m2i`, `sd1i`, `sd2i`, `ni`, `ri` | `MC`, `SMCC`, `SMCR`, `ROMC` |
| Partial and semi-partial correlations, R^2 | `ri`, `ti`, `ni`, `mi`, `r2i` | `PCOR`, `SPCOR`, `R2`, `ZR2` |
| Precomputed | `yi`, `vi` or `sei` | `GEN` |

`metafopy.effects.measures.MEASURES` is the full list, and the [escalc reference](../reference/effects.md) the
required arguments of each. The metafor documentation describes the formulas:
<https://wviechtb.github.io/metafor/reference/escalc.html>.

## Zero cells and bias correction

Counts of zero make log ratios undefined. `add`, `to` and `drop00` control the continuity correction exactly as in
metafor: by default `0.5` is added to all four cells of a study that has a zero cell (`to='only0'`). Set `add=0` to turn
this off, or `to='all'` to correct every study.

```python
from metafopy import escalc

zero = datasets.load('dat.hine1989')
print(escalc(measure='OR', ai='ai', n1i='n1i', ci='ci', n2i='n2i', data=zero)[['yi', 'vi']])
print(escalc(measure='OR', ai='ai', n1i='n1i', ci='ci', n2i='n2i', data=zero, add=0)[['yi', 'vi']])
```

```text
         yi        vi
0  0.819899  1.550837
1  0.000000  0.550000
2  0.453784  0.436002
3  0.326001  0.363800
4  0.847298  0.495608
5  1.004583  0.354944
         yi        vi
0  0.819899  1.550837
1  0.000000  0.550000
2  0.453784  0.436002
3  0.326001  0.363800
4  0.847298  0.495608
5  1.004583  0.354944
```

Measures such as `SMD` apply the small-sample bias correction of Hedges (1981), <https://doi.org/10.3102/10769986006002107>, by default (`correct=True`). For the choice
of sampling variance formula see `vtype`, described in the metafor documentation.

## Correlations and proportions

The raw correlation has a variance that depends on the correlation itself. Meta-analyses of correlations therefore work on
Fisher's z (`ZCOR`) and back-transform for presentation, following Borenstein et al. (2009),
<https://doi.org/10.1002/9780470743386>:

```python
import numpy

from metafopy import predict, rma

mol = datasets.load('dat.molloy2014')
mol = escalc(measure='ZCOR', ri='ri', ni='ni', data=mol)
res = rma('yi', 'vi', data=mol)
print(predict(res, transf=numpy.tanh))
```

```text
     pred   ci.lb   ci.ub   pi.lb   pi.ub
0  0.1488  0.0878  0.2087 -0.0371  0.3248
```

For proportions, the logit (`PLO`) is a common choice, with `transf.ilogit` to go back:

```python
from metafopy import transf

pri = datasets.load('dat.pritz1997')
pri = escalc(measure='PLO', xi='xi', ni='ni', data=pri)
print(predict(rma('yi', 'vi', data=pri), transf=transf.ilogit))
```

```text
     pred   ci.lb   ci.ub   pi.lb   pi.ub
0  0.7575  0.6605  0.8337  0.4661  0.9179
```

## Precomputed effect sizes

If a dataset already has effect sizes and sampling variances, pass them straight to `rma`, or to `escalc` with
`measure='GEN'` to attach labels. Standard errors work too, with `sei=`:

```python
from metafopy import rma

rau = datasets.load('dat.raudenbush1985')
print(rma('yi', 'vi', data=rau))
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
```

## Study labels, subsets and flipping

`slab` gives study labels used in plots and printed output, `subset` selects studies, and `flip` changes the sign of the
effect size of selected studies. All three accept a column name or an array.

```python
stay = escalc(measure='MD', n1i='n1i', n2i='n2i', m1i='m1i', m2i='m2i', sd1i='sd1i', sd2i='sd2i', data=datasets.load('dat.normand1999'), slab='source')
print(stay.attrs['slab'])
```

```text
['Edinburgh', 'Orpington-Mild', 'Orpington-Moderate', 'Orpington-Severe', 'Montreal-Home', 'Montreal-Transfer', 'Newcastle', 'Umea', 'Uppsala']
```

## Summarising

`summary_escalc` adds the standard errors, z values, p values and confidence limits of each study to an `escalc` result:

```python
from metafopy import summary_escalc

print(summary_escalc(stay)[['source', 'yi', 'sei', 'zi', 'pval', 'ci.lb', 'ci.ub']])
```

```text
               source    yi        sei  ...          pval      ci.lb      ci.ub
0           Edinburgh -20.0   6.364591  ...  1.675772e-03 -32.474370  -7.525630
1      Orpington-Mild  -2.0   1.442444  ...  1.655835e-01  -4.827139   0.827139
2  Orpington-Moderate -55.0   3.962121  ...  8.206280e-44 -62.765615 -47.234385
3    Orpington-Severe -71.0  12.256518  ...  6.920755e-09 -95.022333 -46.977667
4       Montreal-Home  -4.0   4.160251  ...  3.363107e-01 -12.153943   4.153943
5   Montreal-Transfer   1.0   1.080436  ...  3.546786e-01  -1.117615   3.117615
6           Newcastle  11.0   9.725694  ...  2.580447e-01  -8.062010  30.062010
7                Umea -10.0   2.512146  ...  6.872424e-05 -14.923716  -5.076284
8             Uppsala   7.0   4.454471  ...  1.160770e-01  -1.730602  15.730602

[9 rows x 7 columns]
```

## Transformations

`metafopy.transf` holds metafor's transformation functions (`transf.ilogit`, `transf.ztor`, `transf.dtocles` and so on, also
imported at the top level as `transf_ilogit`, `transf_ztor`, ...). Pass
them as `transf=` to `predict` and `confint`, or as `atransf=` to the plot functions. NumPy functions such as `numpy.exp` and
`numpy.tanh` work as well.
