# Multilevel and multivariate models

`rma_mv` fits models in which the effect sizes are not independent: several effect sizes per study, studies nested in
districts, several outcomes measured in the same subjects, effects that share a sampling error or are correlated through a
phylogeny. It mirrors metafor's [`rma.mv`](https://wviechtb.github.io/metafor/reference/rma.mv.html) and follows the
interface closely:

- `yi`: the effect sizes, and `V`: their sampling variances or the whole variance-covariance matrix.
- `random`: one or more formulas of the form `~ 1 | factor` (random intercepts) or `~ inner | outer` (a structured
  variance-covariance matrix). Nested factors are written `~ 1 | district/school`.
- `struct`: the structure of the variance-covariance matrix for `~ inner | outer` terms.

## A three-level model

The schools data of Konstantopoulos (2011), <https://doi.org/10.1002/jrsm.35>, have studies (schools) nested in districts:

```python
from metafopy import datasets, rma_mv

dat = datasets.load('dat.konstantopoulos2011')
res = rma_mv(yi='yi', V='vi', random='~ 1 | district/school', data=dat)
print(res)
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
```

The variance components are `sigma^2.1` (between districts) and `sigma^2.2` (between schools within districts).

## Several effect sizes per study

When a study contributes several effect sizes, the sampling errors are correlated. If the correlation is unknown, assume one
and build the sampling variance-covariance matrix with `vcalc`, then check the sensitivity of the conclusions to it:

```python
from metafopy import vcalc

dat = datasets.load('dat.assink2016')
V = vcalc(vi=dat['vi'], cluster=dat['study'], obs=dat['esid'], rho=0.6)
res = rma_mv(yi='yi', V=V, random='~ 1 | study/esid', data=dat)
print(res)
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
```

`rcalc` does the same for correlation coefficients, and `bldiag` and `blsplit` build and split block-diagonal matrices.

## Moderators

Moderators work as in `rma_uni`, with `mods`:

```python
res = rma_mv(yi='yi', V=V, mods='~ year', random='~ 1 | study/esid', data=dat)
print(res)
```

```text
Multivariate Meta-Analysis Model (k = 100; method: REML)

Variance Components:

            estim    sqrt  nlvls  fixed      factor
sigma^2.1  0.0408  0.2019     17     no       study
sigma^2.2  0.1557  0.3945    100     no  study/esid

Test for Residual Heterogeneity:
QE(df = 98) = 743.4621, p-val < .0001

Test of Moderators (coefficient 2):
QM(df = 1) = 6.2292, p-val = 0.0126

Model Results:

         estimate      se     zval    pval    ci.lb    ci.ub
intrcpt    0.3835  0.0825   4.6508  <.0001   0.2219   0.5452  ***
year      -0.0388  0.0155  -2.4958  0.0126  -0.0692  -0.0083    *

---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
```

## Cluster-robust inference

If the dependence structure is only approximately right, use cluster-robust (sandwich) inference. `robust` replaces the
standard errors, tests and confidence intervals of a fit, with the study as the cluster (Hedges et al., 2010,
<https://doi.org/10.1002/jrsm.5>):

```python
from metafopy import robust

print(robust(res, cluster=dat['study']))
```

```text
Multivariate Meta-Analysis Model (k = 100; method: REML)

Variance Components:

            estim    sqrt  nlvls  fixed      factor
sigma^2.1  0.0408  0.2019     17     no       study
sigma^2.2  0.1557  0.3945    100     no  study/esid

Test for Residual Heterogeneity:
QE(df = 98) = 743.4621, p-val < .0001

Number of estimates:   100
Number of clusters:    17
Estimates per cluster: 1-22 (mean: 5.88, median: 5)

Test of Moderators (coefficient 2):¹
F(df1 = 1, df2 = 15) = 3.2611, p-val = 0.0910

Model Results:

         estimate      se¹     tval¹  df¹    pval¹    ci.lb¹   ci.ub¹
intrcpt    0.3835  0.0912    4.2059   15   0.0008    0.1892   0.5779   ***
year      -0.0388  0.0215   -1.8058   15   0.0910   -0.0845   0.0070     .

---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

1) results based on cluster-robust inference (var-cov estimator: CR1,
   approx t/F-tests and confidence intervals, df: residual method)
```

The default is metafor's CR1 estimator with a t-test. `club_sandwich=True` uses the CR0 to CR3 estimators with the
small-sample corrections of the R package [clubSandwich](https://jepusto.github.io/clubSandwich/); see Pustejovsky and
Tipton (2018), <https://doi.org/10.1080/07350015.2016.1247004>.

## Confidence intervals for variance components

```python
from metafopy import confint

print(confint(rma_mv(yi='yi', V='vi', random='~ 1 | study/esid', data=dat), sigma2=1))
```

```text
           estimate   ci.lb   ci.ub
sigma^2.2    0.1120  0.0671  0.1810
sigma.2      0.3347  0.2590  0.4255
```

The profile-likelihood interval for the component with 0-based index `sigma2=1` is computed. Without an index, intervals for
all components are returned.

## Known correlation matrices

For phylogenetic and other known structures, give the correlation matrix as `R={'species.id': matrix}`: a dict of
`DataFrame`s whose row and column names match the levels of the grouping factor. The bundled `dat.moura2021` comes with
its phylogenetic tree (Moura et al., 2021), and `PhyloTree.vcv` turns the tree into a correlation matrix, as `ape::vcv` does in
R. This model has many effect sizes and takes a while to fit:

```python
# docs: skip
tree = datasets.load_tree('dat.moura2021.tree')
dat = escalc(measure='ZCOR', ri='ri', ni='ni', data=datasets.load('dat.moura2021'))
dat['obs'] = range(1, len(dat) + 1)
res = rma_mv(yi='yi', V='vi', random=['~ 1 | species.id', '~ 1 | study.id', '~ 1 | obs'], R={'species.id': tree.vcv(corr=True)}, data=dat)
print(res)
```

See the `R` argument of [`rma_mv`](../reference/models.md#metafopy.rma_mv).

## What works on an rma_mv fit

`predict`, `anova`, `confint` (profile-likelihood intervals for the variance components) and `robust` accept an `rma_mv`
fit. The leave-one-out and influence diagnostics are only available for `rma_uni` fits, see [Diagnostics](diagnostics.md).
