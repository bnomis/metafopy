# metafopy

Meta-analysis in Python: a re-implementation of the R package
[metafor](https://www.metafor-project.org/) (reference version 5.2-1) by Wolfgang Viechtbauer.
Function and argument names follow metafor, with `.` replaced by `_`
(`rma.uni` becomes `rma_uni`). Numerical results are checked against metafor
fixtures generated from R.

```python
from metafopy import datasets, escalc, rma

dat = escalc(measure='RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=datasets.load('dat.bcg'))
print(rma('yi', 'vi', data=dat))
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

## Where to start

- [Installation](install.md) for requirements and a check that the install works.
- [Getting started](getting-started.md) for a complete first analysis, from raw 2x2 tables to a forest plot.
- The user guide, for one task at a time:
    - [Effect sizes](guide/effect-sizes.md)
    - [Fitting models](guide/models.md)
    - [Binary and count data](guide/binary.md)
    - [Multilevel and multivariate models](guide/multilevel.md)
    - [Selection models and location-scale models](guide/advanced.md)
    - [Diagnostics](guide/diagnostics.md)
    - [Publication bias](guide/bias.md)
    - [Plots](guide/plots.md)
    - [Plot gallery](guide/gallery.md), the plots of the metafor documentation next to those of metafopy
    - [Datasets](guide/datasets.md)
- [Analyses](analyses.md) for worked examples that mirror well-known metafor analyses.
- [Coming from R](from-r.md) if you already know metafor.
- The [function reference](reference/index.md), generated from the docstrings.
- [Differences from metafor](differences.md) for the places where metafopy and metafor do not agree.

## Why metafopy

metafor is the most complete meta-analysis package available, but it lives in R. metafopy gives Python users the same
models with the same names, defaults and printed output, so the metafor documentation, the
[metafor analyses gallery](https://www.metafor-project.org/doku.php/analyses) and published code translate directly. Every
fitted model is a plain, frozen dataclass of NumPy arrays, so results drop into the rest of the scientific Python stack.

The trade-off is that metafopy follows metafor, including its conventions and its defaults (REML for the heterogeneity
estimate, Wald-type tests, effect sizes on the log scale for ratios). Where Python or NumPy force a different design, the
[differences page](differences.md) says so.

## Citing

Cite metafor when you use metafopy: Viechtbauer, W. (2010). Conducting meta-analyses in R with the metafor package.
*Journal of Statistical Software*, 36(3), 1-48. <https://doi.org/10.18637/jss.v036.i03>

## License

GPL-2.0-or-later, the same as metafor. The bundled datasets come from the R package
[metadat](https://cran.r-project.org/package=metadat).
