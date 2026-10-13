# Getting started

metafopy requires Python 3.14.7 or later (below 3.15).

```python
from metafopy import confint, datasets, predict, rma

dat = datasets.load('dat.hackshaw1998')
res = rma(data=dat, method='REML')
print(res)
print(confint(res))
print(predict(res))
```

Fixed-effects, random-effects and mixed-effects models, the Mantel-Haenszel and Peto methods, generalized
linear mixed-effects models (`rma_glmm`), multivariate and multilevel models (`rma_mv`), selection models
(`selmodel`) and location-scale models (`rma(..., scale=...)`) share the metafor interface: fit, then call
`predict`, `confint`, `anova`, `profile`, `influence` and the plot functions on the result.
