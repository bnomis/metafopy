# Analyses

Worked examples that mirror the [metafor analyses gallery](https://www.metafor-project.org/doku.php/analyses).

## Random-effects model with a publication-bias check

```python
from metafopy import datasets, regtest, rma, selmodel, trimfill

dat = datasets.load('dat.hackshaw1998')
res = rma(data=dat, method='REML')
print(regtest(res))
print(trimfill(res))
print(selmodel(res, type='halfnorm'))
```

`selmodel` accepts the selection functions of metafor (`beta`, `halfnorm`, `negexp`, `logistic`, `power`,
`negexppow`, `stepfun` and variants). See Vevea & Hedges (1995), <https://doi.org/10.1007/BF02294384>.

## Location-scale model

```python
from metafopy import datasets, rma

dat = datasets.load('dat.bangertdrowns2004')
res = rma(yi, vi, mods=..., scale=..., data=dat)
```

See Viechtbauer & López-López (2022), <https://doi.org/10.1002/jrsm.1562>.
