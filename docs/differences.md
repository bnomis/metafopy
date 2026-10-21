# Differences from metafor

metafopy follows metafor, and for nearly every function it gives the same numbers, to the digits metafor prints. This page shows
where it does not. Each difference has a short summary, the same call in R and in Python with the output of both, and the reason.
The R code and the Python code of an example do the same thing, so any difference in the two outputs is the difference.

Everything that is too small or too specific for an example here is in the [detailed notes](differences-notes.md), grouped by
topic. [Coming from R](from-r.md) lists the rules of the translation.

The outputs below were captured on 2026-10-21 with R 4.3.3, metafor 5.2-1, emmeans 2.0.4 and lme4 2.0.6, and with metafopy at the
commit of the same day. They are a record of that run, not a promise; the tests in `src/tests` and `bin/metafor-audit.sh` are what
keeps the two in line.

## By design

Differences that follow from Python, not from a gap in the port.

### Names and 0-based indices

R's `.` in a name becomes `_`, and every index counts from 0. The study index `ids` and the position that `which.max` returns
differ by one for that reason; the studies are the same.

*Why:* Python counts from 0 and has no `.` in names. Printed `escalc` frames still number their rows from 1, as R's do.

<div class="compare" markdown>
<div markdown>

**R, metafor 5.2-1**

```r
library(metafor)
dat <- escalc(measure="RR", ai=tpos, bi=tneg, ci=cpos, di=cneg, data=dat.bcg)
res <- rma(yi, vi, data=dat)
res$ids[1:3]
which.max(abs(rstudent(res)$z))
```

```text
[1] 1 2 3
[1] 4
```

</div>
<div markdown>

**Python, metafopy**

```python
import numpy

from metafopy import datasets, escalc, rma, rstudent

dat = escalc(measure='RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=datasets.load('dat.bcg'))
res = rma('yi', 'vi', data=dat)
print(res.ids[:3])
print(numpy.argmax(numpy.abs(rstudent(res).z)))
```

```text
[0 1 2]
3
```

</div>
</div>

### Fitted models are read-only

`res$field <- value` changes a fit in R and can add a field that does not exist. A fit in metafopy is a frozen dataclass: use
`dataclasses.replace(res, field=value)` for a changed copy. Assigning to it, or adding an attribute, raises
`FrozenInstanceError`.

*Why:* a fit that quietly changes after the fact is a common source of wrong numbers, and R's list-like access is kept for
reading (`res.tau2`, `res.beta`).

<div class="compare" markdown>
<div markdown>

**R, metafor 5.2-1**

```r
library(metafor)
dat <- escalc(measure="RR", ai=tpos, bi=tneg, ci=cpos, di=cneg, data=dat.bcg)
res <- rma(yi, vi, data=dat)
res$tau2 <- 0
res$tau2
res$zz <- 1
res$zz
```

```text
[1] 0
[1] 1
```

</div>
<div markdown>

**Python, metafopy**

```python
# docs: skip
import dataclasses

from metafopy import datasets, escalc, rma

dat = escalc(measure='RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=datasets.load('dat.bcg'))
res = rma('yi', 'vi', data=dat)
res2 = dataclasses.replace(res, tau2=0.0)
print(res2.tau2)
res.zz = 1
```

```text
0.0
FrozenInstanceError: cannot assign to field 'zz'
```

</div>
</div>

### summary() returns text

In R, `summary(res)` is the fit itself with another print method. In metafopy it is the printed text, as a `str` that also has
the fields of the fit as attributes (`s.beta`), so printing it and reading from it both work.

*Why:* Python has no print methods dispatched on the class of a value; a `str` subclass gives the same printout without a second
fit class.

<div class="compare" markdown>
<div markdown>

**R, metafor 5.2-1**

```r
library(metafor)
dat <- escalc(measure="RR", ai=tpos, bi=tneg, ci=cpos, di=cneg, data=dat.bcg)
res <- rma(yi, vi, data=dat)
s <- summary(res)
is.character(s)
s$beta
```

```text
[1] FALSE
              [,1]
intrcpt -0.7145323
```

</div>
<div markdown>

**Python, metafopy**

```python
from metafopy import datasets, escalc, rma

dat = escalc(measure='RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=datasets.load('dat.bcg'))
res = rma('yi', 'vi', data=dat)
s = res.summary()
print(isinstance(s, str))
print(s.beta)
```

```text
True
[-0.71453235]
```

</div>
</div>

### Frames from escalc() print like R, selections do not

`escalc()` returns a `pandas.DataFrame` subclass that prints as R's `print.escalc` does: rows numbered from 1, the `digits` of the
effect sizes and variances. Any selection of rows or columns, and pandas operations such as `merge`, give a plain `DataFrame`
that keeps the attributes (`dat.attrs['measure']`) but prints the pandas way: rows from 0 and wide frames abbreviated.

*Why:* pandas builds a new plain frame for every selection, and R's `[.escalc` keeps the class. Print the whole frame
(`print(dat)`) to get R's layout.

<div class="compare" markdown>
<div markdown>

**R, metafor 5.2-1**

```r
library(metafor)
dat <- escalc(measure="RR", ai=tpos, bi=tneg, ci=cpos, di=cneg, data=dat.bcg)
dat[1:3, ]
class(dat)
```

```text
  trial           author year tpos tneg cpos cneg ablat  alloc      yi     vi 
1     1          Aronson 1948    4  119   11  128    44 random -0.8893 0.3256 
2     2 Ferguson & Simes 1949    6  300   29  274    55 random -1.5854 0.1946 
3     3  Rosenthal et al 1960    3  228   11  209    42 random -1.3481 0.4154 

[1] "escalc"     "data.frame"
```

</div>
<div markdown>

**Python, metafopy**

```python
from metafopy import datasets, escalc

dat = escalc(measure='RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=datasets.load('dat.bcg'))
print(dat.iloc[:3])
print(type(dat).__name__, dat.attrs['measure'])
```

```text
0      1           Aronson  1948     4  ...     44  random -0.889311  0.325585
1      2  Ferguson & Simes  1949     6  ...     55  random -1.585389  0.194581
2      3   Rosenthal et al  1960     3  ...     42  random -1.348073  0.415368

[3 rows x 11 columns]
EscalcFrame RR
```

</div>
</div>

### digits= where R passes it to print()

R changes the digits of a fit when printing it: `print(res, digits=2)`. A metafopy fit prints with the `digits` it was fitted with, so
pass `digits=` to `rma()`; `predict()`, `cumul()`, `leave1out()` and `influence()` results take a named vector (a dict) as in R,
and `result.with_digits(...)` is the counterpart of `print(result, digits=...)`.

*Why:* Python's `print` takes no arguments. The printed text is the same as R's.

<div class="compare" markdown>
<div markdown>

**R, metafor 5.2-1**

```r
library(metafor)
dat <- escalc(measure="RR", ai=tpos, bi=tneg, ci=cpos, di=cneg, data=dat.bcg)
res <- rma(yi, vi, data=dat)
print(res, digits=2)
print(predict(res), digits=c(pred=1, ci=3))
```

```text
Random-Effects Model (k = 13; tau^2 estimator: REML)

tau^2 (estimated amount of total heterogeneity): 0.31 (SE = 0.17)
tau (square root of estimated tau^2 value):      0.56
I^2 (total heterogeneity / total variability):   92.22%
H^2 (total variability / sampling variability):  12.86

Test for Heterogeneity:
Q(df = 12) = 152.23, p-val < .01

Model Results:

estimate    se   zval  pval  ci.lb  ci.ub      
   -0.71  0.18  -3.97  <.01  -1.07  -0.36  *** 

---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1


    pred     se  ci.lb  ci.ub  pi.lb pi.ub 
 -0.7145 0.1798 -1.067 -0.362 -1.867 0.438 
```

</div>
<div markdown>

**Python, metafopy**

```python
from metafopy import datasets, escalc, predict, rma

dat = escalc(measure='RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=datasets.load('dat.bcg'))
res = rma('yi', 'vi', data=dat, digits=2)
print(res)
print(predict(res, digits=4).with_digits({'pred': 1, 'ci': 3}))
```

```text
Random-Effects Model (k = 13; tau^2 estimator: REML)

tau^2 (estimated amount of total heterogeneity): 0.31 (SE = 0.17)
tau (square root of estimated tau^2 value):      0.56
I^2 (total heterogeneity / total variability):   92.22%
H^2 (total variability / sampling variability):  12.86

Test for Heterogeneity:
Q(df = 12) = 152.23, p-val < .01

Model Results:

estimate    se   zval  pval  ci.lb  ci.ub

---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1


 -0.7145 0.1798 -1.067 -0.362 -1.867 0.438
```

</div>
</div>

### reporter() is not ported

`reporter()` writes an R Markdown report of a fit. There is no counterpart: use a Jupyter notebook, which runs the same
metafopy code and keeps its output.

```python
# docs: skip
import metafopy

metafopy.reporter
```

```text
AttributeError: module 'metafopy' has no attribute 'reporter'
```

## Optimiser stop points

metafor maximises some likelihoods with R's `nlminb` (the PORT library) or `optim`. metafopy uses SciPy's optimisers, and ports
R's `optim` BFGS and Nelder-Mead only for `rma_glmm` (`CM.EL`) and `selmodel`. PORT is not ported. Where the likelihood has a single clear maximum the two agree
to about 1e-4 relative or better. Where it is flat, or has several maxima, the two stop at different points, and the examples
below are those cases.

### rma_mv with struct GEN stops at another maximum

On `dat.konstantopoulos2011` the REML log-likelihood of `~ year | district` has more than one local maximum. R's `nlminb` stops
at a log-likelihood of -32.20462; metafopy's BFGS follows the ridge to its end and finishes higher, at -31.86847, with `rho` at
the boundary. 

*Why:* different optimiser on the same surface. metafopy's value is the higher likelihood, so it is not the worse fit. For the
`struct='GDIAG'` variant, `control={'tau2_init': [0.01, 0.01]}` starts metafopy at R's point; see the
[notes](differences-notes.md#optimisers-and-flat-likelihoods).

<div class="compare" markdown>
<div markdown>

**R, metafor 5.2-1**

```r
library(metafor)
res <- rma.mv(yi, vi, random = ~ year | district, struct = "GEN", data = dat.konstantopoulos2011)
c(logLik = logLik(res))
res$tau2
res$rho
```

```text
   logLik 
-32.20462 
[1] 3.486942e+00 1.167101e-06
[1] -0.9997167
```

</div>
<div markdown>

**Python, metafopy**

```python
from metafopy import datasets, rma_mv

res = rma_mv('yi', 'vi', random='~ year | district', struct='GEN', data=datasets.load('dat.konstantopoulos2011'))
print(res.logLik())
print(res.tau2)
print(res.rho)
```

```text
'log Lik.' -31.86847 (df=4)
[3.13185220e+03 7.94227959e-04]
[-1.]
```

</div>
</div>

### rma_mv with struct UN and a fixed tau2

With one `tau2` fixed at 0.5 on `dat.berkey1998`, metafopy's default fit (a Cholesky parametrisation) stops at a log-likelihood
of -1.5004, R at -0.8600. Passing `control={'cholesky': False}` reaches R's value.

*Why:* a worse local maximum of metafopy's BFGS than R's `nlminb`. For this reason the refits of `profile()` and of the
profile-likelihood `confint()` of `rma_mv` fits start at the estimates of the full fit and keep the higher log-likelihood.

<div class="compare" markdown>
<div markdown>

**R, metafor 5.2-1**

```r
library(metafor)
V <- vcalc(vi=1, cluster=author, rvars=c(v1i, v2i), data=dat.berkey1998)
res <- rma.mv(yi, V, mods = ~ 0 + outcome, random = ~ outcome | trial, struct = "UN",
              tau2 = c(NA, 0.5), data = dat.berkey1998)
c(logLik = logLik(res))
```

```text
    logLik 
-0.8600248 
```

</div>
<div markdown>

**Python, metafopy**

```python
from metafopy import datasets, rma_mv, vcalc

dat = datasets.load('dat.berkey1998')
V = vcalc(vi=1, cluster='author', rvars=['v1i', 'v2i'], data=dat)
res = rma_mv('yi', V, mods='~ 0 + outcome', random='~ outcome | trial', struct='UN', tau2=[None, 0.5], data=dat)
print(res.logLik())
res = rma_mv('yi', V, mods='~ 0 + outcome', random='~ outcome | trial', struct='UN', tau2=[None, 0.5], data=dat, control={'cholesky': False})
print(res.logLik())
```

```text
'log Lik.' -1.500362 (df=4)
'log Lik.' -0.8600248 (df=4)
```

</div>
</div>

### Standard errors of a location-scale model on a flat scale

With `link='identity'` the scale slope of this model is near zero, and the Hessian of the log-likelihood is computed by
numerical differences that shrink to rounding error. R gives 0.539892 for the first scale standard error and metafopy 0.539621,
a relative difference of 5e-4; R's own `r=8` and `r=4` Richardson depths already differ from each other by the same amount.

*Why:* the value is rounding noise in both implementations. metafopy uses depth `hessian_r=4` by default, which agrees with R's
`r=4` to about 3e-6; `control={'hessian_r': 8}` gives the noisy variant.

<div class="compare" markdown>
<div markdown>

**R, metafor 5.2-1**

```r
library(metafor)
dat <- escalc(measure="RR", ai=tpos, bi=tneg, ci=cpos, di=cneg, data=dat.bcg)
res <- rma(yi, vi, scale = ~ ablat, link = "identity", data = dat)
res$se.alpha
```

```text
[1] 0.53989225 0.01556825
```

</div>
<div markdown>

**Python, metafopy**

```python
from metafopy import datasets, escalc, rma

dat = escalc(measure='RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=datasets.load('dat.bcg'))
res = rma('yi', 'vi', scale='~ ablat', link='identity', data=dat)
print(res.se_alpha)
```

```text
[0.53962141 0.0155596 ]
```

</div>
</div>

### selmodel on a flat likelihood

`selmodel(type='beta')` on `dat.baskerville2012` has `tau2` near zero and a `delta` on its lower bound (both languages warn
about it). The log-likelihood is identical to the printed digits and `tau2` agrees to six digits, but `delta`, which sits on its
lower bound, is 1.000e-05 in R and 1.022e-05 here.

*Why:* R's PORT stops on `rel.tol` of 1e-10 on a flat likelihood, and SciPy's optimiser stops a little elsewhere on the same
plateau.

<div class="compare" markdown>
<div markdown>

**R, metafor 5.2-1**

```r
library(metafor)
dat <- dat.baskerville2012
res <- rma(smd, se^2, data = dat)
sel <- selmodel(res, type = "beta", steps = c(0.025, 0.975), control = list(optimizer = "nlminb"))
c(logLik = logLik(sel))
sel$tau2
sel$delta
```

```text
Warning message:
One or more 'delta' estimates are (almost) equal to their lower or upper bound.
Treat results with caution (or consider adjusting 'delta.min' and/or 'delta.max'). 
   logLik 
-1.945572 
[1] 0.05736726
[1] 1.000028e-05 1.741239e+00
```

</div>
<div markdown>

**Python, metafopy**

```python
from metafopy import datasets, rma, selmodel

dat = datasets.load('dat.baskerville2012')
res = rma(dat['smd'], dat['se'] ** 2)
sel = selmodel(res, type='beta', steps=[0.025, 0.975], control={'optimizer': 'nlminb'})
print(sel.logLik())
print(sel.tau2)
print(sel.delta)
```

```text
UserWarning: One or more 'delta' estimates are (almost) equal to their lower or upper bound.
Treat results with caution (or consider adjusting 'delta.min' and/or 'delta.max').
'log Lik.' -1.945572 (df=4)
0.057367159330093655
[1.02207630e-05 1.74123073e+00]
```

</div>
</div>

### rma_glmm with UM.FS: z value in the fourth digit

The z value of the pooled log odds ratio is -4.2436 here against -4.2434 in R. The estimate and the log-likelihood agree, and the
standard error differs in the fifth digit.

*Why:* the cause has not been traced. metafopy maximises the exact likelihood and R fits it with lme4 (its own quadrature and
optimiser), and the difference is within the 2e-3 to 1e-1 relative tolerance that the tests allow for mixed-model fits.

<div class="compare" markdown>
<div markdown>

**R, metafor 5.2-1**

```r
library(metafor)
res <- rma.glmm(measure = "OR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg, model = "UM.FS")
res$zval
res$se
```

```text
[1] -4.243433
[1] 0.1755679
```

</div>
<div markdown>

**Python, metafopy**

```python
from metafopy import datasets, rma_glmm

res = rma_glmm(measure='OR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=datasets.load('dat.bcg'), model='UM.FS')
print(res.zval)
print(res.se)
```

```text
[-4.24356287]
[0.17556082]
```

</div>
</div>

### rma_glmm with UM.RS: a higher log-likelihood

The random-slopes model on `dat.bcg` ends at a log-likelihood of -126.2963 in both languages, but metafopy's is higher by 1e-5
(-126.2962773 against R's -126.2962883), and the estimates differ in the fourth (the pooled log odds ratio
-0.76165 against -0.76158). Both warn that `nAGQ` is set to 1.

*Why:* the objective is the same as lme4's. lme4's `bobyqa` stops a little earlier than metafopy's optimiser. The tests allow
2e-3 to 1e-1 relative for mixed-model fits; see [rma_glmm](differences-notes.md#rma_glmm) for the packages and links.

<div class="compare" markdown>
<div markdown>

**R, metafor 5.2-1**

```r
library(metafor)
res <- rma.glmm(measure = "OR", ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg,
                model = "UM.RS", control = list(package = "lme4"))
c(beta = coef(res), tau2 = res$tau2, logLik = logLik(res))
```

```text
Warning message:
Not possible to fit RE/ME model='UM.RS' with nAGQ > 1 with glmer(). nAGQ automatically set to 1. 
beta.intrcpt         tau2       logLik 
  -0.7615840    0.3198355 -126.2962883 
```

</div>
<div markdown>

**Python, metafopy**

```python
from metafopy import datasets, rma_glmm

res = rma_glmm(measure='OR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=datasets.load('dat.bcg'),
               model='UM.RS', control={'package': 'lme4'})
print(res.beta, res.tau2, res.logLik())
```

```text
UserWarning: Not possible to fit RE/ME model='UM.RS' with nAGQ > 1 with glmer(). nAGQ automatically set to 1.
[-0.76164895] 0.31985975161566554 'log Lik.' -126.2963 (df=4)
```

</div>
</div>

### rma_glmm with CM.EL: standard errors

The conditional model gives a standard error that is 0.5 % lower than R's on the Collins data (0.098986 against 0.099461).

*Why:* R's Hessian is a finite-difference Hessian of an objective that is itself noisy (`integrate()` with `rel.tol` 1.2e-4 and
the truncation of BiasedUrn), so R's value moves in the third digit under tiny shifts and cannot be matched. The default
`dnchgcalc` of metafopy is the smooth density; see the [notes](differences-notes.md#rma_glmm) for the other options.

<div class="compare" markdown>
<div markdown>

**R, metafor 5.2-1**

```r
library(metafor)
res <- rma.glmm(ai = b.xci, n1i = nci, ci = b.xti, n2i = nti, data = dat.collins1985a,
                measure = "OR", model = "CM.EL", method = "FE")
res$se
```

```text
Warning messages:
1: 3 studies with NAs omitted from model fitting. 
2: Some yi/vi values are NA. 
[1] 0.09946104
```

</div>
<div markdown>

**Python, metafopy**

```python
from metafopy import datasets, rma_glmm

res = rma_glmm(ai='b.xci', n1i='nci', ci='b.xti', n2i='nti', data=datasets.load('dat.collins1985a'), measure='OR', model='CM.EL', method='FE')
print(res.se)
```

```text
UserWarning: 3 studies with NAs omitted from model fitting.
UserWarning: Some yi/vi values are NA.
[0.09898644]
```

</div>
</div>

## Random numbers

### Seeded draws reproduce R

`conv_fivenum(method='bc', seed=1)` simulates with a port of R's generator, so it gives the same means and SDs as R's
`set.seed(1)`; so does `gosh(seed=)`. The values are identical here; only the row labels differ (0-based, from pandas). Without
`seed`, metafopy draws from the stream of `set_seed`.

*Why:* an analysis that reports simulated numbers can be checked against its R original.

<div class="compare" markdown>
<div markdown>

**R, metafor 5.2-1**

```r
library(metafor)
dat <- data.frame(n = c(30, 40), mn = c(1, 2), q1 = c(3, 4), md = c(5, 6), q3 = c(8, 9), mx = c(14, 16))
conv.fivenum(min = mn, q1 = q1, median = md, q3 = q3, max = mx, n = n, data = dat, method = "bc", seed = 1, test = FALSE)
```

```text
   n mn q1 md q3 mx     mean       sd
1 30  1  3  5  8 14 5.832434 3.930443
2 40  2  4  6  9 16 6.905464 4.144375
```

</div>
<div markdown>

**Python, metafopy**

```python
import pandas

from metafopy import conv_fivenum

dat = pandas.DataFrame({'n': [30, 40], 'mn': [1, 2], 'q1': [3, 4], 'md': [5, 6], 'q3': [8, 9], 'mx': [14, 16]})
print(conv_fivenum(min='mn', q1='q1', median='md', q3='q3', max='mx', n='n', data=dat, method='bc', seed=1, test=False))
```

```text
0  30   1   3   5   8  14  5.832434  3.930443
1  40   2   4   6   9  16  6.905464  4.144375
```

</div>
</div>

## Stricter than R

Places where metafopy raises an error and R passes on, or the other way round.

### A misspelt control entry raises

R matches the names in `control` against the settings it knows with `pmatch` and ignores the rest, so a typo does nothing. In
metafopy an entry that no function reads raises `ArgumentError`, so the typo is not lost. Entries that another metafor function
reads (`optimizer` for `confint`, for example) are ignored as in R.

<div class="compare" markdown>
<div markdown>

**R, metafor 5.2-1**

```r
library(metafor)
dat <- escalc(measure="RR", ai=tpos, bi=tneg, ci=cpos, di=cneg, data=dat.bcg)
res <- rma(yi, vi, data=dat)
res <- rma(yi, vi, data=dat, control = list(optimzer = "nlminb"))
res$tau2
```

```text
[1] 0.3132433
```

</div>
<div markdown>

**Python, metafopy**

```python
# docs: skip
from metafopy import datasets, escalc, rma

dat = escalc(measure='RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=datasets.load('dat.bcg'))
res = rma('yi', 'vi', data=dat, control={'optimzer': 'nlminb'})
print(res.tau2)
```

```text
ArgumentError: unknown control argument 'optimzer'
```

</div>
</div>

### Unknown plot arguments raise

The `...` of metafor's plot functions is passed on to base graphics, which ignores (and warns about) what it does not know.
The plot functions of metafopy take explicit keywords, so an unknown one is a `TypeError`.

<div class="compare" markdown>
<div markdown>

**R, metafor 5.2-1**

```r
library(metafor)
dat <- escalc(measure="RR", ai=tpos, bi=tneg, ci=cpos, di=cneg, data=dat.bcg)
res <- rma(yi, vi, data=dat)
pdf(NULL)
forest(res, notaparameter = 1)
cat("no error\n")
```

```text
There were 50 or more warnings (use warnings() to see the first 50)
no error
```

</div>
<div markdown>

**Python, metafopy**

```python
# docs: skip
from metafopy import datasets, escalc, forest, rma

dat = escalc(measure='RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=datasets.load('dat.bcg'))
res = rma('yi', 'vi', data=dat)
forest(res, notaparameter=1)
```

```text
TypeError: forest() got an unexpected keyword argument 'notaparameter'
```

</div>
</div>

### escalc() with the reserved measure HR

`HR` and `HD` are names that metafor 5.2-1 reserves without computing anything. R fails because `vi` is missing; metafopy
passes `yi` and `vi` or `sei` through when they are given, and otherwise raises `NotSupportedError` with the reason.

<div class="compare" markdown>
<div markdown>

**R, metafor 5.2-1**

```r
library(metafor)
escalc(measure = "HR", ai = c(1, 2), bi = c(3, 4), ci = c(5, 6), di = c(7, 8))
```

```text
Error in escalc(measure = "HR", ai = c(1, 2), bi = c(3, 4), ci = c(5,  : 
  argument "vi" is missing, with no default
```

</div>
<div markdown>

**Python, metafopy**

```python
# docs: skip
from metafopy import escalc

escalc(measure='HR', ai=[1, 2], bi=[3, 4], ci=[5, 6], di=[7, 8])
```

```text
NotSupportedError: measure='HR' is reserved in metafor but not computed there; supply 'yi' and 'vi' or 'sei'.
```

</div>
</div>

### fsn() of a Mantel-Haenszel fit

Both refuse. R fails with an internal error from `inherits(x, "rma") && x$tau2.fix`; metafopy raises
`NotSupportedError` with the message R's other methods use.

<div class="compare" markdown>
<div markdown>

**R, metafor 5.2-1**

```r
library(metafor)
res <- rma.mh(ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg)
fsn(res)
```

```text
Error in inherits(x, "rma") && x$tau2.fix : invalid 'y' type in 'x && y'
```

</div>
<div markdown>

**Python, metafopy**

```python
# docs: skip
from metafopy import datasets, fsn, rma_mh

res = rma_mh(ai='tpos', bi='tneg', ci='cpos', di='cneg', data=datasets.load('dat.bcg'))
fsn(res)
```

```text
NotSupportedError: Method not available for objects of class "rma.mh".
```

</div>
</div>

## metafopy gives a value where R does not

### emmprep() of an rma_mh fit

R's `emmprep` gives an estimate of 0 with standard error 0 for `rma_mh` (and `rma_peto`) fits, and fails for `selmodel` fits.
metafopy computes the estimated marginal mean from the fit.

*Why:* R's result is a placeholder, not an estimate (the pooled log risk ratio of these data is -0.47). A port that reproduced
zeros would give a misleading answer.

<div class="compare" markdown>
<div markdown>

**R, metafor 5.2-1**

```r
library(metafor)
library(emmeans)
res <- rma.mh(ai = tpos, bi = tneg, ci = cpos, di = cneg, data = dat.bcg)
emmeans(emmprep(res), ~ 1)
```

```text
 1       emmean SE  df asymp.LCL asymp.UCL
 overall      0  0 Inf         0         0

Results are given on the log (not the response) scale. 
Confidence level used: 0.95 
```

</div>
<div markdown>

**Python, metafopy**

```python
from metafopy import datasets, emmeans, emmprep, rma_mh

res = rma_mh(ai='tpos', bi='tneg', ci='cpos', di='cneg', data=datasets.load('dat.bcg'))
print(emmeans(emmprep(res), '~ 1'))
```

```text
 1       emmean    SE  df asymp.LCL asymp.UCL
 overall -0.473 0.041 Inf    -0.554    -0.393

Results are given on the log (not the response) scale. 
Confidence level used: 0.95 
```

</div>
</div>

### fmtx(drop0ifint=TRUE)

R drops trailing zeros only when `digits` is an integer (`2L`); a double `2` does not. Python has no `2` against `2L`, so
metafopy always drops them.

<div class="compare" markdown>
<div markdown>

**R, metafor 5.2-1**

```r
library(metafor)
fmtx(c(1, 2.5, 3.25), digits = 2, drop0ifint = TRUE)
fmtx(c(1, 2.5, 3.25), digits = 2L, drop0ifint = TRUE)
```

```text
[1] 1.00 2.50 3.25
[1]    1  2.5 3.25
```

</div>
<div markdown>

**Python, metafopy**

```python
from metafopy import fmtx

print(fmtx([1, 2.5, 3.25], digits=2, drop0ifint=True))
```

```text
[1]    1  2.5 3.25
```

</div>
</div>

## R's quirks copied on purpose

### blup() of a cluster-robust fit recycles quantiles

For a `robust(..., club_sandwich=True)` fit, R computes the prediction interval of study i with the quantile of coefficient
`i` (modulo the number of coefficients), and warns that the lengths do not match. This is probably a bug in R. metafopy follows
it on purpose, gives the same warning, and the two print the same intervals (study 1: -2.4384 to 0.4335; the normal quantile
would give -1.5480 to -0.4570).

<div class="compare" markdown>
<div markdown>

**R, metafor 5.2-1**

```r
library(metafor)
dat <- escalc(measure="RR", ai=tpos, bi=tneg, ci=cpos, di=cneg, data=dat.bcg)
res <- rma(yi, vi, mods = ~ ablat, data=dat)
rr <- robust(res, cluster=alloc, clubSandwich=TRUE)
head(blup(rr), 2)
```

```text
     pred     se   pi.lb  pi.ub 
1 -1.0025 0.2783 -2.4384 0.4335 
2 -1.4157 0.2747 -3.0415 0.2101 

Warning messages:
1: In crit * se :
  longer object length is not a multiple of shorter object length
2: In crit * se :
  longer object length is not a multiple of shorter object length
```

</div>
<div markdown>

**Python, metafopy**

```python
from metafopy import blup, datasets, escalc, rma, robust

dat = escalc(measure='RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=datasets.load('dat.bcg'))
res = rma('yi', 'vi', mods='~ ablat', data=dat)
rr = robust(res, cluster=dat['alloc'], club_sandwich=True)
print(blup(rr)[:2])
```

```text
UserWarning: longer object length is not a multiple of shorter object length

1 -1.0025 0.2783 -2.4384 0.4335
2 -1.4157 0.2747 -3.0415 0.2101
```

</div>
</div>

### Diagnostics refuse robust fits, as R does

`rstandard`, `rstudent`, `dfbetas`, `influence`, `leave1out`, `cumul`, `permutest` and a few more refuse a `robust()` fit, with
the same message as R. `simulate()`, `cooks_distance()` and `hatvalues()` still work, as in R.

<div class="compare" markdown>
<div markdown>

**R, metafor 5.2-1**

```r
library(metafor)
dat <- escalc(measure="RR", ai=tpos, bi=tneg, ci=cpos, di=cneg, data=dat.bcg)
res <- rma(yi, vi, data=dat)
rr <- robust(res, cluster=dat$alloc)
rstandard(rr)
```

```text
Error: Method not available for objects of class "robust.rma".
```

</div>
<div markdown>

**Python, metafopy**

```python
# docs: skip
from metafopy import datasets, escalc, rma, robust, rstandard

dat = escalc(measure='RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=datasets.load('dat.bcg'))
res = rma('yi', 'vi', data=dat)
rr = robust(res, cluster=dat['alloc'])
rstandard(rr)
```

```text
NotSupportedError: Method not available for objects of class "robust.rma".
```

</div>
</div>

## Everything else

The [detailed notes](differences-notes.md) hold the rest, grouped by topic:

- [Names, results and printing](differences-notes.md#names-results-and-printing)
- [Optimisers and flat likelihoods](differences-notes.md#optimisers-and-flat-likelihoods)
- [rma_glmm](differences-notes.md#rma_glmm)
- [Robust and cluster-robust fits](differences-notes.md#robust-and-cluster-robust-fits)
- [Plots](differences-notes.md#plots)
- [Individual functions](differences-notes.md#individual-functions)
- [Arguments, defaults and errors](differences-notes.md#arguments-defaults-and-errors)
- [Not ported](differences-notes.md#not-ported)
- [The audit](differences-notes.md#the-audit)
