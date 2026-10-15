# Estimated marginal means

metafopy has a native port of the parts of R's [emmeans](https://rvlenth.github.io/emmeans/) (Lenth, 2024,
<https://CRAN.R-project.org/package=emmeans>) that metafor users reach through
[`emmprep`](https://wviechtb.github.io/metafor/reference/emmprep.html): a reference grid, marginal means
(least-squares means) with their covariance matrix, contrasts with multiplicity adjustments, and back-transformation to the
scale of the effect size. The results are checked against emmeans together with metafor 5.2-1 on R-generated fixtures.

A marginal mean is a linear combination `L beta` of the coefficients with covariance `L V L'`. Tests and intervals use the
standard normal distribution, or a t distribution with the residual degrees of freedom when the model was fitted with
`test='knha'` (the Knapp and Hartung, 2003, <https://doi.org/10.1002/sim.1482>, adjustment).

## Reference grid and marginal means

`emmprep` takes a fitted model (`rma_uni`, `rma_mv`, `rma_mh`, `rma_peto`, `rma_glmm`, location-scale and selection
models) and builds the reference grid: every combination of factor levels, with the numeric moderators at their mean.
`emmeans` averages the grid over the factors that are not in `specs`.

```python
from metafopy import contrast, datasets, emmeans, emmprep, escalc, pairs, rma

dat = escalc(measure='RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=datasets.load('dat.bcg'))
res = rma('yi', 'vi', mods='~ alloc + ablat', data=dat)
grid = emmprep(res)
print(grid)
```

```text
     alloc ablat prediction     SE  df asymp.LCL asymp.UCL
 alternate 33.46    -0.6195 0.2935 Inf    -1.195  -0.04416
    random 33.46     -0.887 0.1912 Inf    -1.262   -0.5123
systematic 33.46     -0.561 0.2443 Inf     -1.04  -0.08215

Results are given on the log (not the response) scale.
Confidence level used: 0.95
```

`emmprep` knows the link from the effect size measure (log for the ratio measures, logit for `PLO`, and so on), so
`type='response'` back-transforms the estimates and the intervals; the tests stay on the link scale.

```python
means = emmeans(grid, 'alloc')
print(means.summary(type='response'))
```

```text
     alloc response      SE  df asymp.LCL asymp.UCL
 alternate   0.5382   0.158 Inf    0.3028    0.9568
    random   0.4119 0.07875 Inf    0.2832    0.5991
systematic   0.5706  0.1394 Inf    0.3535    0.9211

Confidence level used: 0.95
Intervals are back-transformed from the log scale
```

`emmeans(res, 'alloc')` on a fitted model calls `emmprep` first; `at` and `cov_reduce` set the covariate values, and `tran` overrides the link.
Several variables (`'alloc'`, `['alloc', 'year']`) give their cross-classification, and `by=` gives separate sets of means.

## Contrasts and multiplicity adjustment

`pairs` and `contrast` take the contrast families of emmeans: `pairwise`, `revpairwise`, `trt_vs_ctrl`, `trt_vs_ctrlk`, `eff`,
`del_eff`, `consec`, `mean_chg`, `poly` and `identity`, or a dict of coefficient vectors. Each family has the default
adjustment of emmeans (`pairwise`: Tukey, `trt_vs_ctrl`: Dunnett, `eff`: FDR, `consec`: multivariate t, `poly`: none), and
`summary(adjust=...)` accepts `tukey`, `dunnettx`, `sidak`, `scheffe`, `mvt` and every method of `p_adjust`. Contrasts of log
means show as ratios, contrasts of logit means as odds ratios.

```python
print(pairs(means).summary(type='response'))
print(contrast(means, 'trt_vs_ctrl', ref='alternate').summary(type='response'))
```

```text
              contrast  ratio     SE  df null z.ratio p.value
    alternate / random  1.307 0.4579 Inf    1  0.7633  0.7255
alternate / systematic 0.9432  0.358 Inf    1  -0.154  0.9870
   random / systematic 0.7218 0.2241 Inf    1   -1.05  0.5452

P value adjustment: tukey method for comparing a family of 3 estimates
Tests are performed on the log scale
              contrast  ratio     SE  df null z.ratio p.value
    random / alternate 0.7653 0.2682 Inf    1 -0.7633  0.6577
systematic / alternate   1.06 0.4024 Inf    1   0.154  0.9750

P value adjustment: dunnettx method for 2 tests
Tests are performed on the log scale
```

The Tukey adjustment is an exact port of R's `ptukey`/`qtukey`, and the Dunnett one follows Lund and Lund (1983),
<https://doi.org/10.2307/2347946>. The `mvt` adjustment integrates the multivariate t distribution with SciPy's quasi-Monte
Carlo routine and so agrees with R to about 1e-3 only. See [Differences from metafor](../differences.md).

## Other options

- `weights` sets how the other factors are averaged: `'equal'`, `'proportional'`, `'outer'`, `'cells'`, `'flat'` or a
  numeric vector.
- `summary(bias_adjust=True, sigma=...)` applies the second-order bias adjustment of the back-transformation (emmeans'
  `bias.adjust`); `sigma` defaults to the estimate of tau (0 for `rma_mv` and location-scale fits).
- `regrid(grid)` re-expresses a grid on the response scale with the delta method (`bias_adjust=True, sigma=...` for the bias adjustment).
- `summary(by=..., cross_adjust=...)` adjusts the p-values within each `by` group and across the groups.
