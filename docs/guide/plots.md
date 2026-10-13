# Plots

metafopy draws plots with matplotlib. Each plot function draws on the current axes (or on an axes object passed as `ax=`)
and returns a result object that holds what was plotted. Save and show figures with the usual matplotlib functions.

```python
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot
import numpy

from metafopy import datasets, escalc, rma

dat = escalc(measure='RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=datasets.load('dat.bcg'))
dat['label'] = dat['author'] + ', ' + dat['year'].astype(str)
res = rma('yi', 'vi', data=dat, slab=dat['label'])
```

## Forest plot

```python
from metafopy import forest

forest(res, atransf=numpy.exp, header=True)
matplotlib.pyplot.savefig('forest.png', dpi=150)
```

`atransf` transforms the axis labels (here to risk ratios) while the plot stays on the log scale, and `transf` transforms the
estimates themselves. Other common arguments are:

- `order`: `'obs'` sorts by effect size, `'prec'` by precision, or give your own values.
- `ilab`, `ilab_lab`, `ilab_xpos`: columns of study information shown next to the plot.
- `addpred=True`: add the prediction interval to the summary.
- `showweights=True`: show the weights.
- `xlim`, `alim`, `at`, `refline`, `xlab`, `digits`, `slab`.

For plots with subgroups, draw the studies with `forest(..., rows=...)` and the summaries with `addpoly`. Examples are in the
metafor documentation: <https://www.metafor-project.org/doku.php/plots:forest_plot_with_subgroups>.

```python
from metafopy import forest

forest(res, atransf=numpy.exp, order='obs', addpred=True, xlab='Risk ratio')
matplotlib.pyplot.savefig('forest-sorted.png')
```

## Funnel plot

```python
from metafopy import funnel

matplotlib.pyplot.figure()
funnel(res)
matplotlib.pyplot.savefig('funnel.png')
```

`yaxis` sets the vertical axis (`'sei'` is the default, also `'vi'`, `'seinv'`, `'vinv'`, `'ni'`, `'ninv'`, `'sqrtni'`,
`'sqrtninv'`, `'lni'`, `'wi'`), `level` and `shade` draw the contours of statistical significance (contour-enhanced funnel
plots, Peters et al., 2008, <https://doi.org/10.1016/j.jclinepi.2007.11.010>), and `funnel(trimfill(res))` shows the filled
studies.

```python
matplotlib.pyplot.figure()
funnel(res, level=[90, 95, 99], shade=['white', 'gray55', 'gray75'], refline=0)
matplotlib.pyplot.savefig('funnel-contours.png')
```

## Other plots

| Function | Plot |
|----------|------|
| `radial` | Radial (Galbraith) plot |
| `labbe` | L'Abbe plot of 2x2 table event rates |
| `baujat` | Baujat plot, which studies drive the heterogeneity |
| `qqnorm` | Normal quantile-quantile plot of the residuals, with a simulated envelope |
| `plot_infl` | Case deletion diagnostics, see [Diagnostics](diagnostics.md) |
| `plot_profile` | Profile likelihood of a variance component, from `profile` |
| `plot_gosh` | GOSH plot |
| `plot_selmodel` | Selection function of a `selmodel` fit |

```python
from metafopy import baujat, qqnorm, radial

matplotlib.pyplot.figure()
radial(res)
matplotlib.pyplot.figure()
baujat(res)
matplotlib.pyplot.figure()
qqnorm(res)
```

## Styling

Fonts and sizes are matplotlib's. `fonts` takes the family names metafor uses (`'sans'`, `'serif'`, `'mono'`), and `cex`,
`cex_lab`, `cex_axis` scale the text as in R. For anything else, pass your own `ax=` and use matplotlib on it.
