# Plot gallery

This page reproduces plots of the [plot gallery of the metafor project](https://www.metafor-project.org/doku.php/plots), with the same
datasets and the same models, so that you can compare the figures of metafopy with those of R metafor. In every figure the
left image is drawn by metafopy and the right image by R metafor 5.2-1, from the code on the corresponding page of the gallery
(Viechtbauer, 2010, <https://doi.org/10.18637/jss.v036.i03>).

The Python code of each figure is run by the test suite, so it cannot rot. The figures are rendered with `python bin/gallery.py` and
`Rscript src/tests/r/gallery.R`, see [Rendering the figures](#rendering-the-figures). The two renderings agree in the geometry of
what is drawn (positions, sizes, labels); they differ in details of the rendering such as the font, which is Helvetica or Arial
in R and DejaVu Sans in matplotlib by default.

Not yet available in metafopy: the cumulative forest plot (`forest` of a `cumul` result), the plot of cumulative results, the
meta-analytic scatterplot (`regplot`), and the BMJ, RevMan and superscript forest plot styles.

```python
import matplotlib

matplotlib.use('Agg')

import matplotlib.colors
import matplotlib.pyplot
import numpy
import pandas
import scipy.special
import scipy.stats

from metafopy import (
    addpoly,
    baujat,
    datasets,
    escalc,
    forest,
    funnel,
    gosh,
    influence,
    labbe,
    plot_gosh,
    plot_infl,
    predict,
    qqnorm,
    radial,
    regtest,
    rma,
    rma_mv,
    trimfill,
    vcalc,
)

matplotlib.rcParams['mathtext.default'] = 'regular'
DPI = 100  # the R figures are rendered at the same size and resolution


def pvalue(p: float) -> str:
    """A p-value as metafor's fmtp2 writes it."""
    if p < 0.001:
        return 'p < .001'
    return f'p = {p:.3f}'.replace('0.', '.') if p < 0.1 else f'p = {p:.2f}'.replace('0.', '.')


def fit_stats(text: str, fit) -> str:
    """The label of a model with its Q-test, I^2 and tau^2 (the helper `mlabfun` of the metafor pages)."""
    return f'{text} (Q = {fit.QE:.2f}, df = {fit.k - fit.p}, {pvalue(fit.QEp)}; $I^2$ = {fit.I2:.1f}%, $\\tau^2$ = {fit.tau2:.2f})'
```

## Forest plots

All forest plots of this section use the BCG vaccine data.

```python
dat = escalc(measure='RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=datasets.load('dat.bcg'))
dat['slab'] = dat['author'] + ', ' + dat['year'].astype(str)
res = rma('yi', 'vi', data=dat, slab=dat['slab'])
counts = dat[['tpos', 'tneg', 'cpos', 'cneg']].to_numpy()
```

### Forest plot with extra annotations

Source: <https://www.metafor-project.org/doku.php/plots:forest_plot>

```python
fig, ax = matplotlib.pyplot.subplots(figsize=(8, 7))
forest(
    res,
    ax=ax,
    atransf=numpy.exp,
    at=numpy.log([0.05, 0.25, 1, 4]),
    xlim=(-16, 6),
    ilab=counts,
    ilab_lab=['TB+', 'TB-', 'TB+', 'TB-'],
    ilab_xpos=[-9.5, -8, -6, -4.5],
    cex=0.75,
    header='Author(s) and Year',
    mlab='',
    shade=True,
)
ax.text(-8.75, res.k + 2.8, 'Vaccinated', ha='center', va='center', fontsize=7.5, fontweight='bold')
ax.text(-5.25, res.k + 2.8, 'Control', ha='center', va='center', fontsize=7.5, fontweight='bold')
ax.text(-16, -1, fit_stats('RE Model', res), ha='left', va='center', fontsize=7.5)
fig.savefig('forest.png', dpi=DPI)
```

| metafopy | R metafor |
|----------|-----------|
| ![forest plot, metafopy](../images/gallery/py/forest.png) | ![forest plot, R](../images/gallery/r/forest.png) |

### Forest plot with adjusted text position

Source: <https://www.metafor-project.org/doku.php/plots:forest_plot_with_textpos>

```python
fig, ax = matplotlib.pyplot.subplots(figsize=(8, 7))
forest(
    res,
    ax=ax,
    atransf=numpy.exp,
    at=numpy.log([0.05, 0.25, 1, 4]),
    xlim=(-20, 2),
    ilab=counts,
    ilab_lab=['TB+', 'TB-', 'TB+', 'TB-'],
    ilab_xpos=[-13.5, -12, -10, -8.5],
    cex=0.75,
    header='Author(s) and Year',
    mlab='',
    textpos=(-20, -3.5),
    shade=True,
)
ax.text(-12.75, res.k + 2.8, 'Vaccinated', ha='center', va='center', fontsize=7.5, fontweight='bold')
ax.text(-9.25, res.k + 2.8, 'Control', ha='center', va='center', fontsize=7.5, fontweight='bold')
ax.text(-20, -1, fit_stats('RE Model', res), ha='left', va='center', fontsize=7.5)
fig.savefig('forest_textpos.png', dpi=DPI)
```

| metafopy | R metafor |
|----------|-----------|
| ![forest plot with adjusted text position, metafopy](../images/gallery/py/forest_textpos.png) | ![forest plot with adjusted text position, R](../images/gallery/r/forest_textpos.png) |

### Forest plot with subgroups

Source: <https://www.metafor-project.org/doku.php/plots:forest_plot_with_subgroups>

The `rows` argument places the studies of the three allocation groups, and `addpoly` adds the summary of each group.

```python
fig, ax = matplotlib.pyplot.subplots(figsize=(8, 7.5))
forest(
    res,
    ax=ax,
    xlim=(-16, 4.6),
    at=numpy.log([0.05, 0.25, 1, 4]),
    atransf=numpy.exp,
    ilab=counts,
    ilab_lab=['TB+', 'TB-', 'TB+', 'TB-'],
    ilab_xpos=[-9.5, -8, -6, -4.5],
    cex=0.75,
    ylim=(-2, 28),
    top=4,
    order=dat['alloc'],
    rows=numpy.r_[3:5, 9:16, 20:24],
    mlab=fit_stats('RE Model for All Studies', res),
    psize=1,
    header='Author(s) and Year',
)
ax.text(-8.75, 27, 'Vaccinated', ha='center', va='center', fontsize=7.5, fontweight='bold')
ax.text(-5.25, 27, 'Control', ha='center', va='center', fontsize=7.5, fontweight='bold')
for y, label in ((24, 'Systematic Allocation'), (16, 'Random Allocation'), (5, 'Alternate Allocation')):
    ax.text(-16, y, label, ha='left', va='center', fontsize=7.5, fontweight='bold', fontstyle='italic')
for name, row in (('systematic', 18.5), ('random', 7.5), ('alternate', 1.5)):
    sub = rma('yi', 'vi', data=dat, subset=(dat['alloc'] == name).to_numpy(), slab=dat['slab'])
    addpoly(sub, ax=ax, row=row, mlab=fit_stats('RE Model for Subgroup', sub))
mods = rma('yi', 'vi', mods='~ alloc', data=dat)
ax.text(-16, -1.8, f'Test for Subgroup Differences: $Q_M$ = {mods.QM:.2f}, df = {mods.p - 1}, {pvalue(mods.QMp)}', ha='left', va='center', fontsize=7.5)
fig.savefig('forest_subgroups.png', dpi=DPI)
```

| metafopy | R metafor |
|----------|-----------|
| ![forest plot with subgroups, metafopy](../images/gallery/py/forest_subgroups.png) | ![forest plot with subgroups, R](../images/gallery/r/forest_subgroups.png) |

### Forest plot with multiple CIs

Source: <https://www.metafor-project.org/doku.php/plots:forest_plot_with_multiple_cis>

The plot shows the 95% CIs of the studies as thin lines and the 80% CIs on top of them as thick lines. `forest` has no line width
argument, so the thick lines are drawn with matplotlib, and the pooled estimate is added with a 99% CI.

```python
fig, ax = matplotlib.pyplot.subplots(figsize=(7, 7))
forest(dat['yi'], dat['vi'], ax=ax, slab=dat['slab'], xlim=(-8, 6), ylim=(-2, len(dat) + 3), psize=1, efac=0, shade=True)
half = scipy.stats.norm.ppf(0.9) * numpy.sqrt(dat['vi'])
ax.hlines(numpy.arange(len(dat), 0, -1), dat['yi'] - half, dat['yi'] + half, color='black', linewidth=3, zorder=5)
addpoly(res, ax=ax, row=-1, level=99, mlab='RE Model (with 99% CI)', annotate=True)
ax.axhline(0, color='black', linewidth=1)
fig.savefig('forest_multiple_cis.png', dpi=DPI)
```

| metafopy | R metafor |
|----------|-----------|
| ![forest plot with multiple CIs, metafopy](../images/gallery/py/forest_multiple_cis.png) | ![forest plot with multiple CIs, R](../images/gallery/r/forest_multiple_cis.png) |

## Funnel plots

### Funnel plot variations

Source: <https://www.metafor-project.org/doku.php/plots:funnel_plot_variations>

```python
res_h = rma('yi', 'vi', data=datasets.load('dat.hackshaw1998'), measure='OR', method='EE')
fig, axes = matplotlib.pyplot.subplots(2, 2, figsize=(9, 9), layout='constrained')
funnel(res_h, ax=axes[0, 0])
axes[0, 0].set_title('Standard Error')
funnel(res_h, yaxis='vi', ax=axes[0, 1])
axes[0, 1].set_title('Sampling Variance')
funnel(res_h, yaxis='seinv', ax=axes[1, 0])
axes[1, 0].set_title('Inverse Standard Error')
funnel(res_h, yaxis='vinv', ax=axes[1, 1])
axes[1, 1].set_title('Inverse Sampling Variance')
fig.savefig('funnel_variations.png', dpi=DPI)
```

| metafopy | R metafor |
|----------|-----------|
| ![funnel plot variations, metafopy](../images/gallery/py/funnel_variations.png) | ![funnel plot variations, R](../images/gallery/r/funnel_variations.png) |

### Contour-enhanced funnel plot

Source: <https://www.metafor-project.org/doku.php/plots:contour_enhanced_funnel_plot>

Contour-enhanced funnel plots (Peters et al., 2008, <https://doi.org/10.1016/j.jclinepi.2007.11.010>) shade the regions of statistical significance.

```python
res_b = rma(measure='RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=datasets.load('dat.bcg'))
fig, ax = matplotlib.pyplot.subplots(figsize=(7, 7))
funnel(res_b, ax=ax, level=[90, 95, 99], refline=0, legend=True)
fig.savefig('funnel_contour.png', dpi=DPI)
```

| metafopy | R metafor |
|----------|-----------|
| ![contour-enhanced funnel plot, metafopy](../images/gallery/py/funnel_contour.png) | ![contour-enhanced funnel plot, R](../images/gallery/r/funnel_contour.png) |

### Contour-enhanced funnel plot with the precision on the vertical axis

Source: <https://www.metafor-project.org/doku.php/plots:contour_enhanced_funnel_plot_2>

```python
dat_g = escalc(measure='OR', ai='ai', n1i='n1i', ci='ci', n2i='n2i', data=datasets.load('dat.graves2010'))
fig, ax = matplotlib.pyplot.subplots(figsize=(7, 7))
funnel(
    dat_g['yi'],
    dat_g['vi'],
    ax=ax,
    yaxis='seinv',
    ylab='Precision (1/se)',
    xlim=(-3, 2),
    ylim=(0.00001, 8),
    level=[0.10, 0.05, 0.01],
    legend=True,
    back='gray90',
    lty=0,
)
ax.set_xlim(-3, 2)
ax.set_ylim(0.00001, 8)
fig.savefig('funnel_contour2.png', dpi=DPI)
```

| metafopy | R metafor |
|----------|-----------|
| ![contour-enhanced funnel plot 2, metafopy](../images/gallery/py/funnel_contour2.png) | ![contour-enhanced funnel plot 2, R](../images/gallery/r/funnel_contour2.png) |

### Funnel plot with limit estimate

Source: <https://www.metafor-project.org/doku.php/plots:funnel_plot_with_limit_estimate>

The regression lines of the regression test for funnel plot asymmetry (`regtest`, Egger et al., 1997, <https://doi.org/10.1136/bmj.315.7109.629>) are added with matplotlib.

```python
res_l = rma('yi', 'vi', data=datasets.load('dat.hackshaw1998'), measure='OR')
fig, ax = matplotlib.pyplot.subplots(figsize=(7, 7))
funnel(res_l, ax=ax, refline=0)
se = numpy.linspace(0, 0.8, 100)
reg = regtest(res_l, ret_fit=True)
ax.plot(reg.fit.beta[0] + reg.fit.beta[1] * se, se, color='black', linewidth=1, label='Standard Errors as Predictor')
reg = regtest(res_l, predictor='vi', ret_fit=True)
ax.plot(reg.fit.beta[0] + reg.fit.beta[1] * se**2, se, color='black', linewidth=2, linestyle=':', label='Sampling Variances as Predictor')
ax.legend(loc='upper right', fontsize=9, framealpha=1)
fig.savefig('funnel_limit.png', dpi=DPI)
```

| metafopy | R metafor |
|----------|-----------|
| ![funnel plot with limit estimate, metafopy](../images/gallery/py/funnel_limit.png) | ![funnel plot with limit estimate, R](../images/gallery/r/funnel_limit.png) |

### Funnel plot with trim and fill

Source: <https://www.metafor-project.org/doku.php/plots:funnel_plot_with_trim_and_fill>

```python
taf = trimfill(rma('yi', 'vi', data=datasets.load('dat.hackshaw1998'), measure='OR'))
fig, ax = matplotlib.pyplot.subplots(figsize=(7, 7))
funnel(taf, ax=ax, legend={'show': 'cis'})
fig.savefig('funnel_trimfill.png', dpi=DPI)
```

| metafopy | R metafor |
|----------|-----------|
| ![funnel plot with trim and fill, metafopy](../images/gallery/py/funnel_trimfill.png) | ![funnel plot with trim and fill, R](../images/gallery/r/funnel_trimfill.png) |

## Other plots

### Radial plot

Source: <https://www.metafor-project.org/doku.php/plots:radial_plot>

```python
res_r = rma('yi', 'vi', data=datasets.load('dat.hackshaw1998'), method='EE')
fig, ax = matplotlib.pyplot.subplots(figsize=(7, 7))
radial(res_r, ax=ax)
fig.savefig('radial.png', dpi=DPI)
```

| metafopy | R metafor |
|----------|-----------|
| ![radial plot, metafopy](../images/gallery/py/radial.png) | ![radial plot, R](../images/gallery/r/radial.png) |

### Baujat plot

Source: <https://www.metafor-project.org/doku.php/plots:baujat_plot>

```python
dat_p = datasets.load('dat.pignon2000')
dat_p['yi'] = dat_p['OmE'] / dat_p['V']
dat_p['vi'] = 1 / dat_p['V']
res_p = rma('yi', 'vi', data=dat_p, method='EE', slab=dat_p['id'])
fig, ax = matplotlib.pyplot.subplots(figsize=(7, 7))
baujat(res_p, ax=ax, xlim=(0, 20), ylim=(0, 0.2))
fig.savefig('baujat.png', dpi=DPI)
```

| metafopy | R metafor |
|----------|-----------|
| ![Baujat plot, metafopy](../images/gallery/py/baujat.png) | ![Baujat plot, R](../images/gallery/r/baujat.png) |

### L'Abbe plots

Source: <https://www.metafor-project.org/doku.php/plots:labbe_plot>

```python
dat_d = datasets.load('dat.damico2009')
res_d = rma(measure='OR', ai='xt', n1i='nt', ci='xc', n2i='nc', data=dat_d)
cells = {'ai': dat_d['xt'], 'bi': dat_d['nt'] - dat_d['xt'], 'ci_counts': dat_d['xc'], 'di': dat_d['nc'] - dat_d['xc']}
fig, axes = matplotlib.pyplot.subplots(2, 2, figsize=(9, 9), layout='constrained')
with matplotlib.rc_context({'font.size': 7}):  # R's par(cex=0.7)
    labbe(res_d, **cells, ax=axes[0, 0])
    labbe(res_d, **cells, ci=True, pi=True, grid=True, legend=True, xlab='Log Odds (Control Group)', ylab='Log Odds (Treatment Group)', ax=axes[0, 1])
    labbe(res_d, **cells, ci=True, pi=True, grid=True, legend=True, transf=numpy.exp, xlab='Odds (Control Group)', ylab='Odds (Treatment Group)', ax=axes[1, 0])
    labbe(
        res_d,
        **cells,
        ci=True,
        pi=True,
        grid=True,
        legend=True,
        transf=scipy.special.expit,
        lim=(0, 1),
        xlab='Risk (Control Group)',
        ylab='Risk (Treatment Group)',
        ax=axes[1, 1],
    )
fig.savefig('labbe.png', dpi=DPI)
```

| metafopy | R metafor |
|----------|-----------|
| ![L'Abbe plots, metafopy](../images/gallery/py/labbe.png) | ![L'Abbe plots, R](../images/gallery/r/labbe.png) |

### Normal quantile-quantile plots

Source: <https://www.metafor-project.org/doku.php/plots:normal_qq_plots>

```python
models = {
    'Equal-Effects Model': rma('yi', 'vi', data=dat, method='EE'),
    'Random-Effects Model': rma('yi', 'vi', data=dat),
    'Fixed-Effects with Moderators Model': rma('yi', 'vi', mods='~ ablat', data=dat, method='FE'),
    'Mixed-Effects Model': rma('yi', 'vi', mods='~ ablat', data=dat),
}
fig, axes = matplotlib.pyplot.subplots(2, 2, figsize=(9, 9), layout='constrained')
for ax, (title, fit) in zip(axes.flat, models.items(), strict=True):
    qqnorm(fit, ax=ax, seed=1234)
    ax.set_title(title)
fig.savefig('qqnorm.png', dpi=DPI)
```

| metafopy | R metafor |
|----------|-----------|
| ![normal QQ plots, metafopy](../images/gallery/py/qqnorm.png) | ![normal QQ plots, R](../images/gallery/r/qqnorm.png) |

### Plot of influence diagnostics

Source: <https://www.metafor-project.org/doku.php/plots:plot_of_influence_diagnostics>

```python
res_i = rma(ri='ri', ni='ni', measure='ZCOR', data=datasets.load('dat.mcdaniel1994'))
fig, ax = matplotlib.pyplot.subplots(figsize=(7, 11))
plot_infl(influence(res_i), ax=ax, mfrow=(8, 1))
fig.savefig('influence.png', dpi=DPI)
```

| metafopy | R metafor |
|----------|-----------|
| ![influence diagnostics, metafopy](../images/gallery/py/influence.png) | ![influence diagnostics, R](../images/gallery/r/influence.png) |

### GOSH plot

Source: <https://www.metafor-project.org/doku.php/plots:gosh_plot>

```python
res_e = rma(measure='OR', ai='ai', n1i='n1i', ci='ci', n2i='n2i', data=datasets.load('dat.egger2001'), method='EE')
sav = gosh(res_e)
fig, ax = matplotlib.pyplot.subplots(figsize=(7, 7))
plot_gosh(sav, out=15, breaks=100, ax=ax)
fig.savefig('gosh.png', dpi=DPI)
```

| metafopy | R metafor |
|----------|-----------|
| ![GOSH plot, metafopy](../images/gallery/py/gosh.png) | ![GOSH plot, R](../images/gallery/r/gosh.png) |

## Plots of multilevel models

### Caterpillar plot

Source: <https://www.metafor-project.org/doku.php/plots:caterpillar_plot>

The data are simulated as on the metafor page, but with the random number generator of NumPy, so the points are not those of
the R figure.

```python
rng = numpy.random.default_rng(5132)
k = 250
vi = rng.chisquare(1, k) * 0.03
yi = rng.normal(rng.normal(0.5, 0.4, k), numpy.sqrt(vi))
res_c = rma(yi, vi)
fig, ax = matplotlib.pyplot.subplots(figsize=(6, 8))
forest(
    yi,
    vi,
    ax=ax,
    header=False,
    xlim=(-2.5, 3.5),
    ylim=(-8, 254),
    order='obs',
    slab=numpy.nan,
    annotate=False,
    efac=(0, 1),
    pch=19,
    colci='gray40',
    psize=2.5,
    cex_lab=1,
    cex_axis=1,
    lty=['solid', 'blank'],
)
addpoly(res_c, ax=ax, mlab='', cex=1)
ax.text(-2, -2, 'RE Model', ha='left', va='center')
fig.savefig('caterpillar.png', dpi=DPI)
```

| metafopy | R metafor |
|----------|-----------|
| ![caterpillar plot, metafopy](../images/gallery/py/caterpillar.png) | ![caterpillar plot, R](../images/gallery/r/caterpillar.png) |

### Forest plot with grouped estimates

Source: <https://www.metafor-project.org/doku.php/plots:forest_plot_with_grouped_estimates>

The effect sizes of a study are drawn on rows close to each other, in a colour per study.

```python
dat_a = datasets.load('dat.assink2016')
V = vcalc(vi=dat_a['vi'], cluster=dat_a['study'], obs=dat_a['esid'], rho=0.6)
res_a = rma_mv(yi='yi', V=V, random='~ 1 | study/esid', data=dat_a)
dat_a = dat_a.sort_values(['study', 'yi'], ascending=False, ignore_index=True)
dat_a['row'] = dat_a['study'].rank(method='dense')
offsets = {1: [0.0], 2: [-0.1, 0.1]}
dat_a['row_seq'] = dat_a['row'] + dat_a.groupby('study')['row'].transform(
    lambda x: pandas.Series(offsets.get(len(x), numpy.linspace(-0.3, 0.3, len(x))), index=x.index)
)
dat_a['slab'] = numpy.where(dat_a['study'].duplicated(), '', 'Study ' + dat_a['study'].astype(str))
palette = matplotlib.colormaps['Dark2'].colors
dat_a['colpt'] = [matplotlib.colors.to_hex(palette[i % len(palette)]) for i in pandas.factorize(dat_a['study'])[0]]
dat_a['colci'] = [matplotlib.colors.to_hex(c, keep_alpha=True)[:7] + '80' for c in dat_a['colpt']]
fig, ax = matplotlib.pyplot.subplots(figsize=(7, 9))
forest(
    dat_a['yi'],
    dat_a['vi'],
    ax=ax,
    rows=dat_a['row_seq'],
    slab=numpy.nan,
    annotate=False,
    xlim=(-4.4, 4.2),
    ylim=(-2.0, dat_a['row'].max() + 2.5),
    cex=1,
    pch=19,
    psize=0.8,
    efac=0,
    col=list(dat_a['colpt']),
    colci=list(dat_a['colci']),
)
for row, label in zip(dat_a['row'], dat_a['slab'], strict=True):
    ax.text(-4.4, row, label, ha='left', va='center_baseline', fontsize=10)
for h in numpy.arange(0, dat_a['row'].max() + 1) + 0.5:
    ax.axhline(h, color='0.2', linewidth=0.2)
addpoly(res_a, ax=ax, row=-0.25, efac=1, mlab='Pooled Estimate', predstyle='bar')
ax.axhline(0.5, color='black', linewidth=1)
fig.savefig('forest_grouped.png', dpi=DPI)
```

| metafopy | R metafor |
|----------|-----------|
| ![forest plot with grouped estimates, metafopy](../images/gallery/py/forest_grouped.png) | ![forest plot with grouped estimates, R](../images/gallery/r/forest_grouped.png) |

### Forest plot in orchard style

Source: <https://www.metafor-project.org/doku.php/plots:forest_plot_orchard_style>

The correlations of the phylogenetic meta-analysis of Moura et al. (2021) are shown per subphylum, with the points of the
studies jittered, and the estimates, CIs and prediction intervals of the model on top.

```python
# docs: skip (the fit takes about a minute and a half; bin/gallery.py runs this block)
dat_m = escalc(measure='ZCOR', ri='ri', ni='ni', data=datasets.load('dat.moura2021'))
levels = sorted(dat_m['subphylum'].unique())
nlvls = len(levels)
res_m = rma_mv('yi', 'vi', mods='~ 0 + subphylum', random=['~ subphylum | study.id', '~ subphylum | effect.size.id'], struct='DIAG', sparse=True, data=dat_m)

# the studies closer to the pooled estimates are jittered more
rng = numpy.random.default_rng(1234)
dat_m['row'] = dat_m['subphylum'].map({name: i + 1 for i, name in enumerate(levels)})
d = numpy.abs(res_m.residuals()) ** (1 / 8)
d = (1 - (d - d.min()) / (d.max() - d.min())) * 0.5
dat_m['row_jitter'] = dat_m['row'] + rng.uniform(-d, d)
palette = matplotlib.colormaps['Dark2'].colors
cols = [matplotlib.colors.to_hex(palette[i % len(palette)]) for i in range(nlvls)]
dat_m['colpt'] = [matplotlib.colors.to_hex(cols[r - 1]) + '99' for r in dat_m['row']]
dat_m['viadj'] = numpy.maximum(dat_m['vi'], dat_m['vi'].quantile(0.05))

fig, ax = matplotlib.pyplot.subplots(figsize=(7, 5))
forest(
    dat_m['yi'],
    dat_m['viadj'],
    ax=ax,
    header='Subphylum',
    rows=dat_m['row_jitter'],
    slab=numpy.nan,
    annotate=False,
    xlim=(-1.8, 1.5),
    ylim=(0.5, nlvls + 2.5),
    cex=1,
    pch=19,
    efac=0,
    col=list(dat_m['colpt']),
    colci='none',
    transf=numpy.tanh,
    digits=1,
    xlab='Correlation Coefficient',
)
for i, name in enumerate(levels, start=1):
    ax.text(-1.8, i, name, ha='left', va='center_baseline', fontsize=10)
for h in numpy.arange(0, nlvls + 1) + 0.5:
    ax.axhline(h, color='0.2', linewidth=0.2)

# points, CIs and PIs of the subphylum levels
pred = predict(res_m, newmods=numpy.eye(res_m.p), tau2_levels=list(range(nlvls)), gamma2_levels=list(range(nlvls)), transf=numpy.tanh)
ypos = numpy.arange(1, nlvls + 1)
ax.hlines(ypos, pred.pi_lb, pred.pi_ub, color='black', linewidth=1)
ax.hlines(ypos, pred.ci_lb, pred.ci_ub, color='black', linewidth=3)
ax.scatter(pred.pred, ypos, s=70, facecolor=cols, edgecolor='black', linewidth=1, zorder=5)
ax.text(1.5, nlvls + 1.5, 'k', ha='right', va='center_baseline', fontsize=10, fontweight='bold')
for i, name in enumerate(levels, start=1):
    ax.text(1.5, i, str((dat_m['subphylum'] == name).sum()), ha='right', va='center_baseline', fontsize=7.5)
fig.savefig('forest_orchard.png', dpi=DPI)
```

| metafopy | R metafor |
|----------|-----------|
| ![forest plot in orchard style, metafopy](../images/gallery/py/forest_orchard.png) | ![forest plot in orchard style, R](../images/gallery/r/forest_orchard.png) |

## Rendering the figures

The image files are committed under `docs/images/gallery`. To render them again, from the repository root:

```bash
source bin/activate.sh
python bin/gallery.py
LC_ALL=C.UTF-8 Rscript src/tests/r/gallery.R
```

The first command writes `docs/images/gallery/py`, the second `docs/images/gallery/r` (it needs R with metafor 5.2-1 and metadat).
