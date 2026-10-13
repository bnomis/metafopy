"""Image regression tests of the plots (pytest-mpl).

The baseline images in src/tests/baseline were generated on Linux with the Agg backend, the ``default`` matplotlib
style and fixed figure sizes. Compare with ``pytest --mpl``; a plain ``pytest`` only runs the plot code. To
regenerate the baselines (after checking the new images by eye) use
``pytest src/tests/test_plots_image.py --mpl-generate-path=src/tests/baseline``.
"""

import sys

import matplotlib.figure
import matplotlib.pyplot
import numpy
import pandas
import pytest
import refutil

import metafopy
import metafopy.plot
import metafopy.plot.baujat
import metafopy.plot.forest
import metafopy.plot.funnel

pytestmark = pytest.mark.skipif(sys.platform != 'linux', reason='baseline images were generated on Linux')

MPL = pytest.mark.mpl_image_compare(style='default', tolerance=6, savefig_kwargs={'dpi': 72}, deterministic=True)


@pytest.fixture(scope='module')
def dat() -> pandas.DataFrame:
    return refutil.bcg()


@pytest.fixture(scope='module')
def res(dat: pandas.DataFrame) -> metafopy.RMAResult:
    return metafopy.rma('yi', 'vi', data=dat, measure='RR', slab='label', ni='n')


@pytest.fixture(scope='module')
def res_mods(dat: pandas.DataFrame) -> metafopy.RMAResult:
    return metafopy.rma('yi', 'vi', mods='~ ablat', data=dat, measure='RR', slab='label', ni='n')


def image_fig(width: float = 7.0, height: float = 6.0) -> matplotlib.figure.Figure:
    fig = matplotlib.pyplot.figure(figsize=(width, height))
    fig.add_subplot(111)
    return fig


@MPL
def test_image_forest(res: metafopy.RMAResult, dat: pandas.DataFrame) -> matplotlib.figure.Figure:
    fig = image_fig(8, 7)
    metafopy.plot.forest(
        res,
        ax=fig.axes[0],
        atransf=numpy.exp,
        at=numpy.log([0.05, 0.25, 1, 4]),
        xlim=(-16, 6),
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        ilab=dat[['tpos', 'tneg', 'cpos', 'cneg']].to_numpy(),
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        ilab_lab=['TB+', 'TB-', 'CTRL+', 'CTRL-'],
        ilab_xpos=[-9.5, -8, -6, -4.5],
        cex=0.75,
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        header=['Author(s) and Year', 'RR [95% CI]'],
        addpred=True,
        showweights=True,
    )
    return fig


@MPL
def test_image_forest_subgroups(dat: pandas.DataFrame) -> matplotlib.figure.Figure:
    fig = image_fig(8, 7)
    order = numpy.argsort(dat['alloc'].to_numpy(), kind='stable')
    fit_all = metafopy.rma('yi', 'vi', data=dat, measure='RR', slab='label', subset=order)
    metafopy.plot.forest(fit_all, rows=numpy.r_[3:9, 12:15, 18:22], ylim=(-1, 25), xlim=(-8, 5), addfit=False, cex=0.8, shade='zebra', ax=fig.axes[0])
    for name, row in (('alternate', 16.5), ('random', 9.5), ('systematic', 1.5)):
        sub = metafopy.rma('yi', 'vi', data=dat, measure='RR', slab='label', subset=(dat['alloc'] == name).to_numpy())
        metafopy.plot.addpoly(sub, row=row, mlab=f'RE Model for {name}', cex=0.8)
    return fig


@MPL
def test_image_funnel_contour(res: metafopy.RMAResult) -> matplotlib.figure.Figure:
    fig = image_fig()
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    metafopy.plot.funnel(res, level=[90, 95, 99], shade=['white', 'gray55', 'gray75'], refline=0, legend=True, label='out', ax=fig.axes[0])
    return fig


@MPL
def test_image_funnel_trimfill(res: metafopy.RMAResult) -> matplotlib.figure.Figure:
    fig = image_fig()
    metafopy.plot.funnel(metafopy.trimfill(res), yaxis='seinv', ax=fig.axes[0])
    return fig


@MPL
def test_image_baujat(res: metafopy.RMAResult) -> matplotlib.figure.Figure:
    fig = image_fig()
    metafopy.plot.baujat(res, ax=fig.axes[0])
    return fig


@MPL
def test_image_radial(res: metafopy.RMAResult) -> matplotlib.figure.Figure:
    fig = image_fig(7, 7)
    metafopy.radial(res, ax=fig.axes[0])
    return fig


@MPL
def test_image_qqnorm(res: metafopy.RMAResult) -> matplotlib.figure.Figure:
    fig = image_fig()
    metafopy.qqnorm(res, seed=1234, label='out', ax=fig.axes[0])
    return fig


@MPL
def test_image_labbe(dat: pandas.DataFrame) -> matplotlib.figure.Figure:
    fig = image_fig()
    fit = metafopy.rma(ai='tpos', bi='tneg', ci='cpos', di='cneg', data=dat, measure='OR', slab='label')
    metafopy.labbe(fit, ai='tpos', bi='tneg', ci_counts='cpos', di='cneg', ci=True, pi=True, grid=True, legend=True, ax=fig.axes[0])
    return fig


@MPL
def test_image_plot_gosh(dat: pandas.DataFrame) -> matplotlib.figure.Figure:
    fig = image_fig(7, 7)
    g = metafopy.gosh(metafopy.rma('yi', 'vi', data=dat.iloc[:9], measure='RR', slab='label'))
    metafopy.plot_gosh(g, out=3, ax=fig.axes[0])
    return fig


@MPL
def test_image_plot_gosh_pairs(dat: pandas.DataFrame) -> matplotlib.figure.Figure:
    fig = image_fig(7, 7)
    g = metafopy.gosh(metafopy.rma('yi', 'vi', mods='~ ablat', data=dat.iloc[:9], measure='RR', slab='label'))
    metafopy.plot_gosh(g, ax=fig.axes[0])
    return fig


@MPL
def test_image_plot_infl(res: metafopy.RMAResult) -> matplotlib.figure.Figure:
    fig = image_fig(8, 9)
    metafopy.plot_infl(metafopy.influence(res), ax=fig.axes[0])
    return fig


@MPL
def test_image_plot_profile(res: metafopy.RMAResult) -> matplotlib.figure.Figure:
    fig = image_fig()
    metafopy.plot_profile(metafopy.profile(res, steps=15, xlim=(0.01, 0.9)), cline=True, ax=fig.axes[0])
    return fig
