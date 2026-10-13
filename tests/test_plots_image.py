"""Image regression tests of the plots (pytest-mpl).

The baseline images in tests/baseline were generated on Linux with the Agg backend, the ``default`` matplotlib
style and fixed figure sizes. Compare with ``pytest --mpl``; a plain ``pytest`` only runs the plot code. To
regenerate the baselines (after checking the new images by eye) use
``pytest tests/test_plots_image.py --mpl-generate-path=tests/baseline``.
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from metafopy import RMAResult, gosh, influence, labbe, plot_gosh, plot_infl, plot_profile, profile, qqnorm, radial, rma, trimfill
from metafopy.plot import addpoly, baujat, forest, funnel
from refutil import bcg

pytestmark = pytest.mark.skipif(sys.platform != 'linux', reason='baseline images were generated on Linux')

MPL = pytest.mark.mpl_image_compare(style='default', tolerance=6, savefig_kwargs={'dpi': 72}, deterministic=True)


@pytest.fixture(scope='module')
def dat() -> pd.DataFrame:
    return bcg()


@pytest.fixture(scope='module')
def res(dat: pd.DataFrame) -> RMAResult:
    return rma('yi', 'vi', data=dat, measure='RR', slab='label', ni='n')


@pytest.fixture(scope='module')
def res_mods(dat: pd.DataFrame) -> RMAResult:
    return rma('yi', 'vi', mods='~ ablat', data=dat, measure='RR', slab='label', ni='n')


def _fig(width: float = 7.0, height: float = 6.0) -> Figure:
    fig = plt.figure(figsize=(width, height))
    fig.add_subplot(111)
    return fig


@MPL
def test_image_forest(res: RMAResult, dat: pd.DataFrame) -> Figure:
    fig = _fig(8, 7)
    forest(
        res,
        ax=fig.axes[0],
        atransf=np.exp,
        at=np.log([0.05, 0.25, 1, 4]),
        xlim=(-16, 6),
        ilab=dat[['tpos', 'tneg', 'cpos', 'cneg']].to_numpy(),
        ilab_lab=['TB+', 'TB-', 'CTRL+', 'CTRL-'],
        ilab_xpos=[-9.5, -8, -6, -4.5],
        cex=0.75,
        header=['Author(s) and Year', 'RR [95% CI]'],
        addpred=True,
        showweights=True,
    )
    return fig


@MPL
def test_image_forest_subgroups(dat: pd.DataFrame) -> Figure:
    fig = _fig(8, 7)
    order = np.argsort(dat['alloc'].to_numpy(), kind='stable')
    fit_all = rma('yi', 'vi', data=dat, measure='RR', slab='label', subset=order)
    forest(fit_all, rows=np.r_[3:9, 12:15, 18:22], ylim=(-1, 25), xlim=(-8, 5), addfit=False, cex=0.8, shade='zebra', ax=fig.axes[0])
    for name, row in (('alternate', 16.5), ('random', 9.5), ('systematic', 1.5)):
        sub = rma('yi', 'vi', data=dat, measure='RR', slab='label', subset=(dat['alloc'] == name).to_numpy())
        addpoly(sub, row=row, mlab=f'RE Model for {name}', cex=0.8)
    return fig


@MPL
def test_image_funnel_contour(res: RMAResult) -> Figure:
    fig = _fig()
    funnel(res, level=[90, 95, 99], shade=['white', 'gray55', 'gray75'], refline=0, legend=True, label='out', ax=fig.axes[0])
    return fig


@MPL
def test_image_funnel_trimfill(res: RMAResult) -> Figure:
    fig = _fig()
    funnel(trimfill(res), yaxis='seinv', ax=fig.axes[0])
    return fig


@MPL
def test_image_baujat(res: RMAResult) -> Figure:
    fig = _fig()
    baujat(res, ax=fig.axes[0])
    return fig


@MPL
def test_image_radial(res: RMAResult) -> Figure:
    fig = _fig(7, 7)
    radial(res, ax=fig.axes[0])
    return fig


@MPL
def test_image_qqnorm(res: RMAResult) -> Figure:
    fig = _fig()
    qqnorm(res, seed=1234, label='out', ax=fig.axes[0])
    return fig


@MPL
def test_image_labbe(dat: pd.DataFrame) -> Figure:
    fig = _fig()
    fit = rma(ai='tpos', bi='tneg', ci='cpos', di='cneg', data=dat, measure='OR', slab='label')
    labbe(fit, ai='tpos', bi='tneg', ci_counts='cpos', di='cneg', ci=True, pi=True, grid=True, legend=True, ax=fig.axes[0])
    return fig


@MPL
def test_image_plot_gosh(dat: pd.DataFrame) -> Figure:
    fig = _fig(7, 7)
    g = gosh(rma('yi', 'vi', data=dat.iloc[:9], measure='RR', slab='label'))
    plot_gosh(g, out=3, ax=fig.axes[0])
    return fig


@MPL
def test_image_plot_gosh_pairs(dat: pd.DataFrame) -> Figure:
    fig = _fig(7, 7)
    g = gosh(rma('yi', 'vi', mods='~ ablat', data=dat.iloc[:9], measure='RR', slab='label'))
    plot_gosh(g, ax=fig.axes[0])
    return fig


@MPL
def test_image_plot_infl(res: RMAResult) -> Figure:
    fig = _fig(8, 9)
    plot_infl(influence(res), ax=fig.axes[0])
    return fig


@MPL
def test_image_plot_profile(res: RMAResult) -> Figure:
    fig = _fig()
    plot_profile(profile(res, steps=15, xlim=(0.01, 0.9)), cline=True, ax=fig.axes[0])
    return fig
