"""forest() and plot_cumul() of cumulative analyses checked against R metafor (tests/reference/phase2/cumul_plots.json, written by tests/r/generate_cumul.R)."""

import collections.abc
import typing

import matplotlib.axes
import matplotlib.pyplot
import numpy
import pandas
import pytest
import refutil

import metafopy
import metafopy.errors
from metafopy.plot import ForestResult, PlotCumulResult


@pytest.fixture(scope='module')
def dat() -> pandas.DataFrame:
    return refutil.bcg()


@pytest.fixture(scope='module')
def cum(dat: pandas.DataFrame) -> metafopy.CumulResult:
    return metafopy.cumul(metafopy.rma('yi', 'vi', data=dat, slab='label'), order='year')


@pytest.fixture(scope='module')
def cum0(dat: pandas.DataFrame) -> metafopy.CumulResult:
    return metafopy.cumul(metafopy.rma('yi', 'vi', data=dat), order='year')


@pytest.fixture
def ax() -> collections.abc.Iterator[matplotlib.axes.Axes]:
    fig, axes = matplotlib.pyplot.subplots(figsize=(7, 7))
    yield axes
    matplotlib.pyplot.close(fig)


def check_forest(out: ForestResult, ref: dict[str, typing.Any]) -> None:
    refutil.close(out.xlim, ref['xlim'], 'xlim')
    refutil.close(out.alim, ref['alim'], 'alim')
    refutil.close(out.at, ref['at'], 'at')
    refutil.close(out.ylim, ref['ylim'], 'ylim')
    refutil.close(out.rows, ref['rows'], 'rows')
    refutil.close(out.textpos, ref['textpos'], 'textpos')
    refutil.close(out.cex, ref['cex'], 'cex')


def check_plot(out: PlotCumulResult, ref: dict[str, typing.Any]) -> None:
    refutil.close(out.estim, ref['estim'], 'estim')
    refutil.close(out.yval, ref['yval'], 'yval')
    assert list(out.col) == [str(c).upper() for c in refutil.as_list(ref['col'])]


def test_forest_matches_metafor(cum: metafopy.CumulResult, ax: matplotlib.axes.Axes) -> None:
    check_forest(metafopy.forest(cum, ax=ax), refutil.load_phase2('cumul_plots')['forest'])


def test_forest_without_study_labels_matches_metafor(cum0: metafopy.CumulResult, ax: matplotlib.axes.Axes) -> None:
    check_forest(metafopy.forest(cum0, ax=ax), refutil.load_phase2('cumul_plots')['forest_slab_null'])


def test_forest_atransf_matches_metafor(cum: metafopy.CumulResult, ax: matplotlib.axes.Axes) -> None:
    out = metafopy.forest(cum, atransf=numpy.exp, at=numpy.log([0.25, 0.5, 1]), xlim=(-6, 3), digits=(2, 3), ax=ax)
    check_forest(out, refutil.load_phase2('cumul_plots')['forest_transf'])


def test_forest_of_transformed_results_matches_metafor(dat: pandas.DataFrame, ax: matplotlib.axes.Axes) -> None:
    res = metafopy.rma('yi', 'vi', data=dat, slab='label')
    check_forest(
        metafopy.forest(metafopy.cumul(res, order='year', transf=numpy.exp), refline=1, steps=3, ax=ax), refutil.load_phase2('cumul_plots')['forest_exp']
    )


def test_forest_rows_and_labels(cum: metafopy.CumulResult, cum0: metafopy.CumulResult, ax: matplotlib.axes.Axes) -> None:
    metafopy.forest_cumul(cum, ax=ax)
    labels = [t.get_text() for t in ax.texts]
    assert 'Study' in labels
    assert 'Overall Estimate' in labels
    assert cum.slab[0] in labels
    assert any(label.startswith('+ ') for label in labels)
    assert not any(label.startswith('+ Study') for label in labels)
    fig, ax0 = matplotlib.pyplot.subplots()
    metafopy.forest_cumul(cum0, ax=ax0)
    labels0 = [t.get_text() for t in ax0.texts]
    assert f'Study {cum0.slab[0]}' in labels0
    assert sum(label.startswith('+ Study') for label in labels0) == 12
    matplotlib.pyplot.close(fig)


def test_plot_matches_metafor(cum: metafopy.CumulResult, ax: matplotlib.axes.Axes) -> None:
    check_plot(metafopy.plot_cumul(cum, ax=ax), refutil.load_phase2('cumul_plots')['plot_tau2'])


def test_plot_i2_transf_matches_metafor(cum: metafopy.CumulResult, ax: matplotlib.axes.Axes) -> None:
    check_plot(metafopy.plot_cumul(cum, yaxis='I2', transf=numpy.exp, ax=ax), refutil.load_phase2('cumul_plots')['plot_I2'])


def test_plot_h2_atransf_matches_metafor(cum: metafopy.CumulResult, ax: matplotlib.axes.Axes) -> None:
    check_plot(metafopy.plot_cumul(cum, yaxis='H2', atransf=numpy.exp, at=numpy.log([0.4, 0.6, 0.8]), ax=ax), refutil.load_phase2('cumul_plots')['plot_H2'])
    assert [t.get_text() for t in ax.get_xticklabels()] == ['0.4', '0.6', '0.8']


def test_plot_fixed_effects_matches_metafor(dat: pandas.DataFrame, ax: matplotlib.axes.Axes) -> None:
    cum_fe = metafopy.cumul(metafopy.rma('yi', 'vi', data=dat, method='FE'), order='year')
    check_plot(metafopy.plot_cumul(cum_fe, ax=ax), refutil.load_phase2('cumul_plots')['plot_fe'])
    with pytest.raises(metafopy.errors.InputError, match='fixed-effects'):
        metafopy.plot_cumul(cum_fe, yaxis='tau2', ax=ax)


def test_plot_axis_labels(cum: metafopy.CumulResult, ax: matplotlib.axes.Axes) -> None:
    metafopy.plot_cumul(cum, ax=ax)
    assert ax.get_ylabel() == r'Amount of Heterogeneity $(\tau^2)$'
    assert ax.get_xlabel() == 'Overall Estimate'


def test_rejects_both_transformations_and_unknown_yaxis(cum: metafopy.CumulResult, ax: matplotlib.axes.Axes) -> None:
    with pytest.raises(metafopy.errors.InputError, match='either'):
        metafopy.plot_cumul(cum, transf=numpy.exp, atransf=numpy.exp, ax=ax)
    with pytest.raises(metafopy.errors.InputError, match='yaxis'):
        metafopy.plot_cumul(cum, yaxis='Q', ax=ax)
