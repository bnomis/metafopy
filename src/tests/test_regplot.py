"""regplot() checked against R metafor (tests/reference/phase2/regplot.json, written by tests/r/generate_regplot.R)."""

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
from metafopy.plot import RegplotResult


@pytest.fixture(scope='module')
def dat() -> pandas.DataFrame:
    return refutil.bcg()


@pytest.fixture(scope='module')
def res(dat: pandas.DataFrame) -> metafopy.RMAResult:
    return metafopy.rma('yi', 'vi', mods='~ ablat', data=dat)


@pytest.fixture
def ax() -> collections.abc.Iterator[matplotlib.axes.Axes]:
    fig, axes = matplotlib.pyplot.subplots(figsize=(7, 7))
    yield axes
    matplotlib.pyplot.close(fig)


def check(out: RegplotResult, ref: dict[str, typing.Any]) -> None:
    refutil.close(out.xi, ref['xi'], 'xi')
    refutil.close(out.yi, ref['yi'], 'yi')
    refutil.close(out.psize, ref['psize'], 'psize')
    refutil.close(out.pred, ref['pred'], 'pred')
    assert out.label.tolist() == refutil.as_list(ref['label'])
    assert out.order.tolist() == [int(v) for v in refutil.as_list(ref['order'])]


def test_default_matches_metafor(res: metafopy.RMAResult, ax: matplotlib.axes.Axes) -> None:
    check(metafopy.regplot(res, ax=ax), refutil.load_phase2('regplot')['default'])


def test_prediction_band_label_and_plim_match_metafor(res: metafopy.RMAResult, ax: matplotlib.axes.Axes) -> None:
    out = metafopy.regplot(res, pi=True, label='piout', plim=(float('nan'), 2.0, 0.5), ax=ax)
    check(out, refutil.load_phase2('regplot')['pi_label'])


def test_ciout_and_seinv_match_metafor(res: metafopy.RMAResult, ax: matplotlib.axes.Axes) -> None:
    check(metafopy.regplot(res, label='ciout', psize='seinv', ax=ax), refutil.load_phase2('regplot')['ciout'])


def test_transf_and_indices_match_metafor(res: metafopy.RMAResult, ax: matplotlib.axes.Axes) -> None:
    out = metafopy.regplot(res, transf=numpy.exp, psize=1.5, label=[0, 3], ax=ax)
    check(out, refutil.load_phase2('regplot')['transf'])


def test_olim_and_xvals_match_metafor(res: metafopy.RMAResult, ax: matplotlib.axes.Axes) -> None:
    out = metafopy.regplot(res, olim=(-1.2, 0), xvals=numpy.linspace(10, 60, 6), ax=ax)
    check(out, refutil.load_phase2('regplot')['olim'])
    assert all(float(numpy.max(numpy.asarray(line.get_ydata()))) <= 0 for line in ax.lines if line.get_gid() == 'line')


def test_second_moderator_matches_metafor(dat: pandas.DataFrame, ax: matplotlib.axes.Axes) -> None:
    res2 = metafopy.rma('yi', 'vi', mods='~ ablat + year', data=dat)
    check(metafopy.regplot(res2, mod='year', label='all', ax=ax), refutil.load_phase2('regplot')['mod2'])


def test_regression_line_and_bands_are_the_predictions(res: metafopy.RMAResult, ax: matplotlib.axes.Axes) -> None:
    ref = refutil.load_phase2('regplot')['band']
    metafopy.regplot(res, pi=True, ax=ax)
    lines = [line for line in ax.lines if line.get_gid() == 'line']
    assert len(lines) == 5
    by_width = {round(line.get_linewidth(), 3): line for line in lines if line.get_linewidth() == 3.0}
    regression = next(iter(by_width.values()))
    refutil.close(numpy.asarray(regression.get_ydata()).sum(), ref['sum_pred'], 'regression line')
    sums = sorted(float(numpy.asarray(line.get_ydata()).sum()) for line in lines if line.get_linewidth() != 3.0)
    expected = sorted([ref['sum_ci_lb'], ref['sum_ci_ub'], ref['sum_pi_lb'], ref['sum_pi_ub']])
    refutil.close(sums, expected, 'bands')
    assert numpy.asarray(regression.get_xdata()).shape[0] == 1000


def test_axis_labels_and_legend(res: metafopy.RMAResult, ax: matplotlib.axes.Axes) -> None:
    metafopy.regplot(res, pi=True, legend=True, ax=ax)
    assert ax.get_xlabel() == 'ablat'
    assert ax.get_ylabel() == 'Observed Outcome'
    legend = ax.get_legend()
    assert legend is not None
    labels = [t.get_text() for t in legend.get_texts()]
    assert labels == ['Studies', 'Regression Line', '95% Confidence Interval', '95% Prediction Interval']


def test_rejects_models_without_moderators(dat: pandas.DataFrame, ax: matplotlib.axes.Axes) -> None:
    with pytest.raises(metafopy.errors.InputError, match='intercept-only'):
        metafopy.regplot(metafopy.rma('yi', 'vi', data=dat), ax=ax)


def test_rejects_ambiguous_moderator_and_both_transformations(dat: pandas.DataFrame, res: metafopy.RMAResult, ax: matplotlib.axes.Axes) -> None:
    res2 = metafopy.rma('yi', 'vi', mods='~ ablat + year', data=dat)
    with pytest.raises(metafopy.errors.InputError, match="Must specify the 'mod'"):
        metafopy.regplot(res2, ax=ax)
    with pytest.raises(metafopy.errors.InputError, match='either'):
        metafopy.regplot(res, transf=numpy.exp, atransf=numpy.exp, ax=ax)
