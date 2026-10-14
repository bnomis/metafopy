"""llplot() checked against R metafor (tests/reference/phase2/llplot.json, written by tests/r/generate_llplot.R)."""

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
from metafopy.plot import LlplotResult

COLUMNS = [0, 99, 249, 399, 499]


@pytest.fixture(scope='module')
def dat() -> pandas.DataFrame:
    return metafopy.escalc('OR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=metafopy.datasets.load('bcg'))


@pytest.fixture
def ax() -> collections.abc.Iterator[matplotlib.axes.Axes]:
    fig, axes = matplotlib.pyplot.subplots(figsize=(7, 7))
    yield axes
    matplotlib.pyplot.close(fig)


def check(out: LlplotResult, ref: dict[str, typing.Any], rtol: float = 1e-8) -> None:
    assert list(out.lls.shape) == refutil.as_list(ref['dim'])
    assert numpy.isnan(out.lls).sum(axis=1).tolist() == [int(v) for v in refutil.as_list(ref['nna'])]
    refutil.close(numpy.nansum(out.lls, axis=1), ref['sums'], 'sums', rtol=rtol)
    refutil.close(out.lls[:, COLUMNS].T.ravel(), ref['at'], 'values', rtol=rtol)
    refutil.close(numpy.nanmax(out.lls), ref['max'], 'maximum', rtol=rtol)


def test_gen_matches_metafor(dat: pandas.DataFrame, ax: matplotlib.axes.Axes) -> None:
    check(metafopy.llplot('GEN', yi='yi', vi='vi', data=dat, xvals=500, ax=ax), refutil.load_phase2('llplot')['gen'])


def test_gen_unscaled_matches_metafor(dat: pandas.DataFrame, ax: matplotlib.axes.Axes) -> None:
    out = metafopy.llplot('GEN', yi='yi', vi='vi', data=dat, xvals=500, scale=False, level=95, xlim=(-3, 2), ax=ax)
    check(out, refutil.load_phase2('llplot')['gen_unscaled'])
    assert ax.get_ylabel() == 'Likelihood'


def test_or_matches_metafor(ax: matplotlib.axes.Axes) -> None:
    bcg = metafopy.datasets.load('bcg')
    out = metafopy.llplot('OR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=bcg, xvals=500, level=99, ax=ax)
    check(out, refutil.load_phase2('llplot')['or'], rtol=1e-6)
    assert ax.get_xlabel() == 'Log Odds Ratio'


def test_or_with_zero_cells_and_subset_matches_metafor(ax: matplotlib.axes.Axes) -> None:
    bcg = metafopy.datasets.load('bcg')
    bcg.loc[1, ['tpos', 'cpos']] = 0
    bcg.loc[4, 'tpos'] = 0
    with pytest.warns(UserWarning, match='omitted'):
        out = metafopy.llplot('OR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=bcg, xvals=500, level=99, subset=[*range(8), *range(9, 13)], ax=ax)
    check(out, refutil.load_phase2('llplot')['or_zero'], rtol=1e-6)
    styles = sorted({typing.cast(typing.Any, line).meta['lty'] for line in ax.lines if line.get_gid() == 'line'})
    assert styles == ['dashed', 'solid']


def test_default_line_widths_and_colours(dat: pandas.DataFrame, ax: matplotlib.axes.Axes) -> None:
    metafopy.llplot('GEN', yi='yi', vi='vi', data=dat, ax=ax)
    lines = [line for line in ax.lines if line.get_gid() == 'line']
    assert len(lines) == 13
    widths = [line.get_linewidth() for line in lines]
    assert widths == sorted(widths)  # the most precise studies (thickest lines) are drawn last
    assert max(widths) == pytest.approx(4.0)
    assert min(widths) == pytest.approx(0.4)


def test_rejects_bad_arguments(ax: matplotlib.axes.Axes) -> None:
    with pytest.raises(metafopy.errors.InputError, match='GEN'):
        metafopy.llplot('RR', ax=ax)
    with pytest.raises(metafopy.errors.InputError, match="'vi' or 'sei'"):
        metafopy.llplot('GEN', yi=[0.1, 0.2], ax=ax)
    with pytest.raises(metafopy.errors.InputError, match='lty'):
        metafopy.llplot('GEN', yi=[0.1, 0.2], vi=[0.1, 0.2], lty=['solid'] * 3, ax=ax)
