"""plot_profile() of the profile lists of rma_mv, location-scale and selection models, and of single location-scale profiles, against R metafor.

metafor draws one panel per component; the reference (tests/r/generate_profile_plots.R) records the geometry of each component's panel.
"""

import collections.abc
import dataclasses
import json
import pathlib
import typing
import warnings

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot
import numpy
import pandas
import plotref
import pytest
import refutil

import metafopy
import metafopy.datasets
import metafopy.errors
import metafopy.plot

REF: dict[str, typing.Any] = json.loads((pathlib.Path(__file__).parent / 'reference' / 'phase11' / 'profile_plots.json').read_text(encoding='utf-8'))
SKIP = {'mtext', 'axis', 'usr'}


@pytest.fixture
def fig() -> collections.abc.Iterator[matplotlib.figure.Figure]:
    figure = matplotlib.pyplot.figure(figsize=(7, 7))
    yield figure
    matplotlib.pyplot.close(figure)


def bangertdrowns() -> pandas.DataFrame:
    d = metafopy.datasets.load('bangertdrowns2004')
    d['ni100'] = d['ni'] / 100
    return d


def profiles() -> dict[str, typing.Any]:
    kon = metafopy.datasets.load('konstantopoulos2011')
    bd = bangertdrowns()
    hack = metafopy.datasets.load('hackshaw1998')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mv = metafopy.profile(metafopy.rma_mv('yi', 'vi', random='~ 1 | district/school', data=kon), steps=5)
        ls = metafopy.rma_uni('yi', 'vi', mods='~ ni100', scale='~ ni100', data=bd)
        ls0 = metafopy.rma_uni('yi', 'vi', scale='~ 1', data=bd)
        sel = metafopy.selmodel(metafopy.rma_uni(hack.yi, hack.vi, method='ML'), 'stepfun', steps=0.025)
        return {
            'mv': mv,
            'ls': metafopy.profile(ls, steps=6),
            'ls_single': metafopy.profile(ls, alpha=1, steps=6),
            'ls_intercept': metafopy.profile(ls0, steps=5),
            'sel': metafopy.profile(sel, steps=5),
        }


def with_r_values(res: typing.Any, key: str) -> typing.Any:
    """The profiles with the grid, log-likelihoods and estimate R computed, so that the drawing is compared free of optimiser differences."""
    out = []
    for r, ref in zip(res if isinstance(res, list) else [res], REF[key]['table'], strict=True):
        grid = numpy.array(refutil.as_list(ref['grid']), dtype=float)
        changes: dict[str, typing.Any] = {
            'll': numpy.array(refutil.as_list(ref['ll']), dtype=float),
            'vc': ref['vc'],
            'maxll': ref['maxll'],
            'ylim': tuple(ref['ylim']),
        }
        if hasattr(r, 'alpha'):
            changes['alpha'] = grid
        else:
            changes.update(tau2=grid, xlim=(float(grid.min()), float(grid.max())))
        out.append(dataclasses.replace(r, **changes))
    return out if isinstance(res, list) else out[0]


@pytest.fixture(scope='module')
def prof() -> dict[str, typing.Any]:
    return profiles()


@pytest.fixture(scope='module')
def rprof(prof: dict[str, typing.Any]) -> dict[str, typing.Any]:
    return {key: with_r_values(res, key) for key, res in prof.items()}


def check_panels(axes: list[matplotlib.axes.Axes], key: str, panels: str = 'panels') -> None:
    recs = REF[key][panels]
    assert len(axes) == len(recs)
    for j, (a, rec) in enumerate(zip(axes, recs, strict=True)):
        plotref.check_all(a, rec, f'{key}[{j}]', skip=SKIP)
        plotref.check_plain(a, rec, f'{key}[{j}]')


def check_table(res: typing.Any, key: str) -> None:
    for r, ref in zip(res if isinstance(res, list) else [res], REF[key]['table'], strict=True):
        refutil.close(r.alpha if hasattr(r, 'alpha') else r.tau2, ref['grid'], f'{key}: grid', rtol=1e-6)
        refutil.close(r.ll, ref['ll'], f'{key}: ll', rtol=1e-6)
        refutil.close(r.ylim, ref['ylim'], f'{key}: ylim', rtol=1e-6)


@pytest.mark.parametrize('key', ['mv', 'ls', 'sel'])
def test_lists_match_metafor(rprof: dict[str, typing.Any], fig: matplotlib.figure.Figure, key: str) -> None:
    res = rprof[key]
    assert isinstance(res, list)
    axes = metafopy.plot.plot_profile(res, ax=fig.add_subplot(111))
    check_panels(axes, key)


def test_list_options(rprof: dict[str, typing.Any], fig: matplotlib.figure.Figure) -> None:
    axes = metafopy.plot.plot_profile(rprof['mv'], cline=True, pch=17, ylim=(-70, -10), ax=fig.add_subplot(111))
    check_panels(axes, 'mv', 'panels_opts')


def test_single_location_scale_profile(rprof: dict[str, typing.Any], fig: matplotlib.figure.Figure) -> None:
    ax = fig.add_subplot(111)
    assert metafopy.plot.plot_profile(rprof['ls_single'], cline=True, refline=False, ax=ax) is ax
    plotref.check_all(ax, REF['ls_single']['panels'][0], 'ls_single', skip=SKIP)
    plotref.check_plain(ax, REF['ls_single']['panels'][0], 'ls_single')
    assert ax.get_xlabel() == r'$\alpha_{2}$ Value'
    assert ax.get_title() == r'Profile Plot for $\alpha_{2}$'


def test_intercept_only_location_scale_profile(rprof: dict[str, typing.Any], fig: matplotlib.figure.Figure) -> None:
    ax = metafopy.plot.plot_profile(rprof['ls_intercept'], ax=fig.add_subplot(111))
    plotref.check_all(ax, REF['ls_intercept']['panels'][0], 'ls_intercept', skip=SKIP)
    plotref.check_plain(ax, REF['ls_intercept']['panels'][0], 'ls_intercept')
    assert ax.get_xlabel() == r'$\alpha$ Value'


def test_profiles_match_metafor(prof: dict[str, typing.Any]) -> None:
    for key in ('mv', 'ls', 'ls_single', 'ls_intercept', 'sel'):
        check_table(prof[key], key)


def test_labels_of_the_panels(prof: dict[str, typing.Any], fig: matplotlib.figure.Figure) -> None:
    axes = metafopy.plot.plot_profile(prof['sel'], ax=fig.add_subplot(111))
    assert [a.get_xlabel() for a in axes] == [r'$\tau^2$ Value', r'$\delta_{2}$ Value']
    assert [a.get_title() for a in axes] == [r'Profile Plot for $\tau^2$', r'Profile Plot for $\delta_{2}$']
    assert {a.get_ylabel() for a in axes} == {'Log-Likelihood'}
    custom = metafopy.plot.plot_profile(prof['mv'], xlab=['a', 'b'], main='m', ylab='y', ax=fig.add_subplot(111))
    assert [a.get_xlabel() for a in custom] == ['a', 'b']
    assert {a.get_title() for a in custom} == {'m'}


def test_panel_label_lengths_are_checked(prof: dict[str, typing.Any], fig: matplotlib.figure.Figure) -> None:
    with pytest.raises(metafopy.errors.InputError, match="'xlab'"):
        metafopy.plot.plot_profile(prof['mv'], xlab=['a', 'b', 'c'], ax=fig.add_subplot(111))
    with pytest.raises(metafopy.errors.InputError, match="'main'"):
        metafopy.plot.plot_profile(prof['ls_single'], main=['a', 'b'], ax=fig.add_subplot(111))


def test_plot_argument_draws_the_grid(fig: matplotlib.figure.Figure) -> None:
    bd = bangertdrowns()
    ls = metafopy.rma_uni('yi', 'vi', mods='~ ni100', scale='~ ni100', data=bd)
    ax = fig.add_subplot(111)
    res: typing.Any = metafopy.profile(ls, steps=4, plot=True, ax=ax)
    assert isinstance(res, list)
    assert sum(1 for a in fig.axes if a.get_xlabel()) == 2
    fig2 = matplotlib.pyplot.figure()
    ax2 = fig2.add_subplot(111)
    single: typing.Any = metafopy.profile(ls, alpha=1, steps=4, plot=True, ax=ax2)
    assert not isinstance(single, list)
    assert ax2.get_xlabel() == r'$\alpha_{2}$ Value'
    matplotlib.pyplot.close(fig2)


def test_pipeline_through_figure_without_axes() -> None:
    kon = metafopy.datasets.load('konstantopoulos2011')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res: typing.Any = metafopy.profile(metafopy.rma_mv('yi', 'vi', random='~ 1 | district/school', data=kon), steps=4, plot=True)
    assert isinstance(res, list)
    assert sum(1 for a in matplotlib.pyplot.gcf().axes if a.get_xlabel()) == 2
    matplotlib.pyplot.close('all')
    numpy.testing.assert_array_equal(numpy.sort(res[0].tau2), res[0].tau2)
