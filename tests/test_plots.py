"""forest(), addpoly(), funnel(), baujat(), radial(), qqnorm(), labbe(), plot_gosh(), plot_infl() and plot_profile() checked against R metafor.

The reference file (tests/reference/phase2/plots.json, written by tests/r/generate_plots.R) holds the values the R
functions return and the geometry they draw (polygons, segments, texts, points, ...), recorded on a null device.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes

from metafopy import RMAResult, rma
from metafopy.bias import trimfill
from metafopy.plot import addpoly, baujat, forest, funnel
from plotref import PLOTS, check_all, check_plain, flat
from refutil import bcg, bcg_na, close


@pytest.fixture(scope='module')
def dat() -> pd.DataFrame:
    return bcg()


@pytest.fixture
def ax() -> Iterator[Axes]:
    fig, axes = plt.subplots(figsize=(7, 7))
    yield axes
    plt.close(fig)


def _ret(res: Any, name: str, ref: dict[str, Any], attr: str | None = None) -> None:
    close(res if attr is None else getattr(res, attr), ref['ret'][name], name)


def check_forest(r: Any, ref: dict[str, Any], what: str, areas: bool = True) -> None:
    ret = ref['ret']
    close(r.xlim, ret['xlim'], f'{what}: xlim')
    close(r.alim, ret['alim'], f'{what}: alim')
    close(r.at, ret['at'], f'{what}: at')
    close(r.ylim, ret['ylim'], f'{what}: ylim')
    close(r.rows, ret['rows'], f'{what}: rows')
    close(r.cex, ret['cex'], f'{what}: cex')
    close(r.cex_lab, ret['cex.lab'], f'{what}: cex.lab')
    close(r.cex_axis, ret['cex.axis'], f'{what}: cex.axis')
    close(r.textpos, ret['textpos'], f'{what}: textpos')
    if ret.get('ilab.xpos') is not None:
        close(r.ilab_xpos, ret['ilab.xpos'], f'{what}: ilab.xpos')
    if areas and ret.get('areas') is not None:
        close(r.areas, ret['areas'], f'{what}: areas')
    check_all(r.ax, ref['rec'], what, skip={'mtext'})
    mt = [t.get_text() for t in r.ax.texts if t.get_gid() == 'mtext']
    assert mt == [str(m['text']) for m in ref['rec'].get('mtext', [])], f'{what}: xlab {mt}'


FOREST_RMA: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}


def _fit(dat: pd.DataFrame, **kw: Any) -> RMAResult:
    return rma('yi', 'vi', data=dat, measure='RR', **kw)


@pytest.fixture(scope='module')
def fits(dat: pd.DataFrame) -> dict[str, RMAResult]:
    return {
        're': _fit(dat, slab='label'),
        'noslab': _fit(dat),
        'mods': _fit(dat, mods='~ ablat', slab='label'),
        'ee': _fit(dat, method='EE', slab='label'),
    }


def _case_forest_exp(dat: pd.DataFrame) -> dict[str, Any]:
    return {
        'atransf': np.exp,
        'at': np.log([0.05, 0.25, 1, 4]),
        'xlim': (-16, 6),
        'ilab': dat[['tpos', 'tneg', 'cpos', 'cneg']].to_numpy(),
        'ilab_lab': ['TB+', 'TB-', 'CTRL+', 'CTRL-'],
        'ilab_xpos': [-9.5, -8, -6, -4.5],
        'cex': 0.75,
        'header': ['Author(s) and Year', 'RR [95% CI]'],
        'digits': 3,
        'efac': (1, 2),
    }


FOREST_CASES: dict[str, tuple[str, Callable[[pd.DataFrame], dict[str, Any]]]] = {
    'forest_re': ('re', lambda d: {}),
    'forest_noslab': ('noslab', lambda d: {}),
    'forest_exp': ('re', _case_forest_exp),
    'forest_transf': (
        're',
        lambda d: {'transf': np.exp, 'alim': (0, 5), 'refline': 1, 'order': 'obs', 'showweights': True, 'psize': 1, 'pch': 18, 'shade': 'zebra2'},
    ),
    'forest_mods': ('mods', lambda d: {'order': 'fit', 'addpred': True, 'shade': True, 'plim': (1, 2), 'colout': 'blue'}),
    'forest_mods_resid': ('mods', lambda d: {'order': 'resid', 'level': 90, 'annotate': False, 'xlab': 'Log RR', 'header': False, 'ylim': (-1, 16)}),
    'forest_pi_polygon': ('re', lambda d: {'predstyle': 'polygon', 'showweights': True, 'mlab': 'RE Model', 'col': ('blue', 'red')}),
    'forest_pi_bar': ('re', lambda d: {'predstyle': 'bar', 'level': 90, 'atransf': np.exp, 'at': np.log([0.1, 1, 10]), 'xlim': (-8, 6)}),
    'forest_pi_line': ('re', lambda d: {'addpred': True, 'steps': 7, 'alim': (-3, 3), 'rows': np.arange(15, 2, -1), 'ylim': (-1, 19)}),
    'forest_ee': ('ee', lambda d: {'addfit': False, 'xlab': ['Left', 'Mid', 'Right'], 'refline': None, 'top': 4}),
    'forest_arrows': (
        're',
        lambda d: {'alim': (-1, 1), 'xlim': (-6, 4), 'olim': (-1.2, 1.2), 'cex': 1.2, 'width': 7, 'annosym': (' (', ' to ', ')')},
    ),
    'forest_dist': (
        're',
        lambda d: {'order': 'prec', 'rowadj': (0.1, -0.1, 0), 'textpos': (-7, 5), 'xlim': (-8, 6), 'lty': ('dashed', 'dashed', 'solid')},
    ),
}


@pytest.mark.parametrize('key', list(FOREST_CASES))
def test_forest_rma_matches_metafor(key: str, dat: pd.DataFrame, fits: dict[str, RMAResult], ax: Axes) -> None:
    fit, mk = FOREST_CASES[key]
    r = forest(fits[fit], ax=ax, **mk(dat))
    check_forest(r, PLOTS[key], key)


def test_forest_subgroups_matches_metafor(dat: pd.DataFrame, ax: Axes) -> None:
    order = np.argsort(dat['alloc'].to_numpy(), kind='stable')
    sub = {a: _fit(dat, slab='label', subset=(dat['alloc'] == a).to_numpy()) for a in ('random', 'alternate', 'systematic')}
    fit_all = _fit(dat, slab='label', subset=order)
    rows_all = np.r_[3:9, 12:15, 18:22]
    r = forest(fit_all, rows=rows_all, ylim=(-1, 25), xlim=(-8, 5), addfit=False, cex=0.8, ax=ax)
    addpoly(sub['alternate'], row=16.5, mlab='RE Model for Subgroup A', cex=0.8, ax=ax)
    addpoly(sub['random'], row=9.5, mlab='RE Model for Subgroup B', cex=0.8, col='red', border='red', ax=ax)
    addpoly(sub['systematic'], row=1.5, mlab='RE Model for Subgroup C', cex=0.8, annotate=False, ax=ax)
    ref = PLOTS['forest_subgroups']
    check_all(ax, ref['rec'], 'forest_subgroups', skip={'mtext'})
    close(r.rows, ref['ret']['rows'])


def _yi_vi(dat: pd.DataFrame) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], list[str]]:
    return dat['yi'].to_numpy(), dat['vi'].to_numpy(), dat['label'].tolist()


def test_forest_default_matches_metafor(dat: pd.DataFrame, ax: Axes) -> None:
    yi, vi, slab = _yi_vi(dat)
    r = forest(yi, vi, slab=slab, measure='RR', ax=ax)
    check_forest(r, PLOTS['forest_default'], 'forest_default', areas=False)


def test_forest_default_options_match_metafor(dat: pd.DataFrame, ax: Axes) -> None:
    yi, vi, slab = _yi_vi(dat)
    r = forest(
        yi,
        sei=np.sqrt(vi),
        slab=slab,
        order='prec',
        showweights=True,
        transf=np.exp,
        alim=(0, 4),
        refline=1,
        ilab=dat['year'].to_numpy(),
        ilab_lab=['Year'],
        ilab_xpos=-3,
        ilab_pos=2,
        col=['black', 'red'] * 6 + ['black'],
        shade='zebra',
        psize=np.arange(1, 14) / 5,
        pch=19,
        header='Trial',
        xlab=['favours A', 'favours B'],
        cex=0.9,
        measure='RR',
        ax=ax,
    )
    check_forest(r, PLOTS['forest_default_opts'], 'forest_default_opts', areas=False)


def test_forest_default_ci_bounds_match_metafor(dat: pd.DataFrame, ax: Axes) -> None:
    yi, _, _ = _yi_vi(dat)
    r = forest(yi, ci_lb=yi - 0.3, ci_ub=yi + 0.5, level=90, slab=np.nan, efac=(1, 1.5), digits=(2, 1, 3), atransf=lambda x: x / 2, measure='RR', ax=ax)
    check_forest(r, PLOTS['forest_default_ci'], 'forest_default_ci', areas=False)


def test_forest_default_missing_and_subset_match_metafor(dat: pd.DataFrame, ax: Axes) -> None:
    yi, vi, slab = _yi_vi(dat)
    yi_na = yi.copy()
    yi_na[[2, 6]] = np.nan
    sub = np.ones(13, dtype=bool)
    sub[0] = False
    r = forest(yi_na, vi, slab=slab, subset=sub, rows=np.arange(13, 1, -1), ylim=(0, 17), measure='RR', ax=ax)
    check_forest(r, PLOTS['forest_default_na'], 'forest_default_na', areas=False)


def test_forest_default_order_vector_and_shade_match_metafor(dat: pd.DataFrame, ax: Axes) -> None:
    yi, vi, slab = _yi_vi(dat)
    year = dat['year'].to_numpy()
    r = forest(yi, vi, slab=slab, order=year, subset=year > 1950, shade=np.ones(13, dtype=bool), measure='RR', ax=ax)
    check_forest(r, PLOTS['forest_default_sub'], 'forest_default_sub', areas=False)


def test_addpoly_default_matches_metafor(dat: pd.DataFrame, ax: Axes) -> None:
    yi, vi, slab = _yi_vi(dat)
    r = forest(yi, vi, slab=slab, ylim=(-4, 17), xlim=(-8, 6), measure='RR', ax=ax)
    addpoly([-0.5, 0.2], sei=[0.1, 0.2], rows=[-1.5, -2.5], mlab=['A', 'B'], cex=0.9, ax=ax)
    addpoly(0.3, ci_lb=-0.2, ci_ub=0.9, pi_lb=-0.8, pi_ub=1.4, rows=-3.2, predstyle='polygon', mlab='C', annotate=False, ax=ax)
    addpoly(-0.3, vi=0.05, rows=-1, transf=np.exp, efac=2, constarea=True, ax=ax)
    ref = PLOTS['addpoly_default']
    check_all(ax, ref['rec'], 'addpoly_default', skip={'mtext'})
    close(r.xlim, ref['ret']['xlim'])


def test_addpoly_rma_matches_metafor(dat: pd.DataFrame, fits: dict[str, RMAResult], ax: Axes) -> None:
    forest(fits['re'], ylim=(-5, 17), addfit=False, ax=ax)
    addpoly(fits['re'], row=-1.5, addpred=True, mlab='RE model', atransf=np.exp, ax=ax)
    addpoly(fits['re'], row=-3, predstyle='bar', level=90, annotate=True, digits=3, col='gray', ax=ax)
    check_all(ax, PLOTS['addpoly_rma']['rec'], 'addpoly_rma', skip={'mtext'})


def test_forest_missing_values_in_fit(dat: pd.DataFrame, ax: Axes) -> None:
    dna = bcg_na()
    res = rma('yi', 'vi', mods='~ ablat', data=dna, slab='label')
    r = forest(res, ax=ax)
    assert len(r.rows) == 11
    assert flat(list(r.rows)) == sorted(flat(list(r.rows)))


def test_forest_rejects_unsupported_options(fits: dict[str, RMAResult], ax: Axes) -> None:
    with pytest.raises(NotImplementedError):
        forest(fits['re'], predstyle='dist', ax=ax)
    with pytest.raises(ValueError, match='not both'):
        forest(fits['re'], transf=np.exp, atransf=np.exp, ax=ax)


# -- funnel ---------------------------------------------------------------------------------------------


@pytest.fixture(scope='module')
def ffits(dat: pd.DataFrame) -> dict[str, RMAResult]:
    return {
        'ni': _fit(dat, slab='label', ni='n'),
        'mods': _fit(dat, mods='~ ablat', slab='label', ni='n'),
        'ee': _fit(dat, method='EE', slab='label'),
        'tf': trimfill(_fit(dat, slab='label', ni='n')),
    }


def check_funnel(r: Any, ref: dict[str, Any], what: str, compare_col: bool = False) -> None:
    ret = ref['ret']
    close(r.x, ret['x'], f'{what}: x')
    close(r.y, ret['y'], f'{what}: y')
    assert list(r.slab) == [str(s) for s in flat(ret['slab'])], what
    if 'fill' in ret:
        assert list(r.fill) == flat(ret['fill'])
    check_all(r.ax, ref['rec'], what, skip={'mtext'}, compare_col=True)


@pytest.mark.parametrize('yaxis', ['sei', 'vi', 'seinv', 'vinv', 'ni', 'ninv', 'sqrtni', 'sqrtninv', 'lni', 'wi'])
def test_funnel_yaxis_matches_metafor(yaxis: str, ffits: dict[str, RMAResult], ax: Axes) -> None:
    r = funnel(ffits['ni'], yaxis=yaxis, ax=ax)
    check_funnel(r, PLOTS[f'funnel_{yaxis}'], yaxis)


FUNNEL_CASES: dict[str, tuple[str, dict[str, Any]]] = {
    'funnel_levels': (
        'ni',
        {
            'level': [90, 95, 99],
            'shade': ['white', 'gray55', 'gray75'],
            'refline': 0,
            'back': 'gray90',
            'hlines': 'white',
            'steps': 7,
            'atransf': np.exp,
            'at': np.log([0.25, 0.5, 1, 2]),
            'digits': (2, 2),
            'label': 'out',
            'xlim': (-2.5, 2),
            'ylim': (0, 1.2),
        },
    ),
    'funnel_levels_ramp': (
        'ni',
        {'level': [90, 95, 99], 'yaxis': 'seinv', 'label': 3, 'offset': 0.2, 'lty': (2, 4), 'pch': 15, 'col': 'blue', 'bg': 'red', 'addtau2': True},
    ),
    'funnel_vi_refline2': ('ni', {'yaxis': 'vi', 'refline': -0.5, 'level': 90, 'xlab': 'Effect', 'ylab': 'V', 'digits': 3, 'label': True}),
    'funnel_mods': ('mods', {'yaxis': 'seinv'}),
    'funnel_mods_stud': ('mods', {'type': 'rstudent', 'label': 2, 'steps': 4}),
    'funnel_ee': ('ee', {'yaxis': 'vinv', 'level': [95, 99]}),
    'funnel_tf': ('tf', {}),
    'funnel_tf_opts': (
        'tf',
        {'yaxis': 'seinv', 'pch_fill': 17, 'col': ['black', 'red'], 'bg': ['gray', 'yellow'], 'level': [90, 95, 99], 'label': 'all'},
    ),
    'funnel_tf_ni': ('tf', {'yaxis': 'ni'}),
    'funnel_pchvec': ('ni', {'pch': [1, 19] * 6 + [1], 'col': ['black', 'blue'] * 6 + ['black'], 'refline': 0.1}),
}


@pytest.mark.parametrize('key', list(FUNNEL_CASES))
def test_funnel_rma_matches_metafor(key: str, ffits: dict[str, RMAResult], ax: Axes) -> None:
    fit, kw = FUNNEL_CASES[key]
    r = funnel(ffits[fit], ax=ax, **kw)
    check_funnel(r, PLOTS[key], key)


def test_funnel_default_matches_metafor(dat: pd.DataFrame, ax: Axes) -> None:
    yi, vi, _ = _yi_vi(dat)
    check_funnel(funnel(yi, vi, ax=ax), PLOTS['funnel_default'], 'funnel_default')


def test_funnel_default_options_match_metafor(dat: pd.DataFrame, ax: Axes) -> None:
    yi, vi, slab = _yi_vi(dat)
    sub = np.ones(13, dtype=bool)
    sub[0] = False
    r = funnel(yi, sei=np.sqrt(vi), yaxis='seinv', level=[90, 99], refline=-0.7, slab=slab, label=4, subset=sub, ax=ax)
    check_funnel(r, PLOTS['funnel_default_opts'], 'funnel_default_opts')


def test_funnel_default_ni_and_wi_match_metafor(dat: pd.DataFrame, ax: Axes) -> None:
    yi, vi, _ = _yi_vi(dat)
    check_funnel(funnel(yi, ni=dat['n'].to_numpy(), yaxis='sqrtni', xlab='A', ax=ax), PLOTS['funnel_default_ni'], 'funnel_default_ni')
    ax.clear()
    check_funnel(funnel(yi, vi, yaxis='wi', ylim=(1, 20), ax=ax), PLOTS['funnel_default_wi'], 'funnel_default_wi')


def test_funnel_default_missing_matches_metafor(dat: pd.DataFrame, ax: Axes) -> None:
    yi, vi, _ = _yi_vi(dat)
    yi_na = yi.copy()
    yi_na[[2, 6]] = np.nan
    check_funnel(funnel(yi_na, vi, ax=ax), PLOTS['funnel_default_na'], 'funnel_default_na')


# -- baujat ---------------------------------------------------------------------------------------------


def check_xy(r: Any, ref: dict[str, Any], what: str) -> None:
    close(r.x, ref['ret']['x'], f'{what}: x')
    close(r.y, ref['ret']['y'], f'{what}: y')
    assert list(r.ids) == [int(v) for v in flat(ref['ret']['ids'])], what
    assert list(r.slab) == [str(v) for v in flat(ref['ret']['slab'])], what


def test_baujat_matches_metafor(fits: dict[str, RMAResult], ax: Axes) -> None:
    r = baujat(fits['re'], ax=ax)
    check_xy(r, PLOTS['baujat_re'], 'baujat_re')
    check_plain(ax, PLOTS['baujat_re']['rec'], 'baujat_re')
    check_all(ax, PLOTS['baujat_re']['rec'], 'baujat_re', skip={'axis', 'usr', 'mtext'})


def test_baujat_options_match_metafor(fits: dict[str, RMAResult], ax: Axes) -> None:
    r = baujat(fits['ee'], symbol='slab', cex=0.6, grid='red', xlab='X', ylab='Y', ax=ax)
    ref = PLOTS['baujat_ee']
    check_xy(r, ref, 'baujat_ee')
    check_plain(ax, ref['rec'], 'baujat_ee')
    check_all(ax, ref['rec'], 'baujat_ee', skip={'axis', 'usr', 'mtext'})


def test_baujat_moderators_and_symbols_match_metafor(fits: dict[str, RMAResult], ax: Axes) -> None:
    r = baujat(fits['mods'], symbol=16, grid=False, xlim=(0, 12), ylim=(0, 3), ax=ax)
    ref = PLOTS['baujat_mods']
    check_xy(r, ref, 'baujat_mods')
    check_plain(ax, ref['rec'], 'baujat_mods')
    check_all(ax, ref['rec'], 'baujat_mods', skip={'axis', 'usr', 'mtext'})
    ax.clear()
    r = baujat(fits['mods'], symbol=[1, 2, 19] * 4 + [1], cex=1.2, ax=ax)
    ref = PLOTS['baujat_symvec']
    check_xy(r, ref, 'baujat_symvec')
    check_all(ax, ref['rec'], 'baujat_symvec', skip={'axis', 'usr', 'mtext'})
