"""forest(), addpoly(), funnel(), baujat(), radial(), qqnorm(), labbe(), plot_gosh(), plot_infl() and plot_profile() checked against R metafor.

The reference file (tests/reference/phase2/plots.json, written by tests/r/generate_plots.R) holds the values the R
functions return and the geometry they draw (polygons, segments, texts, points, ...), recorded on a null device.
"""

import collections.abc
import typing

import matplotlib.axes
import matplotlib.pyplot
import numpy
import pandas
import plotref
import pytest
import refutil
import scipy.stats

import metafopy
import metafopy.bias.trimfill
import metafopy.datasets
import metafopy.errors
import metafopy.plot
import metafopy.plot.baujat
import metafopy.plot.draw
import metafopy.plot.forest
import metafopy.plot.funnel
import metafopy.plot.labbe
import metafopy.plot.plain
import metafopy.plot.qqnorm
import metafopy.plot.radial
import metafopy.plot.rutils
import metafopy.transf


@pytest.fixture(scope='module')
def dat() -> pandas.DataFrame:
    return refutil.bcg()


@pytest.fixture
def ax() -> collections.abc.Iterator[matplotlib.axes.Axes]:
    fig, axes = matplotlib.pyplot.subplots(figsize=(7, 7))
    yield axes
    matplotlib.pyplot.close(fig)


def case_ret(res: typing.Any, name: str, ref: dict[str, typing.Any], attr: str | None = None) -> None:
    refutil.close(res if attr is None else getattr(res, attr), ref['ret'][name], name)


def check_forest(r: typing.Any, ref: dict[str, typing.Any], what: str, areas: bool = True) -> None:
    ret = ref['ret']
    refutil.close(r.xlim, ret['xlim'], f'{what}: xlim')
    refutil.close(r.alim, ret['alim'], f'{what}: alim')
    refutil.close(r.at, ret['at'], f'{what}: at')
    refutil.close(r.ylim, ret['ylim'], f'{what}: ylim')
    refutil.close(r.rows, ret['rows'], f'{what}: rows')
    refutil.close(r.cex, ret['cex'], f'{what}: cex')
    refutil.close(r.cex_lab, ret['cex.lab'], f'{what}: cex.lab')
    refutil.close(r.cex_axis, ret['cex.axis'], f'{what}: cex.axis')
    refutil.close(r.textpos, ret['textpos'], f'{what}: textpos')
    if ret.get('ilab.xpos') is not None:
        refutil.close(r.ilab_xpos, ret['ilab.xpos'], f'{what}: ilab.xpos')
    if areas and ret.get('areas') is not None:
        refutil.close(r.areas, ret['areas'], f'{what}: areas')
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    plotref.check_all(r.ax, ref['rec'], what, skip={'mtext'})
    mt = [t.get_text() for t in r.ax.texts if t.get_gid() == 'mtext']
    assert mt == [str(m['text']) for m in ref['rec'].get('mtext', [])], f'{what}: xlab {mt}'
    for t, m in zip([t for t in r.ax.texts if t.get_gid() == 'mtext'], ref['rec'].get('mtext', []), strict=True):
        assert t.meta['font'] == (m['font'] if m.get('font') is not None else m.get('fontpar', 1)), f'{what}: xlab font'
        assert t.meta['family'] == m.get('family', ''), f'{what}: xlab family'
    check_axis_family(r.ax, ref['rec'], what)


def check_axis_family(ax: matplotlib.axes.Axes, rec: dict[str, typing.Any], what: str) -> None:
    """The labels of the x axis are drawn in the font family R used for them."""
    fam = [a.get('family', '') for a in rec.get('axis', []) if a['side'] == 1 and a.get('at') is not None]
    if fam:
        expected = metafopy.plot.draw.mpl_family(fam[-1]) or matplotlib.rcParams['font.sans-serif'][0]
        got = ax.get_xticklabels()[0].get_fontfamily()[0]
        assert got == (expected if fam[-1] else got), f'{what}: axis family {got} != {expected}'
        if fam[-1]:
            assert got == expected


FOREST_RMA: dict[str, tuple[dict[str, typing.Any], dict[str, typing.Any]]] = {}


def case_fit(dat: pandas.DataFrame, **kw: typing.Any) -> metafopy.RMAResult:
    return typing.cast(metafopy.RMAResult, metafopy.rma('yi', 'vi', data=dat, measure='RR', **kw))


@pytest.fixture(scope='module')
def fits(dat: pandas.DataFrame) -> dict[str, metafopy.RMAResult]:
    return {
        're': case_fit(dat, slab='label'),
        'noslab': case_fit(dat),
        'mods': case_fit(dat, mods='~ ablat', slab='label'),
        'ee': case_fit(dat, method='EE', slab='label'),
        'knha': case_fit(dat, slab='label', test='knha'),
    }


def case_forest_exp(dat: pandas.DataFrame) -> dict[str, typing.Any]:
    return {
        'atransf': numpy.exp,
        'at': numpy.log([0.05, 0.25, 1, 4]),
        'xlim': (-16, 6),
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        'ilab': dat[['tpos', 'tneg', 'cpos', 'cneg']].to_numpy(),
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        'ilab_lab': ['TB+', 'TB-', 'CTRL+', 'CTRL-'],
        'ilab_xpos': [-9.5, -8, -6, -4.5],
        'cex': 0.75,
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        'header': ['Author(s) and Year', 'RR [95% CI]'],
        'digits': 3,
        'efac': (1, 2),
    }


PDXS = numpy.linspace(-3, 2, 301)
PDYS = scipy.stats.norm.pdf(PDXS, -0.7, 0.6)


# auditme: accept[non-alphabetical-dict-keys] order follows the R script that generates the fixtures
FOREST_CASES: dict[str, tuple[str, collections.abc.Callable[[pandas.DataFrame], dict[str, typing.Any]]]] = {
    'forest_re': ('re', lambda d: {}),
    'forest_noslab': ('noslab', lambda d: {}),
    'forest_exp': ('re', case_forest_exp),
    'forest_transf': (
        're',
        lambda d: {'transf': numpy.exp, 'alim': (0, 5), 'refline': 1, 'order': 'obs', 'showweights': True, 'psize': 1, 'pch': 18, 'shade': 'zebra2'},
    ),
    'forest_mods': ('mods', lambda d: {'order': 'fit', 'addpred': True, 'shade': True, 'plim': (1, 2), 'colout': 'blue'}),
    'forest_mods_resid': ('mods', lambda d: {'order': 'resid', 'level': 90, 'annotate': False, 'xlab': 'Log RR', 'header': False, 'ylim': (-1, 16)}),
    'forest_pi_polygon': ('re', lambda d: {'predstyle': 'polygon', 'showweights': True, 'mlab': 'RE Model', 'col': ('blue', 'red')}),
    'forest_pi_bar': ('re', lambda d: {'predstyle': 'bar', 'level': 90, 'atransf': numpy.exp, 'at': numpy.log([0.1, 1, 10]), 'xlim': (-8, 6)}),
    'forest_pi_line': ('re', lambda d: {'addpred': True, 'steps': 7, 'alim': (-3, 3), 'rows': numpy.arange(15, 2, -1), 'ylim': (-1, 19)}),
    'forest_ee': ('ee', lambda d: {'addfit': False, 'xlab': ['Left', 'Mid', 'Right'], 'refline': None, 'top': 4}),
    'forest_arrows': (
        're',
        lambda d: {'alim': (-1, 1), 'xlim': (-6, 4), 'olim': (-1.2, 1.2), 'cex': 1.2, 'width': 7, 'annosym': (' (', ' to ', ')')},
    ),
    'forest_dist': (
        're',
        lambda d: {'order': 'prec', 'rowadj': (0.1, -0.1, 0), 'textpos': (-7, 5), 'xlim': (-8, 6), 'lty': ('dashed', 'dashed', 'solid')},
    ),
    'forest_pi_shade': ('re', lambda d: {'predstyle': 'shade'}),
    'forest_pi_shade_exp': (
        're',
        # auditme: accept[literal-duplication] fixture inputs mirror the R calls that generate the reference
        lambda d: {'predstyle': 'shade', 'transf': numpy.exp, 'alim': (0, 5), 'refline': 1, 'col': ('black', 'darkblue', 'gray90'), 'showweights': True},
    ),
    'forest_pi_shade_knha': (
        'knha',
        lambda d: {'predstyle': 'shade', 'level': 90, 'atransf': numpy.exp, 'at': numpy.log([0.1, 0.5, 1, 2]), 'xlim': (-6, 4)},
    ),
    'forest_pi_dist': ('re', lambda d: {'predstyle': 'dist'}),
    'forest_pi_dist_exp': ('re', lambda d: {'predstyle': 'dist', 'transf': numpy.exp, 'alim': (0, 5), 'refline': 1, 'efac': (1, 2)}),
    'forest_pi_dist_tail': (
        're',
        lambda d: {
            'predstyle': 'dist',
            'predtail': '<-0.5',
            'level': 90,
            'mlab': 'RE',
            'col': ('black', 'gray40', 'lightblue', 'pink'),
            # auditme: accept[literal-duplication] fixture inputs mirror the R calls that generate the reference
            'border': ('black', 'blue'),
            'olim': (-2.5, 1.5),
            'alim': (-2, 1),
            'digits': 1,
        },
    ),
    'forest_pi_dist_knha': ('knha', lambda d: {'predstyle': 'dist', 'predtail': 0.2}),
    'forest_pi_dist_predlim': ('re', lambda d: {'predstyle': 'dist', 'transf': numpy.exp, 'predlim': (0.05, 3), 'alim': (0, 3), 'refline': 1}),
    'forest_pi_dist_ilogit': ('re', lambda d: {'predstyle': 'dist', 'transf': metafopy.transf.transf_ilogit, 'alim': (0, 1), 'refline': 0.5}),
    'forest_pi_dist_preddist': ('re', lambda d: {'predstyle': 'dist', 'preddist': {'x': PDXS, 'y': PDYS, 'level': 90}}),
    'forest_pi_shade_preddist': (
        're',
        lambda d: {'predstyle': 'shade', 'preddist': {'x': PDXS, 'y': PDYS, 'pi_lb': -1.8, 'pi_ub': 0.4}, 'predlim': (-2.5, 1)},
    ),
    'forest_fonts': (
        're',
        # auditme: accept[literal-duplication] fixture inputs mirror the R calls that generate the reference
        lambda d: {'fonts': ('serif', 'mono'), 'ilab': d['year'].to_numpy(), 'ilab_lab': ['Year'], 'ilab_xpos': [-7], 'xlim': (-10, 6)},
    ),
    'forest_fonts_named': (
        're',
        lambda d: {
            'fonts': {'mono': 2, 'serif': 3, 'sans': 4},
            'predstyle': 'polygon',
            'ilab': d['year'].to_numpy(),
            # auditme: accept[literal-duplication] fixture inputs mirror the R calls that generate the reference
            'ilab_lab': ['Year'],
            'ilab_xpos': [-7],
            'xlim': (-10, 6),
            'xlabfont': 3,
        },
    ),
    'forest_fonts_one': ('re', lambda d: {'fonts': 'serif', 'predstyle': 'dist', 'showweights': True}),
}


@pytest.mark.parametrize('key', list(FOREST_CASES))
def test_forest_rma_matches_metafor(key: str, dat: pandas.DataFrame, fits: dict[str, metafopy.RMAResult], ax: matplotlib.axes.Axes) -> None:
    fit, mk = FOREST_CASES[key]
    r = metafopy.plot.forest(fits[fit], ax=ax, **mk(dat))
    check_forest(r, plotref.PLOTS[key], key)
    if key.startswith('forest_pi_'):
        plotref.check_polygons(ax, plotref.PLOTS[key]['rec'], key, compare_col=True)  # also the colours


def test_forest_subgroups_matches_metafor(dat: pandas.DataFrame, ax: matplotlib.axes.Axes) -> None:
    order = numpy.argsort(dat['alloc'].to_numpy(), kind='stable')
    sub = {a: case_fit(dat, slab='label', subset=(dat['alloc'] == a).to_numpy()) for a in ('random', 'alternate', 'systematic')}
    fit_all = case_fit(dat, slab='label', subset=order)
    rows_all = numpy.r_[3:9, 12:15, 18:22]
    r = metafopy.plot.forest(fit_all, rows=rows_all, ylim=(-1, 25), xlim=(-8, 5), addfit=False, cex=0.8, ax=ax)
    metafopy.plot.addpoly(sub['alternate'], row=16.5, mlab='RE Model for Subgroup A', cex=0.8, ax=ax)
    metafopy.plot.addpoly(sub['random'], row=9.5, mlab='RE Model for Subgroup B', cex=0.8, col='red', border='red', ax=ax)
    metafopy.plot.addpoly(sub['systematic'], row=1.5, mlab='RE Model for Subgroup C', cex=0.8, annotate=False, ax=ax)
    ref = plotref.PLOTS['forest_subgroups']
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    plotref.check_all(ax, ref['rec'], 'forest_subgroups', skip={'mtext'})
    refutil.close(r.rows, ref['ret']['rows'])


def yi_vi(dat: pandas.DataFrame) -> tuple[numpy.ndarray[typing.Any, typing.Any], numpy.ndarray[typing.Any, typing.Any], list[str]]:
    return dat['yi'].to_numpy(), dat['vi'].to_numpy(), dat['label'].tolist()


def test_forest_default_matches_metafor(dat: pandas.DataFrame, ax: matplotlib.axes.Axes) -> None:
    yi, vi, slab = yi_vi(dat)
    r = metafopy.plot.forest(yi, vi, slab=slab, measure='RR', ax=ax)
    check_forest(r, plotref.PLOTS['forest_default'], 'forest_default', areas=False)


def test_forest_default_options_match_metafor(dat: pandas.DataFrame, ax: matplotlib.axes.Axes) -> None:
    yi, vi, slab = yi_vi(dat)
    r = metafopy.plot.forest(
        yi,
        sei=numpy.sqrt(vi),
        slab=slab,
        order='prec',
        showweights=True,
        transf=numpy.exp,
        alim=(0, 4),
        refline=1,
        ilab=dat['year'].to_numpy(),
        # auditme: accept[literal-duplication] fixture inputs mirror the R calls that generate the reference
        ilab_lab=['Year'],
        ilab_xpos=-3,
        ilab_pos=2,
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        col=['black', 'red'] * 6 + ['black'],
        shade='zebra',
        psize=numpy.arange(1, 14) / 5,
        pch=19,
        header='Trial',
        xlab=['favours A', 'favours B'],
        cex=0.9,
        measure='RR',
        ax=ax,
    )
    check_forest(r, plotref.PLOTS['forest_default_opts'], 'forest_default_opts', areas=False)


def test_forest_default_ci_bounds_match_metafor(dat: pandas.DataFrame, ax: matplotlib.axes.Axes) -> None:
    yi, _, _ = yi_vi(dat)
    r = metafopy.plot.forest(
        yi, ci_lb=yi - 0.3, ci_ub=yi + 0.5, level=90, slab=numpy.nan, efac=(1, 1.5), digits=(2, 1, 3), atransf=lambda x: x / 2, measure='RR', ax=ax
    )
    check_forest(r, plotref.PLOTS['forest_default_ci'], 'forest_default_ci', areas=False)


def test_forest_default_missing_and_subset_match_metafor(dat: pandas.DataFrame, ax: matplotlib.axes.Axes) -> None:
    yi, vi, slab = yi_vi(dat)
    yi_na = yi.copy()
    yi_na[[2, 6]] = numpy.nan
    sub = numpy.ones(13, dtype=bool)
    sub[0] = False
    r = metafopy.plot.forest(yi_na, vi, slab=slab, subset=sub, rows=numpy.arange(13, 1, -1), ylim=(0, 17), measure='RR', ax=ax)
    check_forest(r, plotref.PLOTS['forest_default_na'], 'forest_default_na', areas=False)


def test_forest_default_order_vector_and_shade_match_metafor(dat: pandas.DataFrame, ax: matplotlib.axes.Axes) -> None:
    yi, vi, slab = yi_vi(dat)
    year = dat['year'].to_numpy()
    r = metafopy.plot.forest(yi, vi, slab=slab, order=year, subset=year > 1950, shade=numpy.ones(13, dtype=bool), measure='RR', ax=ax)
    check_forest(r, plotref.PLOTS['forest_default_sub'], 'forest_default_sub', areas=False)


def test_addpoly_default_matches_metafor(dat: pandas.DataFrame, ax: matplotlib.axes.Axes) -> None:
    yi, vi, slab = yi_vi(dat)
    r = metafopy.plot.forest(yi, vi, slab=slab, ylim=(-4, 17), xlim=(-8, 6), measure='RR', ax=ax)
    metafopy.plot.addpoly([-0.5, 0.2], sei=[0.1, 0.2], rows=[-1.5, -2.5], mlab=['A', 'B'], cex=0.9, ax=ax)
    metafopy.plot.addpoly(0.3, ci_lb=-0.2, ci_ub=0.9, pi_lb=-0.8, pi_ub=1.4, rows=-3.2, predstyle='polygon', mlab='C', annotate=False, ax=ax)
    metafopy.plot.addpoly(-0.3, vi=0.05, rows=-1, transf=numpy.exp, efac=2, constarea=True, ax=ax)
    ref = plotref.PLOTS['addpoly_default']
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    plotref.check_all(ax, ref['rec'], 'addpoly_default', skip={'mtext'})
    refutil.close(r.xlim, ref['ret']['xlim'])


def test_addpoly_rma_matches_metafor(dat: pandas.DataFrame, fits: dict[str, metafopy.RMAResult], ax: matplotlib.axes.Axes) -> None:
    metafopy.plot.forest(fits['re'], ylim=(-5, 17), addfit=False, ax=ax)
    metafopy.plot.addpoly(fits['re'], row=-1.5, addpred=True, mlab='RE model', atransf=numpy.exp, ax=ax)
    metafopy.plot.addpoly(fits['re'], row=-3, predstyle='bar', level=90, annotate=True, digits=3, col='gray', ax=ax)
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    plotref.check_all(ax, plotref.PLOTS['addpoly_rma']['rec'], 'addpoly_rma', skip={'mtext'})


def test_forest_default_fonts_match_metafor(dat: pandas.DataFrame, ax: matplotlib.axes.Axes) -> None:
    yi, vi, slab = yi_vi(dat)
    r = metafopy.plot.forest(yi, vi, slab=slab, fonts={'mono': 1, 'serif': 3}, ilab=dat['year'].to_numpy(), ilab_xpos=[-7], xlim=(-10, 6), measure='RR', ax=ax)
    check_forest(r, plotref.PLOTS['forest_default_fonts'], 'forest_default_fonts', areas=False)


def test_addpoly_rma_pred_styles_match_metafor(ax: matplotlib.axes.Axes, fits: dict[str, metafopy.RMAResult]) -> None:
    # auditme: accept[literal-duplication] fixture inputs mirror the R calls that generate the reference
    metafopy.plot.forest(fits['re'], ylim=(-8, 17), addfit=False, fonts=('serif', 'mono'), ax=ax)
    metafopy.plot.addpoly(fits['re'], row=-1.5, predstyle='shade', ax=ax)
    metafopy.plot.addpoly(fits['re'], row=-4, predstyle='dist', level=90, predtail='>-0.3', atransf=numpy.exp, ax=ax)
    metafopy.plot.addpoly(fits['re'], row=-6.5, predstyle='polygon', fonts=('sans', 'mono'), ax=ax)
    rec = plotref.PLOTS['addpoly_rma_pred']['rec']
    # auditme: accept[literal-duplication] fixture inputs mirror the R calls that generate the reference
    plotref.check_all(ax, rec, 'addpoly_rma_pred', skip={'mtext'}, compare_col=True)


def test_addpoly_rma_pred_styles_transformed_match_metafor(ax: matplotlib.axes.Axes, fits: dict[str, metafopy.RMAResult]) -> None:
    metafopy.plot.forest(fits['re'], ylim=(-8, 17), addfit=False, transf=numpy.exp, alim=(0, 5), ax=ax)
    metafopy.plot.addpoly(fits['re'], row=-1.5, predstyle='dist', transf=numpy.exp, efac=(1, 1.5), ax=ax)
    metafopy.plot.addpoly(fits['knha'], row=-4.5, predstyle='shade', transf=numpy.exp, ax=ax)
    # auditme: accept[literal-duplication] fixture inputs mirror the R calls that generate the reference
    plotref.check_all(ax, plotref.PLOTS['addpoly_rma_pred_exp']['rec'], 'addpoly_rma_pred_exp', skip={'mtext'}, compare_col=True)


def test_addpoly_default_pred_styles_match_metafor(dat: pandas.DataFrame, ax: matplotlib.axes.Axes) -> None:
    yi, vi, slab = yi_vi(dat)
    metafopy.plot.forest(yi, vi, slab=slab, ylim=(-8, 17), xlim=(-8, 6), measure='RR', ax=ax)
    metafopy.plot.addpoly(
        0.3, ci_lb=-0.2, ci_ub=0.9, pi_lb=-0.8, pi_ub=1.4, pi_level=90, pi_dist='t', pi_ddf=6, pi_se=0.55, rows=-2, predstyle='dist', mlab='C', ax=ax
    )
    metafopy.plot.addpoly(
        -0.3, ci_lb=-0.7, ci_ub=0.1, pi_lb=-1.2, pi_ub=0.6, pi_se=0.5, rows=-5, predstyle='shade', annotate=False, fonts={'mono': 2, 'serif': 3}, ax=ax
    )
    metafopy.plot.addpoly(
        -0.3,
        ci_lb=-0.7,
        ci_ub=0.1,
        pi_lb=-1.2,
        pi_ub=0.6,
        rows=-7,
        predstyle='dist',
        preddist=(PDXS, scipy.stats.norm.pdf(PDXS, -0.3, 0.5)),
        predlim=(-2, 1),
        ax=ax,
    )
    # auditme: accept[literal-duplication] fixture inputs mirror the R calls that generate the reference
    plotref.check_all(ax, plotref.PLOTS['addpoly_default_pred']['rec'], 'addpoly_default_pred', skip={'mtext'}, compare_col=True)


def test_forest_missing_values_in_fit(dat: pandas.DataFrame, ax: matplotlib.axes.Axes) -> None:
    dna = refutil.bcg_na()
    res = metafopy.rma('yi', 'vi', mods='~ ablat', data=dna, slab='label')
    r = metafopy.plot.forest(res, ax=ax)
    assert len(r.rows) == 11
    assert plotref.flat(list(r.rows)) == sorted(plotref.flat(list(r.rows)))


def test_forest_rejects_unsupported_options(fits: dict[str, metafopy.RMAResult], ax: matplotlib.axes.Axes) -> None:
    with pytest.raises(ValueError, match='possible with this'):
        metafopy.plot.forest(fits['re'], predstyle='dist', transf=lambda v: v + 1, ax=ax)
    with pytest.raises(ValueError, match='one of'):
        metafopy.plot.forest(fits['re'], predstyle='cloud', ax=ax)
    with pytest.raises(ValueError, match='preddist'):
        metafopy.plot.forest(fits['re'], predstyle='dist', preddist=(PDXS,), ax=ax)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match='fonts'):
        metafopy.plot.forest(fits['re'], fonts=('serif', 7), ax=ax)
    with pytest.raises(ValueError, match='not both'):
        metafopy.plot.forest(fits['re'], transf=numpy.exp, atransf=numpy.exp, ax=ax)


# -- funnel ---------------------------------------------------------------------------------------------


@pytest.fixture(scope='module')
def ffits(dat: pandas.DataFrame) -> dict[str, metafopy.RMAResult]:
    return {
        'ni': case_fit(dat, slab='label', ni='n'),
        'mods': case_fit(dat, mods='~ ablat', slab='label', ni='n'),
        'ee': case_fit(dat, method='EE', slab='label'),
        'tf': metafopy.bias.trimfill(case_fit(dat, slab='label', ni='n')),
    }


def check_funnel(r: typing.Any, ref: dict[str, typing.Any], what: str, compare_col: bool = False) -> None:
    ret = ref['ret']
    refutil.close(r.x, ret['x'], f'{what}: x')
    refutil.close(r.y, ret['y'], f'{what}: y')
    assert list(r.slab) == [str(s) for s in plotref.flat(ret['slab'])], what
    if 'fill' in ret:
        assert list(r.fill) == plotref.flat(ret['fill'])
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    plotref.check_all(r.ax, ref['rec'], what, skip={'mtext'}, compare_col=True)


@pytest.mark.parametrize('yaxis', ['sei', 'vi', 'seinv', 'vinv', 'ni', 'ninv', 'sqrtni', 'sqrtninv', 'lni', 'wi'])
def test_funnel_yaxis_matches_metafor(yaxis: str, ffits: dict[str, metafopy.RMAResult], ax: matplotlib.axes.Axes) -> None:
    r = metafopy.plot.funnel(ffits['ni'], yaxis=yaxis, ax=ax)
    check_funnel(r, plotref.PLOTS[f'funnel_{yaxis}'], yaxis)


# auditme: accept[non-alphabetical-dict-keys] order follows the R script that generates the fixtures
FUNNEL_CASES: dict[str, tuple[str, dict[str, typing.Any]]] = {
    'funnel_levels': (
        'ni',
        {
            'level': [90, 95, 99],
            # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
            'shade': ['white', 'gray55', 'gray75'],
            'refline': 0,
            'back': 'gray90',
            'hlines': 'white',
            'steps': 7,
            'atransf': numpy.exp,
            'at': numpy.log([0.25, 0.5, 1, 2]),
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
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        {'yaxis': 'seinv', 'pch_fill': 17, 'col': ['black', 'red'], 'bg': ['gray', 'yellow'], 'level': [90, 95, 99], 'label': 'all'},
    ),
    'funnel_tf_ni': ('tf', {'yaxis': 'ni'}),
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    'funnel_pchvec': ('ni', {'pch': [1, 19] * 6 + [1], 'col': ['black', 'blue'] * 6 + ['black'], 'refline': 0.1}),
}


@pytest.mark.parametrize('key', list(FUNNEL_CASES))
def test_funnel_rma_matches_metafor(key: str, ffits: dict[str, metafopy.RMAResult], ax: matplotlib.axes.Axes) -> None:
    fit, kw = FUNNEL_CASES[key]
    r = metafopy.plot.funnel(ffits[fit], ax=ax, **kw)
    check_funnel(r, plotref.PLOTS[key], key)


def test_funnel_default_matches_metafor(dat: pandas.DataFrame, ax: matplotlib.axes.Axes) -> None:
    yi, vi, _ = yi_vi(dat)
    check_funnel(metafopy.plot.funnel(yi, vi, ax=ax), plotref.PLOTS['funnel_default'], 'funnel_default')


def test_funnel_default_options_match_metafor(dat: pandas.DataFrame, ax: matplotlib.axes.Axes) -> None:
    yi, vi, slab = yi_vi(dat)
    sub = numpy.ones(13, dtype=bool)
    sub[0] = False
    r = metafopy.plot.funnel(yi, sei=numpy.sqrt(vi), yaxis='seinv', level=[90, 99], refline=-0.7, slab=slab, label=4, subset=sub, ax=ax)
    check_funnel(r, plotref.PLOTS['funnel_default_opts'], 'funnel_default_opts')


def test_funnel_default_ni_and_wi_match_metafor(dat: pandas.DataFrame, ax: matplotlib.axes.Axes) -> None:
    yi, vi, _ = yi_vi(dat)
    check_funnel(metafopy.plot.funnel(yi, ni=dat['n'].to_numpy(), yaxis='sqrtni', xlab='A', ax=ax), plotref.PLOTS['funnel_default_ni'], 'funnel_default_ni')
    ax.clear()
    check_funnel(metafopy.plot.funnel(yi, vi, yaxis='wi', ylim=(1, 20), ax=ax), plotref.PLOTS['funnel_default_wi'], 'funnel_default_wi')


def test_funnel_default_missing_matches_metafor(dat: pandas.DataFrame, ax: matplotlib.axes.Axes) -> None:
    yi, vi, _ = yi_vi(dat)
    yi_na = yi.copy()
    yi_na[[2, 6]] = numpy.nan
    check_funnel(metafopy.plot.funnel(yi_na, vi, ax=ax), plotref.PLOTS['funnel_default_na'], 'funnel_default_na')


# -- baujat ---------------------------------------------------------------------------------------------


def check_xy(r: typing.Any, ref: dict[str, typing.Any], what: str) -> None:
    refutil.close(r.x, ref['ret']['x'], f'{what}: x')
    refutil.close(r.y, ref['ret']['y'], f'{what}: y')
    assert list(r.ids) == [int(v) for v in plotref.flat(ref['ret']['ids'])], what
    assert list(r.slab) == [str(v) for v in plotref.flat(ref['ret']['slab'])], what


def test_baujat_matches_metafor(fits: dict[str, metafopy.RMAResult], ax: matplotlib.axes.Axes) -> None:
    r = metafopy.plot.baujat(fits['re'], ax=ax)
    check_xy(r, plotref.PLOTS['baujat_re'], 'baujat_re')
    plotref.check_plain(ax, plotref.PLOTS['baujat_re']['rec'], 'baujat_re')
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    plotref.check_all(ax, plotref.PLOTS['baujat_re']['rec'], 'baujat_re', skip={'axis', 'usr', 'mtext'})


def test_baujat_options_match_metafor(fits: dict[str, metafopy.RMAResult], ax: matplotlib.axes.Axes) -> None:
    r = metafopy.plot.baujat(fits['ee'], symbol='slab', cex=0.6, grid='red', xlab='X', ylab='Y', ax=ax)
    ref = plotref.PLOTS['baujat_ee']
    check_xy(r, ref, 'baujat_ee')
    plotref.check_plain(ax, ref['rec'], 'baujat_ee')
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    plotref.check_all(ax, ref['rec'], 'baujat_ee', skip={'axis', 'usr', 'mtext'})


def test_baujat_moderators_and_symbols_match_metafor(fits: dict[str, metafopy.RMAResult], ax: matplotlib.axes.Axes) -> None:
    r = metafopy.plot.baujat(fits['mods'], symbol=16, grid=False, xlim=(0, 12), ylim=(0, 3), ax=ax)
    ref = plotref.PLOTS['baujat_mods']
    check_xy(r, ref, 'baujat_mods')
    plotref.check_plain(ax, ref['rec'], 'baujat_mods')
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    plotref.check_all(ax, ref['rec'], 'baujat_mods', skip={'axis', 'usr', 'mtext'})
    ax.clear()
    r = metafopy.plot.baujat(fits['mods'], symbol=[1, 2, 19] * 4 + [1], cex=1.2, ax=ax)
    ref = plotref.PLOTS['baujat_symvec']
    check_xy(r, ref, 'baujat_symvec')
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    plotref.check_all(ax, ref['rec'], 'baujat_symvec', skip={'axis', 'usr', 'mtext'})


# -- radial ---------------------------------------------------------------------------------------------

# auditme: accept[non-alphabetical-dict-keys] order follows the R script that generates the fixtures
RADIAL_CASES: dict[str, tuple[str, dict[str, typing.Any]]] = {
    'radial_re': ('re', {}),
    'radial_ee_center': ('ee', {'center': True, 'steps': 5, 'digits': 3}),
    'radial_opts': (
        're',
        {
            'xlim': (0, 12),
            'zlim': (-6, 6),
            'atz': [-4, 0, 4],
            'aty': [-1.5, -1, 0, 0.5],
            'transf': numpy.exp,
            'level': 90,
            'cex': 1.2,
            'pch': 19,
            'arc_res': (60, 30),
        },
    ),
    'radial_labels': ('re', {'xlab': 'Precision', 'zlab': 'Z', 'back': 'gray90', 'col': 'blue', 'bg': 'red', 'cex_axis': 0.8, 'cex_lab': 1.1}),
}


@pytest.mark.parametrize('key', list(RADIAL_CASES))
def test_radial_matches_metafor(key: str, fits: dict[str, metafopy.RMAResult], ax: matplotlib.axes.Axes) -> None:
    fit, kw = RADIAL_CASES[key]
    r = metafopy.plot.radial(fits[fit], ax=ax, **kw)
    ref = plotref.PLOTS[key]
    check_xy(r, ref, key)
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    plotref.check_all(ax, ref['rec'], key, skip={'mtext', 'axis', 'usr'}, compare_col=True)
    refutil.close(ax.get_xlim(), ref['rec']['usr'][:2])
    refutil.close(ax.get_ylim(), ref['rec']['usr'][2:])
    refutil.close(ax.get_xticks(), ref['rec']['axticks']['x'])
    last = [a for a in ref['rec']['axis'] if a['side'] == 2][-1]
    refutil.close(ax.get_yticks(), last['at'])


# -- qqnorm ---------------------------------------------------------------------------------------------

# auditme: accept[non-alphabetical-dict-keys] order follows the R script that generates the fixtures
QQ_CASES: dict[str, tuple[str, dict[str, typing.Any]]] = {
    'qqnorm_re': ('re', {'seed': 1234}),
    'qqnorm_stud': ('mods', {'type': 'rstudent', 'seed': 42, 'label': True, 'grid': True, 'level': 90, 'reps': 300, 'pos': 24}),
    'qqnorm_bonf': (
        'ee',
        {'seed': 7, 'bonferroni': True, 'smooth': False, 'label': 'out', 'envelope': 'gray85', 'lty': ('dashed', 'solid'), 'pch': 19, 'col': 'blue'},
    ),
    'qqnorm_label': ('re', {'seed': 99, 'label': 3, 'offset': 0.5, 'pos': 13, 'bass': 2, 'reps': 200}),
    'qqnorm_noenv': ('mods', {'envelope': False, 'label': 2, 'pos': 4}),
}


@pytest.mark.parametrize('key', list(QQ_CASES))
def test_qqnorm_matches_metafor(key: str, fits: dict[str, metafopy.RMAResult], ax: matplotlib.axes.Axes) -> None:
    fit, kw = QQ_CASES[key]
    r = metafopy.plot.qqnorm(fits[fit], ax=ax, **kw)
    ref = plotref.PLOTS[key]
    refutil.close(r.x, ref['ret']['x'], f'{key}: x')
    refutil.close(r.y, ref['ret']['y'], f'{key}: y')
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    plotref.check_all(ax, ref['rec'], key, skip={'mtext', 'axis', 'usr'}, compare_col=True)
    plotref.check_plain(ax, ref['rec'], key)


# -- labbe ----------------------------------------------------------------------------------------------


@pytest.fixture(scope='module')
def raw(dat: pandas.DataFrame) -> pandas.DataFrame:
    out = dat.copy()
    out['x1i'], out['x2i'] = out['tpos'], out['cpos']
    out['t1i'], out['t2i'] = (out['tpos'] + out['tneg']) / 100, (out['cpos'] + out['cneg']) / 100
    return out


def arm_fit(raw: pandas.DataFrame, measure: str) -> metafopy.RMAResult:
    if measure in ('IRR', 'IRD', 'IRSD'):
        return metafopy.rma(x1i='x1i', t1i='t1i', x2i='x2i', t2i='t2i', data=raw, measure=measure, slab='label')
    return metafopy.rma(ai='tpos', bi='tneg', ci='cpos', di='cneg', data=raw, measure=measure, slab='label')


# auditme: accept[non-alphabetical-dict-keys] order follows the R script that generates the fixtures
COUNTS: dict[str, typing.Any] = {'ai': 'tpos', 'bi': 'tneg', 'ci_counts': 'cpos', 'di': 'cneg', 'x1i': 'x1i', 'x2i': 'x2i', 't1i': 't1i', 't2i': 't2i'}

# auditme: accept[non-alphabetical-dict-keys] order follows the R script that generates the fixtures
LABBE_CASES: dict[str, tuple[str, dict[str, typing.Any]]] = {
    'labbe_rr': ('RR', {}),
    'labbe_or_ci': ('OR', {'ci': True, 'pi': True, 'grid': True, 'transf': numpy.exp, 'flip': True, 'pch': 19, 'col': 'blue', 'lty': ('dotted', 'solid')}),
    'labbe_rd': ('RD', {'plim': (numpy.nan, 4, 1), 'ci': 'gray80', 'lim': (0, 0.2), 'xlab': 'A', 'ylab': 'B', 'add': 0.25}),
    'labbe_as': ('AS', {'psize': numpy.arange(1, 14) / 4, 'pi': True, 'to': 'all', 'legend': True}),
    'labbe_irr': ('IRR', {'xlim': (-8, -2), 'ylim': (-8, -2), 'grid': 'red'}),
    'labbe_ird': ('IRD', {'ci': True, 'bg': 'yellow', 'plim': (1, numpy.nan, 2)}),
    'labbe_irsd': ('IRSD', {'flip': True, 'llim': (-0.05, 0.1), 'transf': lambda v: v * 100}),
}


@pytest.mark.parametrize('key', list(LABBE_CASES))
def test_labbe_matches_metafor(key: str, raw: pandas.DataFrame, ax: matplotlib.axes.Axes) -> None:
    measure, kw = LABBE_CASES[key]
    r = metafopy.plot.labbe(arm_fit(raw, measure), ax=ax, **COUNTS, **kw)
    ref = plotref.PLOTS[key]
    ret = ref['ret']
    refutil.close(r.x, ret['x'], f'{key}: x')
    refutil.close(r.y, ret['y'], f'{key}: y')
    refutil.close(r.cex, ret['cex'], f'{key}: cex')
    refutil.close(r.pch, ret['pch'], f'{key}: pch')
    assert list(r.col) == [str(v) for v in plotref.flat(ret['col'])], key
    assert list(r.ids) == [int(v) for v in plotref.flat(ret['ids'])], key
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    plotref.check_all(ax, ref['rec'], key, skip={'mtext', 'axis', 'usr'}, compare_col=True)
    plotref.check_plain(ax, ref['rec'], key)


# -- plot_profile ---------------------------------------------------------------------------------------


def test_plot_profile_matches_metafor(fits: dict[str, metafopy.RMAResult], ax: matplotlib.axes.Axes) -> None:
    pf = metafopy.profile(fits['re'], steps=15, xlim=(0.01, 0.9))
    assert metafopy.plot.plot_profile(pf, cline=True, ax=ax) is ax
    ref = plotref.PLOTS['profile_re']['rec']
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    plotref.check_all(ax, ref, 'profile_re', skip={'mtext', 'axis', 'usr'})
    plotref.check_plain(ax, ref, 'profile_re')


def test_plot_profile_options_match_metafor(fits: dict[str, metafopy.RMAResult], ax: matplotlib.axes.Axes) -> None:
    pf = metafopy.profile(fits['re'], steps=9, xlim=(0.4, 0.9), sqrt=True)
    metafopy.plot.plot_profile(pf, refline=False, pch=17, ylim=(-14, -12), cline=0.1, ax=ax)
    ref = plotref.PLOTS['profile_sqrt']['rec']
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    plotref.check_all(ax, ref, 'profile_sqrt', skip={'mtext', 'axis', 'usr'})
    plotref.check_plain(ax, ref, 'profile_sqrt')
    ax.clear()
    pf = metafopy.profile(fits['mods'], steps=8, xlim=(0.01, 0.5), exp=True)
    metafopy.plot.plot_profile(pf, xlab='Tau squared', ylab='L', main='Title', xlim=(0, 0.6), cline=True, ax=ax)
    ref = plotref.PLOTS['profile_exp']['rec']
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    plotref.check_all(ax, ref, 'profile_exp', skip={'mtext', 'axis', 'usr'})
    plotref.check_plain(ax, ref, 'profile_exp')


def test_profile_plot_argument_draws(fits: dict[str, metafopy.RMAResult], ax: matplotlib.axes.Axes) -> None:
    metafopy.profile(fits['re'], steps=5, xlim=(0.05, 0.6), plot=True, ax=ax)
    assert ax.get_ylabel() == 'Restricted Log-Likelihood'
    assert len(ax.get_title()) > 0


# -- plot_infl ------------------------------------------------------------------------------------------


def check_infl(r: typing.Any, ref: dict[str, typing.Any], what: str, dfbs: bool = False) -> None:
    axes = list(r.axes) + list(r.dfbs_axes)
    rec = ref['rec']
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    plotref.check_all(axes, rec, what, skip={'axis', 'usr', 'mtext'}, compare_col=False)
    # the labels asked for, whether or not overlapping ones are hidden when drawn
    tick_labels = [[tick.label1.get_text() for tick in a.xaxis.get_major_ticks()] for a in axes if a.get_xticks().size]
    exp = [[str(v) for v in plotref.flat(a['labels'])] for a in rec['axis'] if a['side'] == 1]
    assert tick_labels == exp, f'{what}: x tick labels'
    plots = rec['plot']
    assert len(plots) == len(axes), f'{what}: number of panels'
    last = axes[-1]
    refutil.close(last.get_xlim(), rec['usr'][:2], f'{what}: last xlim')
    refutil.close(last.get_ylim(), rec['usr'][2:], f'{what}: last ylim')


def test_plot_infl_all_matches_metafor(fits: dict[str, metafopy.RMAResult], ax: matplotlib.axes.Axes) -> None:
    r = metafopy.plot.plot_infl(metafopy.influence(fits['re']), ax=ax)
    assert len(r.axes) == 8
    check_infl(r, plotref.PLOTS['infl_all'], 'infl_all')


def test_plot_infl_options_match_metafor(fits: dict[str, metafopy.RMAResult], ax: matplotlib.axes.Axes) -> None:
    r = metafopy.plot.plot_infl(metafopy.influence(fits['re']), plotinf=[1, 3, 4, 7], logcov=False, las=2, bg='blue', bg_infl='green', slab_style=2, ax=ax)
    check_infl(r, plotref.PLOTS['infl_sub'], 'infl_sub')


def test_plot_infl_dfbs_matches_metafor(fits: dict[str, metafopy.RMAResult], ax: matplotlib.axes.Axes) -> None:
    inf = metafopy.influence(fits['mods'])
    r = metafopy.plot.plot_infl(inf, plotinf=False, plotdfbs=True, ax=ax)
    assert len(r.axes) == 0
    assert len(r.dfbs_axes) == 2
    check_infl(r, plotref.PLOTS['infl_dfbs'], 'infl_dfbs')
    _, axes = matplotlib.pyplot.subplots()
    r = metafopy.plot.plot_infl(inf, plotinf=[2, 5], plotdfbs=2, pch=19, ax=axes)
    check_infl(r, plotref.PLOTS['infl_dfbs_only2'], 'infl_dfbs_only2')
    matplotlib.pyplot.close('all')


def test_plot_infl_missing_values_match_metafor(dat: pandas.DataFrame, ax: matplotlib.axes.Axes) -> None:
    dna = refutil.bcg_na()
    res = metafopy.rma('yi', 'vi', mods='~ ablat', data=dna, slab='label')
    ids = numpy.flatnonzero(res.data.keep) + 1
    r = metafopy.plot.plot_infl(metafopy.influence(res), plotinf=[1, 2, 6], ids=ids, ax=ax)
    check_infl(r, plotref.PLOTS['infl_na_omit'], 'infl_na_omit')
    _, axes = matplotlib.pyplot.subplots()
    r = metafopy.plot.plot_infl(metafopy.influence(res, na_action='exclude'), plotinf=[1, 2, 6, 8], na_action='exclude', ax=axes)
    check_infl(r, plotref.PLOTS['infl_na_exclude'], 'infl_na_exclude')
    matplotlib.pyplot.close('all')


def test_plot_infl_mfrow(fits: dict[str, metafopy.RMAResult], ax: matplotlib.axes.Axes) -> None:
    r = metafopy.plot.plot_infl(metafopy.influence(fits['re']), plotinf=[1, 2, 3], mfrow=(3, 1), ax=ax)
    assert len({round(a.get_position().x0, 6) for a in r.axes}) == 1
    assert len({round(a.get_position().y0, 6) for a in r.axes}) == 3
    with pytest.raises(metafopy.errors.InputError, match='too few panels'):
        metafopy.plot.plot_infl(metafopy.influence(fits['re']), mfrow=(1, 2))


def test_plot_infl_text_size_follows_the_layout(fits: dict[str, metafopy.RMAResult], ax: matplotlib.axes.Axes) -> None:
    base = float(matplotlib.rcParams['font.size'])
    r = metafopy.plot.plot_infl(metafopy.influence(fits['re']), ax=ax)
    assert r.axes[0].title.get_fontsize() == pytest.approx(base * 0.66 * 1.2)
    _, single = matplotlib.pyplot.subplots()
    r = metafopy.plot.plot_infl(metafopy.influence(fits['re']), plotinf=[1], ax=single)
    assert r.axes[0].title.get_fontsize() == pytest.approx(base * 1.2)
    matplotlib.pyplot.close('all')


def test_plot_infl_one_column_layout_matches_the_gap_of_r(fits: dict[str, metafopy.RMAResult]) -> None:
    fig, area = matplotlib.pyplot.subplots(figsize=(7, 11))
    area.set_position((0.08, 0.04, 0.9, 0.92))
    r = metafopy.plot.plot_infl(metafopy.influence(fits['re']), mfrow=(8, 1), ax=area)
    gaps = [(a.get_position().y0 - b.get_position().y1) * 11 for a, b in zip(r.axes[:-1], r.axes[1:], strict=True)]
    assert gaps == pytest.approx([1.06 * 0.66] * 7)
    shown = [len([t for t in a.get_yticklabels() if t.get_visible()]) for a in r.axes]
    assert all(2 <= n <= 6 for n in shown)
    matplotlib.pyplot.close(fig)


def test_ax_ticks_zero_is_exact() -> None:
    at = metafopy.plot.rutils.ax_ticks(-0.3, 0.5)
    assert 0.0 in at
    assert metafopy.plot.rutils.rformat(at) == ['-0.2', '0.0', '0.2', '0.4']


def test_overlapping_x_labels_are_hidden(ax: matplotlib.axes.Axes) -> None:
    n = 160
    pos = numpy.arange(1, n + 1, dtype=float)
    metafopy.plot.plain.plain_plot(ax, (1.0, float(n)), (0.0, 1.0), xat=pos, xlabels=[str(i) for i in range(1, n + 1)])
    shown = [t.get_text() for t in ax.get_xticklabels()]
    assert shown[0] == '1'
    assert 10 < len(shown) < n / 3
    assert len(ax.get_xticks()) == n  # the ticks stay
    _, few = matplotlib.pyplot.subplots()
    metafopy.plot.plain.plain_plot(few, (1.0, 13.0), (0.0, 1.0), xat=pos[:13], xlabels=[str(i) for i in range(1, 14)])
    assert len(few.get_xticklabels()) == 13
    matplotlib.pyplot.close('all')


def test_addpoly_rma_mv_equals_the_numbers(ax: matplotlib.axes.Axes) -> None:
    dat = metafopy.datasets.load('dat.assink2016')
    v = metafopy.vcalc(vi=dat['vi'], cluster=dat['study'], obs=dat['esid'], rho=0.6)
    res = metafopy.rma_mv(yi='yi', V=v, random='~ 1 | study/esid', data=dat)
    pr = metafopy.predict(res)
    _, other = matplotlib.pyplot.subplots()
    for axes in (ax, other):
        metafopy.plot.forest(dat['yi'], dat['vi'], ax=axes, xlim=(-4, 4))
    metafopy.plot.addpoly(res, ax=ax, row=-1.5, predstyle='bar', mlab='Pooled')
    metafopy.plot.addpoly(pr.pred, ci_lb=pr.ci_lb, ci_ub=pr.ci_ub, pi_lb=pr.pi_lb, pi_ub=pr.pi_ub, ax=other, rows=-1.5, predstyle='bar', mlab='Pooled')
    a, b = (metafopy.plot.draw.collect(axes) for axes in (ax, other))
    assert len(a['rect']) == len(b['rect'])
    for ra, rb in zip(a['rect'][-2:], b['rect'][-2:], strict=True):
        numpy.testing.assert_allclose(
            [ra.get_x(), ra.get_width(), ra.get_y() + ra.get_height() / 2], [rb.get_x(), rb.get_width(), rb.get_y() + rb.get_height() / 2]
        )
    assert [t.get_text() for t in a['text']] == [t.get_text() for t in b['text']]
    matplotlib.pyplot.close('all')


# -- plot_gosh ------------------------------------------------------------------------------------------


@pytest.fixture(scope='module')
def gosh_fits(dat: pandas.DataFrame) -> dict[str, typing.Any]:
    d8 = dat.iloc[:8].reset_index(drop=True)
    return {'int': metafopy.gosh(case_fit(d8, slab='label')), 'mods': metafopy.gosh(case_fit(d8, mods='~ ablat', slab='label'))}


def test_gosh_results_match_metafor(gosh_fits: dict[str, typing.Any]) -> None:
    info = plotref.PLOTS['gosh_info']
    assert len(gosh_fits['int'].res) == info['k']
    assert len(gosh_fits['mods'].res) == info['km']
    assert list(gosh_fits['mods'].res.columns) == info['cols']
    refutil.close(gosh_fits['int'].res['estimate'], info['est'])
    refutil.close(gosh_fits['int'].res['I2'], info['I2'])


def scatter_summary(art: typing.Any, ref: dict[str, typing.Any], what: str) -> None:
    p = ref
    m = art.meta
    for k, key in (('x', 'xdat'), ('y', 'ydat')):
        v = numpy.asarray(m[k], dtype=float)
        e = p[key]
        assert len(v) == e['n'], what
        refutil.close(float(numpy.nansum(v)), e['sum'], f'{what}: sum {k}', rtol=1e-7)
        refutil.close(v[:3], e['head'], f'{what}: head {k}')
        refutil.close(v[-3:], e['tail'], f'{what}: tail {k}')
    assert m['pch'] == p['pch'], what
    refutil.close(m['cex'], p['cex'], f'{what}: cex')
    assert set(m['cols']) == set(plotref.flat(p['cols'])), f'{what}: colours {set(m["cols"])}'


def check_gosh_int(r: typing.Any, key: str) -> None:
    rec = plotref.PLOTS[key]['rec']
    plots = rec['plot']
    main = r.main
    scat = next(p for p in plots if p.get('xdat') is not None)
    arts = metafopy.plot.draw.collect(main)['scatter']
    assert len(arts) == 1
    scatter_summary(arts[0], scat, key)
    margins = [a for a in (r.xhist, r.yhist) if a is not None]
    plotref.check_rects(margins, rec, key)
    plotref.check_lines(margins, rec, key)
    assert len(margins) + 1 == len(plots), key
    # user coordinates of the panels, in the order they were plotted
    axes = [a for a in (r.xhist, r.main, r.yhist) if a is not None]
    for a, p in zip(axes, plots, strict=True):
        if p.get('xlim') is not None and a is not r.yhist:
            refutil.close(a.get_xlim(), list(metafopy.plot.rutils.usr_range(*plotref.flat(p['xlim']))), f'{key}: xlim')
        if p.get('ylim') is not None and a is not r.xhist:
            refutil.close(a.get_ylim(), list(metafopy.plot.rutils.usr_range(*plotref.flat(p['ylim']))), f'{key}: ylim')
    if r.xhist is not None:
        refutil.close(r.xhist.get_ylim(), list(metafopy.plot.rutils.usr_range(*plotref.flat(plots[0]['ylim']))), f'{key}: xhist ylim')
    if r.yhist is not None:
        refutil.close(r.yhist.get_xlim(), list(metafopy.plot.rutils.usr_range(*plotref.flat(plots[-1]['xlim']))), f'{key}: yhist xlim')


# auditme: accept[non-alphabetical-dict-keys] order follows the R script that generates the fixtures
GOSH_INT: dict[str, dict[str, typing.Any]] = {
    'gosh_default': {},
    'gosh_opts': {
        'het': 'tau2',
        'out': 2,
        'hh': (0.2, 0.4),
        'breaks': (6, 10),
        'adjust': (1, 2),
        'lwd': (1, 0),
        'alpha': (0.3, 0.5, 0.8),
        'cex': 1.5,
        'pch': 1,
        'border': ('gray', 'black'),
        'labels': ('Est', 'T2'),
    },
    'gosh_nox': {'het': 'QE', 'xhist': False, 'col': 'blue', 'xlim': (-1.5, 0.5), 'trim': 0.05},
    'gosh_noy': {'het': 'H2', 'yhist': False, 'breaks': 'Scott'},
    'gosh_nohist': {'xhist': False, 'yhist': False, 'ylim': (0, 100)},
    'gosh_fd': {'het': 'tau', 'breaks': ('FD', 12), 'out': 1, 'col': ('red', 'green')},
}


@pytest.mark.parametrize('key', list(GOSH_INT))
def test_plot_gosh_matches_metafor(key: str, gosh_fits: dict[str, typing.Any], ax: matplotlib.axes.Axes) -> None:
    r = metafopy.plot.plot_gosh(gosh_fits['int'], ax=ax, **GOSH_INT[key])
    check_gosh_int(r, key)


def check_gosh_pairs(r: typing.Any, key: str) -> None:
    rec = plotref.PLOTS[key]['rec']
    pr = rec['pairs'][0]
    nc = int(pr['nc'])
    panels = r.panels
    assert len(panels) == nc
    for j in range(nc):
        i = 1 if j == 0 else 0
        m = metafopy.plot.draw.collect(panels[i][j])['scatter'][0].meta
        assert len(m['x']) == pr['nr']
        refutil.close(float(numpy.nansum(m['x'])), plotref.flat(pr['colsum'])[j], f'{key}: column {j} sum', rtol=1e-7)
        assert m['pch'] == pr['pch']
        refutil.close(m['cex'], pr['cex'])
        assert set(m['cols']) == set(plotref.flat(pr['cols'])), key
    diag = [panels[i][i] for i in range(nc)]
    plotref.check_rects(diag, rec, key)
    plotref.check_lines(diag, rec, key)
    labs = [t.get_text() for a in diag for t in a.texts if t.get_gid() == 'mtext']
    assert len(labs) == nc


def test_plot_gosh_pairs_matches_metafor(gosh_fits: dict[str, typing.Any], ax: matplotlib.axes.Axes) -> None:
    r = metafopy.plot.plot_gosh(gosh_fits['mods'], ax=ax)
    check_gosh_pairs(r, 'gosh_pairs')


def test_plot_gosh_pairs_options_match_metafor(gosh_fits: dict[str, typing.Any], ax: matplotlib.axes.Axes) -> None:
    r = metafopy.plot.plot_gosh(
        gosh_fits['mods'],
        het='H2',
        out=1,
        alpha=0.4,
        breaks=7,
        adjust=2,
        lwd=1,
        trim=[0.02, [0.1, 0.05], 0],
        border='red',
        hh=0.5,
        labels=['H', 'a', 'b'],
        ax=ax,
    )
    check_gosh_pairs(r, 'gosh_pairs_opts')


# -- defaults and errors --------------------------------------------------------------------------------


def test_plots_draw_on_the_current_axes_by_default(fits: dict[str, metafopy.RMAResult]) -> None:
    matplotlib.pyplot.close('all')
    r = metafopy.plot.funnel(fits['ni'] if 'ni' in fits else fits['re'])
    assert r.ax is matplotlib.pyplot.gca()
    matplotlib.pyplot.close('all')
    assert metafopy.plot.baujat(fits['re']).ax is matplotlib.pyplot.gca()
    matplotlib.pyplot.close('all')


def test_plots_reject_invalid_arguments(fits: dict[str, metafopy.RMAResult], raw: pandas.DataFrame, dat: pandas.DataFrame, ax: matplotlib.axes.Axes) -> None:
    with pytest.raises(ValueError, match='yaxis'):
        metafopy.plot.funnel(fits['re'], yaxis='bad', ax=ax)
    with pytest.raises(ValueError, match='without moderators'):
        metafopy.plot.radial(fits['mods'], ax=ax)
    with pytest.raises(ValueError, match='type'):
        metafopy.plot.qqnorm(fits['re'], type='bad', ax=ax)
    with pytest.raises(ValueError, match='Out of range'):
        metafopy.plot.qqnorm(fits['re'], label=99, seed=1, ax=ax)
    with pytest.raises(ValueError, match='without moderators'):
        metafopy.plot.labbe(fits['mods'], ax=ax)
    with pytest.raises(ValueError, match="'measure'"):
        metafopy.plot.labbe(metafopy.rma('yi', 'vi', data=dat), ax=ax)
    with pytest.raises(ValueError, match='not available'):
        metafopy.plot.labbe(arm_fit(raw, 'RR'), ax=ax)
    with pytest.raises(ValueError, match='plotinf'):
        metafopy.plot.plot_infl(metafopy.influence(fits['re']), plotinf=[20], ax=ax)
    with pytest.raises(ValueError, match='At least one'):
        metafopy.plot.plot_infl(metafopy.influence(fits['re']), plotinf=False, ax=ax)


def test_plot_gosh_rejects_tau2_for_equal_effects_models(dat: pandas.DataFrame, ax: matplotlib.axes.Axes) -> None:
    g = metafopy.gosh(case_fit(dat.iloc[:6].reset_index(drop=True), method='EE'))
    with pytest.raises(ValueError, match='tau2'):
        metafopy.plot.plot_gosh(g, het='tau2', ax=ax)
    with pytest.raises(ValueError, match='het'):
        metafopy.plot.plot_gosh(g, het='bad', ax=ax)
