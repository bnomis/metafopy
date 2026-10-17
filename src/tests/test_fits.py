"""The functions metafor writes once for all model classes, run on rma_mh, rma_peto, rma_mv, rma_glmm, location-scale and selection models.

forest(), funnel(), radial(), baujat(), qqnorm(), labbe(), addpoly() of predict() results, plot_rma(), ranktest(), predict(),
tes() and gosh() checked against R metafor (tests/reference/phase7/fits.json, written by tests/r/generate_fits.R), and the
errors raised for the classes metafor has no method for.
"""

import collections.abc
import functools
import json
import pathlib
import re
import typing

import matplotlib.axes
import matplotlib.pyplot
import numpy
import plotref
import pytest
import refutil

import metafopy
import metafopy.errors
import metafopy.methods.fitview
import metafopy.plot
import metafopy.plot.draw
import metafopy.plot.rutils

FIXTURE = pathlib.Path(__file__).parent / 'reference' / 'phase7' / 'fits.json'
# rma_glmm and selmodel fits are found by numerical optimisation, which agrees with R's to about 1e-6
RTOL_OPT = 1e-5


# auditme: accept[paired-sibling-helpers] each test module keeps its own fixture loader
@functools.cache
def ref() -> dict[str, typing.Any]:
    data: dict[str, typing.Any] = json.loads(FIXTURE.read_text(encoding='utf-8'))
    return data


CASE_ARGS = 'key,fit,kw'  # the arguments of the parametrized tests
BINARY = ['mh', 'peto']
# auditme: accept[literal-duplication] the primitives the plot tests of the repository leave out
SKIP_TEXT = {'mtext'}  # the texts are compared one by one
# auditme: accept[literal-duplication] the primitives the plot tests of the repository leave out
SKIP_AXES = {'axis', 'mtext', 'usr'}  # the texts are compared one by one, the axes depend on the device size


# auditme: accept[paired-sibling-helpers] each test module has its own figure fixture
@pytest.fixture
def ax() -> collections.abc.Iterator[matplotlib.axes.Axes]:
    fig, axes = matplotlib.pyplot.subplots(figsize=(7, 7))
    yield axes
    matplotlib.pyplot.close(fig)


TABLES: dict[str, typing.Any] = {'ai': 'tpos', 'bi': 'tneg', 'ci': 'cpos', 'di': 'cneg'}


@pytest.fixture(scope='module')
def fits() -> dict[str, typing.Any]:
    dat = refutil.bcg()
    dat.attrs['measure'] = 'RR'  # as escalc() sets it, so that the models know the outcome measure
    na = dat.copy()
    na.loc[2, 'tpos'] = numpy.nan
    label = list(dat['label'])
    with pytest.warns(UserWarning, match='NAs omitted'):
        mh_na = metafopy.rma_mh(**TABLES, data=na, slab=label)
    return {
        'mh': metafopy.rma_mh(**TABLES, data=dat, slab=label),
        'mh0': metafopy.rma_mh(**TABLES, data=dat),
        'mh_rr': metafopy.rma_mh(**TABLES, data=dat, slab=label, measure='RR'),
        'mh_na': mh_na,
        'peto': metafopy.rma_peto(**TABLES, data=dat, slab=label),
        'mv': metafopy.rma_mv(yi='yi', V='vi', random='~ 1 | trial', data=dat, slab='label'),
        'mvm': metafopy.rma_mv(yi='yi', V='vi', mods='~ ablat', random='~ 1 | trial', data=dat, slab='label'),
        'ls': metafopy.rma('yi', 'vi', scale='~ 1', data=dat, slab='label'),
        'sel': metafopy.selmodel(metafopy.rma('yi', 'vi', data=dat, slab='label'), type='logistic'),
        'glmm': metafopy.rma_glmm(measure='OR', **TABLES, data=dat, slab=label),
        'uni': metafopy.rma('yi', 'vi', data=dat, slab='label'),
        'unim': metafopy.rma('yi', 'vi', mods='~ ablat', data=dat, slab='label'),
        'robust': metafopy.robust(metafopy.rma('yi', 'vi', data=dat), cluster=list(range(13))),
    }


def rtol_for(key: str) -> float:
    """The tolerance of the values of an iteratively fitted model (``tau^2`` found by optimisation, not in closed form)."""
    return RTOL_OPT if any(s in key for s in ('glmm', 'sel', '_mv', '_ls')) else refutil.RTOL


# -- forest ----------------------------------------------------------------------------------------------------


# auditme: accept[non-alphabetical-dict-keys] order follows the R script that generates the fixtures
FOREST_CASES: dict[str, tuple[str, dict[str, typing.Any]]] = {
    'forest_mh': ('mh', {}),
    'forest_mh0': ('mh0', {}),
    'forest_mh_na': ('mh_na', {}),
    'forest_peto': ('peto', {}),
    'forest_mv': ('mv', {}),
    'forest_mvm': ('mvm', {}),
    'forest_sel': ('sel', {}),
    'forest_glmm': ('glmm', {}),
    'forest_mh_opts': ('mh', {'atransf': numpy.exp, 'at': numpy.log([0.05, 0.25, 1, 4]), 'xlim': (-16, 6), 'showweights': True, 'order': 'obs'}),
}


@pytest.mark.parametrize('key', list(FOREST_CASES))
def test_forest_matches_metafor(key: str, fits: dict[str, typing.Any], ax: matplotlib.axes.Axes) -> None:
    fit, kw = FOREST_CASES[key]
    r = metafopy.forest(fits[fit], ax=ax, **kw)
    case = ref()[key]
    ret = case['ret']
    rtol = rtol_for(key)
    for name in ('xlim', 'alim', 'at', 'ylim', 'rows', 'textpos'):
        refutil.close(getattr(r, name), ret[name], f'{key}: {name}', rtol=rtol)
    if rtol == refutil.RTOL:
        plotref.check_all(r.ax, case['rec'], key, skip=SKIP_TEXT)
    else:
        assert [t.get_text() for t in r.ax.texts if t.get_gid() != 'mtext' and t.get_text()] == [
            t['label'] for t in plotref.r_texts(case['rec']) if t['label']
        ], key


# -- funnel ----------------------------------------------------------------------------------------------------

# auditme: accept[non-alphabetical-dict-keys] order follows the R script that generates the fixtures
FUNNEL_CASES: dict[str, tuple[str, dict[str, typing.Any]]] = {
    'funnel_mh': ('mh', {}),
    'funnel_peto': ('peto', {}),
    'funnel_mv': ('mv', {}),
    'funnel_mvm': ('mvm', {}),
    'funnel_ls': ('ls', {}),
    'funnel_sel': ('sel', {}),
    'funnel_glmm': ('glmm', {}),
    'funnel_mh_wi': ('mh', {'yaxis': 'wi'}),
    # auditme: accept[literal-duplication] the shades of metafor's funnel example
    'funnel_mh_ni': ('mh', {'yaxis': 'ni', 'level': [90, 95, 99], 'shade': ['white', 'gray55', 'gray75']}),
}


def has_mv_residuals(fit: metafopy.RMAMVResult) -> bool:
    try:
        metafopy.rstandard(fit)
    except metafopy.errors.ArgumentError:
        return False
    return True


@pytest.mark.parametrize('key', list(FUNNEL_CASES))
def test_funnel_matches_metafor(key: str, fits: dict[str, typing.Any], ax: matplotlib.axes.Axes) -> None:
    fit, kw = FUNNEL_CASES[key]
    if fit == 'mvm' and not has_mv_residuals(fits['mvm']):
        pytest.skip('rstandard() of rma_mv fits is not available yet')
    r = metafopy.funnel(fits[fit], ax=ax, **kw)
    case = ref()[key]
    rtol = rtol_for(key)
    refutil.close(r.x, case['ret']['x'], f'{key}: x', rtol=rtol)
    refutil.close(r.y, case['ret']['y'], f'{key}: y', rtol=rtol)
    assert list(r.slab) == [str(s) for s in plotref.flat(case['ret']['slab'])], key
    if rtol == refutil.RTOL:
        plotref.check_all(r.ax, case['rec'], key, skip=SKIP_TEXT, compare_col=True)


def test_funnel_addtau2_is_ignored_for_rma_mv_and_moderators(fits: dict[str, typing.Any], ax: matplotlib.axes.Axes) -> None:
    with pytest.warns(UserWarning, match="'addtau2' ignored for 'rma.mv' models"):
        metafopy.funnel(fits['mv'], addtau2=True, ax=ax)
    with pytest.warns(UserWarning, match="'addtau2' ignored for models that contain moderators"):
        metafopy.funnel(fits['unim'], addtau2=True, ax=ax)
    with pytest.warns(UserWarning, match="'addtau2' ignored for 'rma.ls' models"):
        metafopy.funnel(fits['ls'], addtau2=True, ax=ax)


# -- radial, baujat, qqnorm, labbe -------------------------------------------------------------------------------------


@pytest.mark.parametrize('fit', ['mh', 'peto', 'glmm'])
def test_radial_matches_metafor(fit: str, fits: dict[str, typing.Any], ax: matplotlib.axes.Axes) -> None:
    key = f'radial_{fit}'
    r = metafopy.radial(fits[fit], ax=ax)
    case = ref()[key]
    rtol = rtol_for(key)
    refutil.close(r.x, case['ret']['x'], f'{key}: x', rtol=rtol)
    refutil.close(r.y, case['ret']['y'], f'{key}: y', rtol=rtol)
    assert list(r.ids) == [int(v) for v in plotref.flat(case['ret']['ids'])]
    assert list(r.slab) == [str(v) for v in plotref.flat(case['ret']['slab'])]
    if rtol == refutil.RTOL:
        plotref.check_all(ax, case['rec'], key, skip=SKIP_AXES, compare_col=True)


@pytest.mark.parametrize(
    CASE_ARGS, [('baujat_mh', 'mh', {}), ('baujat_mh_na', 'mh_na', {}), ('baujat_peto', 'peto', {}), ('baujat_mh_slab', 'mh', {'symbol': 'slab'})]
)
def test_baujat_matches_metafor(key: str, fit: str, kw: dict[str, typing.Any], fits: dict[str, typing.Any], ax: matplotlib.axes.Axes) -> None:
    r = metafopy.baujat(fits[fit], ax=ax, **kw)
    case = ref()[key]
    refutil.close(r.x, case['ret']['x'], f'{key}: x')
    refutil.close(r.y, case['ret']['y'], f'{key}: y')
    assert list(r.ids) == [int(v) for v in plotref.flat(case['ret']['ids'])]
    assert list(r.slab) == [str(v) for v in plotref.flat(case['ret']['slab'])]
    plotref.check_plain(ax, case['rec'], key)
    if fit != 'mh_na':  # R also draws (nowhere) the label of the table with a missing count
        plotref.check_all(ax, case['rec'], key, skip=SKIP_AXES)


@pytest.mark.parametrize(CASE_ARGS, [('qqnorm_mh', 'mh', {'label': True}), ('qqnorm_peto', 'peto', {'type': 'rstudent', 'grid': True})])
def test_qqnorm_matches_metafor(key: str, fit: str, kw: dict[str, typing.Any], fits: dict[str, typing.Any], ax: matplotlib.axes.Axes) -> None:
    r = metafopy.qqnorm(fits[fit], ax=ax, **kw)
    case = ref()[key]
    refutil.close(r.x, case['ret']['x'], f'{key}: x')
    refutil.close(r.y, case['ret']['y'], f'{key}: y')
    assert r.lb is None
    plotref.check_all(ax, case['rec'], key, skip=SKIP_AXES, compare_col=True)


# auditme: accept[non-alphabetical-dict-keys] order follows the R script that generates the fixtures
LABBE_CASES: dict[str, tuple[str, dict[str, typing.Any]]] = {
    'labbe_mh': ('mh', {}),
    'labbe_mh_rr': ('mh_rr', {'ci': True, 'transf': numpy.exp}),
    'labbe_glmm': ('glmm', {'pi': True, 'ci': True}),
}


@pytest.mark.parametrize('key', list(LABBE_CASES))
def test_labbe_matches_metafor(key: str, fits: dict[str, typing.Any], ax: matplotlib.axes.Axes) -> None:
    fit, kw = LABBE_CASES[key]
    r = metafopy.labbe(fits[fit], ax=ax, **kw)
    ret = ref()[key]['ret']
    refutil.close(r.x, ret['x'], f'{key}: x')
    refutil.close(r.y, ret['y'], f'{key}: y')
    refutil.close(r.cex, ret['cex'], f'{key}: cex')
    assert list(r.ids) == [int(v) for v in plotref.flat(ret['ids'])], key
    assert list(r.slab) == [str(v) for v in plotref.flat(ret['slab'])], key
    if rtol_for(key) == refutil.RTOL:
        plotref.check_all(ax, ref()[key]['rec'], key, skip=SKIP_AXES, compare_col=True)
    else:
        poly = [p for p in ax.patches if p.get_gid() == 'polygon']
        assert len(poly) == len(ref()[key]['rec']['polygon'])
        for p, rp in zip(poly, ref()[key]['rec']['polygon'], strict=True):
            ys = p.meta['xy'][:, 1]  # type: ignore[attr-defined]
            refutil.close(ys.sum(), rp['y']['sum'], key, rtol=RTOL_OPT)
            refutil.close(ys[:3], rp['y']['head'], key, rtol=RTOL_OPT)


# -- addpoly of predict() ------------------------------------------------------------------------------------------------


# auditme: accept[non-alphabetical-dict-keys] order follows the R script that generates the fixtures
ADDPOLY_CASES: dict[str, tuple[str, dict[str, typing.Any]]] = {
    'addpoly_pred_mh': ('mh', {'rows': -3, 'mlab': 'MH'}),
    'addpoly_pred_uni': ('uni', {'addpred': True}),
    'addpoly_pred_mods': ('unim', {}),
}


@pytest.mark.parametrize('key', list(ADDPOLY_CASES))
def test_addpoly_of_predictions_matches_metafor(key: str, fits: dict[str, typing.Any], ax: matplotlib.axes.Axes) -> None:
    fit, kw = ADDPOLY_CASES[key]
    metafopy.forest(fits['uni'], ylim=(-5, 16), ax=ax)
    n_before = {g: len(v) for g, v in metafopy.plot.draw.collect(ax).items()}
    metafopy.addpoly(metafopy.predict(fits[fit]), ax=ax, **kw)
    after = metafopy.plot.draw.collect(ax)
    assert sum(len(v) for v in after.values()) > sum(n_before.values())
    rec = ref()[key]['rec']
    texts = [t.get_text() for t in after['text'][n_before.get('text', 0) :]]
    assert texts == [t['label'] for t in plotref.r_texts(rec)], key
    polys = after['polygon'][n_before.get('polygon', 0) :]
    assert len(polys) == len(rec['polygon'])
    for p, rp in zip(polys, rec['polygon'], strict=True):
        plotref.same_summary(p.meta['xy'][:, 0], rp['x'], key)
        plotref.same_summary(p.meta['xy'][:, 1], rp['y'], key)


def test_addpoly_of_scale_predictions_is_an_error(ax: matplotlib.axes.Axes) -> None:
    dat = refutil.bcg()
    ls = metafopy.rma('yi', 'vi', scale='~ ablat', data=dat)
    metafopy.forest(dat['yi'], dat['vi'], ax=ax)
    with pytest.raises(metafopy.errors.InputError, match='predicted scale values'):
        metafopy.addpoly(metafopy.predict(ls, newscale=[10, 20]), ax=ax)


# -- plot_rma --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize('fit', ['uni', 'unim', 'mh', 'peto'])
def test_plot_rma_matches_metafor(fit: str, fits: dict[str, typing.Any]) -> None:
    fig = matplotlib.pyplot.figure(figsize=(10, 10))
    try:
        r = metafopy.plot_rma(fits[fit], ax=fig.add_subplot(111))
        assert isinstance(r, metafopy.plot.PlotRmaResult)
        rec = ref()[f'plot_{fit}']
        assert [a.get_title() for a in r.axes] == [t['main'] for t in rec['title']]
        last = rec['points'][-1]
        refutil.close(numpy.arange(1, r.z.shape[0] + 1), last['x'])
        refutil.close(r.z, last['y'], 'z')
        assert [str(i) for i in r.ids] == plotref.flat(rec['axis'][-1]['labels'])
        assert [t.get_text() for t in r.axes[3].get_xticklabels()] == plotref.flat(rec['axis'][-1]['labels'])
        hs = sorted(h for a in rec['abline'] for h in plotref.flat(a.get('h')))
        got = sorted(float(numpy.asarray(ln.get_ydata())[0]) for a in r.axes[2:] for ln in a.lines if ln.get_gid() == 'hline')
        if fit == 'unim':
            refutil.close(r.fitted, rec['points'][0]['x'], 'fitted')
            refutil.close(r.z, rec['points'][0]['y'], 'z (fitted panel)')
            assert r.radial is None
        else:
            assert r.radial is not None
            got = sorted(float(numpy.asarray(ln.get_ydata())[0]) for ln in r.axes[3].lines if ln.get_gid() == 'hline')
        refutil.close(got, hs, 'reference lines')
        for a, p in zip(r.axes[4 - len(rec['plot']) :], rec['plot'], strict=True):
            refutil.close(a.get_xlim(), list(metafopy.plot.rutils.usr_range(*p['xlim'])), 'xlim')
            refutil.close(a.get_ylim(), list(metafopy.plot.rutils.usr_range(*p['ylim'])), 'ylim')
            assert a.get_xlabel() == p['xlab']
    finally:
        matplotlib.pyplot.close(fig)


def test_plot_method_and_qqplot(fits: dict[str, typing.Any]) -> None:
    matplotlib.pyplot.figure(figsize=(10, 10))
    try:
        r = fits['mh'].plot(qqplot=True)
        assert isinstance(r, metafopy.plot.PlotRmaResult)
        assert r.qqnorm is not None
        assert r.z.shape == (0,)
        refutil.close(numpy.sort(r.qqnorm.y), numpy.sort(metafopy.rstandard(fits['mh']).z).tolist())
        r2 = fits['uni'].plot(qqplot=True, seed=1)
        assert isinstance(r2, metafopy.plot.PlotRmaResult)
        assert r2.qqnorm is not None
        assert r2.qqnorm.lb is not None
        assert isinstance(metafopy.plot_rma(fits['sel'], draw=False), metafopy.plot.SelPlotResult)
    finally:
        matplotlib.pyplot.close('all')


# -- ranktest, predict, tes, gosh ------------------------------------------------------------------------------------


@pytest.mark.parametrize('fit', ['mh', 'peto', 'mv', 'ls', 'sel', 'glmm'])
def test_ranktest_matches_metafor(fit: str, fits: dict[str, typing.Any]) -> None:
    r = metafopy.ranktest(fits[fit])
    exp = ref()['ranktest'][fit]
    refutil.close(r.tau, exp['tau'])
    refutil.close(r.pval, exp['pval'])


def test_ranktest_needs_a_model_without_moderators(fits: dict[str, typing.Any]) -> None:
    with pytest.raises(metafopy.errors.InputError, match='without moderators'):
        metafopy.ranktest(fits['mvm'])


@pytest.mark.parametrize(
    CASE_ARGS,
    [('mh', 'mh', {}), ('mh_exp', 'mh', {'transf': numpy.exp, 'level': 90}), ('peto', 'peto', {}), ('mh_rr', 'mh_rr', {'transf': numpy.exp})],
)
def test_predict_fixed_binary_matches_metafor(key: str, fit: str, kw: dict[str, typing.Any], fits: dict[str, typing.Any]) -> None:
    p = metafopy.predict(fits[fit], **kw)
    exp = ref()['predict'][key]
    for name in ('pred', 'se', 'ci_lb', 'ci_ub'):
        refutil.close(getattr(p, name), exp[name.replace('_', '.')], f'{key}: {name}')
    assert numpy.isnan(p.pi_lb).all()
    assert numpy.isnan(p.pi_ub).all()
    with pytest.raises(metafopy.errors.InputError, match=re.escape(ref()['errors']['predict_mh_newmods'])):
        metafopy.predict(fits[fit], newmods=1)


@pytest.mark.parametrize('fit', BINARY)
def test_tes_matches_metafor(fit: str, fits: dict[str, typing.Any]) -> None:
    t = metafopy.tes(fits[fit])
    exp = ref()['tes'][fit]
    assert t.k == exp['k']
    assert exp['O'] == t.O
    refutil.close(t.theta, exp['theta'])
    for name in ('E', 'OEratio', 'X2', 'pval', 'power'):
        refutil.close(getattr(t, name), exp[name], f'{fit}: {name}', rtol=1e-6)  # the integrals over the power are computed numerically, as in test_bias


@pytest.mark.parametrize('fit', BINARY)
def test_gosh_matches_metafor(fit: str) -> None:
    dat = refutil.bcg().iloc[:8]
    model = metafopy.rma_mh(**TABLES, data=dat) if fit == 'mh' else metafopy.rma_peto(**TABLES, data=dat)
    g = metafopy.gosh(model)
    exp = ref()['gosh'][fit]
    assert g.res.shape[0] == exp['nrow']
    assert list(g.res.columns) == exp['cols']
    for col in ('estimate', 'QE', 'I2', 'H2', 'k'):
        refutil.close(g.res[col].to_numpy(), exp[col if col != 'estimate' else 'est'], f'{fit}: {col}')
    assert list(g.fit) == [bool(v) for v in exp['fit']]


# -- classes without a method --------------------------------------------------------------------------------------


def call(name: str, fits: dict[str, typing.Any]) -> collections.abc.Callable[[], typing.Any]:
    fn, cls = name.split('_', 1)
    calls: dict[str, collections.abc.Callable[[typing.Any], typing.Any]] = {
        'baujat': metafopy.baujat,
        'fsn': metafopy.fsn,
        'gosh': metafopy.gosh,
        'hc': metafopy.hc,
        'labbe': metafopy.labbe,
        'plot': metafopy.plot_rma,
        'qqnorm': metafopy.qqnorm,
        'radial': metafopy.radial,
        'regtest': metafopy.regtest,
        'tes': metafopy.tes,
        'trimfill': metafopy.trimfill,
    }
    if fn == 'forest':
        if cls == 'ls':
            return lambda: metafopy.forest(fits['ls'])
        return lambda: metafopy.forest(fits[cls.removesuffix('_weights')], showweights=True)
    return lambda: calls[fn](fits[cls])


ERROR_CASES = [k for k in json.loads(FIXTURE.read_text(encoding='utf-8'))['errors'] if not k.startswith('predict')]


@pytest.mark.parametrize('key', ERROR_CASES)
def test_unsupported_classes_raise_metafors_error(key: str, fits: dict[str, typing.Any]) -> None:
    message = ref()['errors'][key]
    with matplotlib.pyplot.ioff(), pytest.raises(metafopy.errors.MetaforError, match=f'^{re.escape(message)}$'):
        call(key, fits)()
    matplotlib.pyplot.close('all')


def test_class_names(fits: dict[str, typing.Any]) -> None:
    assert metafopy.methods.fitview.r_class(fits['robust']) == ('robust.rma', 'rma', 'rma.uni')
    assert metafopy.methods.fitview.r_class(fits['sel']) == ('rma.uni.selmodel', 'rma.uni', 'rma')
    assert metafopy.methods.fitview.r_class(numpy.zeros(2)) == ()
    with pytest.raises(metafopy.errors.ArgumentError, match=re.escape('must be an object of class "rma.uni"')):
        metafopy.methods.fitview.check_class(fits['mh'], must='rma.uni')
    with pytest.raises(metafopy.errors.ArgumentError):
        metafopy.methods.fitview.fit_view(numpy.zeros(2))


def test_fit_view_of_tables_with_missing_counts(fits: dict[str, typing.Any]) -> None:
    v = metafopy.methods.fitview.fit_view(fits['mh_na'])
    assert v.k == 12
    assert v.na_rows is not None
    assert list(numpy.flatnonzero(v.na_rows)) == [2]
    assert not v.keep[2]
    assert v.slab[2] == fits['mh'].slab[3]
    assert v.weights is not None
    numpy.testing.assert_allclose(v.weights.sum(), 100.0)
