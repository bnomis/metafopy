"""vif() of rma_mv and selmodel fits, regplot() of rma_mv, location-scale and selmodel fits, points_regplot(), forest(tabfig=) and the forest layout
of selmodel fits, checked against R metafor 5.2-1 (tests/r/generate_audit4_plots.R)."""

import collections.abc
import typing

import matplotlib.axes
import matplotlib.pyplot
import numpy
import pandas
import pytest
import refutil

import metafopy
import metafopy.datasets
import metafopy.errors
import metafopy.methods.ls_vif
import metafopy.plot.cumul
from metafopy.plot import ForestResult, RegplotResult

pytestmark = pytest.mark.filterwarnings('ignore:One or more intervals do not contain')

REF: dict[str, typing.Any] = refutil.load_phase12('audit4_plots')
VIF: dict[str, typing.Any] = REF['vif']
REG: dict[str, typing.Any] = REF['regplot']
FOREST: dict[str, typing.Any] = REF['forest']
# vif() of rma_mv fits agrees with R to 8e-9 (default), 2e-9 (sim) and 2e-6 (reestimate), the refits being the less precise
MV_RTOL = 1e-5


@pytest.fixture(scope='module')
def dat() -> pandas.DataFrame:
    return metafopy.escalc('RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=metafopy.datasets.load('dat.bcg'))


@pytest.fixture(scope='module')
def kon() -> pandas.DataFrame:
    d = metafopy.datasets.load('konstantopoulos2011').copy()
    d['z'] = numpy.sin(numpy.arange(1, 57))
    return d


@pytest.fixture(scope='module')
def mv(kon: pandas.DataFrame) -> metafopy.RMAMVResult:
    return metafopy.rma_mv('yi', 'vi', mods='~ year + z', random='~ 1 | district/school', data=kon)


@pytest.fixture(scope='module')
def mv_t(kon: pandas.DataFrame) -> metafopy.RMAMVResult:
    return metafopy.rma_mv('yi', 'vi', mods='~ year + z', random='~ 1 | district/school', data=kon, test='t')


@pytest.fixture(scope='module')
def uni(dat: pandas.DataFrame) -> metafopy.RMAResult:
    return metafopy.rma('yi', 'vi', mods='~ ablat + year', data=dat)


@pytest.fixture(scope='module')
def sel(uni: metafopy.RMAResult) -> metafopy.SelmodelResult:
    return typing.cast(metafopy.SelmodelResult, metafopy.selmodel(uni, type='halfnorm'))


def check_vif(out: metafopy.VIFResult, ref: dict[str, typing.Any], what: str, rtol: float = 1e-8, text: bool = False) -> None:
    refutil.close(out.vifs.to_numpy(), ref['vif'], f'{what}: vif', rtol=rtol)
    refutil.close([e.sif for e in out.vif], ref['sif'], f'{what}: sif', rtol=rtol)
    assert [e.m for e in out.vif] == refutil.as_list(ref['m'])
    assert [e.coefs for e in out.vif] == refutil.as_list(ref['coefs'])
    assert out.names == refutil.as_list(ref['names'])
    if text:
        assert refutil.same_print(str(out), ref['print']), f'{what}\n{out}\n{ref["print"]}'


def check_sim(out: metafopy.VIFResult, ref: dict[str, typing.Any], what: str, rtol: float) -> None:
    assert out.sim is not None
    assert out.sim_type == ref['sim_type']
    refutil.close(out.sim, numpy.array(ref['sim']).reshape(out.sim.shape).tolist(), f'{what}: sim', rtol=rtol)
    refutil.close(out.prop, ref['prop'], f'{what}: prop')


def check_table(out: metafopy.VIFResult, columns: collections.abc.Sequence[str]) -> None:
    assert out.table is not None
    assert list(out.table.columns) == list(columns)


# -- vif of rma_mv fits ---------------------------------------------------------------


def test_vif_mv_default(mv: metafopy.RMAMVResult) -> None:
    check_vif(metafopy.vif(mv), VIF['mv_default'], 'default', rtol=MV_RTOL, text=True)
    check_vif(mv.vif(), VIF['mv_default'], 'method', rtol=MV_RTOL)


def test_vif_mv_table(mv: metafopy.RMAMVResult, mv_t: metafopy.RMAMVResult) -> None:
    out = metafopy.vif(mv, table=True)
    check_vif(out, VIF['mv_table'], 'table', rtol=MV_RTOL, text=True)
    check_table(out, ['estimate', 'se', 'zval', 'pval', 'ci.lb', 'ci.ub'])
    out_t = metafopy.vif(mv_t, table=True)
    check_vif(out_t, VIF['mv_table_t'], 'table, test t', rtol=MV_RTOL)
    check_table(out_t, ['estimate', 'se', 'tval', 'df', 'pval', 'ci.lb', 'ci.ub'])


def test_vif_mv_intercept(mv: metafopy.RMAMVResult) -> None:
    check_vif(metafopy.vif(mv, intercept=True), VIF['mv_intercept'], 'intercept', rtol=MV_RTOL)
    out = metafopy.vif(mv, intercept=True, table=True)
    check_vif(out, VIF['mv_intercept_table'], 'intercept, table', rtol=MV_RTOL)
    assert out.table is not None
    assert len(out.table) == len(out.vif) == 3


def test_vif_mv_btt(mv: metafopy.RMAMVResult) -> None:
    check_vif(metafopy.vif(mv, btt=[1, 2]), VIF['mv_btt'], 'btt', rtol=MV_RTOL, text=True)
    check_vif(metafopy.vif(mv, btt='z'), VIF['mv_btt_name'], 'btt name', rtol=MV_RTOL, text=True)
    check_vif(metafopy.vif(mv, btt='y.'), VIF['mv_btt_grep'], 'btt regex', rtol=MV_RTOL, text=True)
    check_vif(metafopy.vif(mv, btt=[1, [2], [1, 2]]), VIF['mv_btt_list'], 'btt list', rtol=MV_RTOL, text=True)
    assert 'error' in VIF['mv_btt_fixed']
    with pytest.raises(metafopy.errors.InputError):
        metafopy.vif(mv, btt='y.', fixed=True)
    # a literal string that is found in 'year' only gives the row of 'year' of the default result
    ref = VIF['mv_default']
    check_vif(
        metafopy.vif(mv, btt='ea', fixed=True), {'vif': ref['vif'][:1], 'sif': ref['sif'][:1], 'm': [1], 'coefs': ['2'], 'names': ['2']}, 'fixed', rtol=MV_RTOL
    )


def test_vif_mv_intercept_and_errors(mv: metafopy.RMAMVResult) -> None:
    with pytest.raises(metafopy.errors.InputError, match='Cannot compute VIF'):
        metafopy.vif(mv, btt=0)
    # with intercept=True the intercept can be given
    assert metafopy.vif(mv, btt=0, intercept=True).vif[0].coefs == '1'
    with pytest.raises(metafopy.errors.InputError, match="Argument 'att' only relevant"):
        metafopy.vif(mv, att=1)
    with pytest.raises(metafopy.errors.InputError, match="Argument 'sim' must be >= 2"):
        metafopy.vif(mv, sim=1)
    intercept_only = metafopy.rma_mv('yi', 'vi', random='~ 1 | district/school', data=metafopy.datasets.load('konstantopoulos2011'))
    with pytest.raises(metafopy.errors.InputError, match='intercept-only'):
        metafopy.vif(intercept_only)


@pytest.mark.slow
def test_vif_mv_reestimate(mv: metafopy.RMAMVResult) -> None:
    check_vif(metafopy.vif(mv, reestimate=True), VIF['mv_reestimate'], 'reestimate', rtol=1e-4, text=False)
    check_vif(metafopy.vif(mv, btt=[1, 2], reestimate=True, intercept=True), VIF['mv_reestimate_btt'], 'reestimate, intercept', rtol=1e-4)


@pytest.mark.slow
def test_vif_mv_sim(mv: metafopy.RMAMVResult) -> None:
    out = metafopy.vif(mv, sim=4, seed=1, progbar=False)
    check_vif(out, VIF['mv_sim'], 'sim', rtol=MV_RTOL, text=True)
    check_sim(out, VIF['mv_sim'], 'sim', rtol=1e-4)
    out = metafopy.vif(mv, sim=3, seed=2, reestimate=True, progbar=False)
    check_vif(out, VIF['mv_sim_reestimate'], 'sim, reestimate', rtol=1e-4)
    check_sim(out, VIF['mv_sim_reestimate'], 'sim, reestimate', rtol=1e-3)


def test_vif_mv_sim_on_matrix(kon: pandas.DataFrame) -> None:
    """A model given a model matrix has its columns permuted (type 'X'), as in metafor."""
    X = numpy.column_stack([kon['year'].to_numpy(), kon['z'].to_numpy()])
    res = metafopy.rma_mv('yi', 'vi', mods=X, random='~ 1 | district/school', data=kon)
    out = metafopy.vif(res, sim=2, seed=1, progbar=False)
    assert out.sim is not None
    assert out.sim_type == 'X'
    assert out.sim.shape == (2, 2)


# -- vif of selmodel fits -------------------------------------------------------------


def test_vif_selmodel(sel: metafopy.SelmodelResult) -> None:
    check_vif(metafopy.vif(sel), VIF['sel_default'], 'default', text=True)
    check_vif(sel.vif(), VIF['sel_default'], 'method')
    out = metafopy.vif(sel, table=True)
    check_vif(out, VIF['sel_table'], 'table', text=True)
    check_table(out, ['estimate', 'se', 'zval', 'pval', 'ci.lb', 'ci.ub'])
    check_vif(metafopy.vif(sel, intercept=True), VIF['sel_intercept'], 'intercept')
    check_vif(metafopy.vif(sel, btt=[1, 2]), VIF['sel_btt'], 'btt', text=True)


def test_vif_selmodel_stepfun(uni: metafopy.RMAResult, dat: pandas.DataFrame) -> None:
    fit = metafopy.selmodel(metafopy.rma('yi', 'vi', mods='~ ablat + year', data=dat), type='stepfun', steps=0.025)
    check_vif(metafopy.vif(fit, btt='ablat'), VIF['sel_btt_name'], 'btt name', rtol=1e-6)


def test_vif_selmodel_refuses_sim_and_reestimate(sel: metafopy.SelmodelResult) -> None:
    assert VIF['sel_sim']['error'] == "Cannot use 'sim' with models of class 'rma.uni.selmodel'."
    assert VIF['sel_reestimate']['error'] == "Cannot use 'restimate=TRUE' with models of class 'rma.uni.selmodel'."
    with pytest.raises(metafopy.errors.InputError, match=r"Cannot use 'sim' with models of class 'rma\.uni\.selmodel'\."):
        metafopy.vif(sel, sim=3)
    with pytest.raises(metafopy.errors.InputError, match=r"Cannot use 'restimate=TRUE' with models of class 'rma\.uni\.selmodel'\."):
        metafopy.vif(sel, reestimate=True)


# -- vif(intercept=, fixed=) of the other classes -------------------------------------------


def test_vif_uni_intercept(uni: metafopy.RMAResult) -> None:
    check_vif(metafopy.vif(uni, intercept=True), VIF['uni_intercept'], 'intercept')
    out = metafopy.vif(uni, intercept=True, table=True)
    check_vif(out, VIF['uni_intercept_table'], 'intercept, table')
    check_vif(metafopy.vif(uni, intercept=True, reestimate=True), VIF['uni_intercept_reestimate'], 'intercept, reestimate', rtol=1e-6)


def test_vif_uni_intercept_sim(uni: metafopy.RMAResult) -> None:
    out = metafopy.vif(uni, intercept=True, sim=4, seed=3, progbar=False)
    check_vif(out, VIF['uni_sim_intercept'], 'intercept, sim')
    check_sim(out, VIF['uni_sim_intercept'], 'intercept, sim', rtol=1e-6)


def test_vif_uni_fixed(uni: metafopy.RMAResult) -> None:
    check_vif(metafopy.vif(uni, btt='ab', fixed=True), VIF['uni_fixed'], 'fixed')
    assert 'error' in VIF['uni_fixed_none']
    with pytest.raises(metafopy.errors.InputError):
        metafopy.vif(uni, btt='.b', fixed=True)
    # as a regular expression '.b' finds the coefficient 'ablat'
    assert metafopy.vif(uni, btt='.b').vif[0].coefs == '2'


def test_vif_ls_intercept_and_fixed(dat: pandas.DataFrame) -> None:
    res = metafopy.rma('yi', 'vi', mods='~ ablat + year', scale='~ ablat', data=dat)
    out = typing.cast(metafopy.methods.ls_vif.LSVIFResult, metafopy.vif(res, intercept=True))
    check_vif(out.beta, VIF['ls_intercept']['beta'], 'ls beta intercept', rtol=1e-5)
    # the scale coefficients are almost collinear (VIF 114), which magnifies the 1e-6 differences of the scale variance-covariance matrix
    check_vif(out.alpha, VIF['ls_intercept']['alpha'], 'ls alpha intercept', rtol=5e-3)
    out = typing.cast(metafopy.methods.ls_vif.LSVIFResult, metafopy.vif(res, btt='ab', att='ab', fixed=True))
    check_vif(out.beta, VIF['ls_fixed']['beta'], 'ls beta fixed', rtol=1e-5)
    check_vif(out.alpha, VIF['ls_fixed']['alpha'], 'ls alpha fixed', rtol=1e-5)


# -- regplot of rma_mv, location-scale and selmodel fits ---------------------------------------


@pytest.fixture
def ax() -> collections.abc.Iterator[matplotlib.axes.Axes]:
    fig, axes = matplotlib.pyplot.subplots(figsize=(7, 7))
    yield axes
    matplotlib.pyplot.close(fig)


def fit_one(kind: str, dat: pandas.DataFrame, formula: str = '~ ablat') -> typing.Any:
    if kind == 'mv':
        return metafopy.rma_mv('yi', 'vi', mods=formula, random='~ 1 | trial', data=dat)
    if kind == 'ls':
        return metafopy.rma('yi', 'vi', mods=formula, scale='~ ablat' if formula == '~ ablat' else '~ year', data=dat)
    return metafopy.selmodel(metafopy.rma('yi', 'vi', mods=formula, data=dat), type='halfnorm')


def check_regplot(out: RegplotResult, ref: dict[str, typing.Any], rtol: float = 1e-8) -> None:
    refutil.close(out.xi, ref['xi'], 'xi')
    refutil.close(out.yi, ref['yi'], 'yi', rtol=rtol)
    refutil.close(out.psize, ref['psize'], 'psize', rtol=rtol)
    refutil.close(out.pred, ref['pred'], 'pred', rtol=rtol)
    assert out.label.tolist() == refutil.as_list(ref['label'])
    assert out.order.tolist() == [int(v) for v in refutil.as_list(ref['order'])]
    assert out.ids.tolist() == [int(v) for v in refutil.as_list(ref['ids'])]
    assert list(out.pch) == refutil.as_list(ref['pch'])
    assert [str(c) for c in out.col] == refutil.as_list(ref['col'])
    assert out.slab == tuple(refutil.as_list(ref['slab']))
    refutil.close(out.offset, ref['offset'], 'offset')
    assert out.labsize == ref['labsize']


def band_sums(ax: matplotlib.axes.Axes, undo: collections.abc.Callable[[typing.Any], typing.Any] = lambda y: y) -> dict[str, typing.Any]:
    """Sums of the regression line and the bounds drawn on the axes (``undo`` reverses a transformation of the plot)."""
    lines = [
        undo(numpy.asarray(line.get_ydata(), dtype=float)) for line in ax.lines if line.get_gid() == 'line' and numpy.asarray(line.get_ydata()).shape[0] == 1000
    ]
    widths = [line.get_linewidth() for line in ax.lines if line.get_gid() == 'line' and numpy.asarray(line.get_ydata()).shape[0] == 1000]
    regression = next(y for y, w in zip(lines, widths, strict=True) if w == 3.0)
    others = [y for y, w in zip(lines, widths, strict=True) if w != 3.0]
    return {'pred': float(regression.sum()), 'bounds': sorted(float(y.sum()) for y in others)}


def check_band(ax: matplotlib.axes.Axes, ref: dict[str, typing.Any], pi: bool, rtol: float = 1e-8, transformed: bool = False) -> None:
    got = band_sums(ax, numpy.log if transformed else lambda y: y)
    refutil.close(got['pred'], ref['sum_pred'], 'regression line', rtol=rtol)
    expected = [ref['sum_ci_lb'], ref['sum_ci_ub']]
    if pi and ref.get('sum_pi_lb') is not None:
        expected += [ref['sum_pi_lb'], ref['sum_pi_ub']]
    refutil.close(got['bounds'], sorted(expected), 'bounds', rtol=rtol)


# the variance components of the rma_mv and selmodel fits are only reproduced to about 1e-4 / 1e-5
REGPLOT_CASES = [
    ('mv_default', 'mv', '~ ablat', {}, 1e-5),
    ('mv_pi_label', 'mv', '~ ablat', {'pi': True, 'label': 'piout'}, 1e-5),
    ('mv_mod', 'mv', '~ ablat + year', {'mod': 'year', 'label': 'ciout', 'psize': 'seinv'}, 1e-5),
    ('ls_default', 'ls', '~ ablat', {}, 1e-5),
    ('ls_mod', 'ls', '~ ablat + year', {'mod': 'year', 'label': 'all', 'transf': numpy.exp}, 1e-5),
    ('sel_default', 'sel', '~ ablat', {}, 1e-4),
    ('sel_pi_label', 'sel', '~ ablat', {'pi': True, 'label': 'piout'}, 1e-4),
]


@pytest.mark.parametrize(('key', 'kind', 'formula', 'kwargs', 'rtol'), REGPLOT_CASES, ids=[c[0] for c in REGPLOT_CASES])
def test_regplot_matches_metafor(
    dat: pandas.DataFrame, ax: matplotlib.axes.Axes, key: str, kind: str, formula: str, kwargs: dict[str, typing.Any], rtol: float
) -> None:
    fit = fit_one(kind, dat, formula)
    out = metafopy.regplot(fit, ax=ax, **kwargs)
    check_regplot(out, REG[key], rtol=rtol)
    check_band(ax, REG[key]['band'], bool(kwargs.get('pi')), rtol=rtol, transformed='transf' in kwargs)
    assert out.ax is ax


@pytest.mark.filterwarnings('ignore:Cannot label points')
def test_regplot_ls_without_prediction_interval(dat: pandas.DataFrame, ax: matplotlib.axes.Axes) -> None:
    """metafor draws no prediction interval for a location-scale model (and labels nothing for label='piout')."""
    fit = fit_one('ls', dat)
    with pytest.warns(UserWarning, match='Cannot draw prediction interval for the given model'):
        out = metafopy.regplot(fit, pi=True, label='piout', ax=ax)
    check_regplot(out, REG['ls_label'], rtol=1e-5)
    assert not out.label.any()


def test_regplot_selmodel_with_xvals(dat: pandas.DataFrame, ax: matplotlib.axes.Axes) -> None:
    fit = fit_one('sel', dat, '~ ablat + year')
    out = metafopy.regplot(fit, mod='year', label=[0, 4, 8], xvals=numpy.linspace(1950, 1990, 5), ax=ax)
    ref = REG['sel_mod']
    check_regplot(out, ref, rtol=1e-4)
    # the line is drawn at the five values
    assert any(numpy.asarray(line.get_xdata()).shape[0] == 5 for line in ax.lines)


def test_regplot_mod_grep_and_fixed(uni: metafopy.RMAResult, ax: matplotlib.axes.Axes) -> None:
    check_regplot(metafopy.regplot(uni, mod='ea', grep=True, ax=ax), REG['grep'])
    matplotlib.pyplot.close('all')
    check_regplot(metafopy.regplot(uni, mod='ea', grep=True, fixed=True), REG['grep_fixed'])
    matplotlib.pyplot.close('all')
    check_regplot(metafopy.regplot(uni, mod='^a.', grep=True), REG['grep_regex'])
    matplotlib.pyplot.close('all')
    assert 'error' in REG['grep_regex_fixed']
    with pytest.raises(metafopy.errors.InputError, match='Could not find or uniquely identify'):
        metafopy.regplot(uni, mod='^a.', grep=True, fixed=True)
    assert 'error' in REG['grep_ambiguous']
    with pytest.raises(metafopy.errors.InputError, match='Could not find or uniquely identify'):
        metafopy.regplot(uni, mod='a.', grep=True)
    # without grep the name has to match from the start, as with charmatch()
    with pytest.raises(metafopy.errors.InputError, match='Could not find or uniquely identify'):
        metafopy.regplot(uni, mod='ea')
    matplotlib.pyplot.close('all')


def test_regplot_refuses_unsupported_fits(dat: pandas.DataFrame) -> None:
    with pytest.raises(metafopy.errors.NotSupportedError):
        metafopy.regplot(metafopy.rma_mh(ai=dat['tpos'], bi=dat['tneg'], ci=dat['cpos'], di=dat['cneg'], measure='RR'))
    with pytest.raises(metafopy.errors.InputError, match='intercept-only'):
        metafopy.regplot(metafopy.rma_mv('yi', 'vi', random='~ 1 | trial', data=dat))
    with pytest.raises(metafopy.errors.InputError, match='intercept-only'):
        metafopy.regplot(typing.cast(metafopy.SelmodelResult, metafopy.selmodel(metafopy.rma('yi', 'vi', data=dat), type='halfnorm')))


# -- points_regplot ---------------------------------------------------------------------------


@pytest.mark.parametrize('kind', ['uni', 'mv', 'ls', 'sel'])
def test_points_regplot_redraws_points_and_labels(dat: pandas.DataFrame, ax: matplotlib.axes.Axes, kind: str) -> None:
    fit = metafopy.rma('yi', 'vi', mods='~ ablat', data=dat) if kind == 'uni' else fit_one(kind, dat)
    out = metafopy.regplot(fit, label='all', ax=ax)
    n_artists, n_texts = len(ax.lines) + len(ax.collections) + len(ax.patches), len(ax.texts)
    metafopy.points_regplot(out)
    k = len(out.xi)
    assert len(ax.texts) == n_texts + k
    assert len(ax.lines) + len(ax.collections) + len(ax.patches) == n_artists + k
    # the new labels are those of the plot, in the same places
    first, second = ax.texts[:k], ax.texts[k:]
    assert [t.get_text() for t in first] == [t.get_text() for t in second]
    assert [t.get_position() for t in first] == [t.get_position() for t in second]
    out.points()
    assert len(ax.texts) == n_texts + 2 * k


def test_points_regplot_without_labels_and_on_other_axes(dat: pandas.DataFrame, ax: matplotlib.axes.Axes) -> None:
    out = metafopy.regplot(fit_one('mv', dat), ax=ax)
    n_texts = len(ax.texts)
    fig, other = matplotlib.pyplot.subplots()
    try:
        metafopy.points_regplot(out, ax=other)
        assert len(ax.texts) == n_texts
        assert len(other.texts) == 0
        assert len(other.lines) + len(other.collections) + len(other.patches) >= len(out.xi)
    finally:
        matplotlib.pyplot.close(fig)
    with pytest.raises(metafopy.errors.InputError, match='regplot'):
        metafopy.points_regplot(typing.cast(typing.Any, 'not a regplot'))


def test_regplot_result_frame_has_the_columns_of_metafor(dat: pandas.DataFrame, ax: matplotlib.axes.Axes) -> None:
    out = metafopy.regplot(fit_one('mv', dat), ax=ax)
    assert list(out.to_frame().columns) == ['slab', 'ids', 'xi', 'yi', 'pch', 'psize', 'col', 'bg', 'label', 'pred', 'order']


# -- forest(tabfig=) and the layout for selmodel fits ----------------------------------------------


@pytest.fixture(scope='module')
def plain(dat: pandas.DataFrame) -> metafopy.RMAResult:
    return metafopy.rma('yi', 'vi', data=dat)


def annotations(result: ForestResult) -> list[str]:
    return [t.get_text() for t in result.ax.texts if '[' in t.get_text()]


def check_layout(out: ForestResult, ref: dict[str, typing.Any]) -> None:
    refutil.close(out.xlim, ref['xlim'], 'xlim', rtol=1e-6)
    refutil.close(out.alim, ref['alim'], 'alim', rtol=1e-6)
    refutil.close(out.textpos, ref['textpos'], 'textpos', rtol=1e-6)
    refutil.close(out.areas, ref['areas'], 'areas')


@pytest.mark.parametrize('tabfig', [1, 2, 3])
def test_forest_tabfig(plain: metafopy.RMAResult, ax: matplotlib.axes.Axes, tabfig: int) -> None:
    out = metafopy.forest(plain, tabfig=tabfig, ax=ax)
    ref = FOREST[f'tabfig{tabfig}']
    assert annotations(out) == ref['annotations']
    check_layout(out, ref)


def test_forest_tabfig_with_prediction_interval_and_annosym(plain: metafopy.RMAResult, ax: matplotlib.axes.Axes) -> None:
    out = metafopy.forest(plain, tabfig=1, addpred=True, ax=ax)
    assert annotations(out) == FOREST['tabfig1_prediction']['annotations']
    matplotlib.pyplot.close('all')
    # tabfig takes precedence over annosym
    out = metafopy.forest(plain, tabfig=3, annosym=[' (', ' / ', ')'])
    assert annotations(out) == FOREST['tabfig_over_annosym']['annotations']
    matplotlib.pyplot.close('all')
    out = metafopy.forest(plain, tabfig=None)
    assert annotations(out) == FOREST['default']['annotations']
    matplotlib.pyplot.close('all')
    # any other value leaves annosym alone
    out = metafopy.forest(plain, tabfig=4)
    assert annotations(out) == FOREST['default']['annotations']
    matplotlib.pyplot.close('all')


def test_forest_tabfig_for_estimates_and_cumulative_fits(dat: pandas.DataFrame, plain: metafopy.RMAResult, ax: matplotlib.axes.Axes) -> None:
    out = metafopy.forest(dat['yi'], dat['vi'], tabfig=1, ax=ax)
    assert annotations(out) == FOREST['tabfig1_default']['annotations']
    matplotlib.pyplot.close('all')
    cum = metafopy.cumul(plain, order=numpy.argsort(dat['year'].to_numpy(), kind='stable'))
    out = metafopy.forest(cum, tabfig=1)
    assert annotations(out) == FOREST['tabfig1_cumul']['annotations']
    matplotlib.pyplot.close('all')
    out = metafopy.plot.cumul.forest_cumul(cum, tabfig=1)
    assert annotations(out) == FOREST['tabfig1_cumul']['annotations']
    matplotlib.pyplot.close('all')


def test_forest_tabfig_is_kept_for_addpoly(plain: metafopy.RMAResult, ax: matplotlib.axes.Axes) -> None:
    metafopy.forest(plain, tabfig=1, addfit=False, ax=ax)
    n = len(ax.texts)
    metafopy.addpoly(plain, row=-1, ax=ax)
    added = [t.get_text() for t in ax.texts[n:] if '[' in t.get_text()]
    assert added
    assert all('\u2009[' in a and ',\u2009' in a for a in added)


def test_forest_of_selmodel_fits_reserves_room_for_the_labels(sel: metafopy.SelmodelResult, dat: pandas.DataFrame, ax: matplotlib.axes.Axes) -> None:
    sel_one = metafopy.selmodel(metafopy.rma('yi', 'vi', mods='~ ablat', data=dat), type='halfnorm')
    out = metafopy.forest(sel_one, ax=ax)
    check_layout(out, FOREST['selmodel'])
    assert annotations(out) == FOREST['selmodel']['annotations']
    matplotlib.pyplot.close('all')
    out = metafopy.forest(sel_one, slab=[f'Trial {t}' for t in dat['trial']])
    check_layout(out, FOREST['selmodel_slab'])
    matplotlib.pyplot.close('all')
    out = metafopy.forest(sel)
    check_layout(out, FOREST['selmodel_mods'])
    matplotlib.pyplot.close('all')
    out = metafopy.forest(metafopy.selmodel(metafopy.rma('yi', 'vi', data=dat), type='halfnorm'), addpred=True)
    check_layout(out, FOREST['selmodel_addpred'])
    matplotlib.pyplot.close('all')


def test_forest_of_selmodel_fits_labels_the_studies_by_number(sel: metafopy.SelmodelResult, ax: matplotlib.axes.Axes) -> None:
    """A selmodel fit has slab.null = FALSE, so the labels are the study numbers, not 'Study 1'."""
    out = metafopy.forest(sel, ax=ax)
    texts = [t.get_text() for t in out.ax.texts]
    assert '1' in texts
    assert 'Study 1' not in texts


def test_unchanged_layout_without_selmodel(plain: metafopy.RMAResult, ax: matplotlib.axes.Axes) -> None:
    check_layout(metafopy.forest(plain, ax=ax), FOREST['default'])
