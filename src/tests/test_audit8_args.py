"""Arguments added after audit 8, against R (src/tests/r/generate_audit8_args.R)."""

import contextlib
import io
import math
import warnings

import matplotlib
import matplotlib.pyplot
import numpy
import pandas
import pytest
import refutil

import metafopy
import metafopy.core.timing
import metafopy.errors
import metafopy.methods.rcalc

REF = refutil.load_phase7('audit8_args')


@pytest.fixture(scope='module')
def bcg() -> pandas.DataFrame:
    dat = metafopy.datasets.load('dat.bcg')
    return metafopy.escalc('RR', ai=dat.tpos, bi=dat.tneg, ci=dat.cpos, di=dat.cneg, data=dat)


def test_vif_joinb(bcg: pandas.DataFrame) -> None:
    res = metafopy.rma('yi', 'vi', mods='~ ablat + year + alloc', data=bcg)
    out = metafopy.vif(res, sim=20, seed=7, joinb=[1, 2, 3, 4, 4])
    ref = REF['vif_joinb']
    refutil.close(out.vifs.to_numpy(), ref['vif'], 'vif')
    numpy.testing.assert_allclose(numpy.asarray(out.sim), numpy.asarray(ref['sim']), rtol=1e-8)
    refutil.close(numpy.asarray(out.prop), ref['prop'], 'prop')
    with pytest.raises(metafopy.errors.InputError):
        metafopy.vif(res, sim=5, seed=1, joinb=[1, 2])


def test_vif_lb_is_accepted(bcg: pandas.DataFrame) -> None:
    res = metafopy.rma('yi', 'vi', mods='~ ablat + year', data=bcg)
    a = metafopy.vif(res, sim=5, seed=3)
    b = metafopy.vif(res, sim=5, seed=3, LB=True)
    numpy.testing.assert_allclose(numpy.asarray(a.sim), numpy.asarray(b.sim))


@pytest.mark.parametrize(
    ('x', 'text'),
    [
        (0.123, '\nProcessing time: 0 hours, 0 minutes, 0.12 seconds\n'),
        (61.4, '\nProcessing time: 0 hours, 1 minute, 1 second\n'),
        (3700.0, '\nProcessing time: 1 hour, 1 minute, 40 seconds\n'),
    ],
)
def test_format_time(x: float, text: str) -> None:
    assert metafopy.core.timing.format_time(x) == text


def test_time_prints_after_the_result(bcg: pandas.DataFrame) -> None:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = metafopy.rma('yi', 'vi', data=bcg, time=True)
        metafopy.confint(res, time=True)
    assert buf.getvalue().count('Processing time:') == 2
    quiet = io.StringIO()
    with contextlib.redirect_stdout(quiet):
        metafopy.rma('yi', 'vi', data=bcg)
    assert quiet.getvalue() == ''


def test_anova_ls_arguments(bcg: pandas.DataFrame) -> None:
    ls1 = metafopy.rma('yi', 'vi', mods='~ ablat + year', scale='~ ablat', data=bcg, method='ML')
    ls0 = metafopy.rma('yi', 'vi', mods='~ ablat', scale='~ 1', data=bcg, method='ML')
    out = metafopy.anova(ls1, ls0, df=2)
    ref = REF['ls_lrt_df']
    assert (out.df_full, out.df_reduced) == (ref['parms_f'], ref['parms_r'])
    refutil.close([out.LRT, out.pval], [ref['LRT'], ref['pval']], 'lrt df', rtol=1e-4)
    assert metafopy.anova(ls1, ls0, df=3).df_full == REF['ls_lrt_df3']['parms_f']
    r1 = metafopy.rma('yi', 'vi', mods='~ ablat + year', scale='~ ablat', data=bcg)
    r0 = metafopy.rma('yi', 'vi', mods='~ ablat', scale='~ ablat', data=bcg)
    wald = metafopy.anova(r1, r0, test='Wald')
    refutil.close([wald.QM, wald.QMp], [REF['ls_wald']['QM'], REF['ls_wald']['QMp']], 'wald between', rtol=1e-4)
    lin = metafopy.anova(r1, L=[[0, 1, 0], [0, 0, 1]])
    refutil.close([lin.QM], [REF['ls_L']['QM']], 'L', rtol=1e-4)
    refutil.close(lin.pval, REF['ls_L']['pval'], 'L pval', rtol=1e-4)
    fixed = metafopy.anova(r1, btt='year', fixed=True)
    refutil.close([fixed.QM, fixed.QMp], [REF['ls_fixed']['QM'], REF['ls_fixed']['QMp']], 'fixed', rtol=1e-4)


def test_anova_att_and_z_need_a_location_scale_model(bcg: pandas.DataFrame) -> None:
    res = metafopy.rma('yi', 'vi', mods='~ ablat', data=bcg)
    with pytest.raises(metafopy.errors.InputError, match="Can only specify 'att'"):
        metafopy.anova(res, att=1)
    with pytest.raises(metafopy.errors.InputError, match="Can only specify 'Z'"):
        metafopy.anova(res, Z=[[1.0, 0.0]])


def test_anova_selmodel_df() -> None:
    hack = metafopy.datasets.load('dat.hackshaw1998')
    base = metafopy.rma('yi', 'vi', data=hack, method='ML')
    se1 = metafopy.selmodel(base, type='stepfun', steps=[0.025, 0.1, 0.5])
    se0 = metafopy.selmodel(base, type='stepfun', steps=[0.025, 0.5])
    out = metafopy.anova(se1, se0, df=2)
    ref = REF['sel_lrt_df']
    assert (out.df_full, out.df_reduced) == (ref['parms_f'], ref['parms_r'])
    refutil.close([out.LRT, out.pval], [ref['LRT'], ref['pval']], 'sel lrt df')


def test_confint_extint(bcg: pandas.DataFrame) -> None:
    ml = metafopy.rma('yi', 'vi', data=bcg, method='ML')
    ref = REF['confint_extint_uni']
    without = metafopy.confint(ml, type='pl', xlim=(0, 0.5)).random
    with_ext = metafopy.confint(ml, type='pl', xlim=(0, 0.5), extint=True).random
    assert without is not None
    assert with_ext is not None
    refutil.close(without.loc['tau^2'].to_numpy(), ref['without'], 'uni without', rtol=1e-5)
    refutil.close(with_ext.loc['tau^2'].to_numpy(), ref['with'], 'uni with', rtol=1e-5)


def test_confint_extint_mv_and_ls(bcg: pandas.DataFrame) -> None:
    mv = metafopy.rma_mv('yi', 'vi', random='~ 1 | trial', data=bcg, method='ML')
    ref = REF['confint_extint_mv']
    got = metafopy.confint(mv, sigma2=0, xlim=(0, 0.5), extint=True).random
    assert got is not None
    refutil.close(got.to_numpy().T.ravel(), ref['with'], 'mv with', rtol=1e-5)
    ls = metafopy.rma('yi', 'vi', scale='~ ablat', data=bcg, method='ML')
    ref = REF['confint_extint_ls']
    got = metafopy.confint(ls, alpha=1, xlim=(-0.1, 0.1), extint=True).random
    assert got is not None
    refutil.close(got.to_numpy().ravel(), ref['with'], 'ls with', rtol=1e-4)
    got = metafopy.confint(ls, alpha=1, xlim=(-0.1, 0.1)).random
    assert got is not None
    refutil.close(got.to_numpy().ravel(), ref['without'], 'ls without', rtol=1e-4)


def test_confint_parm_warns_and_digits(bcg: pandas.DataFrame) -> None:
    res = metafopy.rma('yi', 'vi', data=bcg)
    with pytest.warns(UserWarning, match="'parm' .currently. ignored"):
        metafopy.confint(res, parm=1)
    mh = metafopy.rma_mh(ai=bcg.tpos, bi=bcg.tneg, ci=bcg.cpos, di=bcg.cneg, measure='OR')
    with pytest.warns(UserWarning, match="'parm' .currently. ignored"):
        metafopy.confint(mh, parm=1)
    assert metafopy.confint(mh, digits=2).digits == 2


def test_permutest_ls_permci_warns(bcg: pandas.DataFrame) -> None:
    ls = metafopy.rma('yi', 'vi', mods='~ ablat', scale='~ ablat', data=bcg)
    with pytest.warns(UserWarning, match='Permutation-based CIs for location-scale models'):
        metafopy.permutest(ls, iter=20, seed=1, permci=True)


def test_rma_knha_and_dfs(bcg: pandas.DataFrame) -> None:
    k1 = metafopy.rma('yi', 'vi', data=bcg, knha=True)
    ref = REF['knha']
    assert (k1.test, k1.ddf) == (ref['test'], ref['ddf'])
    refutil.close(k1.pval, ref['pval'], 'knha pval')
    refutil.close([k1.ci_lb[0], k1.ci_ub[0]], ref['ci'], 'knha ci')
    assert metafopy.rma('yi', 'vi', data=bcg, test='knha', knha=False).test == REF['knha_false']['test']
    d5 = metafopy.rma('yi', 'vi', mods='~ ablat', data=bcg, test='t', dfs=5)
    ref = REF['dfs']
    assert d5.ddf == ref['ddf']
    refutil.close(d5.pval, ref['pval'], 'dfs pval')
    refutil.close([d5.QMp], [ref['QMp']], 'dfs QMp')
    refutil.close(d5.ci_lb, ref['ci_lb'], 'dfs ci')
    ls = metafopy.rma('yi', 'vi', mods='~ ablat', scale='~ ablat', data=bcg, test='t', dfs=5)
    assert (ls.ddf, ls.ddf_alpha) == (REF['dfs_ls']['ddf'], REF['dfs_ls']['ddf_alpha'])
    refutil.close(ls.pval, REF['dfs_ls']['pval'], 'dfs ls pval', rtol=1e-4)


def test_rma_i2def(bcg: pandas.DataFrame) -> None:
    res = metafopy.rma('yi', 'vi', data=bcg, i2def='2')
    refutil.close([res.I2, res.H2], [REF['i2def']['I2'], REF['i2def']['H2']], 'i2def')
    res = metafopy.rma('yi', 'vi', mods='~ ablat', data=bcg, i2def='2')
    refutil.close([res.I2, res.H2], [REF['i2def_mods']['I2'], REF['i2def_mods']['H2']], 'i2def mods')
    with pytest.raises(metafopy.errors.InputError):
        metafopy.rma('yi', 'vi', data=bcg, i2def='3')  # type: ignore[call-overload]


@pytest.mark.parametrize('r2def', ['1', '1v', '2', '2v', '3', '3v', '4', '4w', '5', '6', '7', '8'])
def test_rma_r2def(bcg: pandas.DataFrame, r2def: str) -> None:
    ref = REF['r2def'][r2def]
    for method, key in (('REML', 'reml'), ('ML', 'ml')):
        res = metafopy.rma('yi', 'vi', mods='~ ablat', data=bcg, method=method, r2def=r2def)
        assert res.R2 is not None
        refutil.close([res.R2], [ref[key]], f'r2def {r2def} {method}', rtol=1e-6)


def test_rma_skipr2_and_abbrev(bcg: pandas.DataFrame) -> None:
    assert metafopy.rma('yi', 'vi', mods='~ ablat', data=bcg, skipr2=True).R2 is None
    ab = metafopy.rma('yi', 'vi', mods='~ factor(alloc) + I(ablat^2)', data=bcg, abbrev=True)
    assert list(ab.coef_names) == REF['abbrev']
    ls = metafopy.rma('yi', 'vi', mods='~ poly(ablat, degree=2, raw=True)', scale='~ factor(alloc)', data=bcg, abbrev=True)
    assert list(ls.coef_names) == REF['abbrev_ls']['beta']
    assert list(ls.alpha_names) == REF['abbrev_ls']['alpha']


def test_regplot_at_lab_and_box_lty(bcg: pandas.DataFrame) -> None:
    matplotlib.use('Agg')
    res = metafopy.rma('yi', 'vi', mods='~ ablat', data=bcg)
    fig, ax = matplotlib.pyplot.subplots()
    metafopy.regplot(res, mod='ablat', at=[-1.0, -0.5, 0.0], at_lab=['low', 'mid', 'high'], legend=True, box_lty='dashed', ax=ax)
    assert [t.get_text() for t in ax.get_yticklabels()] == ['low', 'mid', 'high']
    legend = ax.get_legend()
    assert legend is not None
    assert legend.get_frame().get_linestyle() == '--'
    with pytest.raises(metafopy.errors.InputError):
        metafopy.regplot(res, mod='ablat', at=[-1.0, -0.5, 0.0], at_lab=['low'], ax=matplotlib.pyplot.subplots()[1])
    matplotlib.pyplot.close('all')
    del fig


def correlation_list() -> dict[str, pandas.DataFrame]:
    r1 = pandas.DataFrame([[1, 0.4, 0.3], [0.4, 1, 0.2], [0.3, 0.2, 1]], index=['a', 'b', 'c'], columns=['a', 'b', 'c'])
    r2 = pandas.DataFrame([[1, 0.5], [0.5, 1]], index=['a', 'b'], columns=['a', 'b'])
    return {'s1': r1, 's2': r2}


def test_rcalc_rowid_vnames_noid() -> None:
    rl = correlation_list()
    base = metafopy.rcalc(rl, ni=[40, 60])
    assert isinstance(base, metafopy.methods.rcalc.RCalcResult)
    assert base.dat is not None
    numpy.testing.assert_allclose(base.dat['yi'], [r['yi'] for r in REF['rcalc_base']['dat']], rtol=1e-10)
    numpy.testing.assert_allclose(numpy.asarray(base.V), numpy.array(REF['rcalc_base']['V']), rtol=1e-8, atol=1e-12)
    ids = ['s2.a.s2.b', 's1.b.s1.c', 's1.a.s1.b', 's1.a.s1.c']
    res = metafopy.rcalc(rl, ni=[40, 60], rowid=ids, vnames=['ri', 'x', 'y', 'study'], noid=True)
    assert isinstance(res, metafopy.methods.rcalc.RCalcResult)
    assert res.dat is not None
    assert list(res.dat.columns) == REF['rcalc_args']['names']
    numpy.testing.assert_allclose(res.dat['yi'], [r['yi'] for r in REF['rcalc_args']['dat']], rtol=1e-10)
    assert list(res.dat['x.y']) == [r['x.y'] for r in REF['rcalc_args']['dat']]
    numpy.testing.assert_allclose(numpy.asarray(res.V), numpy.array(REF['rcalc_args']['V']), rtol=1e-8, atol=1e-12)


def test_pairmat_fixed(bcg: pandas.DataFrame) -> None:
    res = metafopy.rma('yi', 'vi', mods='~ ablat + year + alloc', data=bcg)
    pm = metafopy.pairmat(res, btt='alloc', fixed=True)
    assert list(pm.index) == [REF['pairmat_fixed']['rows']]
    numpy.testing.assert_allclose(pm.to_numpy(), numpy.array(REF['pairmat_fixed']['values']), atol=1e-12)


def test_funnel_transf_warns_and_lb_accepted(bcg: pandas.DataFrame) -> None:
    matplotlib.use('Agg')
    res = metafopy.rma('yi', 'vi', data=bcg)
    with pytest.warns(UserWarning, match="does not have a 'transf' argument"):
        metafopy.funnel(res, transf=math.exp, ax=matplotlib.pyplot.subplots()[1])
    matplotlib.pyplot.close('all')
    a = metafopy.gosh(res, 50, seed=1)
    b = metafopy.gosh(res, 50, seed=1, LB=True)
    numpy.testing.assert_allclose(a.res['estimate'], b.res['estimate'])
    ref = metafopy.vif(metafopy.rma('yi', 'vi', mods='~ ablat + year', data=bcg))
    again = metafopy.vif(metafopy.rma('yi', 'vi', mods='~ ablat + year', data=bcg), joina=['x'], LB=True)
    numpy.testing.assert_allclose([e.vif for e in ref.vif], [e.vif for e in again.vif])


def test_addpoly_cr_aliases(bcg: pandas.DataFrame) -> None:
    matplotlib.use('Agg')
    res = metafopy.rma('yi', 'vi', data=bcg)
    fig, ax = matplotlib.pyplot.subplots()
    metafopy.forest(res, ax=ax)
    metafopy.addpoly(0.1, ci_lb=0.0, ci_ub=0.2, pi_lb=-0.1, pi_ub=0.3, ax=ax)
    n = len(ax.lines) + len(ax.patches)
    assert n > 0
    metafopy.addpoly(0.1, ci_lb=0.0, ci_ub=0.2, cr_lb=-0.1, cr_ub=0.3, ax=ax)
    assert len(ax.lines) + len(ax.patches) > n
    matplotlib.pyplot.close(fig)
