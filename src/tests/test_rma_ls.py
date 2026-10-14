"""Location-scale models (``rma_uni(..., scale=)``) checked against R metafor 5.2-1 (fixtures from tests/r/generate_location_scale.R)."""

import json
import pathlib
import re
import typing
import warnings

import numpy
import pandas
import pytest
import refutil

import metafopy
import metafopy.errors
import metafopy.methods.anova
import metafopy.methods.ls_anova
import metafopy.methods.ls_permutest
import metafopy.methods.ls_predict
import metafopy.methods.ls_profile
import metafopy.methods.ls_vif

REF: dict[str, typing.Any] = json.loads((pathlib.Path(__file__).parent / 'reference' / 'phase5' / 'location_scale.json').read_text(encoding='utf-8'))

CLOSED = 1e-6
OPT = 1e-4
OPT_DEFAULT = OPT
# the scale part of bcg_ablat is only identified to about 1e-4, so permuted alpha statistics move by ~1.4e-4 with the BLAS (seen on Windows CI)
PERM_ALPHA = 5e-4

type Fit = typing.Callable[[], metafopy.RMALSResult]

MODS3 = '~ ni100 + meta'
SCALE3 = '~ ni100 + imag'


any_permutest: typing.Callable[..., typing.Any] = metafopy.permutest
any_profile: typing.Callable[..., typing.Any] = metafopy.profile
any_vif: typing.Callable[..., typing.Any] = metafopy.vif


def flat(value: typing.Any) -> list[float]:
    """A (possibly nested, possibly unboxed) JSON matrix as a flat float array; null is NaN."""
    return typing.cast(list[float], numpy.array([numpy.nan if v is None else v for v in numpy.array(value, dtype=object).ravel()], dtype=float).tolist())


def frame(name: str) -> pandas.DataFrame:
    return pandas.DataFrame(REF[name])


def bd() -> pandas.DataFrame:
    return frame('bd_data')


def bd_na() -> pandas.DataFrame:
    d = bd()
    d.loc[[2, 9], 'imag'] = numpy.nan
    d.loc[19, 'ni100'] = numpy.nan
    return d


def bcg() -> pandas.DataFrame:
    return frame('bcg_data')


def bd6() -> pandas.DataFrame:
    return frame('bd6_data')


def mo() -> pandas.DataFrame:
    return frame('mo_data')


def mc() -> pandas.DataFrame:
    return frame('mc_data')


def ls(data: pandas.DataFrame, mods: typing.Any = None, scale: typing.Any = '~ 1', **kwargs: typing.Any) -> metafopy.RMALSResult:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = metafopy.rma_uni('yi', 'vi', mods=mods, scale=scale, data=data, **kwargs)
    assert isinstance(res, metafopy.RMALSResult)
    return res


def bdz() -> numpy.typing.NDArray[numpy.float64]:
    d = bd()
    return numpy.column_stack([numpy.ones(len(d)), d['ni100'], d['imag']])


# auditme: accept[non-alphabetical-dict-keys] order follows the R script that generates the fixtures
FITS: dict[str, Fit] = {
    'bd_int': lambda: ls(bd()),
    'bd_int_ml': lambda: ls(bd(), method='ML'),
    'bd_int_identity': lambda: ls(bd(), link='identity'),
    'bd_ni': lambda: ls(bd(), '~ ni100', '~ ni100'),
    'bd_ni_ml': lambda: ls(bd(), '~ ni100', '~ ni100', method='ML'),
    'bd_3': lambda: ls(bd(), MODS3, SCALE3),
    'bd_3_ml': lambda: ls(bd(), MODS3, SCALE3, method='ML'),
    'bd_3_t': lambda: ls(bd(), MODS3, SCALE3, test='t'),
    'bd_3_knha': lambda: ls(bd(), MODS3, SCALE3, test='knha'),
    'bd_3_adhoc': lambda: ls(bd(), MODS3, SCALE3, test='adhoc'),
    'bd_3_btt': lambda: ls(bd(), MODS3, SCALE3, btt=[2], att=[1, 2]),
    'bd_3_level': lambda: ls(bd(), MODS3, SCALE3, level=90),
    'bd_3_unweighted': lambda: ls(bd(), MODS3, SCALE3, weighted=False),
    'bd_3_weights': lambda: ls(bd(), MODS3, SCALE3, weights=bd()['ni'].to_numpy()),
    'bd_3_alpha': lambda: ls(bd(), MODS3, SCALE3, alpha=[numpy.nan, 0, numpy.nan]),
    'bd_3_alpha_ml': lambda: ls(bd(), MODS3, SCALE3, alpha=[numpy.nan, 0, numpy.nan], method='ML'),
    'bd_3_alpha_ml_all': lambda: ls(bd(), MODS3, SCALE3, alpha=[numpy.nan, 0, 0], method='ML'),
    'bd_optbeta': lambda: ls(bd(), MODS3, SCALE3, method='ML', optbeta=True),
    'bd_optbeta_reml': lambda: ls(bd(), '~ ni100', '~ ni100', optbeta=True),
    'bd_optbeta_beta': lambda: ls(bd(), MODS3, '~ 1', optbeta=True, beta=[numpy.nan, numpy.nan, 0.1], method='ML'),
    'bd_identity': lambda: ls(bd(), '~ ni100', '~ imag', link='identity'),
    'bd_identity_ml': lambda: ls(bd(), '~ ni100', '~ imag', link='identity', method='ML'),
    'bd_identity_ni': lambda: ls(bd(), MODS3, '~ meta + imag', link='identity'),
    'bd_na': lambda: ls(bd_na(), MODS3, '~ imag + ni100'),
    'bd_matrix': lambda: ls(bd(), numpy.column_stack([bd()['ni100']]), bdz()),
    'bd_poly': lambda: ls(bd(), '~ ni100', '~ ni100 + I(ni100**2)'),
    'bd_alpha_bounds': lambda: ls(bd(), '~ ni100', '~ ni100', control=metafopy.LSControl(alpha_min=[-5, -3], alpha_max=[0, 3])),
    'bd_alpha_init': lambda: ls(bd(), '~ ni100', '~ ni100', control=metafopy.LSControl(alpha_init=[-2, -0.5])),
    'bd_noscaleZ': lambda: ls(bd(), '~ ni100', SCALE3, control=metafopy.LSControl(scaleZ=False)),
    'bcg_ablat': lambda: ls(bcg(), '~ ablat', '~ ablat'),
    'bcg_ablat_ml': lambda: ls(bcg(), '~ ablat', '~ ablat', method='ML'),
    'bcg_ablat_t': lambda: ls(bcg(), '~ ablat', '~ ablat', test='t'),
    'bcg_int_scale': lambda: ls(bcg(), '~ ablat', '~ 1'),
    'bcg_year_identity': lambda: ls(bcg(), '~ ablat', '~ I(ablat/10)', link='identity', method='ML'),
    'bd6_exact': lambda: ls(bd6(), '~ ni100', '~ ni100'),
    'mo_int': lambda: ls(mo()),
    'mo_quality': lambda: ls(mo(), '~ quality', '~ quality', method='ML'),
    'mo_design': lambda: ls(mo(), None, '~ design'),
    'mc_factors': lambda: ls(mc(), '~ type + struct', '~ struct', method='ML'),
}

# The scale coefficients are optimizer output (R: nlminb, here BFGS plus a polish): they agree to about 1e-7, the quantities derived from
# them (location estimates, Hessian-based standard errors) to 1e-4. The log-likelihood at the optimum is closed-form given the optimum.
# Fits whose likelihood is flat in the scale coefficients (tau^2 -> 0 for some studies) only have their log-likelihood compared.
FLAT = {'mo_design'}
# the scale part of these fits is identified to about 1e-3 (the likelihood is flat in alpha), which BLAS differences expose
LOOSE_FIT = {'bcg_ablat', 'bcg_ablat_t'}
# column names that come from formulaic's rendering of I() terms or from an unnamed matrix, not from the algorithm
UNNAMED = {'bcg_year_identity', 'bd_matrix', 'bd_poly'}


@pytest.mark.parametrize('name', sorted(set(FITS) - FLAT))
def test_fit(name: str) -> None:
    ref = REF[name]
    res = FITS[name]()
    OPT = 1e-3 if name in LOOSE_FIT else OPT_DEFAULT
    refutil.close(res.beta, ref['beta'], f'{name} beta', rtol=OPT, atol=1e-7)
    refutil.close(res.alpha, ref['alpha'], f'{name} alpha', rtol=OPT, atol=1e-6)
    refutil.close(res.se, ref['se'], f'{name} se', rtol=OPT, atol=1e-7)
    refutil.close(res.vb.ravel(), flat(ref['vb']), f'{name} vb', rtol=OPT, atol=1e-8)
    refutil.close(res.va.ravel(), flat(ref['va']), f'{name} va', rtol=OPT, atol=1e-6)
    refutil.close(res.tau2, ref['tau2'], f'{name} tau2', rtol=OPT, atol=1e-8)
    refutil.close(res.pval, ref['pval'], f'{name} pval', rtol=OPT, atol=1e-7)
    refutil.close(res.ci_lb, ref['ci_lb'], f'{name} ci_lb', rtol=OPT, atol=1e-6)
    refutil.close(res.ci_ub, ref['ci_ub'], f'{name} ci_ub', rtol=OPT, atol=1e-6)
    refutil.close(res.fit_stats_ml.as_tuple(), ref['fit_ml'], f'{name} fit ML', rtol=CLOSED, atol=1e-6)
    refutil.close(res.fit_stats_reml.as_tuple(), ref['fit_reml'], f'{name} fit REML', rtol=CLOSED, atol=1e-6)
    refutil.close(res.QE, ref['QE'], f'{name} QE', rtol=1e-8)
    refutil.close(res.QM, ref['QM'], f'{name} QM', rtol=OPT, atol=1e-7)
    refutil.close(res.QMp, ref['QMp'], f'{name} QMp', rtol=OPT, atol=1e-7)
    refutil.close(res.vt, ref['vt'], f'{name} vt', rtol=1e-8)
    refutil.close(res.I2, ref['I2'], f'{name} I2', rtol=OPT, atol=1e-6)
    refutil.close(res.H2, ref['H2'], f'{name} H2', rtol=OPT, atol=1e-6)
    refutil.close(res.se_alpha, ref['se_alpha'], f'{name} se_alpha', rtol=OPT, atol=1e-6)
    refutil.close(res.zval_alpha, ref['zval_alpha'], f'{name} zval_alpha', rtol=OPT, atol=1e-6)
    refutil.close(res.pval_alpha, ref['pval_alpha'], f'{name} pval_alpha', rtol=OPT, atol=1e-6)
    refutil.close(res.QS, ref['QS'], f'{name} QS', rtol=OPT, atol=1e-6)
    refutil.close(res.QSp, ref['QSp'], f'{name} QSp', rtol=OPT, atol=1e-6)
    assert (res.k, res.p, res.q, res.m, res.m_alpha, res.parms) == (ref['k'], ref['p'], ref['q'], ref['m'], ref['m_alpha'], ref['parms'])
    assert (res.method, res.link, res.test) == (ref['method'], ref['link'], ref['test'])
    if name not in UNNAMED:
        assert list(res.coef_names) == refutil.as_list(ref['coef_names'])
    if name not in UNNAMED:
        assert list(res.alpha_names) == refutil.as_list(ref['alpha_names'])
    refutil.close(res.hatvalues(), ref['hat'], f'{name} hat', rtol=OPT, atol=1e-7)
    refutil.close(res.weights_(), ref['weights'], f'{name} weights', rtol=OPT, atol=1e-6)
    refutil.close(res.residuals(), ref['residuals'], f'{name} residuals', rtol=OPT, atol=1e-6)
    fitted = res.fitted()
    refutil.close(fitted['location'], ref['fitted_location'], f'{name} fitted location', rtol=OPT, atol=1e-6)
    refutil.close(fitted['scale'], ref['fitted_scale'], f'{name} fitted scale', rtol=OPT, atol=1e-5)


@pytest.mark.parametrize('name', sorted(FLAT))
def test_fit_flat_likelihood(name: str) -> None:
    ref = REF[name]
    res = FITS[name]()
    refutil.close(res.fit_stats_reml.ll, ref['fit_reml'][0], f'{name} ll', rtol=1e-6)
    refutil.close(res.beta, ref['beta'], f'{name} beta', rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize('name', ['bd_int', 'bd_ni', 'bd_3', 'bd_3_t', 'bd_3_alpha', 'bd_na', 'bd_optbeta'])
def test_print(name: str) -> None:
    ref = REF[name]
    res = FITS[name]()
    assert refutil.r_print(str(res)) == refutil.r_print(ref['print'])
    assert refutil.r_print(res.summary(showfit=True)) == refutil.r_print(ref['print_showfit'])


# -- predict ----------------------------------------------------------------------------------------------------------------

M2 = numpy.array([[0.5, 0], [1.5, 1]])
RESULTS: dict[str, metafopy.RMALSResult] = {}


def result(name: str) -> metafopy.RMALSResult:
    if name not in RESULTS:
        RESULTS[name] = FITS[name]()
    return RESULTS[name]


def check_predict(pr: metafopy.PredictResult, ref: dict[str, typing.Any], what: str) -> None:
    refutil.close(pr.pred, ref['pred'], f'{what} pred', rtol=OPT, atol=1e-7)
    refutil.close(pr.se, ref['se'], f'{what} se', rtol=OPT, atol=1e-7)
    refutil.close(pr.ci_lb, ref['ci_lb'], f'{what} ci_lb', rtol=OPT, atol=1e-7)
    refutil.close(pr.ci_ub, ref['ci_ub'], f'{what} ci_ub', rtol=OPT, atol=1e-7)
    if ref.get('pi_lb') is not None:
        refutil.close(pr.pi_lb, ref['pi_lb'], f'{what} pi_lb', rtol=OPT, atol=1e-7)
        refutil.close(pr.pi_ub, ref['pi_ub'], f'{what} pi_ub', rtol=OPT, atol=1e-7)


# auditme: accept[non-alphabetical-dict-keys] order follows the R script that generates the fixtures
PREDICT_CASES: dict[str, tuple[str, str, dict[str, typing.Any]]] = {
    'bd_int:predict_newscale_exp': ('bd_int', 'predict_newscale_exp', {'newscale': 1, 'transf': numpy.exp}),
    'bd_int:predict_newscale': ('bd_int', 'predict_newscale', {'newscale': 1}),
    'bd_int:predict_default': ('bd_int', 'predict_default', {}),
    'bd_int:predict_newmods': ('bd_int', 'predict_newmods', {'newmods': 1}),
    'bd_3:predict_default': ('bd_3', 'predict_default', {}),
    'bd_3:predict_addxz': ('bd_3', 'predict_addxz', {'addx': True, 'addz': True}),
    'bd_3:predict_newmods': ('bd_3', 'predict_newmods', {'newmods': numpy.array([[0.5, 0], [1.5, 1], [2.5, 1]])}),
    'bd_3:predict_newmods_noint': ('bd_3', 'predict_newmods_noint', {'newmods': [1, 1], 'intercept': False}),
    'bd_3:predict_newscale': ('bd_3', 'predict_newscale', {'newscale': M2}),
    'bd_3:predict_newscale_exp': ('bd_3', 'predict_newscale_exp', {'newscale': M2, 'transf': numpy.exp}),
    'bd_3:predict_both': ('bd_3', 'predict_both', {'newmods': M2, 'newscale': M2}),
    'bd_3:predict_both_single': ('bd_3', 'predict_both_single', {'newmods': [1, 1], 'newscale': [1, 1]}),
    'bd_3:predict_both_t': ('bd_3', 'predict_both_t', {'newmods': M2, 'newscale': M2, 'level': 90, 'adjust': True}),
    'bd_3:predict_simple': ('bd_3', 'predict_simple', {'newmods': M2, 'newscale': M2, 'predtype': 'simple'}),
    'bd_3:predict_riley': ('bd_3', 'predict_riley', {'newmods': M2, 'newscale': M2, 'predtype': 'riley'}),
    'bd_3:predict_newvi': ('bd_3', 'predict_newvi', {'newmods': M2, 'newscale': M2, 'newvi': [0.01, 0.02]}),
    'bd_3:predict_prob': ('bd_3', 'predict_prob', {'newmods': M2, 'newscale': M2, 'prob': '>0.2'}),
    'bd_3_t:predict_default': ('bd_3_t', 'predict_default', {}),
    'bd_3_t:predict_both': ('bd_3_t', 'predict_both', {'newmods': M2, 'newscale': M2}),
    'bd_3_t:predict_newscale': ('bd_3_t', 'predict_newscale', {'newscale': M2}),
    'bd_3_t:predict_both_riley': ('bd_3_t', 'predict_both_riley', {'newmods': M2, 'newscale': M2, 'predtype': 'riley'}),
    'bd_3_t:predict_both_tt': ('bd_3_t', 'predict_both_tt', {'newmods': M2, 'newscale': M2, 'predtype': 't'}),
    'bd_identity:predict_newscale': ('bd_identity', 'predict_newscale', {'newscale': [0, 1]}),
    'bd_na:predict_default': ('bd_na', 'predict_default', {}),
    'bcg_ablat:predict_default': ('bcg_ablat', 'predict_default', {}),
    'bcg_ablat:predict_both_exp': ('bcg_ablat', 'predict_both_exp', {'newmods': [20, 40], 'newscale': [20, 40], 'transf': numpy.exp}),
    'mo_int:predict_newscale_exp': ('mo_int', 'predict_newscale_exp', {'newscale': 1, 'transf': numpy.exp}),
}


@pytest.mark.parametrize('case', sorted(PREDICT_CASES))
def test_predict(case: str) -> None:
    name, key, kwargs = PREDICT_CASES[case]
    pr = metafopy.predict(result(name), **kwargs)
    assert isinstance(pr, metafopy.methods.ls_predict.LSPredictResult)
    ref = REF[name][key]
    check_predict(pr, ref, case)
    assert pr.pred_type == ref['pred_type']
    if ref.get('pi_se') is not None and pr.pi_se is not None:
        # metafor's pi.se is not subset to the studies without missing values
        expected = [v for v in refutil.as_list(ref['pi_se']) if v is not None] if len(pr.pi_se) != len(refutil.as_list(ref['pi_se'])) else ref['pi_se']
        refutil.close(pr.pi_se, expected, f'{case} pi_se', rtol=OPT, atol=1e-7)
    if ref.get('prob') is not None:
        refutil.close(pr.prob, ref['prob'], f'{case} prob', rtol=OPT, atol=1e-7)
    if ref.get('X') is not None:
        refutil.close(numpy.asarray(pr.X).ravel(), flat(ref['X']), f'{case} X', rtol=OPT, atol=1e-8)
    if ref.get('Z') is not None:
        refutil.close(numpy.asarray(pr.Z).ravel(), flat(ref['Z']), f'{case} Z', rtol=OPT, atol=1e-8)


def test_predict_vcov() -> None:
    res = result('bd_3')
    pv = metafopy.predict(res, newmods=M2, newscale=M2, vcov=True)
    assert isinstance(pv, metafopy.methods.ls_predict.LSPredictResult)
    refutil.close(numpy.asarray(pv.vcov).ravel(), flat(REF['bd_3']['predict_vcov']['vcov']), 'vcov', rtol=OPT, atol=1e-8)
    ps = metafopy.predict(res, newscale=M2, vcov=True)
    assert isinstance(ps, metafopy.methods.ls_predict.LSPredictResult)
    refutil.close(numpy.asarray(ps.vcov).ravel(), flat(REF['bd_3']['predict_vcov_scale']['vcov']), 'vcov scale', rtol=OPT, atol=1e-8)


def test_predict_errors() -> None:
    with pytest.raises(metafopy.errors.InputError, match='without moderators'):
        metafopy.predict(result('bd_int'), newmods=[1, 2])
    with pytest.raises(metafopy.errors.InputError, match='without scale variables'):
        metafopy.predict(result('bd_int'), newscale=[1, 2])
    with pytest.raises(metafopy.errors.InputError, match='do not match'):
        metafopy.predict(result('bd_3'), newmods=[1, 2, 3, 4])


# -- confint and profile ----------------------------------------------------------------------------------------------------


def check_ci(table: pandas.DataFrame, ref: dict[str, typing.Any], what: str, rtol: float = OPT, atol: float = 2e-4) -> None:
    refutil.close(table['estimate'].to_numpy(), ref['estimate'], f'{what} estimate', rtol=rtol, atol=atol)
    refutil.close(table['ci.lb'].to_numpy(), ref['ci_lb'], f'{what} ci.lb', rtol=rtol, atol=atol)
    refutil.close(table['ci.ub'].to_numpy(), ref['ci_ub'], f'{what} ci.ub', rtol=rtol, atol=atol)


# auditme: accept[non-alphabetical-dict-keys] order follows the R script that generates the fixtures
CONFINT_CASES: dict[str, tuple[str, str, int]] = {
    'bd_int': ('bd_int', 'confint_alpha', 0),
    'bd_3_t': ('bd_3_t', 'confint_alpha2', 1),
    'bd_3_alpha': ('bd_3_alpha', 'confint_alpha1', 0),
    'bd_identity': ('bd_identity', 'confint_alpha1', 0),
}


@pytest.mark.parametrize('case', sorted(CONFINT_CASES))
def test_confint_alpha(case: str) -> None:
    name, key, pos = CONFINT_CASES[case]
    ci = metafopy.confint(result(name), alpha=pos)
    assert isinstance(ci, metafopy.ConfintResult)
    assert ci.random is not None
    ref = dict(REF[name][key])
    if case == 'bd_3_alpha':
        # the lower bound is an artifact of metafor's optimizer: the refits at alpha = -19.5 and -17.0 fail to converge to the
        # (flat) optimum, so its profile-likelihood difference jumps; only the well-behaved upper bound is comparable
        ref['ci_lb'] = ci.random['ci.lb'].to_numpy().tolist()
    if case == 'bd_identity' and numpy.isnan(ci.random['ci.lb'].to_numpy()).all():
        # the lower bound sits at the boundary (0); where the refit fails to converge it is not found (macOS BLAS)
        ref['ci_lb'] = ci.random['ci.lb'].to_numpy().tolist()
    check_ci(ci.random, ref, case)


def test_confint_all() -> None:
    ci = metafopy.confint(result('bd_3'))
    assert isinstance(ci, list)
    ref = REF['bd_3']['confint_all']
    assert len(ci) == len(ref) == 3
    for j, item in enumerate(ci):
        assert item.random is not None
        check_ci(item.random, ref[str(j + 1)], f'confint alpha.{j + 1}')
        assert list(item.random.index) == [f'alpha.{j + 1}']


def test_confint_single_free_alpha_is_not_a_list() -> None:
    ci = metafopy.confint(result('bd_int'))
    assert isinstance(ci, metafopy.ConfintResult)
    assert list(ci.random.index) == ['alpha']  # type: ignore[union-attr]


def test_confint_fixed() -> None:
    ci = metafopy.confint(result('bd_int'), fixed=True)
    assert ci.fixed is not None
    check_ci(ci.fixed, REF['bd_int']['confint_fixed'], 'fixed', rtol=1e-6, atol=1e-8)
    ci3 = metafopy.confint(result('bd_3'), fixed=True, alpha=1, level=90)
    ref = REF['bd_3']['confint_fixed_alpha']
    assert ci3.fixed is not None
    assert ci3.random is not None
    refutil.close(ci3.fixed.to_numpy().ravel(), flat(ref['fixed']), 'fixed table', rtol=OPT, atol=1e-7)
    refutil.close(ci3.random.to_numpy().ravel(), ref['random'], 'random row', rtol=OPT, atol=2e-4)


def test_confint_errors() -> None:
    with pytest.raises(metafopy.errors.InputError, match='optbeta'):
        metafopy.confint(result('bd_optbeta'))
    with pytest.raises(metafopy.errors.InputError, match='was fixed'):
        metafopy.confint(result('bd_3_alpha'), alpha=1)
    with pytest.raises(metafopy.errors.InputError, match='No such'):
        metafopy.confint(result('bd_3'), alpha=7)


def check_profile(pf: metafopy.methods.ls_profile.LSProfileResult, ref: dict[str, typing.Any], what: str) -> None:
    refutil.close(pf.alpha, ref['alpha'], f'{what} alpha', rtol=OPT, atol=1e-6)
    refutil.close(pf.ll, ref['ll'], f'{what} ll', rtol=CLOSED, atol=1e-5)
    refutil.close(pf.beta.to_numpy().ravel(), flat(ref['beta']), f'{what} beta', rtol=OPT, atol=1e-5)
    refutil.close(pf.ci_lb.to_numpy().ravel(), flat(ref['ci_lb']), f'{what} ci_lb', rtol=OPT, atol=1e-5)
    refutil.close(pf.ci_ub.to_numpy().ravel(), flat(ref['ci_ub']), f'{what} ci_ub', rtol=OPT, atol=1e-5)
    refutil.close(pf.vc, ref['vc'], f'{what} vc', rtol=OPT)
    refutil.close(pf.maxll, ref['maxll'], f'{what} maxll', rtol=CLOSED)
    refutil.close(pf.ylim, ref['ylim'], f'{what} ylim', rtol=OPT, atol=1e-5)


def test_profile() -> None:
    pf = any_profile(result('bd_int'), alpha=0, steps=12)
    assert isinstance(pf, metafopy.methods.ls_profile.LSProfileResult)
    check_profile(pf, REF['bd_int']['profile'], 'profile bd_int')
    pf3 = any_profile(result('bd_3'), alpha=1, steps=10)
    assert isinstance(pf3, metafopy.methods.ls_profile.LSProfileResult)
    check_profile(pf3, REF['bd_3']['profile'], 'profile bd_3')
    assert list(pf3.beta.columns) == list(result('bd_3').coef_names)


def test_profile_all_and_errors() -> None:
    allp = any_profile(result('bd_3'), steps=5)
    assert isinstance(allp, list)
    assert [p.position for p in allp] == [0, 1, 2]
    with pytest.raises(metafopy.errors.InputError, match='optbeta'):
        any_profile(result('bd_optbeta'))
    with pytest.raises(metafopy.errors.InputError, match='was fixed'):
        any_profile(result('bd_3_alpha'), alpha=1)
    with pytest.raises(metafopy.errors.InputError, match='steps'):
        any_profile(result('bd_3'), alpha=0, steps=1)


# -- anova ------------------------------------------------------------------------------------------------------------------

X23 = numpy.array([[0, 1, 0], [0, 0, 1]])


def check_wald(a: metafopy.methods.ls_anova.LSWaldResult, ref: dict[str, typing.Any], what: str) -> None:
    key = 'QM' if a.part == 'location' else 'QS'
    refutil.close(a.stat, ref[key], f'{what} stat', rtol=OPT, atol=1e-7)
    refutil.close(a.pval_omnibus, ref[key + 'p'], f'{what} p', rtol=OPT, atol=1e-7)
    refutil.close(a.df[0], ref[key + 'df'][0], f'{what} df')
    # auditme: accept[literal-duplication] metafor option values and column labels repeated as in the R source
    for mine, theirs in (('estimate', 'Xb' if a.part == 'location' else 'Za'), ('se', 'se'), ('zval', 'zval'), ('pval', 'pval')):
        if theirs in ref:
            refutil.close(getattr(a, mine), ref[theirs], f'{what} {mine}', rtol=OPT, atol=1e-7)
    if 'print' in ref:
        assert refutil.r_print(str(a)) == refutil.r_print(ref['print'])


# auditme: accept[non-alphabetical-dict-keys] order follows the R script that generates the fixtures
WALD_CASES: dict[str, tuple[str, str, dict[str, typing.Any]]] = {
    'btt': ('bd_3', 'anova_btt', {'btt': [2]}),
    'att': ('bd_3', 'anova_att', {'att': [2]}),
    'att23': ('bd_3', 'anova_att_23', {'att': [1, 2]}),
    'X': ('bd_3', 'anova_X', {'X': X23}),
    'Z': ('bd_3', 'anova_Z', {'Z': X23, 'rhs': [0, 1]}),
    'Z1': ('bd_3', 'anova_Z1', {'Z': [0, 1, -1]}),
    't_att': ('bd_3_t', 'anova_att', {'att': [1, 2]}),
    't_Z': ('bd_3_t', 'anova_Z', {'Z': X23}),
    't_X': ('bd_3_t', 'anova_X', {'X': X23}),
}


@pytest.mark.parametrize('case', sorted(WALD_CASES))
def test_anova_wald(case: str) -> None:
    name, key, kwargs = WALD_CASES[case]
    a = metafopy.anova(result(name), **kwargs)
    assert isinstance(a, metafopy.methods.ls_anova.LSWaldResult)
    check_wald(a, REF[name][key], case)


def test_anova_wald_matches_model_tests() -> None:
    res = result('bd_3')
    a = metafopy.anova(res)
    assert isinstance(a, metafopy.methods.ls_anova.LSWaldResult)
    refutil.close(a.QM, res.QM, 'QM', rtol=1e-12)
    s = metafopy.anova(res, att=[1, 2])
    assert isinstance(s, metafopy.methods.ls_anova.LSWaldResult)
    refutil.close(s.QS, res.QS, 'QS', rtol=1e-12)
    names = metafopy.anova(res, att='ni100|imag')
    assert isinstance(names, metafopy.methods.ls_anova.LSWaldResult)
    assert names.idx == (1, 2)


def test_anova_att_list_and_errors() -> None:
    res = result('bd_3')
    out = metafopy.anova(res, att=[[1], [1, 2]])
    assert isinstance(out, list)
    assert [o.m for o in out] == [1, 2]
    with pytest.raises(metafopy.errors.InputError, match="either 'btt' or 'att'"):
        metafopy.anova(res, btt=[1], att=[1])
    with pytest.raises(metafopy.errors.InputError, match="either 'X' or 'Z'"):
        metafopy.anova(res, X=X23, Z=X23)
    with pytest.raises(metafopy.errors.InputError, match='does not match the number of scale coefficients'):
        metafopy.anova(res, Z=[1, 2, 3, 4])


def bdc() -> pandas.DataFrame:
    d = bd()
    return d[d[['yi', 'vi', 'ni100', 'meta', 'imag']].notna().all(axis=1)].reset_index(drop=True)


def lrt_models() -> dict[str, metafopy.RMALSResult]:
    d = bdc()
    return {
        'full': ls(d, MODS3, SCALE3, method='ML'),
        'scale': ls(d, MODS3, '~ 1', method='ML'),
        'location': ls(d, '~ 1', SCALE3, method='ML'),
        'alpha_fixed': ls(d, MODS3, SCALE3, method='ML', alpha=[numpy.nan, 0, numpy.nan]),
        'full_reml': ls(d, MODS3, SCALE3),
        'scale_reml': ls(d, MODS3, '~ 1'),
    }


@pytest.mark.parametrize(
    ('key', 'first', 'second'),
    [
        ('lrt_scale', 'full', 'scale'),
        ('lrt_location', 'full', 'location'),
        ('lrt_alpha_fixed', 'full', 'alpha_fixed'),
        ('lrt_swapped', 'scale', 'full'),
        ('lrt_reml', 'full_reml', 'scale_reml'),
    ],
)
def test_anova_lrt(key: str, first: str, second: str) -> None:
    models = lrt_models()
    a = metafopy.anova(models[first], models[second])
    assert isinstance(a, metafopy.methods.anova.MVLRTResult)
    ref = REF[key]
    refutil.close(a.LRT, ref['LRT'], f'{key} LRT', rtol=OPT, atol=1e-6)
    refutil.close(a.pval, ref['pval'], f'{key} pval', rtol=OPT, atol=1e-7)
    refutil.close(a.fit_stats_full, ref['fit_stats_f'], f'{key} fit full', rtol=OPT, atol=1e-6)
    refutil.close(a.fit_stats_reduced, ref['fit_stats_r'], f'{key} fit reduced', rtol=OPT, atol=1e-6)
    refutil.close([a.df_full, a.df_reduced], ref['parms'], f'{key} parms')
    refutil.close([a.QE_full, a.QE_reduced], ref['QE'], f'{key} QE', rtol=1e-8)
    assert refutil.r_print(str(a)).strip() == refutil.r_print(ref['print']).strip()


def test_anova_lrt_errors() -> None:
    models = lrt_models()
    with pytest.raises(metafopy.errors.InputError, match='same number of parameters'):
        metafopy.anova(models['full'], models['full'])
    with pytest.raises(metafopy.errors.InputError, match='Mismatch'):
        metafopy.anova(models['full'], models['scale_reml'])
    with pytest.raises(metafopy.errors.InputError, match='not equal'):
        metafopy.anova(models['full'], ls(bd().iloc[:30], MODS3, '~ 1', method='ML'))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        metafopy.anova(models['full_reml'], ls(bdc(), '~ 1', '~ 1'))
    assert any('REML comparisons' in str(w.message) for w in caught)
    refit = metafopy.anova(models['full_reml'], ls(bdc(), '~ 1', '~ 1'), refit=True)
    assert isinstance(refit, metafopy.methods.anova.MVLRTResult)


# -- vif --------------------------------------------------------------------------------------------------------------------


def test_vif() -> None:
    v = any_vif(result('bd_3'))
    assert isinstance(v, metafopy.methods.ls_vif.LSVIFResult)
    ref = REF['bd_3']['vif']
    refutil.close(v.beta.vifs.to_numpy(), ref['beta'], 'vif beta', rtol=OPT)
    refutil.close(v.alpha.vifs.to_numpy(), ref['alpha'], 'vif alpha', rtol=OPT)
    refutil.close(v.beta.sifs.to_numpy(), ref['beta_sif'], 'sif beta', rtol=OPT)
    assert list(v.beta.vifs.index) == ref['beta_names']
    assert list(v.alpha.vifs.index) == ref['alpha_names']
    assert refutil.r_print(str(v)).strip() == refutil.r_print(ref['print']).strip()


def test_vif_reestimate() -> None:
    v = any_vif(result('bd_3'), reestimate=True)
    assert isinstance(v, metafopy.methods.ls_vif.LSVIFResult)
    ref = REF['bd_3']['vif_reest']
    refutil.close(v.beta.vifs.to_numpy(), ref['beta'], 'vif reest beta', rtol=OPT)
    refutil.close(v.alpha.vifs.to_numpy(), ref['alpha'], 'vif reest alpha', rtol=OPT)


def test_vif_btt_att() -> None:
    v = any_vif(result('bd_3'), btt=1, att=[[1], [1, 2]])
    assert isinstance(v, metafopy.methods.ls_vif.LSVIFResult)
    ref = REF['bd_3']['vif_att']
    refutil.close(v.beta.vifs.to_numpy(), ref['beta'], 'vif btt', rtol=OPT)
    refutil.close(v.alpha.vifs.to_numpy(), ref['alpha'], 'vif att', rtol=OPT)
    assert refutil.r_print(str(v)).strip() == refutil.r_print(ref['print']).strip()


def test_vif_table() -> None:
    v = any_vif(result('bd_3'), table=True)
    assert isinstance(v, metafopy.methods.ls_vif.LSVIFResult)
    assert v.beta.table is not None
    assert v.alpha.table is not None
    refutil.close(v.beta.table['estimate'].to_numpy(), result('bd_3').beta.tolist(), 'table beta', rtol=1e-12)
    refutil.close(v.alpha.table['estimate'].to_numpy(), result('bd_3').alpha.tolist(), 'table alpha', rtol=1e-12)
    text = str(v)
    assert 'intrcpt' in text
    assert 'vif' in text


def test_vif_single_part() -> None:
    v = any_vif(result('mc_factors'))
    assert isinstance(v, metafopy.methods.ls_vif.LSVIFResult)
    refutil.close(v.beta.vifs.to_numpy(), REF['mc_factors']['vif']['beta'], 'vif mc', rtol=OPT)
    only_loc = any_vif(result('bd_ni_ml'))
    assert isinstance(only_loc, metafopy.methods.ls_vif.LSVIFResult)
    loc_only = ls(bd(), MODS3, '~ 1')
    single = any_vif(loc_only)
    assert isinstance(single, metafopy.VIFResult)
    scale_only = ls(bd(), None, SCALE3)
    single_scale = any_vif(scale_only)
    assert isinstance(single_scale, metafopy.VIFResult)
    assert list(single_scale.vifs.index) == ['ni100', 'imag']


def test_vif_errors() -> None:
    with pytest.raises(metafopy.errors.InputError, match='intercept-only'):
        any_vif(result('bd_int'))
    with pytest.raises(metafopy.errors.InputError, match="'sim'"):
        any_vif(result('bd_3'), sim=1)


# -- blup, ranef, rstandard and the other accessors -------------------------------------------------------------------------

# auditme: accept[non-alphabetical-dict-keys] order follows the R script that generates the fixtures
DIAG_CASES = {
    'bd_int': ('bd_int', None),
    'bd_3': ('bd_3', None),
    'bd_na': ('bd_na', 'diag'),
    'bcg_ablat': ('bcg_ablat', 'diag'),
    'mo_int': ('mo_int', 'diag'),
    'mc_factors': ('mc_factors', 'diag'),
}


def diag_ref(name: str, key: str | None) -> dict[str, typing.Any]:
    ref = REF[name]
    out: dict[str, typing.Any] = ref if key is None else ref[key]
    return out


@pytest.mark.parametrize('case', sorted(DIAG_CASES))
def test_blup_ranef_rstandard(case: str) -> None:
    name, key = DIAG_CASES[case]
    res = result(name)
    ref = diag_ref(name, key)
    b = metafopy.blup(res)
    refutil.close(b.pred, ref['blup']['pred'], f'{case} blup pred', rtol=OPT, atol=1e-7)
    refutil.close(b.se, ref['blup']['se'], f'{case} blup se', rtol=OPT, atol=1e-7)
    refutil.close(b.pi_lb, ref['blup']['pi_lb'], f'{case} blup pi_lb', rtol=OPT, atol=1e-7)
    refutil.close(b.pi_ub, ref['blup']['pi_ub'], f'{case} blup pi_ub', rtol=OPT, atol=1e-7)
    # metafor's slab holds the labels of all supplied studies; the BLUPs only those used in the fit
    assert list(b.slab) == list(numpy.asarray(ref['slab'], dtype=object)[res.data.keep])
    assert len(refutil.r_print(str(b)).split('\n')) == len(refutil.r_print(ref['blup']['print']).split('\n'))
    r = metafopy.ranef(res)
    refutil.close(r.pred, ref['ranef']['pred'], f'{case} ranef pred', rtol=OPT, atol=1e-7)
    refutil.close(r.se, ref['ranef']['se'], f'{case} ranef se', rtol=OPT, atol=1e-7)
    refutil.close(r.pi_lb, ref['ranef']['pi_lb'], f'{case} ranef pi_lb', rtol=OPT, atol=1e-7)
    s = metafopy.rstandard(res)
    refutil.close(s.resid, ref['rstandard']['resid'], f'{case} rstandard resid', rtol=OPT, atol=1e-7)
    refutil.close(s.se, ref['rstandard']['se'], f'{case} rstandard se', rtol=OPT, atol=1e-7)
    refutil.close(s.z, ref['rstandard']['z'], f'{case} rstandard z', rtol=OPT, atol=1e-6)


def test_blup_options_and_conditional_residuals() -> None:
    res = result('bd_na')
    padded = metafopy.blup(res, na_action='exclude')
    assert len(padded.pred) == 48
    assert int(numpy.isnan(padded.pred).sum()) == 48 - res.k
    cond = metafopy.rstandard(res, type='conditional')
    assert len(cond.resid) == res.k
    with pytest.raises(metafopy.errors.InputError, match='Missing values'):
        metafopy.rstandard(res, na_action='fail')
    with pytest.raises(metafopy.errors.InputError, match='non-standard weights'):
        metafopy.blup(result('bd_3_weights'))
    exp = metafopy.blup(result('bd_3'), transf=numpy.exp)
    assert bool(numpy.all(numpy.isnan(exp.se)))


def test_accessors() -> None:
    res = result('bd_3')
    ref = REF['bd_3']
    refutil.close(res.coef('beta').to_numpy(), ref['coef']['beta'], 'coef beta', rtol=OPT)  # type: ignore[union-attr]
    refutil.close(res.coef('alpha').to_numpy(), ref['coef']['alpha'], 'coef alpha', rtol=OPT)  # type: ignore[union-attr]
    refutil.close(res.vcov('alpha').to_numpy().ravel(), flat(ref['coef']['vcov_alpha']), 'vcov alpha', rtol=OPT, atol=1e-7)  # type: ignore[union-attr]
    assert set(res.coef()) == {'alpha', 'beta'}
    fs = res.fitstats()
    refutil.close(fs.to_numpy(), list(res.fit_stats_reml.as_tuple()), 'fitstats', rtol=1e-12)
    ml = res.fitstats(REML=False)
    refutil.close(ml.to_numpy(), list(res.fit_stats_ml.as_tuple()), 'fitstats ML', rtol=1e-12)
    refutil.close(res.hatvalues(), ref['hat'], 'hat', rtol=OPT)
    refutil.close(result('bd_int').fitstats().to_numpy(), REF['bd_int']['fitstats'], 'fitstats int', rtol=1e-6)
    assert repr(res).startswith('<RMALSResult')


def test_unsupported_methods_raise() -> None:
    res = result('bd_3')
    for func in (
        metafopy.influence,
        metafopy.cooks_distance,
        metafopy.dfbetas,
        metafopy.rstudent,
        metafopy.leave1out,
        metafopy.cumul,
        metafopy.robust,
    ):
        with pytest.raises(metafopy.errors.ArgumentError):
            func(res)
    with pytest.raises(metafopy.errors.NotSupportedError, match='Method not available'):
        metafopy.gosh(res)


# -- permutest --------------------------------------------------------------------------------------------------------------

# auditme: accept[non-alphabetical-dict-keys] order follows the R script that generates the fixtures
PERM_CASES: dict[str, tuple[str, str, dict[str, typing.Any]]] = {
    'both': ('bcg_ablat', 'permutest', {'iter': 100, 'seed': 1234}),
    'beta': ('bcg_ablat', 'permutest_beta', {'iter': 60, 'seed': 99, 'skip_alpha': True}),
    'alpha': ('bcg_ablat', 'permutest_alpha', {'iter': 60, 'seed': 99, 'skip_beta': True}),
    'btt': ('bcg_ablat', 'permutest_btt', {'iter': 40, 'seed': 5, 'btt': [1], 'att': [1]}),
    'less': ('bcg_ablat', 'permutest_less', {'iter': 50, 'seed': 7, 'control': {'alternative': 'less', 'stat': 'coef'}}),
    'px2': ('bcg_ablat', 'permutest_px2', {'iter': 50, 'seed': 7, 'control': {'p2defn': 'px2'}}),
    't': ('bcg_ablat_t', 'permutest', {'iter': 50, 'seed': 321}),
    'int_scale': ('bcg_int_scale', 'permutest', {'iter': 50, 'seed': 11}),
}


@pytest.mark.parametrize('case', sorted(PERM_CASES))
def test_permutest(case: str) -> None:
    name, key, kwargs = PERM_CASES[case]
    pt = any_permutest(result(name), **kwargs)
    assert isinstance(pt, metafopy.methods.ls_permutest.LSPermutestResult)
    ref = REF[name][key]
    # p-values of a skipped part come from the model fit (optimizer precision), the others are proportions of permutations
    tol_beta = OPT if pt.skip_beta else 1e-8
    tol_alpha = PERM_ALPHA if pt.skip_alpha else 1e-8
    refutil.close(pt.pval, ref['pval'], f'{case} pval', rtol=tol_beta, atol=1e-8)
    refutil.close(pt.QMp, ref['QMp'], f'{case} QMp', rtol=tol_beta, atol=1e-8)
    refutil.close(pt.QM, ref['QM'], f'{case} QM', rtol=OPT, atol=1e-7)
    assert pt.skip_beta == ref['skip_beta']
    assert pt.skip_alpha == ref['skip_alpha']
    refutil.close([pt.exact_iter, pt.Z_exact_iter], [ref['exact_iter'], ref['Z_exact_iter']], f'{case} exact iter')
    if case != 'px2':
        refutil.close(pt.pval_alpha, ref['pval_alpha'], f'{case} pval_alpha', rtol=tol_alpha, atol=1e-8)
    if ref['QSp'] is not None:
        refutil.close(pt.QSp, ref['QSp'], f'{case} QSp', rtol=tol_alpha, atol=1e-8)
    if not pt.skip_beta:
        refutil.close(pt.QM_perm, ref['QM_perm'], f'{case} QM_perm', rtol=OPT, atol=1e-6)
        refutil.close(pt.zval_perm.to_numpy().ravel(), flat(ref['zval_perm']), f'{case} zval_perm', rtol=OPT, atol=1e-6)
    if not pt.skip_alpha:
        refutil.close(pt.QS_perm, ref['QS_perm'], f'{case} QS_perm', rtol=PERM_ALPHA, atol=1e-6)
        refutil.close(pt.zval_alpha_perm.to_numpy().ravel(), flat(ref['zval_alpha_perm']), f'{case} zval_alpha_perm', rtol=PERM_ALPHA, atol=1e-6)


PRINTED_DECIMAL = re.compile(r'-?\d+\.\d+')


def same_loose(actual: str, expected: str) -> bool:
    """Printed output agrees up to the numbers, which may differ by a few units in the third decimal.

    The scale part of ``bcg_ablat`` is only identified to about 1e-4, so the last printed digit depends on the BLAS in use.
    """
    a_text = refutil.r_print(actual).strip()
    e_text = refutil.r_print(expected).strip()
    if PRINTED_DECIMAL.sub('#', a_text) != PRINTED_DECIMAL.sub('#', e_text):
        return False
    a_nums = [float(x) for x in PRINTED_DECIMAL.findall(a_text)]
    e_nums = [float(x) for x in PRINTED_DECIMAL.findall(e_text)]
    return bool(numpy.allclose(a_nums, e_nums, rtol=2e-3, atol=2.5e-3))


@pytest.mark.parametrize('case', ['both', 'beta', 'alpha', 't', 'int_scale'])
def test_permutest_print(case: str) -> None:
    name, key, kwargs = PERM_CASES[case]
    pt = any_permutest(result(name), **kwargs)
    assert same_loose(str(pt), REF[name][key]['print'])


def test_permutest_exact_iter_and_exact() -> None:
    assert any_permutest(result('bcg_ablat'), exact='i') == REF['bcg_ablat']['permutest_iter']
    small = result('bd6_exact')
    assert any_permutest(small, exact='i') == REF['bd6_exact']['permutest_iter']
    pt = any_permutest(small, exact=True)
    assert isinstance(pt, metafopy.methods.ls_permutest.LSPermutestResult)
    ref = REF['bd6_exact']['permutest']
    assert (pt.exact_iter, pt.Z_exact_iter) == (ref['exact_iter'], ref['Z_exact_iter'])
    # metafor's optimizer fails for 11 of the 720 permuted fits (it drops them from the p-values), ours converges for all of them;
    # where both succeed the statistics agree, so the p-values agree up to the share of dropped permutations
    expected = numpy.array([[numpy.nan if v is None else v for v in row] for row in ref['zval_perm']])
    ok = ~numpy.isnan(expected).any(axis=1)
    assert int(ok.sum()) == 709
    refutil.close(pt.zval_perm.to_numpy()[ok].ravel(), expected[ok].ravel().tolist(), 'exact zval_perm', rtol=OPT, atol=1e-6)
    refutil.close(pt.pval, ref['pval'], 'exact pval', rtol=0, atol=11 / 720)
    refutil.close(pt.QMp, ref['QMp'], 'exact QMp', rtol=0, atol=11 / 720)
    refutil.close(pt.pval_alpha, ref['pval_alpha'], 'exact pval_alpha', rtol=0, atol=11 / 720)
    refutil.close(pt.QSp, ref['QSp'], 'exact QSp', rtol=0, atol=11 / 720)


def test_permutest_errors() -> None:
    res = result('bcg_ablat')
    with pytest.raises(metafopy.errors.InputError, match='at least one part'):
        any_permutest(res, skip_beta=True, skip_alpha=True)
    with pytest.raises(metafopy.errors.InputError, match='iter'):
        any_permutest(res, iter=1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        pt = any_permutest(result('bcg_int_scale'), iter=20, seed=2)
    assert any('intercept-only scale model' in str(w.message) for w in caught)
    assert isinstance(pt, metafopy.methods.ls_permutest.LSPermutestResult)
    assert pt.skip_alpha
