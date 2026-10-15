"""anova() of cluster-robust fits and its remaining arguments, blup()/ranef() of clubSandwich fits and ``rma_glmm(model='CM.EL')`` with other optimisers,
checked against R metafor 5.2-1 and clubSandwich 0.7.0 (tests/r/generate_audit4_robust.R)."""

import collections.abc
import math
import typing
import warnings

import numpy
import pandas
import pytest
import refutil

import metafopy
import metafopy.datasets
import metafopy.errors

REF: dict[str, typing.Any] = refutil.load_phase12('audit4_robust')
RECYCLE = 'longer object length is not a multiple of shorter object length'
# auditme: accept[non-alphabetical-literal] the order follows the R fixture generator
MODS = ('~ ablat', '~ ablat + year')


@pytest.fixture(scope='module')
def dat() -> pandas.DataFrame:
    return metafopy.escalc('RR', ai='tpos', bi='tneg', ci='cpos', di='cneg', data=metafopy.datasets.load('dat.bcg'))


@pytest.fixture(scope='module')
def fits(dat: pandas.DataFrame) -> dict[str, typing.Any]:
    """The robust fits of the generator, by the names used there."""
    res1 = metafopy.rma('yi', 'vi', mods='~ ablat', data=dat)
    res0 = metafopy.rma('yi', 'vi', data=dat)
    res2 = metafopy.rma('yi', 'vi', mods='~ ablat + year', data=dat)
    return {
        'res1': res1,
        'rb1': metafopy.robust(res1, 'alloc', club_sandwich=True),
        'rb0': metafopy.robust(res0, 'alloc', club_sandwich=True),
        'rb2': metafopy.robust(res2, 'alloc', club_sandwich=True),
        'res2': res2,
    }


def check_wald(got: typing.Any, ref: dict[str, typing.Any], what: str, *, rtol: float = 1e-6) -> None:
    refutil.close(got.QM, ref['QM'], f'{what} QM', rtol=rtol)
    refutil.close(got.QMp, ref['QMp'], f'{what} QMp', rtol=rtol)
    expected = [v for v in refutil.as_list(ref['QMdf'])] if ref.get('QMdf') is not None else [None]
    if not math.isnan(got.QM):
        assert got.QMdf[0] == expected[0], what
        if len(expected) > 1 and expected[1] is not None:
            refutil.close(got.QMdf[1], expected[1], f'{what} df2', rtol=rtol)
        else:
            assert got.QMdf[1] is None or math.isinf(got.QMdf[1]), what
    if ref.get('btt') is not None:
        assert tuple(i + 1 for i in got.btt) == tuple(refutil.as_list(ref['btt'])), what
    if ref.get('Xb') is not None:
        refutil.close(got.Xb, ref['Xb'], f'{what} Xb', rtol=rtol)
        refutil.close(got.se, ref['se'], f'{what} se', rtol=rtol)
        refutil.close(got.zval, ref['zval'], f'{what} zval', rtol=rtol)
        refutil.close(got.pval, ref['pval'], f'{what} pval', rtol=rtol)
        refutil.close(numpy.where(numpy.isinf(got.ddf), numpy.nan, got.ddf), ref['ddf'], f'{what} ddf', rtol=rtol)


def check_print(got: typing.Any, ref: dict[str, typing.Any], what: str) -> None:
    assert refutil.r_print(str(got)).strip() == refutil.r_print(ref['print']).strip(), what


# auditme: accept[non-alphabetical-dict-keys] the order follows the R fixture generator
CASES: dict[str, tuple[str, dict[str, typing.Any]]] = {
    'default': ('rb1', {}),
    'btt1': ('rb1', {'btt': 0}),
    'btt12': ('rb1', {'btt': [0, 1]}),
    'btt_name': ('rb1', {'btt': 'abl'}),
    'btt_fixed': ('rb1', {'btt': 'ablat', 'fixed': True}),
    'intercept_only': ('rb0', {}),
    'x_vec': ('rb1', {'X': [1, 30]}),
    'x_mat': ('rb1', {'X': [[1, 30], [1, 40]]}),
    'x_noint': ('rb1', {'X': [30]}),
    'l_vec': ('rb1', {'L': [1, 30]}),
    'btt_rhs': ('rb1', {'btt': 1, 'rhs': 0.1}),
    'adjust': ('rb1', {'btt': 1, 'adjust': True}),
    'adjust_x': ('rb1', {'X': [[1, 30], [1, 40]], 'adjust': 'holm'}),
    'p3_btt': ('rb2', {}),
    'p3_x': ('rb2', {'X': [[1, 30, 70], [1, 10, 70]]}),
    'p3_x_short': ('rb2', {'X': [[30, 70], [10, 70]]}),
}


@pytest.mark.parametrize('name', list(CASES))
def test_anova_uni_club(fits: dict[str, typing.Any], name: str) -> None:
    key, kwargs = CASES[name]
    ref = REF['anova_uni'][name]
    got = metafopy.anova(fits[key], **kwargs)
    check_wald(got, ref, name)
    if 'Xb' not in ref and 'btt' in ref:
        check_print(got, ref, name)


def test_anova_uni_club_print(fits: dict[str, typing.Any]) -> None:
    assert str(metafopy.anova(fits['rb1'])) == 'Test of Moderators (coefficient 2):\nF(df1 = 1, df2 = 1.52) = 41.6109, p-val = 0.0440'
    got = metafopy.anova(fits['rb1'], X=[1, 30])
    assert got.QMdf[0] == 1
    refutil.close([got.QM, got.QMdf[1], got.QMp], [37.3383296597104, 1.8284292366209, 0.0317760943779251], 'x', rtol=1e-6)


def test_anova_uni_club_rhs_warning(fits: dict[str, typing.Any]) -> None:
    with pytest.warns(UserWarning, match="Cannot use 'rhs' argument for 'robust.rma' objects based on 'clubSandwich'"):
        got = metafopy.anova(fits['rb1'], X=[1, 30], rhs=1)
    check_wald(got, REF['anova_uni']['x_rhs'], 'x_rhs')


def test_anova_uni_club_rank_deficient(fits: dict[str, typing.Any]) -> None:
    got = metafopy.anova(fits['rb1'], X=[[1, 30], [2, 60]])
    assert math.isnan(got.QM)
    assert math.isnan(got.QMp)
    assert REF['anova_uni']['x_rank']['QM'] is None
    refutil.close(got.ddf, [1.8284292366209, 1.8284292366209], 'ddf', rtol=1e-6)


@pytest.mark.parametrize('name', ['btt_list', 'btt_list_adjust'])
def test_anova_uni_club_list(fits: dict[str, typing.Any], name: str) -> None:
    got = metafopy.anova(fits['rb1'], btt=[[0], [1]], adjust='bonferroni' if name.endswith('adjust') else None)
    for one, ref in zip(got, REF['anova_uni'][name].values() if isinstance(REF['anova_uni'][name], dict) else REF['anova_uni'][name], strict=True):
        refutil.close(one.QM, ref['QM'], name, rtol=1e-6)
        refutil.close(one.QMp, ref['QMp'], name, rtol=1e-6)
        refutil.close(one.QMdf[1], refutil.as_list(ref['QMdf'])[1], name, rtol=1e-6)


# auditme: accept[non-alphabetical-dict-keys] the order follows the R fixture generator
WALD_TESTS: dict[str, tuple[str, str, dict[str, typing.Any]]] = {
    'wald_chisq': ('res1', 'chi-sq', {}),
    'wald_naivef': ('res1', 'Naive-F', {}),
    'wald_hta': ('res2', 'HTA', {}),
    'wald_edf': ('res2', 'EDF', {}),
}


@pytest.mark.parametrize('name', list(WALD_TESTS))
def test_anova_uni_club_wald_tests(fits: dict[str, typing.Any], dat: pandas.DataFrame, name: str) -> None:
    key, wald_test, kwargs = WALD_TESTS[name]
    rb = metafopy.robust(fits[key], 'alloc', club_sandwich=True, wald_test=wald_test)
    check_wald(metafopy.anova(rb, **kwargs), REF['anova_uni'][name], name)


@pytest.mark.parametrize(('name', 'options'), [('coef_naive', {'coef_test': 'naive-t'}), ('coef_z', {'coef_test': 'z'})])
def test_anova_uni_club_coef_tests(fits: dict[str, typing.Any], name: str, options: dict[str, str]) -> None:
    rb = metafopy.robust(fits['res1'], 'alloc', club_sandwich=True, **options)
    check_wald(metafopy.anova(rb, X=[1, 30]), REF['anova_uni'][name], name)


def test_anova_uni_club_cr1(fits: dict[str, typing.Any]) -> None:
    rb = metafopy.robust(fits['res1'], 'alloc', club_sandwich=True, vcov='CR1')
    check_wald(metafopy.anova(rb, btt=1), REF['anova_uni']['cr1'], 'cr1')


def test_anova_uni_plain_robust_unchanged(fits: dict[str, typing.Any]) -> None:
    rb = metafopy.robust(fits['res1'], 'alloc')
    check_wald(metafopy.anova(rb), REF['anova_uni']['plain_robust'], 'plain')
    check_wald(metafopy.anova(rb, X=[1, 30]), REF['anova_uni']['plain_robust_x'], 'plain x')


@pytest.fixture(scope='module')
def mv_fit() -> typing.Any:
    kon = metafopy.datasets.load('konstantopoulos2011')
    return metafopy.rma_mv('yi', 'vi', mods='~ I(year - 1990)', random='~ 1 | district/school', data=kon)


# auditme: accept[non-alphabetical-dict-keys] the order follows the R fixture generator
MV_CASES: dict[str, dict[str, typing.Any]] = {
    'default': {},
    'x': {'X': [1, 5]},
    'x_mat': {'X': [[1, -5], [1, 5]]},
    'btt1': {'btt': 0},
    'l_vec': {'L': [1, 5]},
}


@pytest.mark.parametrize('name', list(MV_CASES))
def test_anova_mv_club(mv_fit: typing.Any, name: str) -> None:
    rb = metafopy.robust(mv_fit, 'district', club_sandwich=True)
    check_wald(metafopy.anova(rb, **MV_CASES[name]), REF['anova_mv'][name], name)


def test_anova_mv_plain_robust_unchanged(mv_fit: typing.Any) -> None:
    check_wald(metafopy.anova(metafopy.robust(mv_fit, 'district')), REF['anova_mv']['plain'], 'mv plain')


@pytest.fixture(scope='module')
def two_models(dat: pandas.DataFrame) -> dict[str, typing.Any]:
    def fit(mods: str, method: str) -> typing.Any:
        return metafopy.rma('yi', 'vi', mods=mods, data=dat, method=method)

    return {
        'full_ml': fit('~ ablat + year', 'ML'),
        'red_ml': fit('~ ablat', 'ML'),
        'full_reml': fit('~ ablat + year', 'REML'),
        'red_reml': fit('~ ablat', 'REML'),
    }


@pytest.mark.parametrize(
    ('name', 'first', 'second', 'kwargs'),
    [
        ('wald_reml', 'full_reml', 'red_reml', {'test': 'Wald'}),
        ('wald_reml_swapped', 'red_reml', 'full_reml', {'test': 'Wald'}),
        ('wald_ml', 'full_ml', 'red_ml', {'test': 'Wald'}),
        ('wald_prefix', 'full_ml', 'red_ml', {'test': 'W'}),
    ],
)
def test_anova_wald_two_models(two_models: dict[str, typing.Any], name: str, first: str, second: str, kwargs: dict[str, str]) -> None:
    got = metafopy.anova(two_models[first], two_models[second], **kwargs)
    ref = REF['anova_two'][name]
    refutil.close(got.QM, ref['QM'], name, rtol=1e-6)
    refutil.close(got.QMp, ref['QMp'], name, rtol=1e-6)
    assert tuple(i + 1 for i in got.btt) == tuple(refutil.as_list(ref['btt']))
    check_print(got, ref, name)


@pytest.mark.parametrize(
    ('name', 'first', 'second', 'kwargs'),
    [
        ('wald_same', 'red_ml', 'red_ml', {'test': 'Wald', 'df': 1}),
        ('wald_methods', 'full_reml', 'red_ml', {'test': 'Wald'}),
        ('lrt_bad_test', 'full_ml', 'red_ml', {'test': 'x'}),
    ],
)
def test_anova_two_models_errors(two_models: dict[str, typing.Any], name: str, first: str, second: str, kwargs: dict[str, typing.Any]) -> None:
    message = REF['anova_two'][name]['error']
    with pytest.raises(metafopy.errors.MetaforError) as info:
        metafopy.anova(two_models[first], two_models[second], **kwargs)
    assert str(info.value) == message


def test_anova_two_models_same_parameters(two_models: dict[str, typing.Any]) -> None:
    with pytest.raises(metafopy.errors.InputError, match='same number of parameters'):
        metafopy.anova(two_models['red_ml'], two_models['red_ml'], test='Wald')


@pytest.mark.parametrize(
    ('name', 'first', 'second', 'df'), [('lrt_df', 'full_ml', 'red_ml', 3), ('lrt_df_same', 'red_ml', 'red_ml', 2), ('lrt_default', 'full_ml', 'red_ml', None)]
)
def test_anova_lrt_df(two_models: dict[str, typing.Any], name: str, first: str, second: str, df: int | None) -> None:
    got = metafopy.anova(two_models[first], two_models[second], df=df)
    ref = REF['anova_two'][name]
    refutil.close(got.LRT, ref['LRT'], name, rtol=1e-6)
    refutil.close(got.pval, ref['pval'], name, rtol=1e-6)
    assert got.df_full == ref['parms_f']
    assert got.df_reduced == ref['parms_r']


def blup_fits(dat: pandas.DataFrame) -> dict[str, typing.Any]:
    dat12 = dat.iloc[:12]
    dat_na = dat.copy()
    dat_na.loc[2, 'yi'] = numpy.nan

    def club(data: pandas.DataFrame, mods: str | None = '~ ablat', **kwargs: typing.Any) -> typing.Any:
        return metafopy.robust(metafopy.rma('yi', 'vi', mods=mods, data=data), 'alloc', club_sandwich=True, **kwargs)

    return {
        'mods': club(dat),
        'mods_k12': club(dat12),
        'intercept': club(dat, None),
        'p3': club(dat, '~ ablat + year'),
        'cr1': club(dat, vcov='CR1'),
        'coef_z': club(dat, coef_test='z'),
        'na': club(dat_na),
        'plain': metafopy.robust(metafopy.rma('yi', 'vi', mods='~ ablat', data=dat), 'alloc'),
    }


@pytest.fixture(scope='module')
def blup_models(dat: pandas.DataFrame) -> dict[str, typing.Any]:
    return blup_fits(dat)


def call(fn: collections.abc.Callable[..., typing.Any], *args: typing.Any, **kwargs: typing.Any) -> tuple[typing.Any, list[warnings.WarningMessage]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        out = fn(*args, **kwargs)
    return out, caught


# auditme: accept[non-alphabetical-dict-keys] the order follows the R fixture generator
BLUP_CASES: tuple[tuple[str, str, dict[str, typing.Any], dict[str, typing.Any]], ...] = (
    ('mods', 'blup', {}, REF['blup']['mods']['blup']),
    ('mods', 'ranef', {}, REF['blup']['mods']['ranef']),
    ('mods', 'blup', {'level': 90}, REF['blup']['mods']['level90']),
    ('mods_k12', 'blup', {}, REF['blup']['mods_k12']['blup']),
    ('mods_k12', 'ranef', {}, REF['blup']['mods_k12']['ranef']),
    ('intercept', 'blup', {}, REF['blup']['intercept']['blup']),
    ('intercept', 'ranef', {}, REF['blup']['intercept']['ranef']),
    ('p3', 'blup', {}, REF['blup']['p3']['blup']),
    ('mods', 'blup', {'transf': numpy.exp}, REF['blup']['transf']),
    ('cr1', 'blup', {}, REF['blup']['cr1']),
    ('coef_z', 'blup', {}, REF['blup']['coef_z']),
    ('na', 'blup', {}, REF['blup']['na_omit']),
    ('plain', 'blup', {}, REF['blup']['plain']),
)


@pytest.mark.parametrize(('model', 'function', 'kwargs', 'ref'), BLUP_CASES, ids=[f'{c[0]}-{c[1]}-{i}' for i, c in enumerate(BLUP_CASES)])
def test_blup_club(blup_models: dict[str, typing.Any], model: str, function: str, kwargs: dict[str, typing.Any], ref: dict[str, typing.Any]) -> None:
    fn = metafopy.blup if function == 'blup' else metafopy.ranef
    got, caught = call(fn, blup_models[model], **kwargs)
    if kwargs.get('transf') is not None:
        refutil.close(got.pred, ref['pred'], 'pred', rtol=1e-6)
        refutil.close(got.pi_lb, ref['pi_lb'], 'pi_lb', rtol=1e-6)
        refutil.close(got.pi_ub, ref['pi_ub'], 'pi_ub', rtol=1e-6)
    else:
        refutil.close(got.pred, ref['pred'], 'pred', rtol=1e-6)
        refutil.close(got.se, ref['se'], 'se', rtol=1e-6)
        refutil.close(got.pi_lb, ref['pi_lb'], 'pi_lb', rtol=1e-6)
        refutil.close(got.pi_ub, ref['pi_ub'], 'pi_ub', rtol=1e-6)
    warned = [str(w.message) for w in caught if RECYCLE in str(w.message)]
    assert len(warned) >= len(ref['warnings'][:1])
    assert set(warned) <= {RECYCLE}
    assert bool(warned) == bool(ref['warnings'])


def test_blup_club_bounds_of_the_first_study(blup_models: dict[str, typing.Any]) -> None:
    got, _ = call(metafopy.blup, blup_models['mods'])
    refutil.close([got.pi_lb[0], got.pi_ub[0]], [-2.4384, 0.4335], 'study 1', rtol=1e-3)


def test_blup_club_na_exclude(blup_models: dict[str, typing.Any]) -> None:
    got, caught = call(metafopy.blup, blup_models['na'], na_action='exclude')
    ref = REF['blup']['na_exclude']
    refutil.close(got.pred, ref['pred'], 'pred', rtol=1e-6)
    refutil.close(got.pi_lb, ref['pi_lb'], 'pi_lb', rtol=1e-6)
    refutil.close(got.pi_ub, ref['pi_ub'], 'pi_ub', rtol=1e-6)
    assert any(RECYCLE in str(w.message) for w in caught)


# auditme: accept[non-alphabetical-literal] the order follows the R fixture generator
GLMM_ARGS: dict[str, typing.Any] = {'measure': 'OR', 'ai': 'tpos', 'bi': 'tneg', 'ci': 'cpos', 'di': 'cneg', 'model': 'CM.EL', 'method': 'FE'}


def glmm(control: dict[str, typing.Any], **kwargs: typing.Any) -> tuple[typing.Any, list[str]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        res = metafopy.rma_glmm(data=metafopy.datasets.load('bcg'), control=control, **{**GLMM_ARGS, **kwargs})
    return res, [str(w.message) for w in caught]


# the two optimisers of optim() that are ports of R's routines reproduce R's estimates and its failure to fit the saturated model
# auditme: accept[non-alphabetical-dict-keys] the order follows the R fixture generator
EXACT_OPTIM: dict[str, tuple[dict[str, typing.Any], dict[str, typing.Any]]] = {
    'fe_optim': ({'optimizer': 'optim'}, {}),
    'fe_BFGS': ({'optimizer': 'BFGS'}, {}),
    'fe_Nelder-Mead': ({'optimizer': 'Nelder-Mead'}, {}),
    'fe_optim_nm': ({'optimizer': 'optim', 'optmethod': 'Nelder-Mead'}, {}),
    'fe_optim_maxit': ({'optimizer': 'optim', 'optCtrl': {'maxit': 5}}, {}),
    'fe_mods_bfgs': ({'optimizer': 'BFGS'}, {'mods': '~ ablat'}),
    'fe_mods_nm': ({'optimizer': 'Nelder-Mead'}, {'mods': '~ ablat'}),
}


@pytest.mark.parametrize('name', list(EXACT_OPTIM))
def test_glmm_cm_el_optim(name: str) -> None:
    control, kwargs = EXACT_OPTIM[name]
    ref = REF['glmm'][name]
    res, warned = glmm(control, **kwargs)
    refutil.close(res.beta, ref['beta'], 'beta', rtol=1e-6)
    refutil.close(res.fit_stats_ml.ll, ref['ll'], 'll', rtol=1e-9)
    # the standard errors come from a finite-difference Hessian of an integral evaluated to 1e-10 in R
    refutil.close(res.se, ref['se'], 'se', rtol=5e-3)
    refutil.close(res.vb.ravel('F'), ref['vb'], 'vb', rtol=1e-2)
    if ref['QE_LRT'] is None:
        assert math.isnan(res.QE_LRT)
        assert math.isnan(res.QE_Wld)
        assert [w for w in warned if w.startswith('Cannot fit saturated model')] == list(ref['warnings'])
    else:
        refutil.close(res.QE_LRT, ref['QE_LRT'], 'QE_LRT', rtol=1e-6)
        assert math.isfinite(res.QE_Wld)
        assert not warned


# the other optimisers are SciPy stand-ins: the estimates and the log-likelihood agree with R's, the heterogeneity test of the exact conditional
# model of this ill-conditioned data set (QE.Wld: 68 with bobyqa, 199 with optim, 269 with nlminb) does not
@pytest.mark.parametrize('optimizer', ['nlminb', 'CG', 'L-BFGS-B', 'bobyqa', 'uobyqa', 'newuoa', 'nlm'])
def test_glmm_cm_el_standins(optimizer: str) -> None:
    ref = REF['glmm'][f'fe_{optimizer}']
    res, _ = glmm({'optimizer': optimizer})
    refutil.close(res.beta, ref['beta'], 'beta', rtol=1e-5)
    refutil.close(res.fit_stats_ml.ll, ref['ll'], 'll', rtol=1e-9)
    refutil.close(res.se, ref['se'], 'se', rtol=5e-3)
    if ref['QE_LRT'] is not None:
        refutil.close(res.QE_LRT, ref['QE_LRT'], 'QE_LRT', rtol=1e-3)


def test_glmm_cm_el_optimizer_choices() -> None:
    with pytest.raises(metafopy.errors.InputError, match="'optimizer' should be one of"):
        metafopy.rma_glmm(data=metafopy.datasets.load('bcg'), control={'optimizer': 'nope'}, **GLMM_ARGS)
    with pytest.raises(metafopy.errors.InputError, match="'optmethod' should be one of"):
        metafopy.rma_glmm(data=metafopy.datasets.load('bcg'), control={'optmethod': 'nope'}, **GLMM_ARGS)
    assert REF['glmm']['bad_optimizer']['error'].startswith("'arg' should be one of")


def test_glmm_cm_el_ml_optim_agrees_with_default() -> None:
    kw = {'method': 'ML'}
    default, _ = glmm({}, **kw)
    bfgs, _ = glmm({'optimizer': 'BFGS'}, **kw)
    refutil.close(bfgs.beta, default.beta.tolist(), 'beta', rtol=1e-3)
    refutil.close(bfgs.tau2, default.tau2, 'tau2', rtol=5e-2)
    refutil.close(bfgs.fit_stats_ml.ll, default.fit_stats_ml.ll, 'll', rtol=1e-6)
