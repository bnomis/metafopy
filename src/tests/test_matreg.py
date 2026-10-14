"""matreg() and its methods against R metafor (fixtures from src/tests/r/generate_matreg.R)."""

import collections.abc
import functools
import json
import math
import pathlib
import typing
import warnings

import numpy
import pandas
import pytest
import refutil

import metafopy
import metafopy.errors

# auditme: accept[non-alphabetical-literal] predictor order of the R example sets the coefficient order
MT_X = ['hp', 'wt', 'am']
COLDITZ_NAMES = ['con', 'exp']
FIXTURE = pathlib.Path(__file__).parent / 'reference' / 'phase7' / 'matreg.json'


# auditme: accept[paired-sibling-helpers] each test module keeps its own fixture loader
@functools.cache
def ref() -> dict[str, typing.Any]:
    data: dict[str, typing.Any] = json.loads(FIXTURE.read_text(encoding='utf-8'))
    return data


def frame(values: typing.Any, names: list[str]) -> pandas.DataFrame:
    return pandas.DataFrame(numpy.array(values, dtype=float), index=names, columns=names)


def mtcars(kind: str) -> pandas.DataFrame:
    mc = ref()['mtcars']
    return frame(mc[kind], mc['names'])


def r_text(text: str) -> str:
    """R's output in a UTF-8 locale quotes the significance codes with curly quotes; trailing spaces are dropped."""
    return refutil.r_print(text.replace(chr(0x2018), "'").replace(chr(0x2019), "'")).strip('\n')


def same_text(actual: str, expected: str) -> None:
    assert refutil.r_print(actual).strip('\n') == r_text(expected)


def close_matrix(actual: pandas.DataFrame, expected: typing.Any, what: str, rtol: float = refutil.RTOL, atol: float = 1e-9) -> None:
    exp = numpy.array([[numpy.nan if v is None else v for v in row] for row in expected], dtype=float)
    numpy.testing.assert_allclose(actual.to_numpy(dtype=float), exp, rtol=rtol, atol=atol, equal_nan=True, err_msg=what)


def check_fit(res: metafopy.MatregResult, exp: dict[str, typing.Any], rtol: float = refutil.RTOL) -> None:
    assert list(res.tab.index) == refutil.as_list(exp['names'])
    assert res.test == exp['test']
    # auditme: accept[literal-duplication] R column names paired with the fixture keys
    for col, key in (('beta', 'beta'), ('se', 'se'), ('pval', 'pval'), ('ci.lb', 'ci_lb'), ('ci.ub', 'ci_ub')):
        refutil.close(res.tab[col], exp[key], col, rtol=rtol)
    close_matrix(res.vb, exp['vb'], 'vb', rtol=rtol)
    refutil.close(res.R2, exp['R2'], 'R2', rtol=rtol)
    refutil.close(res.coef(), exp['coef'], 'coef', rtol=rtol)
    refutil.close(res.confint(level=90).tab.to_numpy().ravel(order='F'), exp['confint90'], 'confint', rtol=rtol)
    if res.test == 't':
        refutil.close(res.tab['tval'], exp['tval'], 'tval', rtol=rtol)
        refutil.close(res.df_residual, exp['df'])
        refutil.close(res.R2adj, exp['R2adj'], 'R2adj', rtol=rtol)
        refutil.close(res.F, exp['F'], 'F', rtol=rtol)
        refutil.close(res.Fp, exp['Fp'], 'Fp', rtol=rtol, atol=1e-12)
        assert res.fit_stats is not None
        close_matrix(res.fit_stats, exp['fit_stats'], 'fit_stats', rtol=rtol)
        refutil.close([res.sigma(), res.sigma(REML=False)], exp['sigma'], 'sigma', rtol=rtol)
        refutil.close([res.logLik(), res.logLik(REML=True)], exp['logLik'], 'logLik', rtol=rtol)
        aic = [res.AIC(), res.AIC(correct=True), res.AIC(REML=True), res.AIC(REML=True, correct=True)]
        refutil.close(aic, exp['AIC'], 'AIC', rtol=rtol)
        refutil.close([res.BIC(), res.BIC(REML=True)], exp['BIC'], 'BIC', rtol=rtol)
        assert res.parms == exp['parms']
    else:
        refutil.close(res.tab['zval'], exp['zval'], 'zval', rtol=rtol)
        refutil.close(res.QM, exp['QM'], 'QM', rtol=rtol)
        refutil.close(res.QMp, exp['QMp'], 'QMp', rtol=rtol, atol=1e-12)


def check_print(res: metafopy.MatregResult, exp: dict[str, typing.Any]) -> None:
    same_text(str(res), exp['print'])
    same_text(res.summary(), exp['summary'])
    same_text(str(res.confint()), exp['confint_print'])


def check_pred(pred: metafopy.MatregPredictResult, exp: dict[str, typing.Any]) -> None:
    refutil.close(pred.pred, exp['pred'], 'pred')
    refutil.close(pred.se, exp['se'], 'se')
    refutil.close(pred.ci_lb, exp['ci_lb'], 'ci_lb')
    refutil.close(pred.ci_ub, exp['ci_ub'], 'ci_ub')
    assert list(pred.slab or ()) == refutil.as_list(exp['slab'])
    if 'vcov' in exp:
        assert pred.vcov is not None
        close_matrix(pred.vcov, exp['vcov'], 'vcov')
    if exp.get('X'):
        assert pred.X is not None
        close_matrix(pred.X, exp['X'], 'X')
    same_text(str(pred), exp['print'])


def cov_fit() -> metafopy.MatregResult:
    mc = ref()['mtcars']
    return metafopy.matreg('mpg ~ hp + wt + am', R=mtcars('S'), cov=True, means=mc['means'], n=mc['n'])


def test_covariance_matrix_with_n() -> None:
    res = cov_fit()
    check_fit(res, ref()['cov_n'])
    check_print(res, ref()['cov_n'])


@pytest.mark.parametrize(
    ('y', 'x'),
    [('mpg', MT_X), (0, [3, 5, 8])],
)
def test_names_and_indices_give_the_same_fit(y: int | str, x: list[int] | list[str]) -> None:
    mc = ref()['mtcars']
    res = metafopy.matreg(y, x, R=mtcars('S'), cov=True, means=mc['means'], n=mc['n'])
    check_fit(res, ref()['cov_n'])


def test_covariance_matrix_without_means() -> None:
    res = metafopy.matreg(0, [3, 5, 8], R=mtcars('S'), cov=True, n=ref()['mtcars']['n'])
    check_fit(res, ref()['cov_n_nomeans'])
    check_print(res, ref()['cov_n_nomeans'])


@pytest.mark.parametrize('key', ['pred_one', 'pred_two', 'pred_named', 'pred_noint', 'pred_full', 'pred_transf'])
def test_predict_covariance_fit(key: str) -> None:
    res = cov_fit()
    calls: dict[str, collections.abc.Callable[[], metafopy.MatregPredictResult]] = {
        'pred_one': lambda: res.predict([100, 3, 1]),
        'pred_two': lambda: res.predict([[100, 3, 1], [150, 2.5, 0]], vcov=True, addx=True),
        # auditme: accept[literal-duplication] the row names of the R example
        'pred_named': lambda: res.predict(pandas.DataFrame({'wt': [3, 2.5], 'am': [1, 0], 'hp': [100, 150]}, index=['a', 'b']), level=90, adjust=True),
        'pred_noint': lambda: res.predict([100, 3, 1], intercept=False),
        'pred_full': lambda: res.predict([1, 100, 3, 1]),
        'pred_transf': lambda: res.predict([100, 3, 1], transf=numpy.log),
    }
    check_pred(calls[key](), ref()['cov_n'][key])


def test_aic_bic_of_several_fits() -> None:
    mc = ref()['mtcars']
    res = cov_fit()
    res2 = metafopy.matreg('mpg ~ hp + wt', R=mtcars('S'), cov=True, means=mc['means'], n=mc['n'])
    exp = ref()['cov_n']
    for got, key in ((res.AIC(res2), 'AIC_multi'), (res.BIC(res2), 'BIC_multi'), (res.AIC(res2, correct=True, REML=True), 'AICc_multi')):
        assert isinstance(got, pandas.DataFrame)
        refutil.close(got.to_numpy().ravel(order='F'), exp[key], key)
    multi = res.AIC(res2, correct=True)
    assert isinstance(multi, pandas.DataFrame)
    assert list(multi.columns) == ['df', 'AICc']


def test_correlation_matrix_with_n() -> None:
    res = metafopy.matreg('mpg', MT_X, R=mtcars('R'), n=ref()['mtcars']['n'])
    exp = ref()['cor_n']
    check_fit(res, exp)
    check_print(res, exp)
    check_pred(res.predict([[1, 0, 0], [0.5, -0.5, 1]]), exp['pred'])
    with pytest.warns(UserWarning, match='intercept'):
        pred = res.predict([1, 1, 1], intercept=False)
    check_pred(pred, exp['pred_noint'])


def test_correlation_matrix_with_v() -> None:
    res = metafopy.matreg('mpg', MT_X, R=mtcars('R'), V=ref()['mtcars']['V'])
    exp = ref()['cor_V']
    check_fit(res, exp)
    check_print(res, exp)
    check_pred(res.predict([1, -1, 0.5], vcov=True), exp['pred'])
    check_pred(res.predict([[1, -1, 0.5], [0, 1, 0]], vcov=True), exp['pred_two'])


def test_rcalc_v_gives_the_same_fit() -> None:
    R = mtcars('R')
    rc = metafopy.rcalc(R, ni=ref()['mtcars']['n'])
    assert isinstance(rc, metafopy.RCalcResult)
    assert isinstance(rc.V, numpy.ndarray)
    check_fit(metafopy.matreg('mpg', MT_X, R=R, V=rc.V), ref()['cor_V'])


def test_covariance_matrix_with_v_without_means() -> None:
    exp = ref()['cov_V_nomeans']
    res = metafopy.matreg(0, [3, 5], R=mtcars('S'), cov=True, V=numpy.diag(exp['V']))
    check_fit(res, exp)
    check_print(res, exp)


def test_all_other_variables_are_the_default_predictors() -> None:
    R = mtcars('R').iloc[:4, :4]
    res = metafopy.matreg(0, R=R, n=32)
    check_fit(res, ref()['cor_n_all'])


def craft_fit(key: str) -> metafopy.MatregResult:
    craft = ref()['craft']
    R = frame(craft['R'], craft['names'])
    Z = frame(craft['Z'], craft['names'])
    calls = {
        'fit': lambda: metafopy.matreg(0, [1, 2, 3], R=R, V=craft['V']),
        # auditme: accept[literal-duplication] test cases spell out their own R arguments
        'fit_n': lambda: metafopy.matreg('perf', ['acog', 'asom', 'conf'], R=R, n=500),
        'fit_ztor': lambda: metafopy.matreg(0, [1, 2, 3], R=Z, V=craft['VZ'], ztor=True),
        'fit_ztor_n': lambda: metafopy.matreg(0, [1, 2, 3], R=Z, n=500, ztor=True),
    }
    return calls[key]()


@pytest.mark.parametrize('key', ['fit', 'fit_n', 'fit_ztor', 'fit_ztor_n'])
def test_craft2003(key: str) -> None:
    res = craft_fit(key)
    exp = ref()['craft'][key]
    check_fit(res, exp)
    check_print(res, exp)
    if 'pred' in exp:
        check_pred(res.predict([0.2, 0.1, 0.3]), exp['pred'])


def test_colditz1994_with_n() -> None:
    c = ref()['colditz']
    G = frame(c['G'], COLDITZ_NAMES)
    res = metafopy.matreg(1, 0, R=G, cov=True, means=c['means'], n=c['n'])
    exp = c['fit_n']
    check_fit(res, exp)
    check_print(res, exp)
    check_pred(res.predict([-4, -3, -2]), exp['pred'])
    check_pred(res.predict(numpy.array([[-4.0], [-3.0]]), intercept=False), exp['pred_matrix'])


def test_colditz1994_with_v() -> None:
    c = ref()['colditz']
    res = metafopy.matreg(1, [0], R=frame(c['G'], COLDITZ_NAMES), cov=True, means=c['means'], V=c['vvc'])
    exp = c['fit_V']
    check_fit(res, exp)
    check_print(res, exp)
    check_pred(res.predict([-4, -3, -2]), exp['pred'])


def test_nearpd() -> None:
    exp = ref()['nearpd']
    R = numpy.array(exp['R'])
    with pytest.raises(metafopy.errors.EstimationError, match=r'Cannot invert R\[x,x\] matrix\.'):
        metafopy.matreg(0, [1, 2, 3], R=R, n=100)
    assert exp['error'] == 'Cannot invert R[x,x] matrix.'
    with pytest.warns(UserWarning, match='nearPD'):
        res = metafopy.matreg(0, [1, 2, 3], R=R, n=100, nearpd=True)
    # nearPD leaves R[x,x] with a smallest eigenvalue of about 1e-8 times the largest (its posd.tol), so the
    # coefficients are of the order 1e6 and rounding differences are amplified by a condition number of about 1e8
    rtol = 1e-7
    refutil.close(res.tab['beta'], exp['fit']['beta'], 'beta', rtol=rtol)
    refutil.close(res.R2, exp['fit']['R2'], 'R2', rtol=rtol)
    refutil.close(res.F, exp['fit']['F'], 'F', rtol=rtol)
    assert numpy.isnan(res.tab['se']).all()


def test_v_models_have_no_likelihood() -> None:
    res = craft_fit('fit')
    for call in (res.sigma, res.logLik, res.AIC, res.BIC):
        with pytest.warns(UserWarning, match='sigma|log-likelihood|AIC|BIC'):
            assert call() is None


def test_confint_parm_is_ignored() -> None:
    res = craft_fit('fit')
    with pytest.warns(UserWarning, match='parm'):
        ci = metafopy.confint(res, parm=1)
    refutil.close(ci.tab['estimate'], ref()['craft']['fit']['beta'])


def test_generic_predict_dispatches() -> None:
    res = craft_fit('fit')
    pred = metafopy.predict(res, [0.2, 0.1, 0.3])
    assert isinstance(pred, metafopy.MatregPredictResult)
    check_pred(pred, ref()['craft']['fit']['pred'])


def test_predict_omits_na_predictions() -> None:
    res = metafopy.matreg(0, [3, 5, 8], R=mtcars('S'), cov=True, n=32)
    pred = res.predict([[100, 3, 1], [150, 2.5, 0]])
    assert pred.pred.size == 0
    kept = res.predict([[100, 3, 1], [150, 2.5, 0]], na_action='pass')
    assert numpy.isnan(kept.pred).all()
    noint = res.predict([100, 3, 1], intercept=False)
    assert math.isfinite(float(noint.pred[0]))


@pytest.mark.parametrize(
    ('kwargs', 'error', 'match'),
    [
        ({'y': 0, 'R': None}, metafopy.errors.InputError, "Must specify the 'R' argument"),
        ({'y': 0, 'R': numpy.ones((2, 3)), 'n': 10}, metafopy.errors.InputError, 'square'),
        ({'y': 0, 'R': numpy.ones((1, 1)), 'n': 10}, metafopy.errors.InputError, 'at least of size 2x2'),
        ({'y': 'a', 'R': numpy.eye(3), 'n': 10}, metafopy.errors.InputError, 'dimension names'),
        ({'y': 'zz', 'n': 10}, metafopy.errors.InputError, "Could not find variable 'zz'"),
        ({'y': 'mpg', 'x': ['hp', 'zz', 'yy'], 'n': 10}, metafopy.errors.InputError, "variables 'zz, yy'"),
        ({'y': 5, 'R': numpy.eye(3), 'n': 10}, metafopy.errors.InputError, "Index 'y'"),
        ({'y': 0, 'x': [1, 1], 'n': 10}, metafopy.errors.InputError, 'duplicated'),
        ({'y': 0, 'x': [0, 1], 'n': 10}, metafopy.errors.InputError, "Variable 'y' should not be an element"),
        ({'y': 0, 'x': [1, 99], 'n': 10}, metafopy.errors.InputError, "Indices in 'x'"),
        ({'y': 0}, metafopy.errors.InputError, "Either 'V' or 'n' must be specified."),
        ({'y': 0, 'n': 10, 'V': numpy.eye(55)}, metafopy.errors.InputError, 'not both'),
        ({'y': 0, 'V': numpy.eye(3)}, metafopy.errors.InputError, "Dimensions of 'V'"),
        ({'y': 0, 'n': 10, 'cov': True, 'ztor': True}, metafopy.errors.InputError, 'ztor'),
        ({'y': 0, 'n': 10, 'cov': 1}, metafopy.errors.ArgumentError, "'cov'"),
        ({'y': 0, 'n': [10, 20]}, metafopy.errors.InputError, "'n' should be a single number"),
        ({'y': 0, 'n': 3}, metafopy.errors.EstimationError, 'equal to or less than the number of regression coefficients'),
        ({'y': 0, 'n': 10, 'cov': True, 'means': [1, 2]}, metafopy.errors.InputError, "Length of 'means'"),
        ({'y': '~ hp', 'n': 10}, metafopy.errors.InputError, 'left-hand side'),
        ({'y': 0, 'R': numpy.array([[1, 2], [2, 1]]), 'n': 10}, metafopy.errors.InputError, r'outside \[-1,1\]'),
    ],
)
def test_argument_errors(kwargs: dict[str, typing.Any], error: type[Exception], match: str) -> None:
    kwargs = dict(kwargs)
    kwargs.setdefault('R', mtcars('S' if kwargs.get('cov') is True else 'R'))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with pytest.raises(error, match=match):
            metafopy.matreg(**kwargs)


@pytest.mark.parametrize(
    ('newmods', 'match'),
    [
        (None, "'newmods' must be specified"),
        ([1, 2], "Dimensions of 'newmods'"),
        (pandas.Series({'hp': 1, 'wt': 2, 'zz': 3}), "Could not find variable 'zz'"),
        (pandas.Series({'hp': 1, 'wt': 2}), 'for this variable: am'),
        (pandas.Series({'hp': 1, 'w': 2, 'wt': 3}), 'multiple matches'),
        ([['a', 'b', 'c']], 'numeric'),
    ],
)
def test_predict_errors(newmods: typing.Any, match: str) -> None:
    with pytest.raises(metafopy.errors.InputError, match=match):
        cov_fit().predict(newmods)


def test_named_series_is_matched_to_the_coefficients() -> None:
    res = cov_fit()
    named = res.predict(pandas.Series({'wt': 3, 'am': 1, 'hp': 100}))
    check_pred(named, ref()['cov_n']['pred_one'])


# auditme: accept[non-alphabetical-literal] factor level order of the R example sets the coefficient order
CRAFT_LEVELS = ['acog.perf', 'asom.perf', 'conf.perf', 'acog.asom', 'acog.conf', 'asom.conf']
# iterative REML fits: the tolerances used for rma_mv fits in test_rma_mv.py (coefficients 1e-5, var-cov matrices 1e-4)
MV_RTOL = 1e-5
MV_VC_RTOL = 1e-4


def craft_pipeline(rtoz: bool) -> tuple[metafopy.RCalcResult, metafopy.RMAMVResult]:
    """The matreg help-page pipeline: rcalc() of dat.craft2003, then a multivariate model of the correlations."""
    dat = metafopy.datasets.load('craft2003')
    tmp = metafopy.rcalc('ri ~ var1 + var2 | study', ni='ni', data=dat, rtoz=rtoz)
    assert isinstance(tmp, metafopy.RCalcResult)
    assert tmp.dat is not None
    dd = tmp.dat.copy()
    dd['var1.var2'] = pandas.Categorical(dd['var1.var2'], categories=CRAFT_LEVELS)
    with pytest.warns(UserWarning, match='9 rows with NAs omitted from model fitting'):
        res = metafopy.rma_mv('yi', tmp.V, mods='~ 0 + var1.var2', random='~ var1.var2 | study', struct='UN', data=dd)
    return tmp, res


@pytest.mark.parametrize(
    ('rtoz', 'mv_key', 'fit_key'),
    [(False, 'mv', 'fit'), (True, 'mv_z', 'fit_ztor')],
)
def test_craft2003_pipeline(rtoz: bool, mv_key: str, fit_key: str) -> None:
    craft = ref()['craft']
    exp = craft[mv_key]
    tmp, res = craft_pipeline(rtoz)
    assert isinstance(tmp.V, numpy.ndarray)
    close_matrix(pandas.DataFrame(tmp.V), exp['rcalc_V'], 'rcalc V')
    refutil.close(tmp.dat['yi'] if tmp.dat is not None else [], exp['rcalc_yi'], 'rcalc yi')
    # rcalc leaves NA covariances in rows whose correlations are present; rma.mv drops only the rows .anyNAv() flags
    assert res.k == exp['k']
    refutil.close(res.coef(), exp['beta'], 'beta', rtol=MV_RTOL)
    # atol for the near-zero covariances (about 1e-5, where 1e-4 relative is below the optimiser's precision)
    close_matrix(res.vcov(), exp['vb'], 'vb', rtol=MV_VC_RTOL, atol=1e-7)
    refutil.close(res.tau2, exp['tau2'], 'tau2', rtol=MV_VC_RTOL)
    refutil.close(res.rho, exp['rho'], 'rho', rtol=MV_VC_RTOL)
    refutil.close(res.logLik(), exp['ll'], 'logLik', rtol=MV_RTOL)

    names = craft['names']
    beta = res.coef().to_numpy()
    R = numpy.eye(4)
    # vec2mat(): the coefficients fill the lower triangle column by column
    rows, cols = numpy.triu_indices(4, 1)
    R[cols, rows] = beta
    R[rows, cols] = beta
    mr = metafopy.matreg(0, [1, 2, 3], R=pandas.DataFrame(R, index=names, columns=names), V=res.vcov(), ztor=rtoz)
    fit = craft[fit_key]
    refutil.close(mr.tab['beta'], fit['beta'], 'matreg beta', rtol=MV_VC_RTOL)
    refutil.close(mr.tab['se'], fit['se'], 'matreg se', rtol=MV_VC_RTOL)
    refutil.close(mr.R2, fit['R2'], 'matreg R2', rtol=MV_VC_RTOL)
