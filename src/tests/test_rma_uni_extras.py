"""rma_uni features beyond the per-dataset grid, checked against R metafor."""

import json
import pathlib
import typing
import warnings

import numpy
import pandas
import pytest

import metafopy
import metafopy.datasets
import metafopy.models.rma_uni

REF: dict[str, typing.Any] = json.loads((pathlib.Path(__file__).parent / 'reference' / 'extras' / 'rma_uni.json').read_text())
RTOL = 1e-7


def close(actual: typing.Any, expected: typing.Any, what: str = '') -> None:
    numpy.testing.assert_allclose(
        numpy.atleast_1d(numpy.asarray(actual, dtype=float)),
        numpy.atleast_1d(numpy.asarray(expected, dtype=float)),
        rtol=RTOL,
        atol=1e-10,
        equal_nan=True,
        err_msg=what,
    )


@pytest.fixture(scope='module')
def dat() -> pandas.DataFrame:
    d = metafopy.datasets.load('bcg')
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    a, b, c, e = (d[col].to_numpy(float) for col in ('tpos', 'tneg', 'cpos', 'cneg'))
    d['yi'] = numpy.log(a / (a + b)) - numpy.log(c / (c + e))
    d['vi'] = 1 / a - 1 / (a + b) + 1 / c - 1 / (c + e)
    d['n'] = a + b + c + e
    return d


def check(res: metafopy.models.rma_uni.RMAResult, key: str) -> None:
    ref = REF[key]
    for attr in ('beta', 'se', 'pval', 'ci_lb', 'ci_ub', 'tau2', 'QE', 'QM', 'QMp', 'I2', 'H2'):
        close(getattr(res, attr), ref[attr], f'{key}: {attr}')
    if ref['se_tau2'] is not None:
        close(res.se_tau2, ref['se_tau2'], f'{key}: se_tau2')
    if ref['R2'] is not None:
        close(res.R2, ref['R2'], f'{key}: R2')
    close(res.fit_stats_ml.ll, ref['ll_ml'], f'{key}: ll ML')
    close(res.fit_stats_reml.ll, ref['ll_reml'], f'{key}: ll REML')
    close(res.weights_(), ref['weights'], f'{key}: weights')
    close(res.hatvalues(), ref['hat'], f'{key}: hat')


# auditme: accept[non-alphabetical-dict-keys] order follows the R script that generates the fixtures
CASES: dict[str, dict[str, typing.Any]] = {
    'weights_REML': {'weights': 'n'},
    'weights_knha': {'weights': 'n', 'test': 'knha'},
    'genq': {'weights': 'n', 'method': 'GENQ'},
    'genq_mods': {'weights': 'n', 'mods': '~ ablat', 'method': 'GENQ'},
    'tau2_fixed': {'tau2': 0.2},
    'tau2_fixed_mods': {'tau2': 0.2, 'mods': '~ ablat'},
    'unweighted': {'weighted': False},
    'unweighted_knha': {'weighted': False, 'mods': '~ ablat', 'test': 'knha'},
    'test_t': {'mods': '~ ablat', 'test': 't'},
    'test_adhoc': {'mods': '~ ablat', 'test': 'adhoc'},
    'btt': {'mods': '~ ablat + alloc', 'btt': [2, 3]},
    'dlit': {'method': 'DLIT'},
    'sjit': {'method': 'SJIT'},
    'no_intercept': {'mods': '~ 0 + alloc'},
    'level90': {'level': 90},
    'subset': {'subset': "alloc == 'random'"},
}


@pytest.mark.parametrize('key', list(CASES))
def test_feature_matches_metafor(dat: pandas.DataFrame, key: str) -> None:
    check(metafopy.rma(data=dat, **CASES[key]), key)


def test_anova_lrt(dat: pandas.DataFrame) -> None:
    full = metafopy.rma(data=dat, mods='~ ablat + alloc', method='ML')
    red = metafopy.rma(data=dat, mods='~ ablat', method='ML')
    out = metafopy.anova(full, red)
    assert isinstance(out, metafopy.LRTResult)
    ref = REF['anova_lrt']
    close(out.LRT, ref['LRT'])
    close(out.pval, ref['pval'])
    close(out.R2, ref['R2'])
    assert (out.df_full, out.df_reduced) == (ref['parms_f'], ref['parms_r'])
    # argument order does not matter
    swapped = metafopy.anova(red, full)
    assert isinstance(swapped, metafopy.LRTResult)
    assert swapped.LRT == out.LRT


def test_anova_linear_combinations(dat: pandas.DataFrame) -> None:
    full = metafopy.rma(data=dat, mods='~ ablat + alloc', method='ML')
    out = metafopy.anova(full, X=[[0, 1, 0, 0], [0, 0, 1, -1]])
    ref = REF['anova_X']
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    for attr in ('Xb', 'se', 'zval', 'pval', 'QM', 'QMp'):
        close(getattr(out, attr), ref[attr], attr)


def test_anova_btt_knha(dat: pandas.DataFrame) -> None:
    res = metafopy.rma(data=dat, mods='~ ablat + alloc', test='knha')
    out = res.anova(btt=[2, 3])
    close(out.QM, REF['anova_btt_knha']['QM'])
    close(out.QMp, REF['anova_btt_knha']['QMp'])
    assert 'coefficients 3:4' in str(out)


def test_predict_newmods(dat: pandas.DataFrame) -> None:
    res = metafopy.rma(data=dat, mods='~ ablat')
    pr = metafopy.predict(res, [10, 30, 50])
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    for attr in ('pred', 'se', 'ci_lb', 'ci_ub', 'pi_lb', 'pi_ub'):
        close(getattr(pr, attr), REF['predict_newmods'][attr], attr)
    fitted = metafopy.predict(res)
    close(fitted.pred, REF['predict_fitted']['pred'])
    close(fitted.pi_lb, REF['predict_fitted']['pi_lb'])
    assert fitted.slab == res.slab


def test_predict_transf_drops_se(dat: pandas.DataFrame) -> None:
    pr = metafopy.predict(metafopy.rma(data=dat), transf=numpy.exp)
    assert numpy.isnan(pr.se).all()
    assert pr.ci_lb[0] < pr.pred[0] < pr.ci_ub[0]


def test_confint_fixed_knha(dat: pandas.DataFrame) -> None:
    ci = metafopy.confint(metafopy.rma(data=dat, mods='~ ablat', test='knha'), fixed=True, random=False).fixed
    assert ci is not None
    ref = REF['confint_fixed_knha']
    close(ci['estimate'], ref['estimate'])
    close(ci['ci.lb'], ref['ci_lb'])
    close(ci['ci.ub'], ref['ci_ub'])


def test_method_fallback(dat: pandas.DataFrame) -> None:
    res = metafopy.rma(data=dat, method=['GENQ', 'REML'])  # GENQ needs weights, so falls back
    assert res.method == 'REML'


def test_control_tau2_init_and_names(dat: pandas.DataFrame) -> None:
    a = metafopy.rma(data=dat, control={'tau2.init': 1.0})
    b = metafopy.rma(data=dat, control={'tau2_init': 1.0})
    assert a.tau2 == pytest.approx(b.tau2)
    with pytest.raises(TypeError, match='unknown control'):
        metafopy.rma(data=dat, control={'nope': 1})


def test_redundant_predictor_dropped(dat: pandas.DataFrame) -> None:
    d = dat.assign(ablat2=dat['ablat'] * 2)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        res = metafopy.rma(data=d, mods='~ ablat + ablat2')
    assert res.coef_names == ('intrcpt', 'ablat')
    assert any('Redundant' in str(x.message) for x in w)


def test_print_matches_metafor_layout(dat: pandas.DataFrame) -> None:
    text = str(metafopy.rma(data=dat))
    assert 'Random-Effects Model (k = 13; tau^2 estimator: REML)' in text
    assert 'Q(df = 12) = 152.2330, p-val < .0001' in text
    assert ' -0.7145  0.1798  -3.9744  <.0001  -1.0669  -0.3622  ***' in text


def test_too_many_parameters() -> None:
    with pytest.raises(ValueError, match='larger than the number of observations'):
        metafopy.rma([0.1, 0.2], [0.01, 0.01], mods=[1.0, 2.0])
