"""rma_uni features beyond the per-dataset grid, checked against R metafor."""

import json
import math
import pathlib
import typing
import warnings

import hypothesis
import hypothesis.strategies
import numpy
import pandas
import pytest
import rstrategies

import metafopy
import metafopy.datasets
import metafopy.models.rma_uni
import metafopy.models.tau2

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
    # auditme: accept[literal-duplication] each test case spells out its own values, as the R script that generates the fixtures does
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


@pytest.mark.parametrize('method', metafopy.models.tau2.FIXED_EFFECTS)
def test_identical_effects_equal_effects_i2_is_zero(method: str) -> None:
    fit = metafopy.rma_uni(yi=[0.0, 0.0, 0.0, 0.0], vi=[0.1, 0.2, 0.3, 0.4], method=method)
    assert (fit.QE, fit.I2, fit.H2) == (0.0, 0.0, 0.0)


def test_identical_effects_reml_i2_is_zero() -> None:
    fit = metafopy.rma_uni(yi=[0.0, 0.0, 0.0, 0.0], vi=[0.1, 0.2, 0.3, 0.4], method='REML')
    assert (fit.I2, fit.H2) == (0.0, 1.0)


def test_control_reads_an_ignored_entry_as_an_attribute(dat: pandas.DataFrame) -> None:
    res = metafopy.rma(data=dat, control={'vc.min': 0.25})
    assert res.control.vc_min == 0.25
    with pytest.raises(AttributeError, match='nothere'):
        res.control.__getattr__('nothere')


def test_hat_matrix_diagonal_is_hatvalues(dat: pandas.DataFrame) -> None:
    res = metafopy.rma(data=dat, mods='~ year')
    hat = res.hat_matrix()
    numpy.testing.assert_allclose(numpy.diag(hat), res.hatvalues(), rtol=1e-10)
    numpy.testing.assert_allclose(hat @ hat, hat, atol=1e-8)
    assert numpy.trace(hat) == pytest.approx(2.0)


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(q=rstrategies.finite_floats(1e-6, 1e9), df=rstrategies.finite_floats(0.5, 1e6))
def test_i2_from_q_property_bounds_zero_below_df_and_non_decreasing(q: float, df: float) -> None:
    """Candidate ``src/metafopy/models/tau2.py::i2_from_q``: in [0, 100), 0 whenever ``q <= df``, and non-decreasing in ``q``."""
    value = metafopy.models.tau2.i2_from_q(q, df)
    assert 0.0 <= value < 100.0
    if q <= df:
        assert value == 0.0
    assert metafopy.models.tau2.i2_from_q(q * 1.5, df) >= value - 4 * math.ulp(100.0)


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(level=hypothesis.strategies.floats(min_value=0.0, max_value=100.0, allow_nan=False))
def test_rma_uni_level_to_alpha_is_a_probability(level: float) -> None:
    """models/rma_uni.py::level_to_alpha: 0 gives 1, 1 to 100 a percent, up to 0.5 an alpha already, above 0.5 a proportion; always in [0, 1]."""
    alpha = metafopy.models.rma_uni.level_to_alpha(level)
    assert 0.0 <= alpha <= 1.0
    if level == 0:
        assert alpha == 1.0
    elif level >= 1:
        assert alpha == pytest.approx((100 - level) / 100, abs=1e-12)
    elif level > 0.5:
        assert alpha == pytest.approx(1 - level, abs=1e-12)
    else:
        assert alpha == level


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(
    start=hypothesis.strategies.integers(min_value=0, max_value=50),
    gaps=hypothesis.strategies.lists(hypothesis.strategies.integers(min_value=1, max_value=4), max_size=8),
)
def test_rma_uni_coef_span_round_trips(start: int, gaps: list[int]) -> None:
    """models/rma_uni.py::coef_span: the 1-based indices are written as ``a:b`` runs of two or more and single numbers, in order."""
    idx = [start]
    for g in gaps:
        idx.append(idx[-1] + g)
    out = metafopy.models.rma_uni.coef_span(idx)
    expected = [i + 1 for i in idx]
    if ':' in out:
        a, b = out.split(':')
        assert [int(a), int(b)] == [expected[0], expected[-1]]
        assert expected == list(range(expected[0], expected[-1] + 1))
    else:
        assert [int(v) for v in out.split(',')] == expected


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(start=hypothesis.strategies.integers(min_value=0, max_value=50), size=hypothesis.strategies.integers(min_value=2, max_value=20))
def test_rma_uni_coef_span_writes_a_run_as_a_range(start: int, size: int) -> None:
    """models/rma_uni.py::coef_span: consecutive indices give ``first:last``."""
    assert metafopy.models.rma_uni.coef_span(list(range(start, start + size))) == f'{start + 1}:{start + size}'


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(
    first=hypothesis.strategies.floats(min_value=0.5, max_value=1e6, allow_nan=False, allow_infinity=False),
    rest=hypothesis.strategies.lists(hypothesis.strategies.floats(min_value=0.5, max_value=1e6, allow_nan=False, allow_infinity=False), max_size=3),
)
def test_rma_uni_first_df_keeps_the_first_value(first: float, rest: list[float]) -> None:
    """models/rma_uni.py::first_df: the first value is kept, as an int when whole and unchanged when fractional."""
    for dfs in (first, [first, *rest]):
        out = metafopy.models.rma_uni.first_df(dfs)
        assert out == first
        assert isinstance(out, int) == first.is_integer()


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(df=hypothesis.strategies.integers(min_value=1, max_value=10**6))
def test_rma_uni_first_df_returns_a_whole_number_as_int(df: int) -> None:
    """models/rma_uni.py::first_df: an int stays an int."""
    out = metafopy.models.rma_uni.first_df(df)
    assert out == df
    assert isinstance(out, int)


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(p=rstrategies.probabilities(open_interval=False), places=hypothesis.strategies.integers(min_value=1, max_value=6))
def test_rma_uni_pfmt_writes_the_p_value_with_its_sign(p: float, places: int) -> None:
    """models/rma_uni.py::pfmt: ``< .0..1`` below 10**-digits, else ``= `` and the p-value to ``digits`` decimals."""
    out = metafopy.models.rma_uni.pfmt(p, places)
    if p < 10.0**-places:
        assert out == '< ' + '.' + '0' * (places - 1) + '1'
    else:
        assert out.startswith('= ')
        assert abs(float(out[2:]) - p) <= 0.5 * 10.0**-places * (1 + 1e-9)


@pytest.mark.parametrize('key', ['weights_REML', 'unweighted', 'dlit'])
def test_hat_matrix_matches_the_hat_values_of_metafor(dat: pandas.DataFrame, key: str) -> None:
    """``hat_matrix`` for user weights, ``weighted=False`` and inverse-variance weights: its diagonal is metafor's ``hatvalues``, and with an
    intercept-only model every row sums to 1."""
    hat = metafopy.rma(data=dat, **CASES[key]).hat_matrix()
    close(numpy.diag(hat), REF[key]['hat'], f'{key}: hat')
    numpy.testing.assert_allclose(hat.sum(axis=1), 1.0, rtol=1e-10)
