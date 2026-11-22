"""Tests of the analytic information of the exact conditional likelihood in ``metafopy.core.nchg``."""

import collections.abc
import math
import types
import typing

import hypothesis
import hypothesis.strategies
import numpy
import pandas
import pytest
import scipy.integrate
import scipy.optimize
import scipy.stats

import metafopy
import metafopy.core.nchg
import metafopy.datasets
import metafopy.errors

ARGS: collections.abc.Mapping[str, typing.Any] = types.MappingProxyType(
    {'ai': 'ai', 'ci': 'ci', 'measure': 'OR', 'method': 'FE', 'model': 'CM.EL', 'n1i': 'n1i', 'n2i': 'n2i'}
)
PNEUMONIA = {'ai': [1, 0, 1, 13, 2, 4, 6], 'bi': [155, 89, 43, 64, 157, 43, 76], 'ci': [12, 3, 6, 27, 5, 6, 0], 'di': [69, 40, 32, 53, 165, 43, 148]}
TIGHT = metafopy.core.nchg.IntegrationControl(calc='dnoncenhypergeom')
TIGHT_RANDOM = metafopy.core.nchg.IntegrationControl(rel_tol=1e-13, subdivisions=500, calc='dnoncenhypergeom')


def tables_of(data: pandas.DataFrame) -> metafopy.core.nchg.Tables:
    n1, n2 = data['n1i'].to_numpy(float), data['n2i'].to_numpy(float)
    ai, ci = data['ai'].to_numpy(float), data['ci'].to_numpy(float)
    return metafopy.core.nchg.Tables.from_counts(ai, n1 - ai, ci, n2 - ci)


def pneumonia() -> pandas.DataFrame:
    frame = pandas.DataFrame(PNEUMONIA)
    frame['n1i'], frame['n2i'] = frame['ai'] + frame['bi'], frame['ci'] + frame['di']
    return frame


def collins() -> pandas.DataFrame:
    data = metafopy.datasets.load('collins1985a').rename(columns={'b.xci': 'ai', 'nci': 'n1i', 'b.xti': 'ci', 'nti': 'n2i'})
    return data.dropna(subset=['ai', 'n1i', 'ci', 'n2i']).reset_index(drop=True)


FRAMES = (collins(), metafopy.datasets.load('nielweise2007'), pneumonia())
FRAME_IDS = ('collins1985a', 'nielweise2007', 'pneumonia')
# auditme: accept[non-alphabetical-literal] the names follow the order of the zipped (frame, expected) pairs
FRAME_PARAMS = ('frame', 'expected')


@pytest.mark.parametrize('counts', [(3, 7, 4, 9), (0, 5, 2, 8), (10, 30, 12, 25), (2, 2, 1, 1)])
@pytest.mark.parametrize('log_or', [-1.3, 0.0, 0.4, 2.0])
def test_conditional_moments_match_scipy(counts: tuple[int, int, int, int], log_or: float) -> None:
    a, b, c, d = counts
    tables = metafopy.core.nchg.Tables.from_counts(numpy.array([a]), numpy.array([b]), numpy.array([c]), numpy.array([d]))
    mean, variance = metafopy.core.nchg.conditional_moments(tables, 0, log_or)
    ref = scipy.stats.nchypergeom_fisher(a + b + c + d, a + b, a + c, math.exp(log_or))
    assert mean == pytest.approx(float(ref.mean()), rel=1e-10)
    assert variance == pytest.approx(float(ref.var()), rel=1e-9)


def test_conditional_moments_clamp_gives_zero_variance() -> None:
    tables = metafopy.core.nchg.Tables.from_counts(numpy.array([3.0]), numpy.array([7.0]), numpy.array([4.0]), numpy.array([9.0]))
    mean, variance = metafopy.core.nchg.conditional_moments(tables, 0, 40.0)
    assert variance == 0.0
    assert mean == pytest.approx(7.0)


def second_difference(beta: numpy.ndarray, tables: metafopy.core.nchg.Tables, X: numpy.ndarray, step: float = 3e-4) -> numpy.ndarray:
    p = len(beta)
    hess = numpy.zeros((p, p))

    def f(b: numpy.ndarray) -> float:
        return metafopy.core.nchg.neg_loglik(b, tables, X, False, TIGHT)

    for j in range(p):
        for k in range(p):
            ej, ek = numpy.eye(p)[j] * step, numpy.eye(p)[k] * step
            hess[j, k] = (f(beta + ej + ek) - f(beta + ej - ek) - f(beta - ej + ek) + f(beta - ej - ek)) / (4 * step * step)
    return hess


@pytest.mark.parametrize('moderator', [False, True])
def test_fixed_information_matches_second_difference(moderator: bool) -> None:
    data = metafopy.datasets.load('nielweise2007')
    tables = tables_of(data)
    X = numpy.ones((len(data), 1))
    beta = numpy.array([0.3])
    if moderator:
        X = numpy.column_stack([X, numpy.linspace(-1.0, 1.0, len(data))])
        beta = numpy.array([0.3, -0.2])
    info = metafopy.core.nchg.fixed_information(beta, tables, X)
    numpy.testing.assert_allclose(info, second_difference(beta, tables, X), rtol=1e-7)


@pytest.mark.parametrize(FRAME_PARAMS, zip(FRAMES, (0.09955789399, 0.22403481656, 0.25175350287), strict=True), ids=FRAME_IDS)
def test_fixed_effects_se_matches_spec_measurement(frame: pandas.DataFrame, expected: float) -> None:
    fit = metafopy.rma_glmm(**ARGS, data=frame)
    X = numpy.ones((len(frame), 1))
    info = metafopy.core.nchg.fixed_information(fit.beta, tables_of(frame), X)
    assert math.sqrt(numpy.linalg.inv(info)[0, 0]) == pytest.approx(expected, rel=1e-9)


def second_difference_random(parms: numpy.ndarray, tables: metafopy.core.nchg.Tables, X: numpy.ndarray, step: float = 3e-4) -> numpy.ndarray:
    n = len(parms)
    hess = numpy.zeros((n, n))

    def f(b: numpy.ndarray) -> float:
        return metafopy.core.nchg.neg_loglik(b, tables, X, True, TIGHT_RANDOM)

    for j in range(n):
        for k in range(n):
            ej, ek = numpy.eye(n)[j] * step, numpy.eye(n)[k] * step
            hess[j, k] = (f(parms + ej + ek) - f(parms + ej - ek) - f(parms - ej + ek) + f(parms - ej - ek)) / (4 * step * step)
    return hess


@pytest.mark.parametrize('moderator', [False, True])
@pytest.mark.parametrize('away', [False, True])
def test_marginal_information_matches_second_difference(moderator: bool, away: bool) -> None:
    data = metafopy.datasets.load('nielweise2007')
    tables = tables_of(data)
    X = numpy.ones((len(data), 1))
    parms = numpy.array([0.3, math.log(0.4)])
    if moderator:
        X = numpy.column_stack([X, numpy.linspace(-1.0, 1.0, len(data))])
        parms = numpy.array([0.3, -0.2, math.log(0.4)])
    if away:
        parms = parms + 0.15
    info = metafopy.core.nchg.marginal_information(parms, tables, X, TIGHT_RANDOM)
    numpy.testing.assert_allclose(info, second_difference_random(parms, tables, X), rtol=1e-7)


def test_marginal_information_raises_on_non_finite_integral(monkeypatch: pytest.MonkeyPatch) -> None:
    data = metafopy.datasets.load('nielweise2007')
    monkeypatch.setattr(scipy.integrate, 'quad', lambda *args, **kwargs: (math.nan, 0.0))
    with pytest.raises(metafopy.errors.EstimationError):
        metafopy.core.nchg.marginal_information(numpy.array([0.3, 0.0]), tables_of(data), numpy.ones((len(data), 1)), TIGHT_RANDOM)


ML_SES = (0.13638547727, 0.35104032933, 0.88810229648)


def ml_se(frame: pandas.DataFrame, parms: numpy.ndarray) -> float:
    info = metafopy.core.nchg.marginal_information(parms, tables_of(frame), numpy.ones((len(frame), 1)), TIGHT_RANDOM)
    return math.sqrt(numpy.linalg.inv(info)[0, 0])


@pytest.mark.parametrize(FRAME_PARAMS, zip(FRAMES, ML_SES, strict=True), ids=FRAME_IDS)
def test_ml_se_matches_spec_measurement_at_default_estimate(frame: pandas.DataFrame, expected: float) -> None:
    fit = metafopy.rma_glmm(**{**ARGS, 'method': 'ML'}, data=frame)
    assert ml_se(frame, numpy.append(fit.beta, math.log(fit.tau2))) == pytest.approx(expected, rel=1e-5)


@pytest.mark.parametrize(FRAME_PARAMS, zip(FRAMES, ML_SES, strict=True), ids=FRAME_IDS)
def test_ml_se_matches_spec_measurement_at_tight_optimum(frame: pandas.DataFrame, expected: float) -> None:
    fit = metafopy.rma_glmm(**{**ARGS, 'method': 'ML'}, data=frame)
    tables, X = tables_of(frame), numpy.ones((len(frame), 1))
    start = numpy.append(fit.beta, math.log(fit.tau2))

    def objective(b: numpy.ndarray) -> float:
        return metafopy.core.nchg.neg_loglik(b, tables, X, True, TIGHT_RANDOM)

    result = scipy.optimize.minimize(objective, start, method='Nelder-Mead', options={'xatol': 1e-11, 'fatol': 1e-14, 'maxiter': 5000, 'maxfev': 10000})
    # rel: 10 times the largest deviation measured on 2026-11-17 across the development Mac, the arm64 Linux container and the three CI platforms (2.2e-08,
    # the Nelder-Mead end points differ in the eighth decimal); see agents/specs/cross-platform-numerics/measurements.md
    assert ml_se(frame, result.x) == pytest.approx(expected, rel=2.5e-7)


@hypothesis.strategies.composite
def urn_tables(draw: hypothesis.strategies.DrawFn, max_total: int) -> tuple[int, int, int, int]:
    """``(draws, m, total, x)`` with ``x`` in the support of the hypergeometric table."""
    total = draw(hypothesis.strategies.integers(min_value=1, max_value=max_total))
    m = draw(hypothesis.strategies.integers(min_value=0, max_value=total))
    draws = draw(hypothesis.strategies.integers(min_value=0, max_value=total))
    x = draw(hypothesis.strategies.integers(min_value=max(0, m + draws - total), max_value=min(draws, m)))
    return draws, m, total, x


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(n=hypothesis.strategies.integers(min_value=0, max_value=10**6))
def test_ln_fac_property_matches_lgamma_and_is_non_decreasing(n: int) -> None:
    """Candidate ``src/metafopy/core/nchg.py::ln_fac``: ``lgamma(n + 1)`` on both sides of the table boundary, and non-decreasing in n."""
    value = metafopy.core.nchg.ln_fac(n)
    assert value == pytest.approx(math.lgamma(n + 1), rel=1e-8, abs=1e-12)
    assert metafopy.core.nchg.ln_fac(n + 1) >= value - 4 * math.ulp(value)


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(n=hypothesis.strategies.integers(min_value=metafopy.core.nchg.FAK_LEN - 3, max_value=metafopy.core.nchg.FAK_LEN + 3))
def test_ln_fac_property_matches_lgamma_across_the_table_boundary(n: int) -> None:
    """Candidate ``src/metafopy/core/nchg.py::ln_fac``: the table and the Stirling series agree with ``lgamma`` at ``FAK_LEN``."""
    assert metafopy.core.nchg.ln_fac(n) == pytest.approx(math.lgamma(n + 1), rel=1e-12)


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(table=urn_tables(max_total=40), odds=hypothesis.strategies.floats(min_value=1e-3, max_value=1e3))
def test_fisher_log_probability_property_sums_to_one_over_the_support(table: tuple[int, int, int, int], odds: float) -> None:
    """Candidate ``src/metafopy/core/nchg.py::fisher_log_probability``: ``exp`` of it lies in [0, 1] and sums to 1 over the support."""
    draws, m, total, _ = table
    xmin, xmax = max(0, m + draws - total), min(draws, m)
    probabilities = [math.exp(metafopy.core.nchg.fisher_log_probability(draws, m, total, x, odds, 1e-10)) for x in range(xmin, xmax + 1)]
    assert all(0.0 <= p <= 1.0 + 1e-9 for p in probabilities)
    assert math.fsum(probabilities) == pytest.approx(1.0, abs=1e-8)


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(table=urn_tables(max_total=3000), odds=hypothesis.strategies.floats(min_value=1e-3, max_value=1e3))
def test_fisher_log_probability_property_matches_scipy_fisher_distribution(table: tuple[int, int, int, int], odds: float) -> None:
    """Candidate ``src/metafopy/core/nchg.py::fisher_log_probability``: the log pmf of ``scipy.stats.nchypergeom_fisher`` (total, m, draws, odds)."""
    draws, m, total, x = table
    value = metafopy.core.nchg.fisher_log_probability(draws, m, total, x, odds, 1e-10)
    expected = float(scipy.stats.nchypergeom_fisher.logpmf(x, total, m, draws, odds))
    # scipy loses accuracy in the far tail: at (draws, m, total, x, odds) = (290, 122, 384, 28, 806) it gives -733.91983239 where exact
    # Decimal arithmetic and the port give -733.91984342, so the comparison is made only where the probability is above e^-30.
    hypothesis.assume(expected > -30.0)
    assert value == pytest.approx(expected, rel=1e-8, abs=1e-8)


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(
    table=urn_tables(max_total=40),
    odds=hypothesis.strategies.floats(min_value=1e-3, max_value=1e3),
    data=hypothesis.strategies.data(),
)
def test_fisher_log_probability_property_minus_infinity_outside_the_support(
    table: tuple[int, int, int, int], odds: float, data: hypothesis.strategies.DataObject
) -> None:
    """Candidate ``src/metafopy/core/nchg.py::fisher_log_probability``: ``-inf`` for an ``x`` outside the support."""
    draws, m, total, _ = table
    xmin, xmax = max(0, m + draws - total), min(draws, m)
    outside = data.draw(hypothesis.strategies.one_of(hypothesis.strategies.integers(max_value=xmin - 1), hypothesis.strategies.integers(min_value=xmax + 1)))
    assert metafopy.core.nchg.fisher_log_probability(draws, m, total, outside, odds, 1e-10) == -math.inf


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(table=urn_tables(max_total=60), logodds=hypothesis.strategies.floats(min_value=-6.0, max_value=6.0), data=hypothesis.strategies.data())
def test_tabulated_log_probability_property_non_positive_and_sums_to_one(
    table: tuple[int, int, int, int], logodds: float, data: hypothesis.strategies.DataObject
) -> None:
    """Candidate ``src/metafopy/core/nchg.py::tabulated_log_probability``: ``<= 0``, and ``exp`` of it sums to 1 over the support."""
    draws, m, total, _ = table
    xmin, xmax = max(0, m + draws - total), min(draws, m)
    mean = data.draw(hypothesis.strategies.integers(min_value=xmin, max_value=xmax))
    values = [metafopy.core.nchg.tabulated_log_probability(draws, m, total, x, logodds, mean, 1e-12) for x in range(xmin, xmax + 1)]
    assert all(v <= 1e-12 for v in values)
    assert math.fsum(math.exp(v) for v in values) == pytest.approx(1.0, abs=1e-8)
