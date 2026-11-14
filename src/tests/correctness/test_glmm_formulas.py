"""Unit checks of ``glmm_formulas``: its likelihoods against ``scipy.stats`` and its three integration rules against each other."""

import numpy
import pytest
import scipy.optimize
import scipy.special
import scipy.stats

import correctness.glmm_formulas

EVENTS = numpy.array([410.0, 520.0, 295.0, 610.0, 350.0])
SIZES = numpy.array([1000.0, 1200.0, 800.0, 1500.0, 900.0])
BETA = -0.4
# The Laplace error shrinks with tau2 times the curvature of the counts, so the three rules agree to 1e-6 only for a small variance.
TAU2 = 1e-6


def conditional(u: numpy.ndarray) -> numpy.ndarray:
    return correctness.glmm_formulas.binomial_logpmf(EVENTS, SIZES, BETA + u, 'logit')


def test_binomial_loglik_matches_scipy() -> None:
    eta = numpy.array([-1.0, 0.3, 0.8])
    y, size = numpy.array([2.0, 5.0, 7.0]), numpy.array([6.0, 9.0, 10.0])
    for link, p in (('logit', scipy.special.expit(eta)), ('probit', scipy.stats.norm.cdf(eta)), ('cloglog', 1 - numpy.exp(-numpy.exp(eta)))):
        assert correctness.glmm_formulas.binomial_loglik(y, size, eta, link) == pytest.approx(float(numpy.sum(scipy.stats.binom.logpmf(y, size, p))), rel=1e-12)


def test_poisson_loglik_matches_scipy() -> None:
    eta = numpy.array([0.1, 1.0, 1.5])
    y = numpy.array([1.0, 3.0, 6.0])
    assert correctness.glmm_formulas.poisson_loglik(y, eta, 'log') == pytest.approx(float(numpy.sum(scipy.stats.poisson.logpmf(y, numpy.exp(eta)))), rel=1e-12)


def test_nchg_loglik_matches_scipy() -> None:
    ai, n1i, ci, n2i = numpy.array([3.0, 8.0]), numpy.array([20.0, 30.0]), numpy.array([5.0, 6.0]), numpy.array([22.0, 28.0])
    psi = numpy.array([0.4, -0.3])
    expected = numpy.sum(scipy.stats.nchypergeom_fisher.logpmf(ai, n1i + n2i, n1i, ai + ci, numpy.exp(psi)))
    assert correctness.glmm_formulas.nchg_loglik(ai, n1i, ci, n2i, psi) == pytest.approx(float(expected), rel=1e-10)


def test_rules_agree_on_large_counts() -> None:
    exact = correctness.glmm_formulas.marginal_loglik(conditional, TAU2, 'exact', studies=5)
    assert correctness.glmm_formulas.marginal_loglik(conditional, TAU2, 'agq', 7, studies=5) == pytest.approx(exact, abs=1e-6)
    assert correctness.glmm_formulas.marginal_loglik(conditional, TAU2, 'laplace', studies=5) == pytest.approx(exact, abs=1e-6)


def test_agq_with_one_point_is_laplace() -> None:
    assert correctness.glmm_formulas.marginal_loglik(conditional, TAU2, 'agq', 1, studies=5) == pytest.approx(
        correctness.glmm_formulas.marginal_loglik(conditional, TAU2, 'laplace', studies=5), abs=1e-12
    )


def test_zero_variance_is_the_conditional_sum() -> None:
    assert correctness.glmm_formulas.marginal_loglik(conditional, 0.0, 'agq', studies=5) == pytest.approx(float(conditional(numpy.zeros(5)).sum()))


def test_bivariate_rules_agree() -> None:
    def conditional2(u1: numpy.ndarray, u2: numpy.ndarray) -> numpy.ndarray:
        return correctness.glmm_formulas.binomial_logpmf(EVENTS[:2], SIZES[:2], BETA + u1 + u2, 'logit')

    cov = numpy.array([[0.04, 0.01], [0.01, 0.03]])
    exact = correctness.glmm_formulas.bivariate_marginal_loglik(conditional2, cov, 'exact', studies=2)
    assert correctness.glmm_formulas.bivariate_marginal_loglik(conditional2, cov, 'agq', 7, studies=2) == pytest.approx(exact, abs=1e-5)


def test_forced_integration_failure_raises() -> None:
    def hostile(u: numpy.ndarray) -> numpy.ndarray:
        return 40.0 * numpy.sin(5e3 * u)

    with pytest.raises(ArithmeticError, match='toy'):
        correctness.glmm_formulas.marginal_loglik(hostile, 100.0, 'exact', studies=1, model='toy')


def test_non_converged_search_raises() -> None:
    result = scipy.optimize.minimize(lambda x: float(x[0] ** 2), numpy.array([5.0]), method='Nelder-Mead', options={'maxiter': 1})  # type: ignore[call-overload]
    with pytest.raises(ArithmeticError, match='toy'):
        correctness.glmm_formulas.require_converged(result, 'toy')


def test_unknown_rule_is_refused() -> None:
    with pytest.raises(ValueError, match='unsupported rule'):
        correctness.glmm_formulas.marginal_loglik(conditional, TAU2, 'trapezium', studies=5)
