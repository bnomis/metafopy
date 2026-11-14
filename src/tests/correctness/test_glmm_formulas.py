"""Unit checks of ``glmm_formulas``: its likelihoods against ``scipy.stats`` and its three integration rules against each other."""

import math
import typing

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


AI = numpy.array([4.0, 16.0, 3.0, 19.0, 5.0])
N1 = numpy.array([60.0, 80.0, 50.0, 120.0, 70.0])
CI = numpy.array([18.0, 9.0, 17.0, 15.0, 16.0])
N2 = numpy.array([62.0, 78.0, 52.0, 118.0, 72.0])
T1 = numpy.array([600.0, 800.0, 500.0, 1200.0, 700.0])
T2 = numpy.array([620.0, 780.0, 520.0, 1180.0, 720.0])
MODERATOR = numpy.column_stack([numpy.ones(5), numpy.array([0.0, 1.0, 2.0, 3.0, 1.0])])


def fit(model: str, method: str, **kwargs: typing.Any) -> correctness.glmm_formulas.GLMMReference:
    return correctness.glmm_formulas.glmm_fit(AI, N1 - AI, CI, N2 - CI, model, method, **kwargs)


@pytest.mark.parametrize('method', correctness.glmm_formulas.METHODS)
@pytest.mark.parametrize('model', correctness.glmm_formulas.MODELS)
def test_score_vanishes_at_the_optimum(model: str, method: str) -> None:
    reference = fit(model, method)
    assert reference.gradient_norm < 1e-6
    assert numpy.all(reference.se > 0)
    if method == 'ML':
        assert reference.tau2 > 0


def test_poisson_fits_have_a_vanishing_score() -> None:
    for model in correctness.glmm_formulas.MODELS:
        for method in correctness.glmm_formulas.METHODS:
            assert correctness.glmm_formulas.poisson_glmm_fit(AI, CI, T1, T2, model, method).gradient_norm < 1e-6


def test_irr_approximate_and_exact_conditional_coincide() -> None:
    approximate = correctness.glmm_formulas.poisson_glmm_fit(AI, CI, T1, T2, 'CM.AL', 'ML')
    exact = correctness.glmm_formulas.poisson_glmm_fit(AI, CI, T1, T2, 'CM.EL', 'ML')
    assert approximate.beta == pytest.approx(exact.beta, abs=1e-12)
    assert approximate.loglik == pytest.approx(exact.loglik, abs=1e-12)


def test_closed_forms_equal_the_root_solution() -> None:
    xi, ni = numpy.array([3.0, 8.0, 2.0, 14.0, 5.0]), numpy.array([40.0, 60.0, 30.0, 90.0, 50.0])
    plo = correctness.glmm_formulas.one_group_glmm_fit(xi, ni, 'binomial', 'logit', 'FE')
    assert plo.beta[0] == pytest.approx(correctness.glmm_formulas.closed_form_plo(xi, ni), abs=1e-9)
    irln = correctness.glmm_formulas.one_group_glmm_fit(xi, ni, 'poisson', 'log', 'FE')
    assert irln.beta[0] == pytest.approx(correctness.glmm_formulas.closed_form_irln(xi, ni), abs=1e-9)
    assert plo.se[0] == pytest.approx(1 / math.sqrt(xi.sum() * (1 - xi.sum() / ni.sum())), rel=1e-6)


@pytest.mark.parametrize('link', ['logit', 'probit', 'cloglog', 'log', 'identity'])
def test_one_group_mixed_fit_score(link: str) -> None:
    xi, ni = numpy.array([3.0, 14.0, 2.0, 24.0, 5.0]), numpy.array([40.0, 60.0, 30.0, 90.0, 50.0])
    assert correctness.glmm_formulas.one_group_glmm_fit(xi, ni, 'binomial', link, 'ML').gradient_norm < 1e-6


def test_conditional_information_equals_the_observed_information() -> None:
    analytic = fit('CM.EL', 'FE', X=MODERATOR)
    problem = correctness.glmm_formulas.glmm_problem(AI, N1 - AI, CI, N2 - CI, 'CM.EL', X=MODERATOR)
    structure, loglik = correctness.glmm_formulas.loglik_function(problem, 'FE')
    numeric = numpy.linalg.inv(-correctness.glmm_formulas.fd_hessian(loglik, analytic.theta, 5e-3))
    assert analytic.vb == pytest.approx(numeric, rel=1e-5)
    assert structure.information is not None


def test_moderator_fit_has_one_coefficient_per_column() -> None:
    reference = fit('UM.FS', 'ML', X=MODERATOR)
    assert reference.beta.shape == (2,)
    assert reference.vb.shape == (2, 2)
    assert reference.gradient_norm < 1e-6


def test_rules_give_nearby_but_different_fits() -> None:
    laplace = fit('UM.FS', 'ML', rule='laplace')
    adaptive = fit('UM.FS', 'ML', rule='agq', points=9)
    exact = fit('UM.FS', 'ML', rule='exact')
    assert adaptive.loglik == pytest.approx(exact.loglik, abs=1e-6)
    assert laplace.loglik != pytest.approx(exact.loglik, abs=1e-9)


def test_exact_rule_matches_quadrature() -> None:
    reference = fit('UM.FS', 'ML', rule='exact')
    problem = correctness.glmm_formulas.glmm_problem(AI, N1 - AI, CI, N2 - CI, 'UM.FS')
    structure = correctness.glmm_formulas.structure(problem, 'ML')
    nf = structure.start.size

    def conditional(u: numpy.ndarray) -> numpy.ndarray:
        return structure.cond(reference.theta[:nf], u[:, None, None])[:, 0]

    quad = correctness.glmm_formulas.marginal_loglik(conditional, reference.tau2, 'exact', studies=5)
    assert quad == pytest.approx(reference.loglik, abs=1e-8)


def test_loglik_at_and_profile_agree_with_the_fit() -> None:
    reference = fit('UM.FS', 'ML')
    problem = correctness.glmm_formulas.glmm_problem(AI, N1 - AI, CI, N2 - CI, 'UM.FS')
    value, gradient = correctness.glmm_formulas.loglik_at(problem, 'ML', reference.theta)
    assert value == pytest.approx(reference.loglik, abs=1e-12)
    assert numpy.max(numpy.abs(gradient)) < 1e-6
    profiled = correctness.glmm_formulas.profile_loglik_at(problem, 'ML', reference.beta, reference.tau2)
    assert profiled == pytest.approx(reference.loglik, abs=1e-7)


def test_unrelated_estimates_have_a_lower_profile_loglik() -> None:
    reference = fit('UM.FS', 'ML')
    problem = correctness.glmm_formulas.glmm_problem(AI, N1 - AI, CI, N2 - CI, 'UM.FS')
    assert correctness.glmm_formulas.profile_loglik_at(problem, 'ML', reference.beta + 0.5, reference.tau2) < reference.loglik - 0.1


COUNTS = {'ai': AI, 'bi': N1 - AI, 'ci': CI, 'di': N2 - CI, 'measure': 'OR'}


def test_critical_values() -> None:
    assert correctness.glmm_formulas.critical_value('z', 95, 0) == pytest.approx(1.959963984540054, abs=1e-12)
    assert correctness.glmm_formulas.critical_value('t', 95, 10) == pytest.approx(2.228138851986274, abs=1e-9)
    with pytest.raises(ValueError, match='unsupported test'):
        correctness.glmm_formulas.critical_value('F', 95, 3)


def test_wald_statistic_of_a_diagonal_covariance() -> None:
    assert correctness.glmm_formulas.wald_statistic(numpy.array([1.0, 2.0]), numpy.diag([1.0, 4.0])) == pytest.approx(2.0, abs=1e-14)


def test_saturated_design_has_full_rank_and_keeps_the_model_matrix() -> None:
    saturated = correctness.glmm_formulas.saturated_design(MODERATOR)
    assert saturated.shape == (5, 5)
    assert numpy.linalg.matrix_rank(saturated) == 5
    assert numpy.array_equal(saturated[:, :2], MODERATOR)


def test_typical_variance_of_an_intercept_only_model_is_higgins_and_thompsons() -> None:
    weights = 1.0 / (1.0 / AI + 1.0 / (N1 - AI) + 1.0 / CI + 1.0 / (N2 - CI))
    call = correctness.glmm_formulas.inference_call((), {**COUNTS, 'method': 'FE'})
    problem, _, _ = correctness.glmm_formulas.problem_of(call)
    expected = (5 - 1) / (weights.sum() - (weights**2).sum() / weights.sum())
    assert correctness.glmm_formulas.typical_variance(call, problem) == pytest.approx(expected, rel=1e-12)
    second = correctness.glmm_formulas.inference_call((), {**COUNTS, 'method': 'FE', 'i2def': '2'})
    assert correctness.glmm_formulas.typical_variance(second, problem) == pytest.approx(1.0 / weights.mean(), rel=1e-12)


def test_typical_variance_refuses_other_measures_and_definitions() -> None:
    call = correctness.glmm_formulas.inference_call((), {**COUNTS, 'method': 'FE'})
    problem, _, _ = correctness.glmm_formulas.problem_of(call)
    with pytest.raises(ValueError, match='coded for OR'):
        correctness.glmm_formulas.typical_variance({**call, 'measure': 'RR'}, problem)
    with pytest.raises(ValueError, match='unsupported i2def'):
        correctness.glmm_formulas.typical_variance({**call, 'i2def': '3'}, problem)


def test_identical_studies_have_no_heterogeneity() -> None:
    same = {'ai': numpy.full(4, 6.0), 'bi': numpy.full(4, 34.0), 'ci': numpy.full(4, 12.0), 'di': numpy.full(4, 28.0), 'measure': 'OR', 'model': 'CM.AL'}
    assert correctness.glmm_formulas.rma_glmm_qe_lrt(**same, method='FE') == pytest.approx(0.0, abs=1e-6)
    assert correctness.glmm_formulas.rma_glmm_qe_wld(**same, method='FE') == pytest.approx(0.0, abs=1e-3)


def test_unconditional_random_study_has_no_saturated_fit() -> None:
    with pytest.raises(ValueError, match=r'UM\.RS is not coded'):
        correctness.glmm_formulas.rma_glmm_qe_lrt(**COUNTS, model='UM.RS', method='FE')


def test_tau2_interval_needs_the_exact_conditional_mixed_model() -> None:
    with pytest.raises(ValueError, match=r'CM\.EL with method ML'):
        correctness.glmm_formulas.rma_glmm_se_tau2(**COUNTS, model='CM.AL', method='ML')


def test_the_test_statistics_follow_the_fit() -> None:
    z = correctness.glmm_formulas.rma_glmm_zval(**COUNTS, model='CM.AL', method='FE')
    reference = correctness.glmm_formulas.rma_glmm_fit(**COUNTS, model='CM.AL', method='FE')
    assert z == pytest.approx(reference.beta / reference.se, rel=1e-14)
    assert correctness.glmm_formulas.rma_glmm_qm(**COUNTS, model='CM.AL', method='FE') == pytest.approx(float(z[0]) ** 2, rel=1e-12)
    chi = correctness.glmm_formulas.rma_glmm_pval(**COUNTS, model='CM.AL', method='FE', test='z')
    student = correctness.glmm_formulas.rma_glmm_pval(**COUNTS, model='CM.AL', method='FE', test='t')
    assert chi == pytest.approx(2 * scipy.stats.norm.sf(abs(z)), rel=1e-12)
    assert student > chi


def test_prediction_interval_widens_with_the_variance_and_with_t() -> None:
    lower = correctness.glmm_formulas.rma_glmm_pi_lb(**COUNTS, model='CM.AL', method='ML')
    upper = correctness.glmm_formulas.rma_glmm_pi_ub(**COUNTS, model='CM.AL', method='ML')
    ci_lb = correctness.glmm_formulas.rma_glmm_ci_lb(**COUNTS, model='CM.AL', method='ML')
    assert numpy.all(lower < ci_lb)
    assert numpy.all(upper > correctness.glmm_formulas.rma_glmm_ci_ub(**COUNTS, model='CM.AL', method='ML'))
    wide = correctness.glmm_formulas.rma_glmm_pi_ub(**COUNTS, model='CM.AL', method='ML', test='t')
    assert numpy.all(wide > upper)


def test_moderator_columns_enter_the_model_matrix_of_a_call() -> None:
    mods = numpy.array([0.0, 1.0, 2.0, 3.0, 1.0])
    problem, _, _ = correctness.glmm_formulas.rma_glmm_problem(**COUNTS, model='CM.AL', method='ML', mods=mods)
    assert numpy.array_equal(problem.X, MODERATOR)
    assert correctness.glmm_formulas.rma_glmm_beta(**COUNTS, model='CM.AL', method='ML', mods=mods).shape == (2,)
