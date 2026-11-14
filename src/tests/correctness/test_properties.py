"""``property`` cases: identities, invariances and estimating equations that theory guarantees.

A case that is not a comparison of values names a ``check`` function of this module; it is called with the case and the result and raises
``AssertionError`` when the property fails. The invariances are plain ``hypothesis`` tests over every estimator in ``ESTIMATORS``.
"""

import collections.abc
import dataclasses
import typing

import hypothesis
import hypothesis.strategies
import numpy
import numpy.typing
import pandas
import pytest

import correctness.cases
import correctness.formulas
import correctness.test_derived
import metafopy
import metafopy.errors
import metafopy.models.tau2

type FloatArray = numpy.typing.NDArray[numpy.float64]

CONTROL = {'maxiter': 1000, 'tau2_max': 1e6, 'threshold': 1e-13, 'tol': 1e-13}
# every estimator that estimates tau2, read from the library's own list; the others fix it at zero
RANDOM_EFFECTS = tuple(m for m in metafopy.models.tau2.ESTIMATORS if m not in metafopy.models.tau2.FIXED_EFFECTS)
# the Sidik-Jonkman estimator is a sum of squares and is positive whatever the data, so it never truncates
NEVER_ZERO = ('SJ',)
# the generalised Q estimators weight the coefficients with weights the caller supplies, so their coefficients are not those of the equal-effects fit
# auditme: accept[literal-duplication] formulas.py must not import the tests or metafopy, so it keeps its own copy
SUPPLIED_WEIGHTS = ('GENQ', 'GENQM')


@pytest.mark.parametrize('case', correctness.cases.case_params('property'))
# auditme: accept[paired-sibling-helpers] one test module per kind of evidence, as the spec lays out the suite
def test_property(case: correctness.cases.Case) -> None:
    correctness.cases.check_case(case, correctness.cases.run_case(case))


def i2_follows_h2(case: correctness.cases.Case, result: typing.Any) -> None:
    """``I2`` (a percentage) is ``H2`` as a share of the total variability: ``I2 = 100 (H2 - 1) / H2`` (Higgins and Thompson, 2002)."""
    tolerance = case.tolerance or correctness.cases.PROPERTY_TOLERANCE
    expected = 100 * (result.H2 - 1) / result.H2
    if not numpy.isclose(result.I2, expected, rtol=0, atol=tolerance):
        raise AssertionError(f'{case.id}: I2 is {result.I2!r}, 100 (H2 - 1) / H2 is {expected!r} for H2 = {result.H2!r}')


def fit(yi: FloatArray, vi: FloatArray, method: str) -> typing.Any:
    """``rma_uni`` with ``method`` and tight convergence settings; the generalised Q estimators get the weights ``1, 2, ..., k``."""
    kwargs: dict[str, typing.Any] = {'weights': numpy.arange(1.0, len(yi) + 1)} if method in SUPPLIED_WEIGHTS else {}
    return metafopy.rma_uni(yi=yi, vi=vi, method=method, control=CONTROL, **kwargs)


def fit_or_reject(yi: FloatArray, vi: FloatArray, method: str) -> typing.Any:
    """``fit``, but data on which an iterative estimator does not converge are not examples of the property."""
    try:
        return fit(yi, vi, method)
    except metafopy.errors.EstimationError:
        hypothesis.reject()


def close(where: str, name: str, actual: typing.Any, expected: typing.Any, tolerance: float | None = None) -> None:
    allowed = (tolerance or correctness.cases.PROPERTY_TOLERANCE) * numpy.maximum(1.0, numpy.abs(expected))
    if not numpy.all(numpy.abs(numpy.asarray(actual) - numpy.asarray(expected)) <= allowed):
        raise AssertionError(f'{where}: {name}: metafopy gives {actual!r}, theory gives {expected!r}')


@hypothesis.strategies.composite
def meta_data(draw: collections.abc.Callable[..., typing.Any]) -> tuple[FloatArray, FloatArray]:
    """Effects ``yi`` and sampling variances ``vi`` of a small meta-analysis (4 to 8 studies)."""
    k = draw(hypothesis.strategies.integers(min_value=4, max_value=8))
    effects = draw(hypothesis.strategies.lists(hypothesis.strategies.floats(-2, 2), min_size=k, max_size=k))
    variances = draw(hypothesis.strategies.lists(hypothesis.strategies.floats(0.01, 1), min_size=k, max_size=k))
    # identical effects make the heterogeneity statistic Q zero, which the library does not handle for an equal-effects fit (see TODO.md)
    hypothesis.assume(numpy.ptp(effects) > 0.01)
    return numpy.array(effects), numpy.array(variances)


@pytest.mark.parametrize('method', metafopy.models.tau2.ESTIMATORS)
@hypothesis.settings(max_examples=20, deadline=10000)
@hypothesis.given(data=meta_data(), shift=hypothesis.strategies.floats(-5, 5))
def test_location_invariance(method: str, data: tuple[FloatArray, FloatArray], shift: float) -> None:
    """Adding ``c`` to every effect leaves ``tau2`` unchanged and adds ``c`` to the intercept."""
    yi, vi = data
    before = fit_or_reject(yi, vi, method)
    after = fit(yi + shift, vi, method)
    close(method, 'tau2', after.tau2, before.tau2)
    close(method, 'beta', after.beta, before.beta + shift)


@pytest.mark.parametrize('method', metafopy.models.tau2.ESTIMATORS)
@hypothesis.settings(max_examples=20, deadline=10000)
@hypothesis.given(data=meta_data(), scale=hypothesis.strategies.floats(0.25, 4))
def test_scale_invariance(method: str, data: tuple[FloatArray, FloatArray], scale: float) -> None:
    """Scaling every effect by ``c`` and every variance by ``c^2`` scales ``tau2`` by ``c^2`` and the intercept by ``c``."""
    yi, vi = data
    before = fit_or_reject(yi, vi, method)
    after = fit(yi * scale, vi * scale**2, method)
    close(method, 'tau2', after.tau2, before.tau2 * scale**2)
    close(method, 'beta', after.beta, before.beta * scale)


def truncated_tau2_is_equal_effects(case: correctness.cases.Case, result: typing.Any) -> None:
    """Where the data are as homogeneous as sampling error alone allows, every random-effects estimator truncates ``tau2`` at 0 and so gives the
    equal-effects fit (``tau2 = 0``, weights ``1 / v``); the generalised Q estimators keep their supplied weights and are held to ``tau2 = 0`` only."""
    frame = correctness.cases.case_frame(case)
    assert frame is not None
    yi, vi = frame['yi'].to_numpy(dtype=numpy.float64), frame['vi'].to_numpy(dtype=numpy.float64)
    equal = fit(yi, vi, 'EE')
    for method in RANDOM_EFFECTS:
        if method in NEVER_ZERO:
            continue
        random = fit(yi, vi, method)
        close(case.id, f'{method} tau2', random.tau2, 0.0, case.tolerance)
        if method not in SUPPLIED_WEIGHTS:
            close(case.id, f'{method} beta', random.beta, equal.beta, case.tolerance)
            close(case.id, f'{method} se', random.se, equal.se, case.tolerance)


def reml_score_is_zero(case: correctness.cases.Case, result: typing.Any) -> None:
    """At the REML estimate ``y' P^2 y = tr(P)`` (Viechtbauer, 2005, the REML score equation), ``P`` the projection of ``W = diag(1 / (v + tau2))``."""
    p = correctness.formulas.projection(1 / (result.vi + result.tau2), result.X)
    close(case.id, 'REML score', result.yi @ p @ p @ result.yi - numpy.trace(p), 0.0, case.tolerance)


def ml_score_is_zero(case: correctness.cases.Case, result: typing.Any) -> None:
    """At the ML estimate ``y' P^2 y = tr(W)`` (Hardy and Thompson, 1996, the ML score equation)."""
    weights = 1 / (result.vi + result.tau2)
    p = correctness.formulas.projection(weights, result.X)
    close(case.id, 'ML score', result.yi @ p @ p @ result.yi - weights.sum(), 0.0, case.tolerance)


def pm_q_is_k_minus_p(case: correctness.cases.Case, result: typing.Any) -> None:
    """At the Paule-Mandel estimate ``Q(tau2) = k - p`` (Paule and Mandel, 1982); ``k - p`` is ``k - 1`` for an intercept-only model."""
    q = correctness.formulas.random_effects_q(result.yi, result.vi, result.X, result.tau2)
    close(case.id, 'Q(tau2)', q, result.yi.size - result.X.shape[1], case.tolerance)


def shift_leaves_qe_and_tau2(case: correctness.cases.Case, result: typing.Any) -> None:
    """With an intercept the constant lies in the column space of ``X``, so ``P (y + c) = P y``: a shift of every effect changes neither QE nor tau2."""
    frame = correctness.cases.case_frame(case)
    assert frame is not None
    shifted = frame.assign(yi=frame['yi'] + 1.7)
    for method in metafopy.models.tau2.ESTIMATORS:
        kwargs: dict[str, typing.Any] = {'weights': frame['ni'].to_numpy(dtype=numpy.float64)} if method in SUPPLIED_WEIGHTS else {}
        args = {**case.args, 'method': method, 'control': CONTROL, **kwargs}
        before = metafopy.rma_uni(data=frame, **args)
        after = metafopy.rma_uni(data=shifted, **args)
        close(case.id, f'{method} QE', after.QE, before.QE, case.tolerance)
        close(case.id, f'{method} tau2', after.tau2, before.tau2, case.tolerance)


def qe_of_spanning_dummies_equals_intercept_model(case: correctness.cases.Case, result: typing.Any) -> None:
    """A full set of dummy columns spans the constant, so ``P`` and hence QE are those of the same model with an intercept."""
    frame = correctness.cases.case_frame(case)
    assert frame is not None
    with_intercept = metafopy.rma_uni(data=frame, **{**case.args, 'mods': '~ factor(alloc)', 'intercept': True})
    close(case.id, 'QE', result.QE, with_intercept.QE, case.tolerance)


def i2_follows_h2_every_estimator(case: correctness.cases.Case, result: typing.Any) -> None:
    """``I2 = 100 (H2 - 1) / H2`` for the fit of every estimator in ``ESTIMATORS``."""
    frame = correctness.cases.case_frame(case)
    assert frame is not None
    yi, vi = frame['yi'].to_numpy(dtype=numpy.float64), frame['vi'].to_numpy(dtype=numpy.float64)
    for method in metafopy.models.tau2.ESTIMATORS:
        fitted = fit(yi, vi, method)
        close(case.id, f'{method} I2', fitted.I2, 100 * (fitted.H2 - 1) / fitted.H2, case.tolerance)


def i2_is_zero_for_identical_effects(case: correctness.cases.Case, result: typing.Any) -> None:
    """``I2 = 0`` for effects that are all equal, for every estimator in ``ESTIMATORS`` that fits them (Higgins and Thompson, 2002)."""
    frame = correctness.cases.case_frame(case)
    assert frame is not None
    yi, vi = frame['yi'].to_numpy(dtype=numpy.float64), frame['vi'].to_numpy(dtype=numpy.float64)
    ran = 0
    for method in metafopy.models.tau2.ESTIMATORS:
        try:
            fitted = fit(yi, vi, method)
        except metafopy.errors.EstimationError:
            continue
        ran += 1
        close(case.id, f'{method} I2', fitted.I2, 0.0, case.tolerance)
    if not ran:
        raise AssertionError(f'{case.id}: no estimator fitted the identical effects')


MV_CONTROL = {'gtol': 1e-12, 'maxiter': 10000}
MV_AGREEMENT = 1e-6
# what two fits of one structured model share: the REML log-likelihood, the coefficients, the variances and the correlation
STRUCTURED_FIT = ('beta', 'll', 'rho', 'tau2')
ROBUST_AGREEMENT = 1e-8


def refit_with(case: correctness.cases.Case, **changes: typing.Any) -> typing.Any:
    """The case's fit again, with the ``args`` in ``changes`` replacing or adding to its own (``None`` removes one) and no ``then`` call."""
    args = {**case.args, **changes}
    return correctness.cases.run_case(dataclasses.replace(case, args={k: v for k, v in args.items() if v is not None}, then=None))


def mv_score_is_zero(case: correctness.cases.Case, result: typing.Any) -> None:
    """At the optimum of an ``rma_mv`` fit with ``sigma2`` components, the score of each is zero (Harville 1977; Hardy and Thompson 1996).

    With ``M = V + sum(sigma2_j Z_j Z_j')`` and ``P = M^-1 - M^-1 X (X' M^-1 X)^-1 X' M^-1``, the REML score of ``sigma2_j`` is
    ``y' P Z_j Z_j' P y - tr(P Z_j Z_j')`` and the ML score has ``tr(M^-1 Z_j Z_j')`` for the trace; both are zero at an interior optimum.
    """
    fit = refit_with(case, control=MV_CONTROL)
    s = correctness.test_derived.mv_setup(case, fit)
    inverse = numpy.linalg.inv(s.V + s.extra())
    p = inverse - inverse @ s.X @ numpy.linalg.solve(s.X.T @ inverse @ s.X, s.X.T @ inverse)
    for position, term in enumerate(s.terms):
        if term.struct != correctness.formulas.SIGMA2:
            raise ValueError(f'{case.id}: mv_score_is_zero covers sigma2 components only')
        z = correctness.formulas.term_matrix(term, numpy.ones(1), numpy.zeros(0))
        trace = numpy.trace((p if s.scored == 'REML' else inverse) @ z)
        close(case.id, f'{s.scored} score of sigma2[{position}]', s.yi @ p @ z @ p @ s.yi - trace, 0.0, case.tolerance)


def fit_summary(fit: typing.Any) -> dict[str, typing.Any]:
    """What two fits of one model must share: the REML log-likelihood, the coefficients and every variance component."""
    parts: dict[str, typing.Any] = {'ll': fit.fit_stats_reml.ll, 'beta': fit.beta}
    for name in ('gamma2', 'phi', 'rho', 'sigma2', 'tau2'):
        value = getattr(fit, name)
        if value is not None:
            parts[name] = numpy.atleast_1d(value)
    return parts


def same_fit(case: correctness.cases.Case, first: typing.Any, second: typing.Any, names: tuple[str, ...]) -> None:
    one, two = fit_summary(first), fit_summary(second)
    for name in names:
        close(case.id, name, one[name], two[name], case.tolerance or MV_AGREEMENT)


def mv_one_row_per_id_equals_uni(case: correctness.cases.Case, result: typing.Any) -> None:
    """``rma_mv`` with ``~ 1 | id`` and one row per ``id`` is ``rma_uni``: ``sigma2`` is ``tau2`` and the coefficients, standard errors and the
    log-likelihood are those of the random-effects model (both fit the one marginal model ``N(Xb, diag(vi) + tau2 I)``)."""
    frame = correctness.cases.case_frame(case)
    assert frame is not None
    uni = metafopy.rma_uni(yi=case.args['yi'], vi=case.args['V'], mods=case.args.get('mods'), data=frame, method='REML', control=CONTROL)
    tolerance = case.tolerance or MV_AGREEMENT
    close(case.id, 'beta', result.beta, uni.beta, tolerance)
    close(case.id, 'se', result.se, uni.se, tolerance)
    close(case.id, 'tau2', result.sigma2[0], uni.tau2, tolerance)
    close(case.id, 'log-likelihood', result.fit_stats_reml.ll, uni.fit_stats_reml.ll, tolerance)


def mv_zero_component_equals_reduced_model(case: correctness.cases.Case, result: typing.Any) -> None:
    """A three-level fit with the lowest level's ``sigma2`` held at 0 is the two-level fit of the levels above it: a zero variance adds nothing to
    the marginal covariance, so the likelihood, coefficients and the remaining variance are those of the model without the term."""
    reduced = refit_with(case, random=case.args['random'].split('/')[0], sigma2=None)
    fixed = result.sigma2
    assert fixed[-1] == 0.0, f'{case.id}: the case must fix the last sigma2 at 0'
    tolerance = case.tolerance or MV_AGREEMENT
    close(case.id, 'log-likelihood', result.fit_stats_reml.ll, reduced.fit_stats_reml.ll, tolerance)
    close(case.id, 'beta', result.beta, reduced.beta, tolerance)
    close(case.id, 'sigma2', fixed[:-1], reduced.sigma2, tolerance)


def mv_un_equals_hcs_for_two_levels(case: correctness.cases.Case, result: typing.Any) -> None:
    """With two inner levels, ``UN`` (two variances and one correlation) and ``HCS`` (two variances and one common correlation) are the same model,
    so they reach the same optimum."""
    same_fit(case, result, refit_with(case, struct='HCS'), STRUCTURED_FIT)


def mv_ar_equals_car_at_equal_spacing(case: correctness.cases.Case, result: typing.Any) -> None:
    """``CAR`` has correlation ``rho ** |t_i - t_j|``; at times one apart that is ``AR``'s ``rho ** |i - j|``, so the fits agree."""
    same_fit(case, result, refit_with(case, struct='CAR'), STRUCTURED_FIT)


def mv_optbeta_reaches_same_optimum(case: correctness.cases.Case, result: typing.Any) -> None:
    """Optimising the coefficients with the variance components (``optbeta=True``) reaches the optimum that profiling them out by generalised least
    squares reaches: the REML estimating equations have one solution at an interior optimum."""
    same_fit(case, result, refit_with(case, optbeta=True), ('ll', 'beta', 'sigma2'))


def as_columns(frame: pandas.DataFrame) -> dict[str, list[float]]:
    return {str(name): column.tolist() for name, column in frame.items()}


def mv_location_and_scale_invariance(case: correctness.cases.Case, result: typing.Any) -> None:
    """Adding ``c`` to every effect adds ``c`` to the intercept only; multiplying every effect by ``c`` and ``V`` by ``c^2`` multiplies the coefficients
    by ``c`` and the variance components by ``c^2`` (the model is a location-scale family in ``y``)."""
    frame = correctness.cases.case_frame(case)
    assert frame is not None
    yi, vi = case.args['yi'], case.args['V']
    shift, scale = 3.1, 2.5
    shifted_frame = frame.assign(**{yi: frame[yi] + shift})
    scaled_frame = frame.assign(**{yi: frame[yi] * scale, vi: frame[vi] * scale**2})
    shifted = correctness.cases.run_case(dataclasses.replace(case, then=None, dataset=None, data=as_columns(shifted_frame)))
    scaled = correctness.cases.run_case(dataclasses.replace(case, then=None, dataset=None, data=as_columns(scaled_frame)))
    tolerance = case.tolerance or MV_AGREEMENT
    close(case.id, 'sigma2 after a shift', shifted.sigma2, result.sigma2, tolerance)
    close(case.id, 'beta after a shift', shifted.beta, result.beta + numpy.eye(len(result.beta))[0] * shift, tolerance)
    close(case.id, 'sigma2 after scaling', scaled.sigma2, result.sigma2 * scale**2, tolerance)
    close(case.id, 'beta after scaling', scaled.beta, result.beta * scale, tolerance)
    close(case.id, 'se after scaling', scaled.se, result.se * scale, tolerance)


def cr0_with_singleton_clusters_is_hc0(case: correctness.cases.Case, result: typing.Any) -> None:
    """With one row per cluster, ``CR0`` is the heteroskedasticity-consistent sandwich ``HC0`` (White 1980; Liang and Zeger 1986):
    ``(X'WX)^-1 X'W diag(e^2) W X (X'WX)^-1`` for the diagonal weights ``W`` of the fit and its residuals ``e``."""
    fit = refit_with(case)
    frame = correctness.cases.case_frame(case)
    assert frame is not None
    yi = frame[case.args['yi']].to_numpy(dtype=numpy.float64)
    x = correctness.test_derived.design_matrix(frame, case.args.get('mods'))
    weights = 1 / (frame[case.args['vi']].to_numpy(dtype=numpy.float64) + float(fit.tau2))
    bread = numpy.linalg.inv(x.T @ (weights[:, None] * x))
    resid = yi - x @ (bread @ (x.T @ (weights * yi)))
    hc0 = bread @ (x.T @ (((weights * resid) ** 2)[:, None] * x)) @ bread
    close(case.id, 'vb', result.vb, hc0, case.tolerance or ROBUST_AGREEMENT)


def cr2_is_unbiased_under_the_working_model(case: correctness.cases.Case, result: typing.Any) -> None:
    """When the working model is the true one, ``CR2`` is unbiased for the variance of the estimate (Pustejovsky and Tipton 2018, theorem 1).

    The estimate is ``vb(y)``, a quadratic form in the residuals ``(I - H) y``, so its expectation under ``y ~ N(Xb, Theta)`` is the sum of
    ``vb(L e_i)`` over the unit vectors ``e_i`` for any ``L`` with ``L L' = Theta`` - exact, no simulation. It must equal
    ``(X'WX)^-1 X'W Theta W X (X'WX)^-1``, ``W = Theta^-1``. The fit holds its variance components fixed, so ``W`` does not move with ``y``.
    """
    assert case.then is not None
    s = correctness.test_derived.mv_setup(case, result)
    theta = s.V + s.extra()
    bread = numpy.linalg.inv(s.X.T @ numpy.linalg.solve(theta, s.X))
    factor = numpy.linalg.cholesky(theta)
    total = numpy.zeros_like(bread)
    for column in range(len(s.yi)):
        shocked = refit_with(case, yi=factor[:, column])
        total += metafopy.robust(shocked, **case.then.args).vb
    close(case.id, 'expected CR2 variance', total, bread, case.tolerance or ROBUST_AGREEMENT)
