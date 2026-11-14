"""``property`` cases: identities, invariances and estimating equations that theory guarantees.

A case that is not a comparison of values names a ``check`` function of this module; it is called with the case and the result and raises
``AssertionError`` when the property fails. The invariances are plain ``hypothesis`` tests over every estimator in ``ESTIMATORS``.
"""

import collections.abc
import typing

import hypothesis
import hypothesis.strategies
import numpy
import numpy.typing
import pytest

import correctness.cases
import correctness.formulas
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
