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
import metafopy.models.rma_glmm
import metafopy.models.rma_ls_fit
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
    """Scaling every effect by ``c`` and every variance by ``c^2`` scales ``tau2`` by ``c^2`` and the intercept by ``c``.

    The iterative estimators stop on an absolute change in ``tau2``, so a scaled copy of data that converged in few enough iterations may need more than
    ``maxiter``; such a copy is not an example of the property, as the unscaled data are not when they do not converge."""
    yi, vi = data
    before = fit_or_reject(yi, vi, method)
    after = fit_or_reject(yi * scale, vi * scale**2, method)
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


# The unconditional model with study-specific random effects (``UM.RS``) is left out of the invariances below: its Laplace approximation is not
# invariant to the reparametrisation, so only the models whose integral is exact or adaptive are asserted. Read from the library's own list.
GLMM_INVARIANT_MODELS = tuple(m for m in metafopy.models.rma_glmm.MODELS if m != 'UM.RS')
# the approximate conditional likelihood ``CM.AL`` (Stijnen et al. 2010) is not symmetric in events and non-events, in R metafor 5.2-1 too, so the
# outcome swap runs over the models whose likelihood is exact
OUTCOME_SWAP_MODELS = tuple(m for m in GLMM_INVARIANT_MODELS if m != 'CM.AL')
# each cell argument and the one it swaps with: the groups swap ``ai`` with ``ci`` and ``bi`` with ``di``; the outcomes swap ``ai`` with ``bi``
GROUP_SWAP = {'ai': 'ci', 'bi': 'di', 'ci': 'ai', 'di': 'bi'}
OUTCOME_SWAP = {'ai': 'bi', 'bi': 'ai', 'ci': 'di', 'di': 'ci'}
GLMM_AGREEMENT = 1e-5
SCALE_FACTOR = 7.0
# the person-time columns of a two-group rate-ratio call
PERSON_TIME = ('t1i', 't2i')
CODINGS = (0.5, 1, 0)


def swapped(case: correctness.cases.Case, mapping: dict[str, str]) -> dict[str, typing.Any]:
    """The case's ``args`` with each cell argument in ``mapping`` pointing at the column of the argument it swaps with."""
    return {name: case.args[other] for name, other in mapping.items()}


def glmm_agree(case: correctness.cases.Case, name: str, actual: typing.Any, expected: typing.Any) -> None:
    close(case.id, name, actual, expected, case.tolerance or GLMM_AGREEMENT)


def swapping_groups_negates_beta(case: correctness.cases.Case, result: typing.Any) -> None:
    """Swapping the two groups swaps treatment and control, so the coefficient changes sign and ``tau2`` stays. Run over every model in
    ``GLMM_INVARIANT_MODELS`` with the unconditional model's coding 0.5, the coding that treats the groups alike (``coding`` 1 and 0 make one
    group the reference, so the random-effects structure is not the same after a swap)."""
    for model in GLMM_INVARIANT_MODELS:
        before = refit_with(case, model=model, coding=0.5)
        after = refit_with(case, model=model, coding=0.5, **swapped(case, GROUP_SWAP))
        glmm_agree(case, f'{model} beta after swapping the groups', after.beta, -before.beta)
        glmm_agree(case, f'{model} tau2 after swapping the groups', after.tau2, before.tau2)


def swapping_outcomes_negates_beta(case: correctness.cases.Case, result: typing.Any) -> None:
    """Swapping events and non-events replaces each probability ``p`` by ``1 - p``, so every log odds changes sign and the log odds ratio with it;
    ``tau2`` stays. Run over every model in ``OUTCOME_SWAP_MODELS``."""
    for model in OUTCOME_SWAP_MODELS:
        before = refit_with(case, model=model)
        after = refit_with(case, model=model, **swapped(case, OUTCOME_SWAP))
        glmm_agree(case, f'{model} beta after swapping events and non-events', after.beta, -before.beta)
        glmm_agree(case, f'{model} tau2 after swapping events and non-events', after.tau2, before.tau2)


def scaling_person_time_leaves_irr(case: correctness.cases.Case, result: typing.Any) -> None:
    """Multiplying both person-times by one constant is a change of the time unit: the rate ratio, and so ``beta`` and ``tau2``, are unchanged.
    Run over every model in ``GLMM_INVARIANT_MODELS``."""
    frame = correctness.cases.case_frame(case)
    assert frame is not None
    scaled = frame.assign(**{case.args[name]: frame[case.args[name]] * SCALE_FACTOR for name in PERSON_TIME})
    rescaled = dataclasses.replace(case, dataset=None, data=as_columns(scaled), setup=None, then=None)
    for model in GLMM_INVARIANT_MODELS:
        before = refit_with(case, model=model)
        after = correctness.cases.run_case(dataclasses.replace(rescaled, args={**case.args, 'model': model}))
        glmm_agree(case, f'{model} beta after rescaling the person-time', after.beta, before.beta)
        glmm_agree(case, f'{model} tau2 after rescaling the person-time', after.tau2, before.tau2)


def irr_cm_al_equals_cm_el(case: correctness.cases.Case, result: typing.Any) -> None:
    """For the rate ratio the conditional model is binomial in the first group's count whichever likelihood is asked for (Stijnen et al. 2010), so
    ``CM.AL`` and ``CM.EL`` fit one model: the coefficient, its standard error and ``tau2`` agree."""
    al, el = refit_with(case, model='CM.AL'), refit_with(case, model='CM.EL')
    # auditme: accept[literal-duplication] the attributes of a fit that two calls must share; a test module names its own
    for name in ('beta', 'se', 'tau2'):
        glmm_agree(case, f'CM.AL and CM.EL {name}', getattr(al, name), getattr(el, name))


def codings_give_one_fe_fit(case: correctness.cases.Case, result: typing.Any) -> None:
    """With ``method='FE'`` there is no random effect, so the coding of the group in the random-effects structure changes nothing: the codings in
    ``CODINGS`` give one fit."""
    first = refit_with(case, coding=CODINGS[0])
    for coding in CODINGS[1:]:
        other = refit_with(case, coding=coding)
        glmm_agree(case, f'beta with coding {coding}', other.beta, first.beta)
        glmm_agree(case, f'se with coding {coding}', other.se, first.se)


def ml_below_tau2tol_is_fe(case: correctness.cases.Case, result: typing.Any) -> None:
    """When the ML estimate of ``tau2`` is below ``tau2tol`` the exact conditional fit (``CM.EL``, the one model that uses ``tau2tol``, in metafor
    too) takes the fixed-effects fit (the case sets ``tau2tol`` above the estimate): the coefficient and its standard error are those of ``FE``."""
    assert result.tau2 < case.args['control']['tau2tol'], f'{case.id}: tau2 is not below tau2tol'
    fixed = refit_with(case, method='FE', control=None)
    for name in ('beta', 'se'):
        glmm_agree(case, name, getattr(result, name), getattr(fixed, name))


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


def ls_loglik_not_below_reference(case: correctness.cases.Case, result: typing.Any) -> None:
    """metafopy's log-likelihood is at least that of the independent maximisation, less ``1e-9``: a fit below it has stopped short of the maximum."""
    reference = correctness.test_derived.ls_reference(case)
    actual = float(result.logLik(REML=case.args['method'] == 'REML'))
    if actual < reference['ll'] - correctness.test_derived.LS_LOGLIK:
        raise AssertionError(f'{case.id}: metafopy gives log-likelihood {actual!r}, below the reference maximum {reference["ll"]!r}')


LS_AGREEMENT = 1e-6
# an optimiser that cannot run on a case's link or model is unavailable, not wrong
LS_UNAVAILABLE = (metafopy.errors.InputError, metafopy.errors.NotSupportedError)


def ls_fit_of(case: correctness.cases.Case, frame: pandas.DataFrame, **changes: typing.Any) -> typing.Any:
    """``rma_uni`` with the case's own arguments on ``frame``, those in ``changes`` replacing or adding to them (``None`` removes one)."""
    args = {**case.args, 'data': frame, **changes}
    return metafopy.rma_uni(**{k: v for k, v in args.items() if v is not None})


def ls_loglik_of(fit: typing.Any) -> float:
    return float(fit.logLik(REML=fit.method == 'REML'))


def ls_column(case: correctness.cases.Case) -> str:
    """The one moderator column of a case whose ``mods`` and ``scale`` are both ``~ <column>``."""
    name = case.args['scale'].removeprefix('~').strip()
    assert case.args['mods'].removeprefix('~').strip() == name, f'{case.id}: mods and scale must be the same single column'
    return str(name)


def ls_case_frame(case: correctness.cases.Case) -> pandas.DataFrame:
    frame = correctness.cases.case_frame(case)
    assert frame is not None
    return frame


def ls_reduces_to_rma_uni(case: correctness.cases.Case, result: typing.Any) -> None:
    """With an intercept-only scale model, ``tau2_i = exp(alpha_0)`` is one ``tau2`` and the fit is the random-effects fit with the same method.

    Both maximise one likelihood, so ``beta``, its standard error, ``tau2`` and the log-likelihood agree; the standard error of ``tau2`` is not
    compared, because the location-scale fit gives that of ``alpha_0``.
    """
    uni = ls_fit_of(case, ls_case_frame(case), scale=None, link=None, control=CONTROL)
    tolerance = case.tolerance or LS_AGREEMENT
    close(case.id, 'beta', result.beta, uni.beta, tolerance)
    close(case.id, 'se', result.se, uni.se, tolerance)
    close(case.id, 'tau2', numpy.exp(result.alpha[0]), uni.tau2, tolerance)
    close(case.id, 'log-likelihood', ls_loglik_of(result), ls_loglik_of(uni), tolerance)


def ls_subgroups_are_separate_fits(case: correctness.cases.Case, result: typing.Any) -> None:
    """One categorical moderator in both parts fits each category on its own: ``tau2`` and the fitted mean of a category are those of ``rma_uni`` on its rows.

    The mean and the variance of each category are free parameters, so the joint likelihood is the sum of the categories' likelihoods and is
    maximised category by category.
    """
    frame = ls_case_frame(case)
    column = ls_column(case)
    tolerance = case.tolerance or LS_AGREEMENT
    for level in sorted(frame[column].unique()):
        rows = (frame[column] == level).to_numpy()
        separate = ls_fit_of(case, frame, mods=None, scale=None, link=None, subset=rows, control=CONTROL)
        first = int(numpy.flatnonzero(rows)[0])
        close(case.id, f'mean of {column} = {level}', float(numpy.asarray(result.X)[first] @ numpy.asarray(result.beta)), float(separate.beta[0]), tolerance)
        close(case.id, f'tau2 of {column} = {level}', float(result.tau2[first]), float(separate.tau2), tolerance)


def ls_score_is_zero(case: correctness.cases.Case, result: typing.Any) -> None:
    """At the estimates, the analytic score of the log-likelihood (``formulas.ls_score``, derived from equations 7 and 9) is zero in every ``alpha_j``."""
    yi, vi, X, Z = correctness.test_derived.ls_inputs(case)
    alpha = numpy.asarray(result.alpha, dtype=numpy.float64)
    score = correctness.formulas.ls_score(yi, vi, X, Z, alpha, case.args.get('link', 'log'), case.args['method'])
    close(case.id, 'score', score, numpy.zeros_like(score), case.tolerance or 1e-4)


def ls_invariance(case: correctness.cases.Case, result: typing.Any) -> None:
    """Centring and rescaling the scale moderator ``z`` to ``(z - c) / s`` changes ``alpha`` linearly and nothing else.

    ``alpha_0 + alpha_1 z = alpha_0 + alpha_1 c + (alpha_1 s) (z - c) / s``, so the new coefficients are ``alpha_0 + alpha_1 c`` and ``alpha_1 s``
    (the linear predictor is the same for either link), and ``tau2_i``, ``beta`` and the log-likelihood are unchanged.
    """
    frame = ls_case_frame(case).copy()
    column = case.args['scale'].removeprefix('~').strip()
    centre, factor = float(frame[column].mean()), 2.5
    frame['moved'] = (frame[column] - centre) / factor
    moved = ls_fit_of(case, frame, scale='~ moved')
    tolerance = case.tolerance or LS_AGREEMENT
    alpha = numpy.asarray(result.alpha, dtype=numpy.float64)
    close(case.id, 'alpha', moved.alpha, [alpha[0] + alpha[1] * centre, alpha[1] * factor], tolerance)
    close(case.id, 'tau2', moved.tau2, result.tau2, tolerance)
    close(case.id, 'beta', moved.beta, result.beta, tolerance)
    close(case.id, 'log-likelihood', ls_loglik_of(moved), ls_loglik_of(result), tolerance)


def ls_profile_peaks_at_estimate(case: correctness.cases.Case, result: typing.Any) -> None:
    """Each ``alpha_j``'s profile log-likelihood (section 2.5.3, maximised over the other ``alpha``) is highest at the estimate and equals the fit's there."""
    yi, vi, X, Z = correctness.test_derived.ls_inputs(case)
    link, method = case.args.get('link', 'log'), case.args['method']
    alpha = numpy.asarray(result.alpha, dtype=numpy.float64)
    tolerance = case.tolerance or 1e-6
    for j in range(len(alpha)):
        step = 0.25 * float(result.se_alpha[j])
        peak = correctness.formulas.ls_profile_ll(yi, vi, X, Z, link, method, j, float(alpha[j]), alpha)
        close(case.id, f'profile log-likelihood of alpha[{j}] at the estimate', peak, ls_loglik_of(result), tolerance)
        for side in (-step, step):
            value = correctness.formulas.ls_profile_ll(yi, vi, X, Z, link, method, j, float(alpha[j]) + side, alpha)
            if value > peak + tolerance:
                raise AssertionError(f'{case.id}: the profile of alpha[{j}] is {value!r} at {alpha[j] + side!r}, above {peak!r} at the estimate {alpha[j]!r}')


def ls_optimisers_agree(case: correctness.cases.Case, result: typing.Any) -> None:
    """Every optimiser of ``metafopy.models.rma_ls_fit.OPTIMIZERS`` that runs on the case reaches the same maximum.

    The list is read at test time. An optimiser that raises ``InputError`` or ``NotSupportedError`` (a constrained one on the log link, ``SANN`` and
    ``Brent``) is unavailable; the available ones are compared first, and the case then skips, naming the unavailable ones, so they never look checked.
    """
    frame = ls_case_frame(case)
    tolerance = case.tolerance or LS_AGREEMENT
    unavailable: list[str] = []
    for name in metafopy.models.rma_ls_fit.OPTIMIZERS:
        try:
            other = ls_fit_of(case, frame, control={'optimizer': name})
        except LS_UNAVAILABLE:
            unavailable.append(name)
            continue
        close(case.id, f'log-likelihood with optimizer {name}', ls_loglik_of(other), ls_loglik_of(result), tolerance)
    if unavailable:
        pytest.skip(f'{case.id}: the rest agree; unavailable on this model: {", ".join(unavailable)}')


def ls_links_agree_for_group_indicators(case: correctness.cases.Case, result: typing.Any) -> None:
    """When the scale model is a set of group indicators, ``tau2_g = exp(alpha_g)`` and ``tau2_g = alpha_g`` span the same variances, so the links
    reach the same maximum: the log-likelihood, ``beta`` and ``tau2_i`` agree."""
    other = ls_fit_of(case, ls_case_frame(case), link='identity' if case.args.get('link', 'log') == 'log' else 'log')
    tolerance = case.tolerance or LS_AGREEMENT
    close(case.id, 'log-likelihood', ls_loglik_of(other), ls_loglik_of(result), tolerance)
    close(case.id, 'beta', other.beta, result.beta, tolerance)
    close(case.id, 'tau2', other.tau2, result.tau2, tolerance)


BIAS_AGREEMENT = 1e-6
# the selection-model optimiser stops within about 1e-5 of the maximum, so a fit with no selection agrees with ML to about that
SELMODEL_AGREEMENT = 1e-4
REGTEST_MODELS: tuple[typing.Literal['lm', 'rma'], ...] = ('lm', 'rma')
REGTEST_PREDICTORS: tuple[typing.Literal['sei', 'vi'], ...] = ('sei', 'vi')
LOCATION_SHIFT = 3.0


def uni_fit(case: correctness.cases.Case, frame: pandas.DataFrame) -> typing.Any:
    """The ``rma_uni`` fit of the case's ``setup`` on ``yi`` and ``vi`` of ``frame``."""
    assert case.setup is not None
    return metafopy.rma_uni(yi='yi', vi='vi', data=frame, **case.setup['rma_uni'])


def reflected_fit(case: correctness.cases.Case, shift: float = 0.0, sign: float = 1.0) -> tuple[typing.Any, typing.Any]:
    """The case's model fitted on ``yi`` and again on ``sign * yi + shift``."""
    frame = correctness.cases.case_frame(case)
    assert frame is not None
    changed = frame.assign(yi=sign * frame['yi'].to_numpy(dtype=numpy.float64) + shift)
    return uni_fit(case, frame), uni_fit(case, changed)


def selmodel_without_selection_is_ml_fit(case: correctness.cases.Case, result: typing.Any) -> None:
    """``selmodel`` of any ``type`` with every ``delta`` fixed at its no-selection value (weight 1 everywhere) is the ML ``rma_uni`` fit: the same
    ``beta``, ``tau2`` and log-likelihood (Vevea and Hedges 1995: the model reduces to the usual one when all weights are equal)."""
    frame = correctness.cases.case_frame(case)
    assert frame is not None
    ml = uni_fit(case, frame)
    tolerance = case.tolerance or SELMODEL_AGREEMENT
    close(case.id, 'beta', result.beta, ml.beta, tolerance)
    close(case.id, 'tau2', result.tau2, ml.tau2, tolerance)
    close(case.id, 'log-likelihood', result.ll, float(ml.logLik()), tolerance)


def reflection_negates_regtest(case: correctness.cases.Case, result: typing.Any) -> None:
    """Reflecting ``yi`` to ``-yi`` negates the test statistic ``zval`` and the limit estimate ``est`` of ``regtest``, swaps and negates the limits of
    its interval, and keeps the p-value. Run over ``model`` in ``REGTEST_MODELS`` and ``predictor`` in ``REGTEST_PREDICTORS``."""
    before, after = reflected_fit(case, sign=-1.0)
    tolerance = case.tolerance or BIAS_AGREEMENT
    for model in REGTEST_MODELS:
        for predictor in REGTEST_PREDICTORS:
            first = metafopy.regtest(before, model=model, predictor=predictor)
            second = metafopy.regtest(after, model=model, predictor=predictor)
            where = f'{case.id} (model {model}, predictor {predictor})'
            assert first.est is not None, f'{where}: no limit estimate'
            assert first.ci_lb is not None, f'{where}: no lower limit'
            assert first.ci_ub is not None, f'{where}: no upper limit'
            close(where, 'zval', second.zval, -first.zval, tolerance)
            close(where, 'pval', second.pval, first.pval, tolerance)
            close(where, 'est', second.est, -first.est, tolerance)
            close(where, 'ci_lb', second.ci_lb, -first.ci_ub, tolerance)
            close(where, 'ci_ub', second.ci_ub, -first.ci_lb, tolerance)


def reflection_and_shift_leave_ranktest_p(case: correctness.cases.Case, result: typing.Any) -> None:
    """Reflecting ``yi`` negates Kendall's ``tau`` of ``ranktest`` and keeps its p-value, and adding a constant to ``yi`` changes neither: the
    standardised effects ``(yi - theta) / sqrt(vi - vb)`` are the same up to sign. Run with ``exact`` True and False."""
    tolerance = case.tolerance or BIAS_AGREEMENT
    before, reflected = reflected_fit(case, sign=-1.0)
    _, shifted = reflected_fit(case, shift=LOCATION_SHIFT)
    for exact in (True, False):
        first = metafopy.ranktest(before, exact=exact)
        where = f'{case.id} (exact {exact})'
        close(where, 'tau after reflection', metafopy.ranktest(reflected, exact=exact).tau, -first.tau, tolerance)
        close(where, 'pval after reflection', metafopy.ranktest(reflected, exact=exact).pval, first.pval, tolerance)
        close(where, 'tau after a shift', metafopy.ranktest(shifted, exact=exact).tau, first.tau, tolerance)
        close(where, 'pval after a shift', metafopy.ranktest(shifted, exact=exact).pval, first.pval, tolerance)


def reflection_moves_trimfill_side(case: correctness.cases.Case, result: typing.Any) -> None:
    """Reflecting ``yi`` to ``-yi`` with ``side`` unset (estimated) gives the same ``k0``, the opposite ``side``, a negated pooled estimate and the
    negated filled-in effects. Run with the case's ``estimator``."""
    before, after = reflected_fit(case, sign=-1.0)
    first = metafopy.trimfill(before, **case.args)
    second = metafopy.trimfill(after, **case.args)
    tolerance = case.tolerance or BIAS_AGREEMENT
    assert second.k0 == first.k0, f'{case.id}: k0 is {second.k0} after reflection, {first.k0} before'
    assert first.k0 > 0, f'{case.id}: the case must have missing studies'
    assert {first.side, second.side} == {'left', 'right'}, f'{case.id}: side is {first.side!r} before and {second.side!r} after reflection'
    close(case.id, 'beta', second.beta, -first.beta, tolerance)
    close(case.id, 'se', second.se, first.se, tolerance)
    close(case.id, 'filled-in effects', numpy.sort(second.yi), numpy.sort(-first.yi), tolerance)


def trimfill_without_missing_returns_the_model(case: correctness.cases.Case, result: typing.Any) -> None:
    """When ``trimfill`` finds no missing study (``k0 = 0``) it returns the input model: the same studies, ``beta``, ``se`` and ``tau2``, none filled in."""
    frame = correctness.cases.case_frame(case)
    assert frame is not None
    fit = uni_fit(case, frame)
    tolerance = case.tolerance or BIAS_AGREEMENT
    assert result.k0 == 0, f'{case.id}: k0 is {result.k0}, the case must have no missing studies'
    assert result.k == fit.k, f'{case.id}: {result.k} studies, the model has {fit.k}'
    assert not numpy.any(result.fill), f'{case.id}: {int(numpy.sum(result.fill))} studies filled in'
    # auditme: accept[literal-duplication] the attributes of a fit that two calls must share; a test module names its own
    for name in ('beta', 'se', 'tau2'):
        close(case.id, name, getattr(result, name), getattr(fit, name), tolerance)
    close(case.id, 'yi', result.yi, fit.yi, tolerance)


def hc_with_zero_tau2_is_the_equal_effects_estimate(case: correctness.cases.Case, result: typing.Any) -> None:
    """With a DerSimonian-Laird ``tau2`` of 0 the ``hc`` estimate and standard error are those of the equal-effects model: the weights are ``1 / vi``."""
    frame = correctness.cases.case_frame(case)
    assert frame is not None
    equal = metafopy.rma_uni(yi='yi', vi='vi', data=frame, method='EE')
    tolerance = case.tolerance or BIAS_AGREEMENT
    assert result.tau2 == 0.0, f'{case.id}: the DerSimonian-Laird tau2 is {result.tau2!r}, the case must have 0'
    close(case.id, 'beta', result.beta, equal.beta, tolerance)
    close(case.id, 'se', result.se, equal.se, tolerance)


def hc_with_zero_tau2_is_the_equal_effects_interval(case: correctness.cases.Case, result: typing.Any) -> None:
    """With a DerSimonian-Laird ``tau2`` of 0 the ``hc`` interval is the equal-effects interval (the spec's property)."""
    frame = correctness.cases.case_frame(case)
    assert frame is not None
    equal = metafopy.rma_uni(yi='yi', vi='vi', data=frame, method='EE')
    tolerance = case.tolerance or BIAS_AGREEMENT
    close(case.id, 'ci_lb', result.ci_lb, equal.ci_lb[0], tolerance)
    close(case.id, 'ci_ub', result.ci_ub, equal.ci_ub[0], tolerance)
