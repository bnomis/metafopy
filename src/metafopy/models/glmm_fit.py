"""Model fitting behind ``rma_glmm``: fixed, saturated and mixed-effects GLMMs of metafor's ``rma.glmm``.

The four models follow Stijnen, Hamza and Ozdemir (2010), doi:10.1002/sim.3767,
and Viechtbauer's documentation at https://wviechtb.github.io/metafor/reference/rma.glmm.html:
``UM.FS`` and ``UM.RS`` are unconditional logistic (``OR``) or Poisson (``IRR``)
models with fixed or random study effects, ``CM.AL`` and ``CM.EL`` condition on
the table margins and use the approximate (binomial) or the exact
(non-central hypergeometric) likelihood. ``PLO`` (log odds) and ``IRLN`` (log
rates) are one-group logistic and Poisson mixed models. ``metafor`` fits the mixed models with
``lme4`` (the default), ``GLMMadaptive`` or ``glmmTMB`` (``control=list(package=...)``); here
:func:`fit_mixed` maximises the same likelihood and reports it as the chosen package does. The experimental
measures use binomial models with a log link (``RR``, ``PLN``) or an identity
link (``RD``, ``PR``); ``RR`` and ``RD`` are unconditional two-group models,
``PR`` and ``PLN`` one-group models. This module mirrors
the corresponding branches of
https://github.com/wviechtb/metafor/blob/master/R/rma.glmm.r.
"""

import dataclasses
import math
import sys
import typing
import warnings

import numpy
import numpy.typing
import scipy.integrate
import scipy.optimize
import scipy.stats

import metafopy.core.glm
import metafopy.core.glmm
import metafopy.core.glmmadaptive
import metafopy.core.nchg
import metafopy.core.numderiv
import metafopy.errors

__all__ = ['FitSpec', 'GLMMControl', 'ModelFit', 'fit_model']

type FloatArray = numpy.typing.NDArray[numpy.float64]
type IntArray = numpy.typing.NDArray[numpy.intp]

NAN = math.nan
# auditme: accept[non-alphabetical-literal] the order follows metafor's documented option order
PACKAGES = ('lme4', 'GLMMadaptive', 'glmmTMB')
GLMMADAPTIVE_MAX_COEF = 12.0
GLMMADAPTIVE_MAX_COEF_SATURATED = 50.0
SATURATED_WARNING = "Cannot fit saturated model (set 'verbose=TRUE' to obtain further details)."


@dataclasses.dataclass(frozen=True, slots=True)
class GLMMControl:
    """Control settings of ``rma_glmm`` (names follow metafor's ``control`` list with ``.`` replaced by ``_``).

    ``int_rel_tol`` and ``int_subdivisions`` are the ``rel.tol`` and ``subdivisions`` of the ``intCtrl``
    list used for the integral in the exact conditional model; ``hessian_r`` is numDeriv's ``r``.
    ``package`` is the package that fits the mixed models: ``'lme4'``, ``'GLMMadaptive'`` or ``'glmmTMB'``.
    """

    package: str = 'lme4'
    scale_x: bool = True
    evtol: float = 1e-7
    tau2tol: float = 1e-4
    int_rel_tol: float = sys.float_info.epsilon**0.25
    int_subdivisions: int = 100
    hessian_r: int = 16
    gtol: float = 1e-7


@dataclasses.dataclass(frozen=True, slots=True)
class FitSpec:
    """Everything the model fitting needs: ``X`` (``k x p``, possibly rescaled) and the table counts used in fitting."""

    measure: str
    model: str
    method: str
    X: FloatArray
    cells: dict[str, FloatArray]
    coding: float
    cor: bool
    nagq: int
    alpha: float
    control: GLMMControl


@dataclasses.dataclass(frozen=True, slots=True)
class ModelFit:
    """Estimates and log-likelihoods of one fitted model (``beta`` and ``vb`` refer to the possibly rescaled ``X``)."""

    beta: FloatArray
    vb: FloatArray
    tau2: float
    sigma2: float
    rho: float
    se_tau2: float
    ci_lb_tau2: float
    ci_ub_tau2: float
    ll_fe: float
    ll_qe: float
    ll_ml: float
    b2_qe: FloatArray
    vb2_qe: FloatArray
    qe_wld: float | None
    qe_converged: bool
    parms: int
    p_eff: int
    k_eff: int
    method: str
    se_warn: bool = False


def interleave(first: FloatArray, second: FloatArray) -> FloatArray:
    """``c(rbind(first, second))``: the two groups of each study in adjacent rows."""
    return numpy.column_stack([first, second]).ravel()


def study_dummies(k: int) -> FloatArray:
    """Indicator columns of the ``k`` studies for observations ordered two per study."""
    return numpy.repeat(numpy.eye(k), 2, axis=0)


def empty_vcov() -> FloatArray:
    return numpy.zeros((0, 0))


def fit_mixed(
    spec: FitSpec,
    problem: metafopy.core.glmm.GLMMProblem,
    *,
    nagq: int | None = None,
    theta0: FloatArray | None = None,
    d0: FloatArray | None = None,
    max_coef: float = GLMMADAPTIVE_MAX_COEF,
) -> metafopy.core.glmm.GLMMFit:
    """Fit a mixed model as the package chosen with ``control$package`` does.

    ``lme4`` follows ``glmer()`` (start value ``theta0``), ``glmmTMB`` maximises the Laplace approximation
    (``nAGQ`` is ignored) and ``GLMMadaptive`` repeats ``mixed_model()``'s iterations (start value ``d0`` for the
    covariance matrix of the random effects) with adaptive quadrature in every dimension; it stops with an error
    when a coefficient other than the first exceeds ``max_coef`` in absolute value, which metafor turns into a failed fit.

    Raises:
        metafopy.errors.EstimationError: the model cannot be fitted.
    """
    points = spec.nagq if nagq is None else nagq
    package = spec.control.package
    if package == 'lme4':
        return metafopy.core.glmm.fit_glmm(problem, nagq=points, theta0=theta0, gtol=spec.control.gtol)
    if package == 'GLMMadaptive':
        control = metafopy.core.glmmadaptive.AdaptiveControl(max_coef_value=max_coef)
        return metafopy.core.glmmadaptive.fit_adaptive(problem, nagq=points, d0=d0, control=control)
    observed = dataclasses.replace(problem, observed=True)
    return metafopy.core.glmm.fit_glmm(observed, nagq=1, theta0=theta0, gtol=spec.control.gtol, lme4_conventions=False)


def ml_loglik(spec: FitSpec, ml: metafopy.core.glmm.GLMMFit, ll_qe: float, *, nan_falls_back: bool) -> float:
    """Log-likelihood of the mixed model.

    ``lme4``: the saturated model's minus half the conditional deviance, which makes it equal to the fixed-effects
    model's for ``tau2 = 0`` and is missing when the saturated model is (``logLik()`` instead with ``nan_falls_back``).
    ``GLMMadaptive`` and ``glmmTMB``: always ``logLik()``.
    """
    if spec.control.package != 'lme4' or (nan_falls_back and math.isnan(ll_qe)):
        return ml.loglik
    return ll_qe - 0.5 * ml.cond_deviance


# auditme: accept[non-alphabetical-literal] the order follows metafor's documented option order
BINOMIAL_MEASURES = ('OR', 'RR', 'RD', 'PLO', 'PR', 'PLN')
# auditme: accept[non-alphabetical-literal] the order follows metafor's documented option order
TWO_GROUP_BINOMIAL = ('OR', 'RR', 'RD')
# auditme: accept[non-alphabetical-literal] the order follows metafor's documented option order
ONE_GROUP = ('PLO', 'PR', 'PLN', 'IRLN')
LINKS: dict[str, metafopy.core.glm.Link] = {
    'IRLN': 'log',
    'IRR': 'log',
    'OR': 'logit',
    'PLN': 'log',
    'PLO': 'logit',
    'PR': 'identity',
    'RD': 'identity',
    'RR': 'log',
}


def glm_family(measure: str) -> metafopy.core.glm.Family:
    return 'binomial' if measure in BINOMIAL_MEASURES else 'poisson'


def glm_link(measure: str) -> metafopy.core.glm.Link:
    """The link of the measure's model: logit (``OR``, ``PLO``), log (``RR``, ``PLN``, ``IRR``, ``IRLN``) or identity (``RD``, ``PR``)."""
    return LINKS[measure]


def fixed_glm(
    X: FloatArray, y: FloatArray, size: FloatArray | None, offset: FloatArray, family: metafopy.core.glm.Family, link: metafopy.core.glm.Link
) -> metafopy.core.glm.GLMFit:
    """``glm()`` fit; raises the metafor-style error when it fails.

    Raises:
        metafopy.errors.EstimationError: the model cannot be fitted.
    """
    try:
        return metafopy.core.glm.glm_fit(X, y, family=family, size=size, offset=offset, link=link)
    except (metafopy.errors.EstimationError, numpy.linalg.LinAlgError) as exc:
        raise metafopy.errors.EstimationError('Cannot fit FE model.') from exc


def fixed_effects_result(
    fit: metafopy.core.glm.GLMFit, p: int, parms: int, p_eff: int, k_eff: int, sigma2: float, qe: tuple[float, FloatArray, FloatArray, bool]
) -> ModelFit:
    """``ModelFit`` for the fixed-effects methods taken from a ``glm`` fit (``tau2 = 0``)."""
    ll_qe, b2, vb2, converged = qe
    return ModelFit(
        beta=fit.coef[:p],
        vb=fit.vcov[:p, :p],
        tau2=0.0,
        sigma2=sigma2,
        rho=NAN,
        se_tau2=NAN,
        ci_lb_tau2=NAN,
        ci_ub_tau2=NAN,
        ll_fe=fit.loglik,
        ll_qe=ll_qe,
        ll_ml=NAN,
        b2_qe=b2,
        vb2_qe=vb2,
        qe_wld=None,
        qe_converged=converged,
        parms=parms,
        p_eff=p_eff,
        k_eff=k_eff,
        method='FE',
    )


def saturated_glm(
    X_qe: FloatArray,
    y: FloatArray,
    size: FloatArray | None,
    offset: FloatArray,
    family: metafopy.core.glm.Family,
    link: metafopy.core.glm.Link,
    skip: int,
) -> tuple[float, FloatArray, FloatArray, bool, metafopy.core.glm.GLMFit | None]:
    """Log-likelihood and the coefficients/covariance beyond the first ``skip`` of the saturated model (``glm``).

    Returns NaN values and ``False`` when the fit fails, as metafor only warns then.
    """
    try:
        fit = metafopy.core.glm.glm_fit(X_qe, y, family=family, size=size, offset=offset, link=link)
    except metafopy.errors.EstimationError, numpy.linalg.LinAlgError:
        warnings.warn(SATURATED_WARNING, stacklevel=3)
        return NAN, numpy.zeros(0), empty_vcov(), False, None
    coef = fit.coef[~fit.aliased]
    return fit.loglik, coef[skip:], fit.vcov[skip:, skip:], True, fit


def unconditional(spec: FitSpec) -> ModelFit:
    """Models ``UM.FS`` and ``UM.RS`` for two-group counts (``OR``) or rates (``IRR``).

    Raises:
        metafopy.errors.EstimationError: the models cannot be fitted.
    """
    k = spec.X.shape[0]
    c = spec.cells
    if spec.measure in TWO_GROUP_BINOMIAL:
        family: metafopy.core.glm.Family = 'binomial'
        y = interleave(c['ai'], c['ci'])
        size: FloatArray | None = interleave(c['ai'] + c['bi'], c['ci'] + c['di'])
        offset = numpy.zeros(2 * k)
    else:
        family = 'poisson'
        y = interleave(c['x1i'], c['x2i'])
        size = None
        offset = numpy.log(interleave(c['t1i'], c['t2i']))
    group1 = numpy.tile([1.0, 0.0], k)
    group2 = numpy.tile([0.0, 1.0], k)
    group12 = numpy.tile([0.5, -0.5], k)
    group = {0.5: group12, 1.0: group1, 0.0: group2}[spec.coding]
    study = numpy.repeat(numpy.arange(k), 2)
    const = numpy.ones((2 * k, 1))
    X_fit = group1[:, None] * spec.X[study]
    if spec.model == 'UM.FS':
        return unconditional_fixed_study(spec, family, y, size, offset, X_fit, group1, group, study, const)
    return unconditional_random_study(spec, family, y, size, offset, X_fit, group1, group, study, const)


def unconditional_fixed_study(
    spec: FitSpec,
    family: metafopy.core.glm.Family,
    y: FloatArray,
    size: FloatArray | None,
    offset: FloatArray,
    X_fit: FloatArray,
    group1: FloatArray,
    group: FloatArray,
    study: IntArray,
    const: FloatArray,
) -> ModelFit:
    k, p = spec.X.shape
    dummies = study_dummies(k)
    link = glm_link(spec.measure)
    fe = fixed_glm(numpy.column_stack([X_fit, dummies if k > 1 else const]), y, size, offset, family, link)
    ll_qe, b2, vb2, qe_ok = fe.loglik, numpy.zeros(0), empty_vcov(), True
    if k > 1:
        X_qe = numpy.column_stack([X_fit, dummies, dummies * group1[:, None]])
        ll_qe, b2, vb2, qe_ok, _ = saturated_glm(X_qe, y, size, offset, family, link, k + p)
    if spec.method != 'ML':
        return fixed_effects_result(fe, p, p + k, p + k, 2 * k, NAN, (ll_qe, b2, vb2, qe_ok))
    problem = metafopy.core.glmm.GLMMProblem(numpy.column_stack([X_fit, dummies]), y, size, offset, study, group[:, None], family, 'scalar', link)
    ml = fit_mixed(spec, problem)
    ll_ml = ml_loglik(spec, ml, ll_qe, nan_falls_back=True)
    return ModelFit(
        beta=ml.beta[:p],
        vb=ml.vcov[:p, :p],
        tau2=float(ml.theta[0] ** 2),
        sigma2=NAN,
        rho=NAN,
        se_tau2=NAN,
        ci_lb_tau2=NAN,
        ci_ub_tau2=NAN,
        ll_fe=fe.loglik,
        ll_qe=ll_qe,
        ll_ml=ll_ml,
        b2_qe=b2,
        vb2_qe=vb2,
        qe_wld=None,
        qe_converged=qe_ok,
        parms=p + k + 1,
        p_eff=p + k,
        k_eff=2 * k,
        method='ML',
    )


def unconditional_random_study(
    spec: FitSpec,
    family: metafopy.core.glm.Family,
    y: FloatArray,
    size: FloatArray | None,
    offset: FloatArray,
    X_fit: FloatArray,
    group1: FloatArray,
    group: FloatArray,
    study: IntArray,
    const: FloatArray,
) -> ModelFit:
    k, p = spec.X.shape
    nagq = 1 if spec.method == 'ML' and spec.control.package == 'lme4' else spec.nagq
    ones = const
    link = glm_link(spec.measure)
    fe_problem = metafopy.core.glmm.GLMMProblem(numpy.column_stack([X_fit, const]), y, size, offset, study, ones, family, 'scalar', link)
    fe = fit_mixed(spec, fe_problem, nagq=nagq)
    ll_qe, b2, vb2, qe_ok = fe.loglik, numpy.zeros(0), empty_vcov(), True
    if k > 1:
        interaction = study_dummies(k) * group1[:, None]
        X_qe = numpy.column_stack([X_fit, const, interaction])
        probe = metafopy.core.glm.glm_fit(X_qe, y, family=family, size=size, offset=offset, link=link)
        X_qe = X_qe[:, ~probe.aliased]
        qe_problem = metafopy.core.glmm.GLMMProblem(X_qe, y, size, offset, study, ones, family, 'scalar', link)
        try:
            d0 = fe.covariance() if family == 'binomial' else None
            qe = fit_mixed(spec, qe_problem, nagq=nagq, theta0=fe.theta, d0=d0, max_coef=GLMMADAPTIVE_MAX_COEF_SATURATED)
            ll_qe, b2, vb2 = qe.loglik, qe.beta[p + 1 :], qe.vcov[p + 1 :, p + 1 :]
        except metafopy.errors.EstimationError:
            warnings.warn(SATURATED_WARNING, stacklevel=3)
            ll_qe, b2, vb2, qe_ok = NAN, numpy.zeros(0), empty_vcov(), False
    if spec.method != 'ML':
        return ModelFit(
            beta=fe.beta[:p],
            vb=fe.vcov[:p, :p],
            tau2=0.0,
            sigma2=float(fe.theta[0] ** 2),
            rho=NAN,
            se_tau2=NAN,
            ci_lb_tau2=NAN,
            ci_ub_tau2=NAN,
            ll_fe=fe.loglik,
            ll_qe=ll_qe,
            ll_ml=NAN,
            b2_qe=b2,
            vb2_qe=vb2,
            qe_wld=None,
            qe_converged=qe_ok,
            parms=p + 2,
            p_eff=p + 1,
            k_eff=2 * k,
            method='FE',
        )
    Z = numpy.column_stack([numpy.ones(2 * k), group])
    ml_problem = metafopy.core.glmm.GLMMProblem(
        numpy.column_stack([X_fit, const]), y, size, offset, study, Z, family, 'correlated' if spec.cor else 'independent', link
    )
    ml = fit_mixed(
        spec, ml_problem, nagq=1 if spec.control.package == 'lme4' else spec.nagq, max_coef=GLMMADAPTIVE_MAX_COEF_SATURATED if k > 1 else GLMMADAPTIVE_MAX_COEF
    )
    cov = ml.covariance()
    sigma2, tau2 = float(cov[0, 0]), float(cov[1, 1])
    rho = float(cov[0, 1] / math.sqrt(tau2 * sigma2)) if spec.cor and tau2 * sigma2 > 0 else NAN
    return ModelFit(
        beta=ml.beta[:p],
        vb=ml.vcov[:p, :p],
        tau2=tau2,
        sigma2=sigma2,
        rho=rho,
        se_tau2=NAN,
        ci_lb_tau2=NAN,
        ci_ub_tau2=NAN,
        ll_fe=fe.loglik,
        ll_qe=ll_qe,
        ll_ml=ml.loglik,
        b2_qe=b2,
        vb2_qe=vb2,
        qe_wld=None,
        qe_converged=qe_ok,
        parms=p + 1 + 2,
        p_eff=p + 1,
        k_eff=2 * k,
        method='ML',
    )


@dataclasses.dataclass(frozen=True, slots=True)
class SingleLevel:
    """Intermediate results of the single-random-effect models that the exact conditional model starts from."""

    fit: ModelFit
    fe: metafopy.core.glm.GLMFit
    qe: metafopy.core.glm.GLMFit | None
    X_qe: FloatArray
    ml_beta: FloatArray
    ml_tau2: float


def single_level_data(spec: FitSpec) -> tuple[metafopy.core.glm.Family, FloatArray, FloatArray | None, FloatArray]:
    """Counts ``y``, binomial totals, offset of the conditional (``CM.*``) and one-group (``PLO``, ``PR``, ``PLN``, ``IRLN``) models."""
    c = spec.cells
    k = spec.X.shape[0]
    if spec.measure == 'OR':
        return 'binomial', c['ai'], c['ai'] + c['ci'], numpy.log((c['ai'] + c['bi']) / (c['ci'] + c['di']))
    if spec.measure == 'IRR':
        return 'binomial', c['x1i'], c['x1i'] + c['x2i'], numpy.log(c['t1i'] / c['t2i'])
    if spec.measure in ('PLO', 'PR', 'PLN'):
        return 'binomial', c['xi'], c['xi'] + c['mi'], numpy.zeros(k)
    return 'poisson', c['xi'], None, numpy.log(c['ti'])


def single_level(spec: FitSpec) -> SingleLevel:
    """Fixed-effects, saturated and (for ``method='ML'``) mixed-effects model with one random intercept per table.

    Used for ``CM.AL`` (odds ratios), ``CM.EL`` (incidence rate ratios; also the start values for odds ratios) and the one-group measures.

    Raises:
        metafopy.errors.EstimationError: a model cannot be fitted.
    """
    k, p = spec.X.shape
    family, y, size, offset = single_level_data(spec)
    X = spec.X
    link = glm_link(spec.measure) if spec.measure in ONE_GROUP else metafopy.core.glm.canonical_link(family)
    fe = fixed_glm(X, y, size, offset, family, link)
    ll_qe, b2, vb2, qe_ok = fe.loglik, numpy.zeros(0), empty_vcov(), True
    qe_fit: metafopy.core.glm.GLMFit | None = fe
    X_qe = X
    if k > 1:
        X_qe = numpy.column_stack([X, numpy.eye(k)])
        ll_qe, b2, vb2, qe_ok, qe_fit = saturated_glm(X_qe, y, size, offset, family, link, p)
    if spec.method != 'ML':
        return SingleLevel(fixed_effects_result(fe, p, p, p, k, NAN, (ll_qe, b2, vb2, qe_ok)), fe, qe_fit, X_qe, fe.coef[:p], 0.0)
    problem = metafopy.core.glmm.GLMMProblem(X, y, size, offset, numpy.arange(k), numpy.ones((k, 1)), family, 'scalar', link)
    ml = fit_mixed(spec, problem)
    # auditme: accept[literal-duplication] option and column lists repeat as metafor repeats them
    ll_ml = ml_loglik(spec, ml, ll_qe, nan_falls_back=spec.measure in ('OR', 'IRR'))
    tau2 = float(ml.theta[0] ** 2)
    fit = ModelFit(
        beta=ml.beta[:p],
        vb=ml.vcov[:p, :p],
        tau2=tau2,
        sigma2=NAN,
        rho=NAN,
        se_tau2=NAN,
        ci_lb_tau2=NAN,
        ci_ub_tau2=NAN,
        ll_fe=fe.loglik,
        ll_qe=ll_qe,
        ll_ml=ll_ml,
        b2_qe=b2,
        vb2_qe=vb2,
        qe_wld=None,
        qe_converged=qe_ok,
        parms=p + 1,
        p_eff=p,
        k_eff=k,
        method='ML',
    )
    return SingleLevel(fit, fe, qe_fit, X_qe, ml.beta[:p], tau2)


def minimise(func: typing.Callable[[FloatArray], float], x0: FloatArray, gtol: float) -> scipy.optimize.OptimizeResult:
    """BFGS with central-difference gradients, restarted once from its solution.

    Raises:
        metafopy.errors.EstimationError: the optimiser returns non-finite values.
    """
    with numpy.errstate(all='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = scipy.optimize.minimize(func, x0, method='BFGS', jac='3-point', options={'gtol': gtol, 'maxiter': 500})
        res = scipy.optimize.minimize(func, res.x, method='BFGS', jac='3-point', options={'gtol': gtol, 'maxiter': 500})
    if not numpy.all(numpy.isfinite(res.x)) or not math.isfinite(res.fun):
        raise metafopy.errors.EstimationError('Optimizer did not achieve convergence.')
    return res


def invert_hessian(h: FloatArray, what: str) -> FloatArray:
    """``chol2inv(chol(h))`` with R's fall back to a QR-based solve (a warning is issued then).

    Raises:
        metafopy.errors.EstimationError: the Hessian cannot be inverted.
    """
    if not numpy.all(numpy.isfinite(h)):
        raise metafopy.errors.EstimationError(f'Cannot invert the Hessian for the {what} model.')
    try:
        chol = numpy.linalg.cholesky(h)
    except numpy.linalg.LinAlgError:
        warnings.warn('Choleski factorization of Hessian failed. Trying inversion via QR decomposition.', stacklevel=3)
        try:
            return numpy.asarray(numpy.linalg.inv(h), dtype=numpy.float64)
        except numpy.linalg.LinAlgError as exc:
            raise metafopy.errors.EstimationError(f'Cannot invert the Hessian for the {what} model.') from exc
    inv = numpy.linalg.inv(chol)
    return numpy.asarray(inv.T @ inv, dtype=numpy.float64)


def saturated_wald(par: FloatArray, hessian: FloatArray, p: int) -> float:
    """Wald statistic of the extra coefficients of the saturated exact-likelihood model from its Hessian (Schur complement)."""
    q = par.shape[0]
    zero_col = (hessian == 0).sum(axis=0) == q
    b2 = par[~zero_col][p:]
    h = hessian[numpy.ix_(~zero_col, ~zero_col)]
    h_a, h_b, h_c, h_d = h[:p, :p], h[:p, p:], h[p:, :p], h[p:, p:]
    try:
        chol = numpy.linalg.cholesky(h_a)
    except numpy.linalg.LinAlgError:
        warnings.warn('Cannot invert the Hessian for the saturated model.', stacklevel=3)
        return NAN
    inv = numpy.linalg.inv(chol)
    ivb2 = h_d - h_c @ (inv.T @ inv) @ h_b
    return float(b2 @ ivb2 @ b2)


def exact_conditional(spec: FitSpec, base: SingleLevel) -> ModelFit:
    """``CM.EL`` for odds ratios: maximise the non-central hypergeometric likelihood started from the ``CM.AL`` fit.

    Raises:
        metafopy.errors.EstimationError: an optimisation fails or the Hessian cannot be inverted.
    """
    c = spec.cells
    k, p = spec.X.shape
    ctl = spec.control
    tables = metafopy.core.nchg.Tables.from_counts(c['ai'], c['bi'], c['ci'], c['di'])
    integ = metafopy.core.nchg.IntegrationControl(ctl.int_rel_tol, ctl.int_subdivisions)

    def fixed_objective(X: FloatArray) -> typing.Callable[[FloatArray], float]:
        return lambda b: metafopy.core.nchg.neg_loglik(b, tables, X, False, integ)

    def hess(func: typing.Callable[[FloatArray], float], x: FloatArray) -> FloatArray:
        return metafopy.core.numderiv.hessian(func, x, r=ctl.hessian_r)

    f_fe = fixed_objective(spec.X)
    res_fe = minimise(f_fe, base.fe.coef[:p], ctl.gtol)
    h_fe = hess(f_fe, res_fe.x)
    ll_fe = -float(res_fe.fun)

    ll_qe, b2, vb2 = NAN, numpy.zeros(0), empty_vcov()
    qe_wld: float | None = None
    qe_ok = False
    if base.fit.qe_converged and base.qe is not None:
        if k > 1:
            keep = ~base.qe.aliased
            X_qe = base.X_qe[:, keep]
            f_qe = fixed_objective(X_qe)
            res_qe = minimise(f_qe, base.qe.coef[keep], ctl.gtol)
            h_qe = hess(f_qe, res_qe.x)
            ll_qe = -float(res_qe.fun)
            qe_wld = saturated_wald(res_qe.x, h_qe, p)
        else:
            ll_qe = ll_fe
        qe_ok = True

    if spec.method != 'ML':
        vb = invert_hessian(h_fe, spec.method)
        return ModelFit(res_fe.x, vb, 0.0, NAN, NAN, NAN, NAN, NAN, ll_fe, ll_qe, NAN, b2, vb2, qe_wld, qe_ok, p, p, k, 'FE')

    def f_ml(par: FloatArray) -> float:
        return metafopy.core.nchg.neg_loglik(par, tables, spec.X, True, integ)

    res_ml = minimise(f_ml, numpy.append(base.ml_beta, math.log(base.ml_tau2 + 0.01)), ctl.gtol)
    ll_ml = -float(res_ml.fun)
    tau2 = math.exp(float(res_ml.x[p]))
    if tau2 < ctl.tau2tol:
        vb = invert_hessian(h_fe, 'ML')
        return ModelFit(res_fe.x, vb, tau2, NAN, NAN, 0.0, NAN, NAN, ll_fe, ll_qe, ll_ml, b2, vb2, qe_wld, qe_ok, p + 1, p, k, 'ML')
    h_ml = hess(f_ml, res_ml.x)
    vb_full = invert_hessian(h_ml, 'ML')
    vb = vb_full[:p, :p]
    if numpy.any(numpy.diag(vb) <= 0):
        raise metafopy.errors.EstimationError('Cannot compute var-cov matrix of the fixed effects.')
    se_tau2 = ci_lb = ci_ub = NAN
    if vb_full[p, p] >= 0:
        se_tau2 = math.sqrt(float(vb_full[p, p])) * tau2
        crit = float(scipy.stats.norm.isf(spec.alpha / 2))
        ci_lb = math.exp(float(res_ml.x[p]) - crit * math.sqrt(float(vb_full[p, p])))
        ci_ub = math.exp(float(res_ml.x[p]) + crit * math.sqrt(float(vb_full[p, p])))
    return ModelFit(res_ml.x[:p], vb, tau2, NAN, NAN, se_tau2, ci_lb, ci_ub, ll_fe, ll_qe, ll_ml, b2, vb2, qe_wld, qe_ok, p + 1, p, k, 'ML')


def fit_model(spec: FitSpec) -> ModelFit:
    """Fit the model requested by ``spec`` (``model``, ``method``, ``measure``).

    Raises:
        metafopy.errors.EstimationError: the model cannot be fitted.
    """
    # auditme: accept[literal-duplication] option and column lists repeat as metafor repeats them
    if spec.measure in ONE_GROUP:
        return single_level(spec).fit
    if spec.model in ('UM.FS', 'UM.RS'):
        return unconditional(spec)
    base = single_level(spec)
    if spec.model == 'CM.EL' and spec.measure == 'OR':
        return exact_conditional(spec, base)
    return base.fit
