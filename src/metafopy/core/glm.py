"""Binomial and Poisson generalized linear models fitted like R's ``glm()`` / ``glm.fit()``.

``rma.glmm`` (https://github.com/wviechtb/metafor/blob/master/R/rma.glmm.r) fits
its fixed-effects and saturated models with ``glm()``. Its log-likelihoods,
tests for heterogeneity and the set of aliased columns depend on where the
iteratively reweighted least squares (IRLS) iterations stop, so the algorithm
of ``glm.fit`` is followed step by step: the same starting values
(``mustart``), the same convergence rule (relative change of the deviance below
``epsilon``, at most ``maxit`` iterations), the same clamps of the logit and
log links, and the covariance matrix of the last weighted least squares fit.

Reference: McCullagh and Nelder (1989), *Generalized Linear Models*, 2nd ed.,
and the R source ``src/library/stats/R/glm.R`` and ``src/library/stats/src/family.c``.
"""

import dataclasses
import math
import sys
import typing

import numpy
import numpy.typing
import scipy.special
import scipy.stats

import metafopy.errors

__all__ = ['GLMFit', 'aliased_columns', 'deviance_residuals', 'family_constant', 'glm_fit', 'saturated_loglik']

type FloatArray = numpy.typing.NDArray[numpy.float64]
type BoolArray = numpy.typing.NDArray[numpy.bool_]
type Family = typing.Literal['binomial', 'poisson']

DBL_EPSILON = sys.float_info.epsilon
LOGIT_THRESHOLD = 30.0


@dataclasses.dataclass(frozen=True, slots=True)
class GLMFit:
    """A fitted GLM. ``coef`` holds NaN for aliased columns; ``vcov`` covers the other columns in order."""

    coef: FloatArray
    vcov: FloatArray
    aliased: BoolArray
    eta: FloatArray
    mu: FloatArray
    loglik: float
    deviance: float
    converged: bool
    iterations: int

    def complete_vcov(self) -> FloatArray:
        """``vcov(fit, complete=TRUE)`` with NaN rows and columns for the aliased coefficients."""
        q = self.coef.shape[0]
        full = numpy.full((q, q), numpy.nan)
        keep = numpy.flatnonzero(~self.aliased)
        full[numpy.ix_(keep, keep)] = self.vcov
        return full


def aliased_columns(X: FloatArray, tol: float = 1e-11) -> BoolArray:
    """Columns that are numerically linear combinations of earlier ones (the ``NA`` coefficients of ``glm``).

    LINPACK's ``dqrdc2`` moves a column to the end when its norm after
    orthogonalisation against the preceding columns is below ``tol`` times its
    original norm; the same rule is applied here with Gram-Schmidt steps.
    """
    q = X.shape[1]
    aliased = numpy.zeros(q, dtype=numpy.bool_)
    basis: list[FloatArray] = []
    for j in range(q):
        col = X[:, j].astype(numpy.float64)
        norm0 = float(numpy.linalg.norm(col))
        resid = col.copy()
        for _ in range(2):
            for b in basis:
                resid = resid - b * float(b @ resid)
        norm1 = float(numpy.linalg.norm(resid))
        if norm0 == 0.0 or norm1 < tol * norm0:
            aliased[j] = True
        else:
            basis.append(resid / norm1)
    return aliased


def expit_clamped(eta: FloatArray) -> FloatArray:
    """R's ``binomial()$linkinv``: the logistic function with the linear predictor clamped to ``[-30, 30]``."""
    tmp = numpy.where(eta < -LOGIT_THRESHOLD, DBL_EPSILON, numpy.where(eta > LOGIT_THRESHOLD, 1.0 / DBL_EPSILON, numpy.exp(eta)))
    return numpy.asarray(tmp / (1.0 + tmp), dtype=numpy.float64)


def mu_eta(family: Family, eta: FloatArray) -> FloatArray:
    """``d mu / d eta`` as R's ``mu.eta`` computes it for the canonical link."""
    if family == 'binomial':
        opexp = 1.0 + numpy.exp(eta)
        return numpy.where(numpy.abs(eta) > LOGIT_THRESHOLD, DBL_EPSILON, numpy.exp(eta) / (opexp * opexp))
    return numpy.maximum(numpy.exp(eta), DBL_EPSILON)


def ylogy(y: FloatArray, mu: FloatArray) -> FloatArray:
    """``y * log(y / mu)`` with the convention 0 for ``y == 0``."""
    with numpy.errstate(divide='ignore', invalid='ignore'):
        return numpy.where(y > 0, y * numpy.log(y / mu), 0.0)


def deviance_residuals(family: Family, y: FloatArray, mu: FloatArray, weights: FloatArray) -> FloatArray:
    """Unit deviances ``dev.resids(y, mu, wt)`` of the binomial (``y`` a proportion) or Poisson family."""
    if family == 'binomial':
        return 2.0 * weights * (ylogy(y, mu) + ylogy(1.0 - y, 1.0 - mu))
    return 2.0 * weights * (ylogy(y, mu) - (y - mu))


def saturated_loglik(family: Family, y: FloatArray, size: FloatArray | None) -> float:
    """Log-likelihood without the combinatorial constants of the saturated model (``mu = y`` / ``mu = x / n``).

    Raises:
        metafopy.errors.InternalError: ``size`` is missing for the binomial family.
    """
    if family == 'binomial':
        if size is None:
            raise metafopy.errors.InternalError('binomial models need the group sizes')
        return float(numpy.sum(ylogy(y, size) + ylogy(size - y, size)))
    return float(numpy.sum(ylogy(y, numpy.ones_like(y)) - y))


def family_constant(family: Family, y: FloatArray, size: FloatArray | None) -> float:
    """Sum of the terms the exponential-family log-likelihood has beyond ``y * eta - cumulant``: log binomial coefficients or ``-log(y!)``.

    Raises:
        metafopy.errors.InternalError: ``size`` is missing for the binomial family.
    """
    if family == 'binomial':
        if size is None:
            raise metafopy.errors.InternalError('binomial models need the group sizes')
        return float(numpy.sum(scipy.special.gammaln(size + 1) - scipy.special.gammaln(y + 1) - scipy.special.gammaln(size - y + 1)))
    return float(-numpy.sum(scipy.special.gammaln(y + 1)))


def log_likelihood(family: Family, y: FloatArray, size: FloatArray | None, mu: FloatArray) -> float:
    """Log-likelihood of the observed counts under fitted means ``mu`` (proportions for the binomial family).

    Raises:
        metafopy.errors.InternalError: ``size`` is missing for the binomial family.
    """
    if family == 'binomial':
        if size is None:
            raise metafopy.errors.InternalError('binomial models need the group sizes')
        pos = size > 0
        return float(numpy.sum(scipy.stats.binom.logpmf(numpy.round(y[pos]), numpy.round(size[pos]), mu[pos])))
    return float(numpy.sum(scipy.stats.poisson.logpmf(y, mu)))


def glm_fit(
    X: FloatArray,
    y: FloatArray,
    *,
    family: Family,
    size: FloatArray | None = None,
    offset: FloatArray | None = None,
    epsilon: float = 1e-8,
    maxit: int = 25,
) -> GLMFit:
    """Fit ``glm(cbind(y, size - y) ~ 0 + X, family=binomial)`` or ``glm(y ~ 0 + X, offset=offset, family=poisson)``.

    ``y`` holds the event counts and, for the binomial family, ``size`` the
    number of trials. Aliased columns get a NaN coefficient and are left out of
    ``vcov``. The model is reported as non-converged (``converged=False``) when
    ``maxit`` iterations are used up, as in the saturated models of tables with
    zero cells, where R only warns.

    Raises:
        metafopy.errors.InputError: the arguments have inconsistent shapes or ``size`` is missing.
        metafopy.errors.EstimationError: the iterations produce non-finite values.
    """
    n, q = X.shape
    if y.shape[0] != n:
        raise metafopy.errors.InputError('y and X must have the same number of rows')
    off = numpy.zeros(n) if offset is None else offset
    if family == 'binomial':
        if size is None:
            raise metafopy.errors.InputError("binomial models need 'size'")
        weights = size.astype(numpy.float64)
        with numpy.errstate(divide='ignore', invalid='ignore'):
            resp = numpy.where(weights > 0, y / weights, 0.0)
        mustart = (weights * resp + 0.5) / (weights + 1.0)
        eta = numpy.log(mustart / (1.0 - mustart))
        mu = expit_clamped(eta)
    else:
        weights = numpy.ones(n)
        resp = y.astype(numpy.float64)
        mustart = resp + 0.1
        eta = numpy.log(mustart)
        mu = numpy.maximum(numpy.exp(eta), DBL_EPSILON)

    tol = min(1e-7, epsilon / 1000)
    devold = float(numpy.sum(deviance_residuals(family, resp, mu, weights)))
    coef = numpy.zeros(q)
    aliased = numpy.zeros(q, dtype=numpy.bool_)
    cov_unscaled = numpy.zeros((0, 0))
    converged = False
    iterations = 0
    for it in range(1, maxit + 1):
        iterations = it
        slope = mu_eta(family, eta)
        var = mu * (1.0 - mu) if family == 'binomial' else mu
        good = (weights > 0) & (slope != 0)
        z = (eta - off)[good] + (resp - mu)[good] / slope[good]
        w = numpy.sqrt(weights * slope**2 / var)[good]
        Xw = X[good] * w[:, None]
        aliased = aliased_columns(Xw, tol)
        keep = ~aliased
        qr_q, qr_r = numpy.linalg.qr(Xw[:, keep])
        coef_keep = numpy.linalg.solve(qr_r, qr_q.T @ (z * w))
        rinv = numpy.linalg.solve(qr_r, numpy.eye(qr_r.shape[0]))
        cov_unscaled = rinv @ rinv.T
        coef = numpy.zeros(q)
        coef[keep] = coef_keep
        eta = X @ coef + off
        mu = expit_clamped(eta) if family == 'binomial' else numpy.maximum(numpy.exp(eta), DBL_EPSILON)
        dev = float(numpy.sum(deviance_residuals(family, resp, mu, weights)))
        if not math.isfinite(dev):
            raise metafopy.errors.EstimationError('non-finite deviance in the GLM iterations')
        if abs(dev - devold) / (abs(dev) + 0.1) < epsilon:
            converged = True
            break
        devold = dev

    full = coef.copy()
    full[aliased] = numpy.nan
    return GLMFit(
        coef=full,
        vcov=cov_unscaled,
        aliased=aliased,
        eta=eta,
        mu=mu,
        loglik=log_likelihood(family, y, size, mu),
        deviance=dev,
        converged=converged,
        iterations=iterations,
    )
