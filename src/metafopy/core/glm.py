"""Binomial and Poisson generalized linear models fitted like R's ``glm()`` / ``glm.fit()``.

``rma.glmm`` (https://github.com/wviechtb/metafor/blob/master/R/rma.glmm.r) fits
its fixed-effects and saturated models with ``glm()``. Its log-likelihoods,
tests for heterogeneity and the set of aliased columns depend on where the
iteratively reweighted least squares (IRLS) iterations stop, so the algorithm
of ``glm.fit`` is followed step by step: the same starting values
(``mustart``), the same convergence rule (relative change of the deviance below
``epsilon``, at most ``maxit`` iterations), the same clamps of the logit and
log links, and the covariance matrix of the last weighted least squares fit.

The binomial family is also available with the log and identity links
(``rma.glmm`` uses them for the experimental measures ``RR``, ``PLN`` and
``RD``, ``PR``) and, like the Poisson family, with the probit, cauchit,
complementary log-log and square root links that ``make.link()`` provides
(``rma.glmm(link=...)`` and ``rma.glmm(family=...)`` pass them on). Those links can leave the
parameter space of the family, so ``glm.fit``'s validity checks and step
halving are reproduced as well.

Reference: McCullagh and Nelder (1989), *Generalized Linear Models*, 2nd ed.,
and the R source ``src/library/stats/R/glm.R`` and ``src/library/stats/src/family.c``.
"""

import dataclasses
import math
import sys
import typing
import warnings

import numpy
import numpy.typing
import scipy.special
import scipy.stats

import metafopy.errors

__all__ = [
    'GLMFamily',
    'GLMFit',
    'Link',
    'aliased_columns',
    'binomial',
    'canonical_link',
    'check_link',
    'deviance_residuals',
    'family_constant',
    'glm_fit',
    'link_fun',
    'link_inv',
    'poisson',
    'saturated_loglik',
    'valid_eta',
    'valid_mu',
]

type FloatArray = numpy.typing.NDArray[numpy.float64]
type BoolArray = numpy.typing.NDArray[numpy.bool_]
type Family = typing.Literal['binomial', 'poisson']
type Link = typing.Literal['cauchit', 'cloglog', 'identity', 'log', 'logit', 'probit', 'sqrt']

DBL_EPSILON = sys.float_info.epsilon
LOGIT_THRESHOLD = 30.0
# auditme: accept[non-alphabetical-literal] the order follows metafor's documented option order
LINKS: tuple[Link, ...] = ('logit', 'probit', 'cauchit', 'cloglog', 'log', 'identity', 'sqrt')
# auditme: accept[non-alphabetical-literal] the order follows metafor's documented option order
FAMILIES: tuple[Family, ...] = ('binomial', 'poisson')
PROBIT_THRESHOLD = float(-scipy.stats.norm.ppf(DBL_EPSILON))
CAUCHIT_THRESHOLD = float(-scipy.stats.cauchy.ppf(DBL_EPSILON))
CLOGLOG_ETA_MAX = 700.0


def check_link(link: str) -> Link:
    """The link named by ``link`` (``make.link()``'s names for the links usable with the binomial and Poisson families).

    Raises:
        metafopy.errors.InputError: the link is not one of ``'logit'``, ``'probit'``, ``'cauchit'``, ``'cloglog'``, ``'log'``, ``'identity'``, ``'sqrt'``.
    """
    if link not in LINKS:
        raise metafopy.errors.InputError(f'Unknown link {link!r}; use one of {", ".join(repr(name) for name in LINKS)}.')
    return link


@dataclasses.dataclass(frozen=True, slots=True)
class GLMFamily:
    """A family with its link, the equivalent of R's ``binomial(link=...)`` and ``poisson(link=...)`` objects."""

    name: Family
    link: Link

    def __post_init__(self) -> None:
        if self.name not in FAMILIES:
            raise metafopy.errors.InputError(f'Unknown family {self.name!r}; use one of {", ".join(repr(name) for name in FAMILIES)}.')
        check_link(self.link)

    def __str__(self) -> str:
        return f'{self.name}(link = "{self.link}")'


def binomial(link: str = 'logit') -> GLMFamily:
    """R's ``binomial(link)``: the family with the given link (default ``'logit'``)."""
    return GLMFamily('binomial', check_link(link))


def poisson(link: str = 'log') -> GLMFamily:
    """R's ``poisson(link)``: the family with the given link (default ``'log'``)."""
    return GLMFamily('poisson', check_link(link))


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


def canonical_link(family: Family) -> Link:
    """The canonical link of the family (``logit`` for the binomial, ``log`` for the Poisson family)."""
    return 'logit' if family == 'binomial' else 'log'


def link_fun(link: Link, mu: FloatArray) -> FloatArray:
    """``linkfun(mu)`` of R's ``make.link()``."""
    with numpy.errstate(divide='ignore', invalid='ignore'):
        if link == 'logit':
            return numpy.asarray(numpy.log(mu / (1.0 - mu)), dtype=numpy.float64)
        if link == 'log':
            return numpy.asarray(numpy.log(mu), dtype=numpy.float64)
        if link == 'probit':
            return numpy.asarray(scipy.stats.norm.ppf(mu), dtype=numpy.float64)
        if link == 'cauchit':
            return numpy.asarray(scipy.stats.cauchy.ppf(mu), dtype=numpy.float64)
        if link == 'cloglog':
            return numpy.asarray(numpy.log(-numpy.log(1.0 - mu)), dtype=numpy.float64)
        if link == 'sqrt':
            return numpy.asarray(numpy.sqrt(mu), dtype=numpy.float64)
    return mu.astype(numpy.float64)


def link_inv(link: Link, eta: FloatArray) -> FloatArray:
    """``linkinv(eta)`` of R's ``make.link()`` (the logit and log links are clamped as in R)."""
    if link == 'logit':
        return expit_clamped(eta)
    if link == 'log':
        return numpy.asarray(numpy.maximum(numpy.exp(eta), DBL_EPSILON), dtype=numpy.float64)
    if link == 'probit':
        return numpy.asarray(scipy.stats.norm.cdf(numpy.clip(eta, -PROBIT_THRESHOLD, PROBIT_THRESHOLD)), dtype=numpy.float64)
    if link == 'cauchit':
        return numpy.asarray(scipy.stats.cauchy.cdf(numpy.clip(eta, -CAUCHIT_THRESHOLD, CAUCHIT_THRESHOLD)), dtype=numpy.float64)
    if link == 'cloglog':
        return numpy.asarray(numpy.clip(-numpy.expm1(-numpy.exp(eta)), DBL_EPSILON, 1.0 - DBL_EPSILON), dtype=numpy.float64)
    if link == 'sqrt':
        return numpy.asarray(eta**2, dtype=numpy.float64)
    return eta.astype(numpy.float64)


def mu_eta(link: Link, eta: FloatArray) -> FloatArray:
    """``d mu / d eta`` as R's ``mu.eta`` computes it."""
    if link == 'logit':
        opexp = 1.0 + numpy.exp(eta)
        return numpy.where(numpy.abs(eta) > LOGIT_THRESHOLD, DBL_EPSILON, numpy.exp(eta) / (opexp * opexp))
    if link == 'log':
        return numpy.maximum(numpy.exp(eta), DBL_EPSILON)
    if link == 'probit':
        return numpy.maximum(scipy.stats.norm.pdf(eta), DBL_EPSILON)
    if link == 'cauchit':
        return numpy.maximum(scipy.stats.cauchy.pdf(eta), DBL_EPSILON)
    if link == 'cloglog':
        capped = numpy.minimum(eta, CLOGLOG_ETA_MAX)
        return numpy.asarray(numpy.maximum(numpy.exp(capped) * numpy.exp(-numpy.exp(capped)), DBL_EPSILON), dtype=numpy.float64)
    if link == 'sqrt':
        return 2.0 * eta
    return numpy.ones_like(eta)


def valid_eta(link: Link, eta: FloatArray) -> bool:
    """``family$valideta``: the square root link needs a finite, positive linear predictor, the other links accept any."""
    if link != 'sqrt':
        return True
    return bool(numpy.all(numpy.isfinite(eta)) and numpy.all(eta > 0))


def valid_mu(family: Family, mu: FloatArray) -> bool:
    """``family$validmu``: all means finite and inside ``(0, 1)`` (binomial) or positive (Poisson)."""
    if not bool(numpy.all(numpy.isfinite(mu))):
        return False
    return bool(numpy.all((mu > 0) & (mu < 1))) if family == 'binomial' else bool(numpy.all(mu > 0))


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
    link: Link | None = None,
    epsilon: float = 1e-8,
    maxit: int = 25,
) -> GLMFit:
    """Fit ``glm(cbind(y, size - y) ~ 0 + X, family=binomial)`` or ``glm(y ~ 0 + X, offset=offset, family=poisson)``.

    ``y`` holds the event counts and, for the binomial family, ``size`` the
    number of trials. ``link`` defaults to the canonical link. Aliased columns
    get a NaN coefficient and are left out of ``vcov``. The model is reported
    as non-converged (``converged=False``) when ``maxit`` iterations are used
    up, as in the saturated models of tables with zero cells, where R only
    warns. Like ``glm.fit``, an update that leaves the parameter space is
    halved towards the previous coefficients (with a warning).

    Raises:
        metafopy.errors.InputError: the arguments have inconsistent shapes or ``size`` is missing.
        metafopy.errors.EstimationError: no valid coefficients can be found or the iterations produce non-finite values.
    """
    n, q = X.shape
    if y.shape[0] != n:
        raise metafopy.errors.InputError('y and X must have the same number of rows')
    lnk: Link = canonical_link(family) if link is None else link
    off = numpy.zeros(n) if offset is None else offset
    if family == 'binomial':
        if size is None:
            raise metafopy.errors.InputError("binomial models need 'size'")
        weights = size.astype(numpy.float64)
        with numpy.errstate(divide='ignore', invalid='ignore'):
            resp = numpy.where(weights > 0, y / weights, 0.0)
        mustart = (weights * resp + 0.5) / (weights + 1.0)
    else:
        weights = numpy.ones(n)
        resp = y.astype(numpy.float64)
        mustart = resp + 0.1
    eta = link_fun(lnk, mustart)
    mu = link_inv(lnk, eta)
    if not (valid_mu(family, mu) and valid_eta(lnk, eta)):
        raise metafopy.errors.EstimationError('cannot find valid starting values: please specify some')

    def deviance_of(m: FloatArray) -> float:
        return float(numpy.sum(deviance_residuals(family, resp, m, weights)))

    def predictor(coef_now: FloatArray) -> tuple[FloatArray, FloatArray]:
        e = X @ coef_now + off
        return e, link_inv(lnk, e)

    tol = min(1e-7, epsilon / 1000)
    devold = deviance_of(mu)
    dev = devold
    coef = numpy.zeros(q)
    aliased = numpy.zeros(q, dtype=numpy.bool_)
    cov_unscaled = numpy.zeros((0, 0))
    coefold: FloatArray | None = None
    converged = False
    iterations = 0
    for it in range(1, maxit + 1):
        iterations = it
        slope = mu_eta(lnk, eta)
        var = mu * (1.0 - mu) if family == 'binomial' else mu
        if not numpy.all(numpy.isfinite(var)):
            raise metafopy.errors.EstimationError('NAs in V(mu)')
        if numpy.any(var == 0):
            raise metafopy.errors.EstimationError('0s in V(mu)')
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
        eta, mu = predictor(coef)
        dev = deviance_of(mu)
        if not math.isfinite(dev):
            if coefold is None:
                raise metafopy.errors.EstimationError('no valid set of coefficients has been found: please supply starting values')
            warnings.warn('step size truncated due to divergence', stacklevel=2)
            for _ in range(maxit):
                coef = (coef + coefold) / 2
                eta, mu = predictor(coef)
                dev = deviance_of(mu)
                if math.isfinite(dev):
                    break
            else:
                raise metafopy.errors.EstimationError('inner loop 1; cannot correct step size')
        if not (valid_mu(family, mu) and valid_eta(lnk, eta)):
            if coefold is None:
                raise metafopy.errors.EstimationError('no valid set of coefficients has been found: please supply starting values')
            warnings.warn('step size truncated: out of bounds', stacklevel=2)
            for _ in range(maxit):
                coef = (coef + coefold) / 2
                eta, mu = predictor(coef)
                if valid_mu(family, mu) and valid_eta(lnk, eta):
                    break
            else:
                raise metafopy.errors.EstimationError('inner loop 2; cannot correct step size')
            dev = deviance_of(mu)
        if abs(dev - devold) / (abs(dev) + 0.1) < epsilon:
            converged = True
            break
        devold = dev
        coefold = coef.copy()

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
