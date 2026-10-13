"""Binomial and Poisson mixed-effects models with Gaussian random effects, fitted by maximum likelihood.

``rma.glmm`` (https://github.com/wviechtb/metafor/blob/master/R/rma.glmm.r) fits
its mixed-effects models with ``lme4::glmer()``
(https://github.com/lme4/lme4/blob/master/R/modular.R, Bates et al. 2015,
doi:10.18637/jss.v067.i01). This module maximises the same likelihood:

* one scalar random effect per cluster is integrated out with adaptive
  Gauss-Hermite quadrature (``nAGQ`` points; ``nAGQ = 1`` is the Laplace
  approximation), centred at the conditional mode with the curvature there as
  scale (Pinheiro and Bates 1995, doi:10.1080/10618600.1995.10474663);
* two random effects per cluster (an intercept and a group effect, independent
  or correlated) are handled by the Laplace approximation, as ``glmer`` requires.

The random effects are ``b = Lambda(theta) u`` with ``u ~ N(0, I)``. Unlike
``glmer``, which stops its derivative-free optimiser at a loose tolerance, the
objective is optimised over ``(beta, theta)`` with BFGS until the gradient is
negligible, so estimates agree with lme4's to its optimiser tolerance.
The optimised objective is lme4's deviance, i.e. minus twice the log marginal
likelihood relative to the saturated model
(:func:`metafopy.core.glm.saturated_loglik`); the reported log-likelihood adds
the constants of the family (binomial coefficients, factorials) back, as
``logLik()`` of a ``glmer`` fit does.

Besides the canonical links, the binomial family is supported with the log and
identity links (the experimental measures ``RR``, ``PLN`` and ``RD``, ``PR`` of
``rma.glmm``). As in lme4, the Laplace determinant, the quadrature scale and
the covariance of the fixed effects use the expected (Fisher) information
``prior * (d mu / d eta)^2 / V(mu)``, which equals the observed information
only for canonical links; the likelihood itself is exact. Linear predictors
that put a mean outside the parameter space make the likelihood undefined
(``NaN``), as in lme4, where such steps are rejected or the fit fails.
"""

import collections.abc
import dataclasses
import math
import typing
import warnings

import numpy
import numpy.polynomial.hermite_e
import numpy.typing
import scipy.optimize
import scipy.special

import metafopy.core.glm
import metafopy.core.numderiv
import metafopy.errors

__all__ = ['GLMMFit', 'GLMMProblem', 'fit_glmm']

type FloatArray = numpy.typing.NDArray[numpy.float64]
type IntArray = numpy.typing.NDArray[numpy.intp]
type Structure = typing.Literal['scalar', 'independent', 'correlated']

BOUNDARY_TOL = 1e-5
MODE_TOL = 1e-10
MODE_MAXIT = 100
HALVING_MAXIT = 40
STAGE_ONE_MAX = 20.0
HESSIAN_WARNING = (
    'variance-covariance matrix computed from finite-difference Hessian is\n'
    'not positive definite or contains NA values: falling back to var-cov estimated from RX'
)


@dataclasses.dataclass(frozen=True, slots=True)
class GLMMProblem:
    """Data of a mixed model ``eta = X beta + offset + Z Lambda(theta) u`` with ``u`` independent across clusters.

    ``Z`` has one column for ``structure='scalar'`` and two (intercept first)
    otherwise. ``size`` gives the binomial totals (``None`` for the Poisson family).
    """

    X: FloatArray
    y: FloatArray
    size: FloatArray | None
    offset: FloatArray
    cluster: IntArray
    Z: FloatArray
    family: metafopy.core.glm.Family
    structure: Structure = 'scalar'
    link: metafopy.core.glm.Link | None = None

    @property
    def n_clusters(self) -> int:
        return int(self.cluster.max()) + 1

    @property
    def n_theta(self) -> int:
        return {'scalar': 1, 'independent': 2, 'correlated': 3}[self.structure]


@dataclasses.dataclass(frozen=True, slots=True)
class GLMMFit:
    """Maximum-likelihood fit of a :class:`GLMMProblem`.

    ``loglik`` is the log marginal likelihood including the family constants,
    which is what ``logLik()`` of a ``glmer`` fit reports. ``cond_deviance``
    is the sum of squared deviance residuals at the conditional modes. ``vcov``
    is the covariance of ``beta`` as ``vcov.merMod`` reports it: twice the
    ``beta`` block of the inverse finite-difference Hessian of the deviance, or
    the one from the penalised weighted least squares decomposition when that
    Hessian is not positive definite.
    """

    beta: FloatArray
    theta: FloatArray
    loglik: float
    cond_deviance: float
    vcov: FloatArray
    modes: FloatArray
    converged: bool
    evaluations: int

    def covariance(self) -> FloatArray:
        """Covariance matrix of the random effects ``Lambda @ Lambda.T``."""
        lam = lambda_matrix(self.theta, 'scalar' if self.theta.shape[0] == 1 else ('independent' if self.theta.shape[0] == 2 else 'correlated'))
        return numpy.asarray(lam @ lam.T, dtype=numpy.float64)


def lambda_matrix(theta: FloatArray, structure: Structure) -> FloatArray:
    """Relative covariance factor ``Lambda`` (lower triangular) for the parameters ``theta`` (lme4 order)."""
    if structure == 'scalar':
        return numpy.array([[theta[0]]])
    if structure == 'independent':
        return numpy.diag(theta)
    return numpy.array([[theta[0], 0.0], [theta[1], theta[2]]])


def row_terms(prob: GLMMProblem, eta: FloatArray, y: FloatArray, size: FloatArray | None) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Log-likelihood contribution (without constants), score ``d ell / d eta`` and Fisher information of each row.

    Rows whose mean leaves the parameter space get ``NaN`` for the log-likelihood.

    Raises:
        metafopy.errors.InternalError: ``size`` is missing for the binomial family.
    """
    link = prob.link or metafopy.core.glm.canonical_link(prob.family)
    if prob.family == 'binomial':
        if size is None:
            raise metafopy.errors.InternalError('binomial models need the group sizes')
        if link == 'logit':
            p = scipy.special.expit(eta)
            return y * eta - size * numpy.logaddexp(0.0, eta), y - size * p, size * p * (1.0 - p)
        mu = numpy.exp(eta) if link == 'log' else eta
        invalid = ~((mu > 0) & (mu < 1))
        mu_safe = numpy.where(invalid, 0.5, mu)
        ell = scipy.special.xlogy(y, mu_safe) + scipy.special.xlogy(size - y, 1.0 - mu_safe)
        slope = mu_safe if link == 'log' else numpy.ones_like(mu_safe)
        score = (y - size * mu_safe) / (mu_safe * (1.0 - mu_safe)) * slope
        info = size * slope**2 / (mu_safe * (1.0 - mu_safe))
        return numpy.where(invalid, numpy.nan, ell), score, info
    mu = numpy.exp(eta)
    return y * eta - mu, y - mu, mu


def information(cluster: IntArray, A: FloatArray, w: FloatArray, n_clusters: int) -> FloatArray:
    """Per-cluster matrices ``I + A' W A``."""
    r = A.shape[1]
    H = numpy.zeros((n_clusters, r, r))
    for m in range(r):
        for ell in range(m + 1):
            s = numpy.bincount(cluster, weights=A[:, m] * A[:, ell] * w, minlength=n_clusters)
            H[:, m, ell] = s
            H[:, ell, m] = s
    H += numpy.eye(r)
    return H


@dataclasses.dataclass(frozen=True, slots=True)
class Modes:
    """Conditional modes of the spherical random effects and the quantities evaluated there."""

    u: FloatArray
    eta: FloatArray
    ell: FloatArray
    w: FloatArray
    H: FloatArray


def conditional_modes(prob: GLMMProblem, f: FloatArray, A: FloatArray) -> Modes:
    """Penalised maximum over ``u`` of ``sum ell(eta) - |u|^2 / 2`` for every cluster, by damped Newton iterations.

    The objective is concave for the canonical links used here, so the mode is
    unique; step halving per cluster guarantees convergence.

    Raises:
        metafopy.errors.EstimationError: the objective becomes non-finite.
    """
    cl = prob.cluster
    K = prob.n_clusters
    r = A.shape[1]
    u = numpy.zeros((K, r))

    def penalised(uu: FloatArray) -> tuple[FloatArray, FloatArray]:
        eta = f + numpy.sum(A * uu[cl], axis=1)
        ell, _, _ = row_terms(prob, eta, prob.y, prob.size)
        return numpy.bincount(cl, weights=ell, minlength=K) - 0.5 * numpy.sum(uu**2, axis=1), eta

    obj, _ = penalised(u)
    if not numpy.all(numpy.isfinite(obj)):
        raise metafopy.errors.EstimationError('the linear predictor is outside the parameter space')
    for _ in range(MODE_MAXIT):
        eta = f + numpy.sum(A * u[cl], axis=1)
        _, score, w = row_terms(prob, eta, prob.y, prob.size)
        g = numpy.column_stack([numpy.bincount(cl, weights=A[:, m] * score, minlength=K) for m in range(r)]) - u
        H = information(cl, A, w, K)
        step = numpy.linalg.solve(H, g[:, :, None])[:, :, 0]
        t = numpy.ones(K)
        cand = u + step
        new, _ = penalised(cand)
        for _ in range(HALVING_MAXIT):
            bad = ~(new >= obj - 1e-12 * numpy.abs(obj))
            if not bad.any():
                break
            t = numpy.where(bad, t / 2, t)
            cand = u + t[:, None] * step
            new, _ = penalised(cand)
        if not numpy.all(numpy.isfinite(new)):
            raise metafopy.errors.EstimationError('non-finite objective while finding the conditional modes')
        u, obj = cand, new
        if float(numpy.max(numpy.abs(t[:, None] * step))) < MODE_TOL:
            break
    eta = f + numpy.sum(A * u[cl], axis=1)
    ell, _, w = row_terms(prob, eta, prob.y, prob.size)
    return Modes(u=u, eta=eta, ell=ell, w=w, H=information(cl, A, w, K))


def quadrature_rule(nagq: int) -> tuple[FloatArray, FloatArray]:
    """Nodes and log-weights of the standard-normal Gauss-Hermite rule with ``nagq`` points."""
    if nagq == 1:
        return numpy.zeros(1), numpy.zeros(1)
    nodes, weights = numpy.polynomial.hermite_e.hermegauss(nagq)
    return nodes, numpy.log(weights / math.sqrt(2 * math.pi))


def evaluate(prob: GLMMProblem, beta: FloatArray, theta: FloatArray, nagq: int) -> tuple[float, Modes, FloatArray]:
    """Log marginal likelihood, the conditional modes and the matrix ``A = Z Lambda`` at the given parameters.

    Raises:
        metafopy.errors.InputError: ``nagq > 1`` with more than one random effect per cluster.
    """
    lam = lambda_matrix(theta, prob.structure)
    A = prob.Z @ lam
    f = prob.X @ beta + prob.offset
    modes = conditional_modes(prob, f, A)
    K = prob.n_clusters
    logdet = numpy.linalg.slogdet(modes.H)[1]
    if nagq == 1:
        per_cluster = numpy.bincount(prob.cluster, weights=modes.ell, minlength=K) - 0.5 * numpy.sum(modes.u**2, axis=1) - 0.5 * logdet
        return float(per_cluster.sum()), modes, A
    if prob.Z.shape[1] != 1:
        raise metafopy.errors.InputError('nAGQ > 1 is only possible with a single random effect per cluster')
    nodes, logw = quadrature_rule(nagq)
    sd = 1.0 / numpy.sqrt(modes.H[:, 0, 0])
    terms = numpy.empty((K, nodes.shape[0]))
    for g, (z, lw) in enumerate(zip(nodes, logw, strict=True)):
        ug = modes.u[:, 0] + z * sd
        eta = f + A[:, 0] * ug[prob.cluster]
        ell, _, _ = row_terms(prob, eta, prob.y, prob.size)
        terms[:, g] = lw + numpy.bincount(prob.cluster, weights=ell, minlength=K) - 0.5 * ug**2 + 0.5 * z**2
    per_cluster = scipy.special.logsumexp(terms, axis=1) - 0.5 * logdet
    return float(per_cluster.sum()), modes, A


def beta_covariance(prob: GLMMProblem, modes: Modes, A: FloatArray) -> FloatArray:
    """``(X'WX - X'WA (A'WA + I)^-1 A'WX)^-1`` with the weights at the conditional modes."""
    X, w, cl = prob.X, modes.w, prob.cluster
    K = prob.n_clusters
    q = X.shape[1]
    r = A.shape[1]
    C = numpy.zeros((K, q, r))
    numpy.add.at(C, cl, w[:, None, None] * X[:, :, None] * A[:, None, :])
    schur = (X * w[:, None]).T @ X
    inner = numpy.linalg.solve(modes.H, numpy.transpose(C, (0, 2, 1)))
    schur = schur - numpy.einsum('kqr,krs->qs', C, inner)
    return numpy.asarray(numpy.linalg.inv(schur), dtype=numpy.float64)


def normalise_theta(theta: FloatArray, structure: Structure) -> FloatArray:
    """Make the diagonal of ``Lambda`` non-negative (the likelihood is invariant to these signs) and snap tiny values to the bound 0."""
    out = theta.copy()
    if structure == 'correlated':
        if out[0] < 0:
            out[0], out[1] = -out[0], -out[1]
        out[2] = abs(out[2])
        diag = [0, 2]
    else:
        out = numpy.abs(out)
        diag = list(range(out.shape[0]))
    for pos in diag:
        if out[pos] < BOUNDARY_TOL:
            out[pos] = 0.0
    return out


def hessian_covariance(deviance: collections.abc.Callable[[FloatArray], float], beta: FloatArray, theta: FloatArray, structure: Structure) -> FloatArray | None:
    """Covariance of ``beta`` as ``vcov.merMod`` derives it from the Hessian of the deviance, ``None`` if that is unusable."""
    q = beta.shape[0]
    x = numpy.concatenate([beta, theta])
    lower = numpy.full(x.shape[0], numpy.nan)
    lower[q + numpy.array({'scalar': [0], 'independent': [0, 1], 'correlated': [0, 2]}[structure])] = 0.0
    H = metafopy.core.numderiv.deriv12(deviance, x, lower=lower)
    if not numpy.all(numpy.isfinite(H)):
        return None
    try:
        inv = numpy.linalg.inv(H)
    except numpy.linalg.LinAlgError:
        return None
    block = inv[:q, :q]
    cov = block + block.T
    if not numpy.all(numpy.isfinite(cov)) or numpy.linalg.eigvalsh(cov).min() <= 0:
        return None
    return numpy.asarray(cov, dtype=numpy.float64)


def starting_theta(structure: Structure) -> FloatArray:
    return {'scalar': numpy.array([1.0]), 'independent': numpy.array([1.0, 1.0]), 'correlated': numpy.array([1.0, 0.0, 1.0])}[structure]


def joint_modes(prob: GLMMProblem, theta: FloatArray, beta0: FloatArray) -> tuple[float, FloatArray, Modes, FloatArray]:
    """Joint penalised maximum over the fixed effects and the spherical random effects (the PIRLS of lme4's ``nAGQ = 0`` stage).

    Returns the Laplace value ``sum ell - |u|^2 / 2 - log det(H) / 2``, the fixed effects, the :class:`Modes` and ``A = Z Lambda``.

    Raises:
        metafopy.errors.EstimationError: the objective is undefined at the start or becomes non-finite.
    """
    A = prob.Z @ lambda_matrix(theta, prob.structure)
    X, cl = prob.X, prob.cluster
    K, r, q = prob.n_clusters, A.shape[1], X.shape[1]
    beta = beta0.copy()
    u = numpy.zeros((K, r))

    def objective(b: FloatArray, uu: FloatArray) -> float:
        ell, _, _ = row_terms(prob, X @ b + prob.offset + numpy.sum(A * uu[cl], axis=1), prob.y, prob.size)
        return float(numpy.bincount(cl, weights=ell, minlength=K).sum() - 0.5 * numpy.sum(uu**2))

    obj = objective(beta, u)
    if not math.isfinite(obj):
        raise metafopy.errors.EstimationError('the linear predictor is outside the parameter space')
    for _ in range(MODE_MAXIT):
        eta = X @ beta + prob.offset + numpy.sum(A * u[cl], axis=1)
        _, score, w = row_terms(prob, eta, prob.y, prob.size)
        g_u = numpy.column_stack([numpy.bincount(cl, weights=A[:, m] * score, minlength=K) for m in range(r)]) - u
        g_b = X.T @ score
        H = information(cl, A, w, K)
        C = numpy.zeros((K, q, r))
        numpy.add.at(C, cl, w[:, None, None] * X[:, :, None] * A[:, None, :])
        h_gu = numpy.linalg.solve(H, g_u[:, :, None])[:, :, 0]
        h_ct = numpy.linalg.solve(H, numpy.transpose(C, (0, 2, 1)))
        schur = (X * w[:, None]).T @ X - numpy.einsum('kqr,krs->qs', C, h_ct)
        d_b = numpy.linalg.solve(schur, g_b - numpy.einsum('kqr,kr->q', C, h_gu))
        d_u = h_gu - numpy.einsum('krq,q->kr', h_ct, d_b)
        t = 1.0
        new = objective(beta + d_b, u + d_u)
        for _ in range(HALVING_MAXIT):
            if new >= obj - 1e-12 * abs(obj):
                break
            t /= 2
            new = objective(beta + t * d_b, u + t * d_u)
        if not math.isfinite(new):
            raise metafopy.errors.EstimationError('non-finite objective while finding the joint modes')
        beta, u, obj = beta + t * d_b, u + t * d_u, new
        if max(float(numpy.max(numpy.abs(t * d_b), initial=0.0)), float(numpy.max(numpy.abs(t * d_u), initial=0.0))) < MODE_TOL:
            break
    eta = X @ beta + prob.offset + numpy.sum(A * u[cl], axis=1)
    ell, _, w = row_terms(prob, eta, prob.y, prob.size)
    modes = Modes(u=u, eta=eta, ell=ell, w=w, H=information(cl, A, w, K))
    value = obj - 0.5 * float(numpy.linalg.slogdet(modes.H)[1].sum())
    return value, beta, modes, A


def stage_one(prob: GLMMProblem, beta0: FloatArray) -> tuple[FloatArray, FloatArray, Modes, FloatArray]:
    """lme4's first stage (``nAGQ = 0``) for one scalar random effect: minimise the deviance over ``theta`` with the fixed effects profiled out.

    Raises:
        metafopy.errors.EstimationError: the deviance is undefined for every ``theta``.
    """
    sat = metafopy.core.glm.saturated_loglik(prob.family, prob.y, prob.size)
    cache: dict[float, tuple[float, FloatArray, Modes, FloatArray]] = {}

    def deviance(t: float) -> float:
        try:
            value, beta, modes, A = joint_modes(prob, numpy.array([t]), beta0)
        except numpy.linalg.LinAlgError, metafopy.errors.EstimationError, FloatingPointError:
            return 1e300
        cache[t] = (value, beta, modes, A)
        return -2.0 * (value - sat) if math.isfinite(value) else 1e300

    with numpy.errstate(all='ignore'):
        grid = [0.0, *numpy.geomspace(1e-4, STAGE_ONE_MAX, 30).tolist()]
        values = [deviance(t) for t in grid]
        best = int(numpy.argmin(values))
        if values[best] >= 1e299:
            raise metafopy.errors.EstimationError('the GLMM deviance is undefined for all values of theta')
        lo, hi = grid[max(best - 1, 0)], grid[min(best + 1, len(grid) - 1)]
        res = scipy.optimize.minimize_scalar(deviance, bounds=(lo, hi), method='bounded', options={'xatol': 1e-10})
        theta = float(res.x) if res.fun <= values[best] else grid[best]
        theta = 0.0 if theta < BOUNDARY_TOL else theta
        deviance(theta)
    _, beta, modes, A = cache[theta]
    return beta, numpy.array([theta]), modes, A


def conditional_deviance(prob: GLMMProblem, eta: FloatArray) -> float:
    """Sum of the squared deviance residuals at the linear predictor ``eta`` (what ``deviance()`` of a ``glmer`` fit returns)."""
    mu = metafopy.core.glm.link_inv(prob.link or metafopy.core.glm.canonical_link(prob.family), eta)
    if prob.family == 'binomial':
        size = prob.size if prob.size is not None else numpy.ones_like(prob.y)
        with numpy.errstate(divide='ignore', invalid='ignore'):
            prop = numpy.where(size > 0, prob.y / size, 0.0)
        return float(numpy.sum(metafopy.core.glm.deviance_residuals('binomial', prop, mu, size)))
    return float(numpy.sum(metafopy.core.glm.deviance_residuals('poisson', prob.y, mu, numpy.ones_like(prob.y))))


def two_stage(prob: GLMMProblem) -> bool:
    """Whether lme4's staging matters: adaptive quadrature with a non-canonical link can be undefined away from the first-stage solution."""
    return prob.structure == 'scalar' and prob.link is not None and prob.link != metafopy.core.glm.canonical_link(prob.family)


def fit_glmm(
    prob: GLMMProblem,
    *,
    nagq: int = 1,
    beta0: FloatArray | None = None,
    theta0: FloatArray | None = None,
    gtol: float = 1e-7,
    maxiter: int = 500,
) -> GLMMFit:
    """Maximise the log marginal likelihood over the fixed effects and the random-effects parameters.

    The optimiser works on signed ``theta``; the likelihood only depends on
    ``theta`` through ``Lambda @ Lambda.T`` and the signs are normalised at the end.

    For a non-canonical link and adaptive quadrature (``nagq > 1``) the fit
    follows lme4's two stages: the first minimises the deviance over ``theta``
    with the fixed effects profiled out (``nAGQ = 0``), the second maximises the
    quadrature likelihood from there. When that likelihood is undefined at the
    first-stage solution (quadrature points outside the parameter space), lme4's
    second stage cannot move, so the first-stage estimates are returned with an
    undefined (``-inf``) log-likelihood and the covariance from the penalised
    least squares decomposition, as metafor ends up with.

    Raises:
        metafopy.errors.EstimationError: the optimiser fails to produce a finite optimum.
    """
    q = prob.X.shape[1]
    if beta0 is None:
        start = metafopy.core.glm.glm_fit(prob.X, prob.y, family=prob.family, size=prob.size, offset=prob.offset, link=prob.link)
        beta0 = numpy.where(start.aliased, 0.0, start.coef)
    th0 = starting_theta(prob.structure) if theta0 is None else theta0
    count = 0
    sat = metafopy.core.glm.saturated_loglik(prob.family, prob.y, prob.size)

    def deviance(x: FloatArray) -> float:
        nonlocal count
        count += 1
        try:
            value = evaluate(prob, x[:q], x[q:], nagq)[0] - sat
        except numpy.linalg.LinAlgError, metafopy.errors.EstimationError, FloatingPointError:
            return 1e300
        return -2.0 * value if math.isfinite(value) else 1e300

    staged = nagq > 1 and two_stage(prob)
    x0 = numpy.concatenate([beta0, th0])
    if staged:
        beta1, theta1, modes1, A1 = stage_one(prob, beta0)
        x0 = numpy.concatenate([beta1, theta1])
        if deviance(x0) >= 1e299:
            return stage_one_result(prob, beta1, theta1, modes1, A1, count)
    with numpy.errstate(all='ignore'):
        res = scipy.optimize.minimize(deviance, x0, method='BFGS', jac='3-point', options={'gtol': gtol, 'maxiter': maxiter})
        # restarting BFGS from the solution resets its curvature estimate and removes premature stops
        res = scipy.optimize.minimize(deviance, res.x, method='BFGS', jac='3-point', options={'gtol': gtol, 'maxiter': maxiter})
    if not numpy.all(numpy.isfinite(res.x)) or res.fun >= 1e299:
        if staged:
            return stage_one_result(prob, beta1, theta1, modes1, A1, count)
        raise metafopy.errors.EstimationError('the GLMM likelihood could not be maximised')
    beta = res.x[:q]
    theta = normalise_theta(res.x[q:], prob.structure)
    with numpy.errstate(all='ignore'):
        loglik, modes, A = evaluate(prob, beta, theta, nagq)
        # lme4 reports the true log-likelihood for the Laplace fit but -deviance/2 for adaptive quadrature
        if nagq == 1:
            loglik += metafopy.core.glm.family_constant(prob.family, prob.y, prob.size)
        else:
            loglik -= metafopy.core.glm.saturated_loglik(prob.family, prob.y, prob.size)
        cov = hessian_covariance(deviance, beta, theta, prob.structure)
        if cov is None:
            warnings.warn(HESSIAN_WARNING, stacklevel=3)
            cov = beta_covariance(prob, modes, A)
    return GLMMFit(
        beta=beta,
        theta=theta,
        loglik=loglik,
        cond_deviance=conditional_deviance(prob, modes.eta),
        vcov=cov,
        modes=modes.u,
        converged=bool(res.success),
        evaluations=count,
    )


def stage_one_result(prob: GLMMProblem, beta: FloatArray, theta: FloatArray, modes: Modes, A: FloatArray, count: int) -> GLMMFit:
    """The first-stage estimates as the final fit (lme4 cannot improve them); the log-likelihood is undefined."""
    warnings.warn(HESSIAN_WARNING, stacklevel=4)
    return GLMMFit(
        beta=beta,
        theta=theta,
        loglik=-math.inf,
        cond_deviance=conditional_deviance(prob, modes.eta),
        vcov=beta_covariance(prob, modes, A),
        modes=modes.u,
        converged=True,
        evaluations=count,
    )
