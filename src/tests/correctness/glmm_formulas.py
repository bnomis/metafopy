"""Independent re-derivations of the ``rma_glmm`` likelihoods and fits, coded from the defining publications.

Rule: this module imports ``numpy`` and ``scipy`` only, never ``metafopy`` (``test_coverage.py`` checks it with ``ast``), and each function cites the
equation it codes, so that an expected value never comes from metafopy's or metafor's source.

A ``derived`` case names a function here as ``glmm_formulas.<name>``, the value of an ``expect`` key; it is called with the arguments of the case that
it declares (see ``cases.formula_value``). See ``agents/specs/correctness-glmm/spec.md``.
"""

import collections.abc
import contextlib
import dataclasses
import functools
import math
import typing
import warnings

import numpy
import numpy.typing
import scipy.integrate
import scipy.optimize
import scipy.special
import scipy.stats

type FloatArray = numpy.typing.NDArray[numpy.float64]
type Conditional = collections.abc.Callable[[FloatArray], FloatArray]
type Conditional2 = collections.abc.Callable[[FloatArray, FloatArray], FloatArray]

RULES = ('agq', 'exact', 'laplace')
CURVATURE_STEP = 1e-3


@contextlib.contextmanager
def strict(model: str) -> collections.abc.Iterator[None]:
    """Turn ``scipy.integrate.IntegrationWarning`` into ``ArithmeticError`` naming ``model``, so no reference comes from a failed integral."""
    with warnings.catch_warnings():
        warnings.simplefilter('error', scipy.integrate.IntegrationWarning)
        try:
            yield
        except scipy.integrate.IntegrationWarning as error:
            raise ArithmeticError(f'{model}: integration failed') from error


def require_converged(result: scipy.optimize.OptimizeResult, model: str) -> scipy.optimize.OptimizeResult:
    """Return ``result`` of a ``scipy.optimize`` search, or raise ``ArithmeticError`` naming ``model`` when it did not converge."""
    if not result.success:
        raise ArithmeticError(f'{model}: optimisation did not converge: {result.message}')
    return result


def inverse_link(eta: FloatArray, link: str) -> FloatArray:
    """Mean ``mu = g^-1(eta)`` for the link ``g`` of a generalised linear model (McCullagh and Nelder 1989, Table 2.1)."""
    if link == 'logit':
        return scipy.special.expit(eta)
    if link == 'probit':
        return scipy.special.ndtr(eta)
    if link == 'cauchit':
        return 0.5 + numpy.arctan(eta) / math.pi
    if link == 'cloglog':
        return -numpy.expm1(-numpy.exp(eta))
    if link == 'log':
        return numpy.exp(eta)
    if link == 'identity':
        return numpy.asarray(eta, dtype=numpy.float64)
    if link == 'sqrt':
        return eta**2
    raise ValueError(f'unsupported link: {link}')


def binomial_logpmf(y: FloatArray, size: FloatArray, eta: FloatArray, link: str) -> FloatArray:
    """Binomial log-probability of each ``y`` of ``size`` trials, ``log C(n, y) + y log p + (n - y) log(1 - p)`` with ``p = g^-1(eta)``.

    The binomial coefficient is the family constant that ``logLik()`` of a binomial model reports (Stijnen et al. 2010, Eq. 2; Hamza et al. 2008).
    A mean outside ``[0, 1]`` (``log`` and ``identity`` links) has log-likelihood ``-inf``.
    """
    p = inverse_link(eta, link)
    coefficient = scipy.special.gammaln(size + 1) - scipy.special.gammaln(y + 1) - scipy.special.gammaln(size - y + 1)
    with numpy.errstate(divide='ignore', invalid='ignore'):
        value = coefficient + scipy.special.xlogy(y, p) + scipy.special.xlog1py(size - y, -p)
    return numpy.where((p < 0) | (p > 1), -numpy.inf, value)


def binomial_loglik(y: FloatArray, size: FloatArray, eta: FloatArray, link: str) -> float:
    """Binomial log-likelihood with its binomial coefficients, for the linear predictor ``eta`` under ``link`` (sum of ``binomial_logpmf``)."""
    return float(numpy.sum(binomial_logpmf(y, size, eta, link)))


def poisson_logpmf(y: FloatArray, eta: FloatArray, link: str) -> FloatArray:
    """Poisson log-probability of each count ``y``, ``y log mu - mu - log y!`` with ``mu = g^-1(eta)`` (Bagos and Nikolopoulos 2009, Eq. 1).

    ``eta`` includes the log person-time offset where the model has one. ``log y!`` is the family constant that ``logLik()`` of a Poisson model reports.
    """
    mu = inverse_link(eta, link)
    with numpy.errstate(divide='ignore', invalid='ignore'):
        value = scipy.special.xlogy(y, mu) - mu - scipy.special.gammaln(y + 1)
    return numpy.where(mu < 0, -numpy.inf, value)


def poisson_loglik(y: FloatArray, eta: FloatArray, link: str) -> float:
    """Poisson log-likelihood with its factorial constant, for the linear predictor ``eta`` under ``link`` (sum of ``poisson_logpmf``)."""
    return float(numpy.sum(poisson_logpmf(y, eta, link)))


def nchg_logpmf(ai: FloatArray, n1i: FloatArray, ci: FloatArray, n2i: FloatArray, log_or: FloatArray) -> FloatArray:
    """Fisher's non-central hypergeometric log-probability of each ``ai`` given its margins, by direct summation over the support.

    Study ``i`` has ``ai`` events among ``n1i`` in group 1 and ``ci`` among ``n2i`` in group 2; conditional on ``m = ai + ci`` events,
    ``P(X = x) = C(n1, x) C(n2, m - x) exp(x psi) / sum_u C(n1, u) C(n2, m - u) exp(u psi)`` over ``max(0, m - n2) <= u <= min(n1, m)``, ``psi`` the
    log odds ratio (van Houwelingen et al. 1993; Stijnen et al. 2010, Eq. 5).
    """
    log_or = numpy.broadcast_to(numpy.asarray(log_or, dtype=numpy.float64), ai.shape)
    out = numpy.empty(ai.shape)
    for i in range(ai.size):
        n1, n2, m = n1i[i], n2i[i], ai[i] + ci[i]
        support = numpy.arange(max(0.0, m - n2), min(n1, m) + 1)
        terms = (
            scipy.special.gammaln(n1 + 1)
            - scipy.special.gammaln(support + 1)
            - scipy.special.gammaln(n1 - support + 1)
            + scipy.special.gammaln(n2 + 1)
            - scipy.special.gammaln(m - support + 1)
            - scipy.special.gammaln(n2 - m + support + 1)
            + support * log_or[i]
        )
        x = ai[i]
        own = terms[numpy.argmin(numpy.abs(support - x))]
        out[i] = own - scipy.special.logsumexp(terms)
    return out


def nchg_loglik(ai: FloatArray, n1i: FloatArray, ci: FloatArray, n2i: FloatArray, log_or: FloatArray) -> float:
    """Sum of Fisher's non-central hypergeometric log-probabilities of ``ai`` given the margins (van Houwelingen et al. 1993)."""
    return float(numpy.sum(nchg_logpmf(ai, n1i, ci, n2i, log_or)))


def approximate_conditional_logpmf(ai: FloatArray, n1i: FloatArray, ci: FloatArray, n2i: FloatArray, log_or: FloatArray) -> FloatArray:
    """The approximate conditional (``CM.AL``) log-probability: ``ai`` is binomial with ``ai + ci`` trials and logit ``psi + log(n1 / n2)``.

    Stijnen et al. (2010) replace the hypergeometric by a binomial whose odds are the odds ratio times the ratio of the group sizes.
    """
    return binomial_logpmf(ai, ai + ci, numpy.asarray(log_or) + numpy.log(n1i / n2i), 'logit')


def approximate_conditional_loglik(ai: FloatArray, n1i: FloatArray, ci: FloatArray, n2i: FloatArray, log_or: FloatArray) -> float:
    """Sum of ``approximate_conditional_logpmf``, the ``CM.AL`` conditional log-likelihood at a given log odds ratio."""
    return float(numpy.sum(approximate_conditional_logpmf(ai, n1i, ci, n2i, log_or)))


def study_function(conditional: Conditional, i: int, k: int) -> collections.abc.Callable[[float], float]:
    """Log conditional likelihood of study ``i`` as a function of its own random effect (``conditional`` takes one effect per study)."""
    return lambda u: float(conditional(numpy.full(k, u))[i])


def posterior_mode(conditional: Conditional, tau2: float, k: int) -> tuple[FloatArray, FloatArray]:
    """Mode of ``l_i(u) - u^2 / (2 tau2)`` per study and the scale ``s = 1 / sqrt(-h'')`` of its curvature (Pinheiro and Bates 1995, Eq. 2.3)."""
    mode = numpy.empty(k)
    scale = numpy.empty(k)
    for i in range(k):
        f = study_function(conditional, i, k)

        def negative(u: float, f: collections.abc.Callable[[float], float] = f) -> float:
            return -(f(u) - u * u / (2 * tau2))

        mode[i] = require_converged(scipy.optimize.minimize_scalar(negative, bracket=(-1.0, 1.0), tol=1e-12), 'marginal mode').x
        h = CURVATURE_STEP
        second = (negative(mode[i] + h) - 2 * negative(mode[i]) + negative(mode[i] - h)) / h**2
        if second <= 0:
            raise ArithmeticError('marginal: the conditional likelihood is not concave at its mode')
        scale[i] = 1 / math.sqrt(second)
    return mode, scale


def marginal_loglik(conditional: Conditional, tau2: float, rule: str, points: int = 7, *, studies: int, model: str = 'marginal') -> float:
    """Normal-mixture log-likelihood of a random-intercept model, ``sum_i log int exp(l_i(u)) N(u; 0, tau2) du``.

    ``conditional`` maps one random effect per study to that study's log conditional likelihood. ``rule`` is ``'exact'`` (``scipy.integrate.quad``),
    ``'agq'`` (adaptive Gauss-Hermite quadrature with ``points`` nodes, centred at the conditional mode and scaled by its curvature, Pinheiro and
    Bates 1995, Eq. 2.4) or ``'laplace'`` (one node, Eq. 2.3). A failed integral raises ``ArithmeticError`` naming ``model``.
    """
    if rule not in RULES:
        raise ValueError(f'unsupported rule: {rule}')
    k = studies
    if tau2 <= 0:
        return float(numpy.sum(conditional(numpy.zeros(k))))
    sd = math.sqrt(tau2)
    if rule == 'exact':
        total = 0.0
        with strict(model):
            for i in range(k):
                f = study_function(conditional, i, k)
                reference = f(0.0)
                value, _ = scipy.integrate.quad(
                    lambda u, f=f, reference=reference: math.exp(f(u) - reference) * scipy.stats.norm.pdf(u, scale=sd),
                    -numpy.inf,
                    numpy.inf,
                    epsabs=0,
                    epsrel=1e-12,
                    limit=200,
                )
                total += reference + math.log(value)
        return total
    nodes, weights = numpy.polynomial.hermite.hermgauss(1 if rule == 'laplace' else points)
    mode, scale = posterior_mode(conditional, tau2, k)
    total = 0.0
    for i in range(k):
        u = mode[i] + math.sqrt(2) * scale[i] * nodes
        values = numpy.array([conditional(numpy.full(k, ui))[i] for ui in u])
        terms = values + scipy.stats.norm.logpdf(u, scale=sd) + nodes**2 + numpy.log(weights) + math.log(math.sqrt(2) * scale[i])
        total += float(scipy.special.logsumexp(terms))
    return total


def bivariate_marginal_loglik(
    conditional: Conditional2, cov: FloatArray, rule: str, points: int = 7, *, studies: int, model: str = 'bivariate marginal'
) -> float:
    """Normal-mixture log-likelihood with two correlated random effects per study (the ``UM.RS`` intercept and treatment effect).

    ``conditional(u1, u2)`` maps one pair of effects per study to that study's log conditional likelihood; ``cov`` is the 2 x 2 covariance of the
    effects. ``'exact'`` is ``scipy.integrate.dblquad``; ``'agq'`` is the tensor-product adaptive rule of Pinheiro and Bates (1995) with
    ``points`` nodes per axis, centred at the mode with the Cholesky factor of the inverse Hessian; ``'laplace'`` is its one-node form.
    """
    if rule not in RULES:
        raise ValueError(f'unsupported rule: {rule}')
    precision = numpy.linalg.inv(cov)
    log_norm = -math.log(2 * math.pi) - 0.5 * math.log(numpy.linalg.det(cov))
    total = 0.0
    for i in range(studies):

        def f(u1: float, u2: float, i: int = i) -> float:
            return float(conditional(numpy.full(studies, u1), numpy.full(studies, u2))[i])

        def log_density(u1: float, u2: float, f: collections.abc.Callable[[float, float], float] = f) -> float:
            u = numpy.array([u1, u2])
            return f(u1, u2) + log_norm - 0.5 * float(u @ precision @ u)

        if rule == 'exact':
            reference = log_density(0.0, 0.0)
            sd1, sd2 = math.sqrt(cov[0, 0]), math.sqrt(cov[1, 1])
            with strict(model):
                value, _ = scipy.integrate.dblquad(
                    lambda b, a, reference=reference, log_density=log_density: math.exp(log_density(a, b) - reference),
                    -8 * sd1,
                    8 * sd1,
                    -8 * sd2,
                    8 * sd2,
                    epsabs=0,
                    epsrel=1e-10,
                )
            total += reference + math.log(value)
            continue
        fit = require_converged(
            scipy.optimize.minimize(  # type: ignore[call-overload]
                lambda u, log_density=log_density: -log_density(u[0], u[1]),
                numpy.zeros(2),
                method='Nelder-Mead',
                tol=1e-12,
                options={'maxiter': 4000},
            ),
            model,
        )
        mode = fit.x
        hessian = numpy.empty((2, 2))
        h = CURVATURE_STEP
        for a in range(2):
            for b in range(2):
                ea, eb = numpy.eye(2)[a] * h, numpy.eye(2)[b] * h
                hessian[a, b] = (
                    -log_density(*(mode + ea + eb)) + log_density(*(mode + ea - eb)) + log_density(*(mode - ea + eb)) - log_density(*(mode - ea - eb))
                ) / (4 * h * h)
        factor = numpy.linalg.cholesky(numpy.linalg.inv(hessian))
        nodes, weights = numpy.polynomial.hermite.hermgauss(1 if rule == 'laplace' else points)
        terms = []
        for node_a, wa in zip(nodes, weights, strict=True):
            for node_b, wb in zip(nodes, weights, strict=True):
                z = numpy.array([node_a, node_b])
                u = mode + math.sqrt(2) * factor @ z
                terms.append(log_density(u[0], u[1]) + float(z @ z) + math.log(wa * wb) + math.log(2 * abs(numpy.linalg.det(factor))))
        total += float(scipy.special.logsumexp(terms))
    return total


# ---------------------------------------------------------------------------------------------------------------------------------------------
# The formulas' own fits: every model is a log conditional likelihood per study, integrated over the random effects and maximised with
# ``scipy.optimize`` from the formulas' own start. Nothing here calls metafopy.
# ---------------------------------------------------------------------------------------------------------------------------------------------

type Kernel = collections.abc.Callable[[FloatArray, FloatArray], FloatArray]
type Scalar = collections.abc.Callable[[FloatArray], float]

# auditme: accept[literal-duplication] this module never imports metafopy, so it keeps its own lists of the options it fits
METHODS = ('FE', 'ML')
MODELS = ('CM.AL', 'CM.EL', 'UM.FS', 'UM.RS')
EXACT_POINTS = {1: 101, 2: 41}
VARIANCE_FLOOR = 1e-14
BOUNDARY = 1e-4
GRADIENT_TOLERANCE = 1e-6
CODINGS: dict[float, tuple[float, float]] = {0.5: (0.5, -0.5), 1.0: (1.0, 0.0), 0.0: (0.0, 1.0)}


def link_function(mu: FloatArray, link: str) -> FloatArray:
    """Linear predictor ``g(mu)`` of the mean ``mu`` under ``link`` (McCullagh and Nelder 1989, Table 2.1); only used to start a search."""
    if link == 'logit':
        return scipy.special.logit(mu)
    if link == 'probit':
        return scipy.special.ndtri(mu)
    if link == 'cauchit':
        return numpy.tan(math.pi * (mu - 0.5))
    if link == 'cloglog':
        return numpy.log(-numpy.log1p(-mu))
    if link == 'log':
        return numpy.log(mu)
    if link == 'identity':
        return numpy.asarray(mu, dtype=numpy.float64)
    if link == 'sqrt':
        return numpy.sqrt(mu)
    raise ValueError(f'unsupported link: {link}')


def fd_gradient(f: collections.abc.Callable[[FloatArray], typing.Any], x: FloatArray, h: float) -> FloatArray:
    """Central-difference gradient of ``f`` along the last axis of ``x``."""
    out = []
    for i in range(x.shape[-1]):
        e = numpy.zeros(x.shape[-1])
        e[i] = h
        out.append((numpy.asarray(f(x + e)) - numpy.asarray(f(x - e))) / (2 * h))
    return numpy.stack(out, axis=-1)


def fd_hessian(f: collections.abc.Callable[[FloatArray], typing.Any], x: FloatArray, h: float = 1e-3) -> FloatArray:
    """Hessian of ``f`` along the last axis of ``x``: central second differences at steps ``h / 2`` and ``h``, Richardson-extrapolated (error ``O(h^4)``)."""
    d = x.shape[-1]
    shape = numpy.asarray(f(x)).shape

    def once(step: float) -> FloatArray:
        out = numpy.empty((*shape, d, d))
        eye = numpy.eye(d) * step
        for i in range(d):
            for j in range(i, d):
                value = (f(x + eye[i] + eye[j]) - f(x + eye[i] - eye[j]) - f(x - eye[i] + eye[j]) + f(x - eye[i] - eye[j])) / (4 * step * step)
                out[..., i, j] = value
                out[..., j, i] = value
        return out

    return (4 * once(h / 2) - once(h)) / 3


def solve_mode(objective: collections.abc.Callable[[FloatArray], FloatArray], start: FloatArray, model: str) -> FloatArray:
    """Maximiser of a concave ``objective`` over the last axis (one per study) by damped Newton steps from ``start`` (from zero where ``start`` is invalid)."""
    u = numpy.where(numpy.isfinite(objective(start))[..., None], start, 0.0)
    current = objective(u)
    if not numpy.all(numpy.isfinite(current)):
        raise ArithmeticError(f'{model}: the conditional likelihood is not finite at zero')
    for _ in range(80):
        g = fd_gradient(objective, u, 1e-4)
        try:
            step = numpy.linalg.solve(fd_hessian(objective, u, 1e-3), -g[..., None])[..., 0]
        except numpy.linalg.LinAlgError as error:
            raise ArithmeticError(f'{model}: singular curvature at the conditional mode') from error
        if not numpy.all(numpy.isfinite(step)):
            raise ArithmeticError(f'{model}: the conditional mode search left the support')
        scale = numpy.ones(u.shape[:-1])
        for _ in range(60):
            candidate = u + scale[..., None] * numpy.clip(step, -2.0, 2.0)
            value = objective(candidate)
            bad = ~numpy.isfinite(value) | (value < current - 1e-12)
            if not numpy.any(bad):
                break
            scale = numpy.where(bad, scale / 2, scale)
        else:
            raise ArithmeticError(f'{model}: the conditional mode search found no ascent')
        u, current = candidate, value
        if numpy.max(numpy.abs(scale[..., None] * step)) < 1e-10:
            return u
    raise ArithmeticError(f'{model}: the conditional mode did not converge')


def study_integrals(cond: Kernel, fixed: FloatArray, cov: FloatArray, rule: str, points: int, start: FloatArray, model: str) -> tuple[FloatArray, FloatArray]:
    """Log of each study's integral ``int exp(l_i(u)) N(u; 0, cov) du`` by the adaptive rule of Pinheiro and Bates (1995, Eq. 2.4), with its mode.

    ``cond(fixed, U)`` maps random effects ``U`` of shape ``(k, m, d)`` to log conditional likelihoods of shape ``(k, m)``; ``start`` is the mode to
    begin the Newton search from (shape ``(k, 1, d)``). ``'laplace'`` has one node; ``'exact'`` has many (see ``EXACT_POINTS``), and equals
    ``scipy.integrate.quad`` to the precision that the unit checks pin.
    """
    d = cov.shape[0]
    precision = numpy.linalg.inv(cov)
    constant = -0.5 * (d * math.log(2 * math.pi) + math.log(numpy.linalg.det(cov)))

    def objective(u: FloatArray) -> FloatArray:
        return typing.cast('FloatArray', cond(fixed, u) - 0.5 * numpy.einsum('kmi,ij,kmj->km', u, precision, u) + constant)

    mode = solve_mode(objective, start, model)
    curvature = -fd_hessian(objective, mode, 1e-2)[:, 0]
    try:
        factor = numpy.linalg.cholesky(numpy.linalg.inv(curvature))
    except numpy.linalg.LinAlgError as error:
        raise ArithmeticError(f'{model}: the conditional likelihood is not concave at its mode') from error
    n = 1 if rule == 'laplace' else (EXACT_POINTS[d] if rule == 'exact' else points)
    nodes, weights = numpy.polynomial.hermite.hermgauss(n)
    z = numpy.stack([grid.ravel() for grid in numpy.meshgrid(*([nodes] * d), indexing='ij')], axis=-1)
    log_w = numpy.sum(numpy.log(numpy.stack(numpy.meshgrid(*([weights] * d), indexing='ij'), axis=-1)), axis=-1).ravel()
    u = mode + math.sqrt(2) * numpy.einsum('kij,nj->kni', factor, z)
    log_det = numpy.log(numpy.abs(numpy.linalg.det(factor)))
    terms = objective(u) + numpy.sum(z * z, axis=-1) + log_w + d * math.log(math.sqrt(2)) + log_det[:, None]
    return scipy.special.logsumexp(terms, axis=1), mode


@dataclasses.dataclass(frozen=True, eq=False)
class Structure:
    """One model at one ``method``: its parameters, log conditional likelihood and variance parametrisation.

    The parameter vector is ``(fixed, variance)``; ``fixed`` starts with the ``p`` moderator coefficients, and ``variance`` holds standard deviations
    (and, for correlated effects, ``atanh(rho)``), so a variance of zero is a stationary point and not a boundary.
    """

    model: str
    method: str
    p: int
    k: int
    d: int
    start: FloatArray
    cond: Kernel
    variance_start: FloatArray
    correlated: bool = False
    information: collections.abc.Callable[[FloatArray], FloatArray] | None = None

    def covariance(self, variance: FloatArray) -> FloatArray:
        s2 = variance[: self.d] ** 2 + VARIANCE_FLOOR
        cov = numpy.diag(s2)
        if self.d == 2 and self.correlated:
            cov[0, 1] = cov[1, 0] = math.tanh(variance[2]) * math.sqrt(s2[0] * s2[1])
        return cov

    def parameters(self, variance: FloatArray) -> tuple[float, float, float]:
        """``(tau2, sigma2, rho)`` of a variance vector; ``nan`` where the model has none."""
        nan = math.nan
        if self.model == 'UM.RS':
            sigma2 = float(variance[0] ** 2)
            if self.method == 'FE':
                return 0.0, sigma2, nan
            return float(variance[1] ** 2), sigma2, (math.tanh(variance[2]) if self.correlated else nan)
        return (float(variance[0] ** 2) if self.method == 'ML' else 0.0), nan, nan


@dataclasses.dataclass(frozen=True, eq=False)
class GLMMReference:
    """A fit by the formulas: ``beta`` with standard errors from the observed information, variances, log-likelihood and gradient at the optimum."""

    beta: FloatArray
    se: FloatArray
    vb: FloatArray
    tau2: float
    sigma2: float
    rho: float
    loglik: float
    gradient_norm: float
    theta: FloatArray
    rule: str
    points: int


@dataclasses.dataclass(frozen=True, eq=False)
class Problem:
    """Data and model of one fit; build it with ``glmm_problem``, ``poisson_problem`` or ``one_group_problem``.

    ``y`` holds the counts of each study (one column per group), ``size`` the binomial trials, ``offset`` the log person-times; ``n1`` and ``n2`` are the
    group sizes (person-times) that ``CM.*`` conditions on.
    """

    kind: str
    model: str
    family: str
    link: str
    y: FloatArray
    size: FloatArray
    offset: FloatArray
    X: FloatArray
    coding: float
    cor: bool
    n1: FloatArray
    n2: FloatArray


def design(k: int, X: FloatArray | None) -> FloatArray:
    return numpy.ones((k, 1)) if X is None else numpy.asarray(X, dtype=numpy.float64)


def check_model(model: str) -> None:
    if model not in MODELS:
        raise ValueError(f'unsupported model: {model}')


def glmm_problem(
    ai: FloatArray,
    bi: FloatArray,
    ci: FloatArray,
    di: FloatArray,
    model: str,
    *,
    X: FloatArray | None = None,
    coding: float = 0.5,
    cor: bool = False,
    link: str = 'logit',
) -> Problem:
    """Two-group binomial data (``OR``): ``ai`` events among ``ai + bi`` in group 1 and ``ci`` among ``ci + di`` in group 2."""
    ai, bi, ci, di = (numpy.asarray(v, dtype=numpy.float64) for v in (ai, bi, ci, di))
    check_model(model)
    return Problem(
        'two_group',
        model,
        'binomial',
        link,
        numpy.column_stack([ai, ci]),
        numpy.column_stack([ai + bi, ci + di]),
        numpy.zeros((ai.size, 2)),
        design(ai.size, X),
        coding,
        cor,
        ai + bi,
        ci + di,
    )


def poisson_problem(
    x1i: FloatArray,
    x2i: FloatArray,
    t1i: FloatArray,
    t2i: FloatArray,
    model: str,
    *,
    X: FloatArray | None = None,
    coding: float = 0.5,
    cor: bool = False,
    link: str = 'log',
) -> Problem:
    """Two-group Poisson data (``IRR``): ``x1i`` events in ``t1i`` person-time in group 1, ``x2i`` in ``t2i`` in group 2 (Bagos and Nikolopoulos 2009)."""
    x1i, x2i, t1i, t2i = (numpy.asarray(v, dtype=numpy.float64) for v in (x1i, x2i, t1i, t2i))
    check_model(model)
    return Problem(
        'two_group',
        model,
        'poisson',
        link,
        numpy.column_stack([x1i, x2i]),
        numpy.zeros((x1i.size, 2)),
        numpy.log(numpy.column_stack([t1i, t2i])),
        design(x1i.size, X),
        coding,
        cor,
        t1i,
        t2i,
    )


def one_group_problem(xi: FloatArray, ni: FloatArray, family: str, link: str, *, X: FloatArray | None = None) -> Problem:
    """One-group data: ``xi`` events among ``ni`` (``family='binomial'``: ``PLO``, ``PR``, ``PLN``) or in ``ni`` person-time (``'poisson'``: ``IRLN``)."""
    xi, ni = numpy.asarray(xi, dtype=numpy.float64), numpy.asarray(ni, dtype=numpy.float64)
    binomial = family == 'binomial'
    return Problem(
        'one_group',
        'none',
        family,
        link,
        xi[:, None],
        (ni if binomial else numpy.zeros_like(xi))[:, None],
        (numpy.zeros_like(xi) if binomial else numpy.log(ni))[:, None],
        design(xi.size, X),
        1.0,
        False,
        ni,
        ni,
    )


def group_logpmf(problem: Problem, j: int, eta: FloatArray) -> FloatArray:
    """Log-probability of the counts of group ``j`` for linear predictor ``eta`` of shape ``(k, m)``."""
    y = problem.y[:, j : j + 1]
    if problem.family == 'binomial':
        return binomial_logpmf(y, problem.size[:, j : j + 1], eta, problem.link)
    return poisson_logpmf(y, eta, problem.link)


def nchg_support(n1: float, n2: float, m: float) -> tuple[FloatArray, FloatArray]:
    """Support of the central hypergeometric table with margins ``n1``, ``n2``, ``m`` and its log weights ``log C(n1, x) C(n2, m - x)``."""
    support = numpy.arange(max(0.0, m - n2), min(n1, m) + 1)
    log_w = (
        scipy.special.gammaln(n1 + 1)
        - scipy.special.gammaln(support + 1)
        - scipy.special.gammaln(n1 - support + 1)
        + scipy.special.gammaln(n2 + 1)
        - scipy.special.gammaln(m - support + 1)
        - scipy.special.gammaln(n2 - m + support + 1)
    )
    return support, log_w


def conditional_kernel(problem: Problem) -> tuple[Kernel, FloatArray]:
    """Log conditional likelihood of ``CM.AL`` / ``CM.EL`` given ``psi_i + u_i``, and a start for ``beta`` (the pooled log ratio)."""
    k, p = problem.X.shape
    ai, ci = problem.y[:, 0], problem.y[:, 1]
    n1, n2 = problem.n1, problem.n2
    approximate = problem.model == 'CM.AL' or problem.family == 'poisson'
    shift = numpy.log(n1 / n2)
    supports = [nchg_support(n1[i], n2[i], ai[i] + ci[i]) for i in range(k)]
    owns = [int(numpy.argmin(numpy.abs(supports[i][0] - ai[i]))) for i in range(k)]

    def cond(fixed: FloatArray, u: FloatArray) -> FloatArray:
        psi = (problem.X @ fixed[:p])[:, None] + numpy.sum(u, axis=-1)
        if approximate:
            return binomial_logpmf(ai[:, None], (ai + ci)[:, None], psi + shift[:, None], 'logit')
        out = numpy.empty(psi.shape)
        for i, (support, log_w) in enumerate(supports):
            out[i] = log_w[owns[i]] + ai[i] * psi[i] - scipy.special.logsumexp(log_w[None, :] + support[None, :] * psi[i][:, None], axis=1)
        return out

    pooled = (ai.sum() + 0.5) * (n2.sum() + 1) / ((ci.sum() + 0.5) * (n1.sum() + 1))
    return cond, numpy.concatenate([[math.log(pooled)], numpy.zeros(p - 1)])


def conditional_information(problem: Problem, fixed: FloatArray) -> FloatArray:
    """Information of the ``CM.EL`` fixed-effects fit: ``sum_i Var_i(ai) x_i x_i'``, the conditional variance of ``ai`` given its margins (Breslow 1981)."""
    k, p = problem.X.shape
    ai, ci = problem.y[:, 0], problem.y[:, 1]
    info = numpy.zeros((p, p))
    psi = problem.X @ fixed[:p]
    for i in range(k):
        support, log_w = nchg_support(problem.n1[i], problem.n2[i], ai[i] + ci[i])
        log_p = log_w + support * psi[i]
        prob = numpy.exp(log_p - scipy.special.logsumexp(log_p))
        mean = float(prob @ support)
        info += float(prob @ (support - mean) ** 2) * numpy.outer(problem.X[i], problem.X[i])
    return info


def group_vectors(problem: Problem, d: int, method: str) -> FloatArray:
    """Rows ``Z_j`` of the random-effects design of group ``j`` (``UM.FS``: the coding; ``UM.RS``: intercept and coding)."""
    c = CODINGS[problem.coding]
    if problem.model == 'UM.FS':
        return numpy.array([[c[0]], [c[1]]]) if d else numpy.zeros((2, 0))
    if method == 'FE':
        return numpy.ones((2, 1))
    return numpy.array([[1.0, c[0]], [1.0, c[1]]])


def mean_start(problem: Problem, j: int) -> FloatArray:
    """Linear predictor, without offset, of each study's group-``j`` rate with 0.5 added to the count: a start for the intercepts."""
    y = problem.y[:, j]
    if problem.family == 'binomial':
        return link_function(numpy.clip((y + 0.5) / (problem.size[:, j] + 1), 1e-6, 1 - 1e-6), problem.link)
    return numpy.log(y + 0.5) - problem.offset[:, j]


def structure(problem: Problem, method: str) -> Structure:
    """The model ``problem`` at ``method`` as a ``Structure`` (see its docstring for the parameter layout)."""
    if method not in METHODS:
        raise ValueError(f'unsupported method: {method}')
    k, p = problem.X.shape
    ml = method == 'ML'
    one_variance = numpy.array([0.3]) if ml else numpy.zeros(0)
    if problem.kind == 'one_group':
        start = numpy.concatenate([[float(numpy.mean(mean_start(problem, 0)))], numpy.zeros(p - 1)])

        def cond1(fixed: FloatArray, u: FloatArray) -> FloatArray:
            eta = (problem.X @ fixed[:p] + problem.offset[:, 0])[:, None] + numpy.sum(u, axis=-1)
            return group_logpmf(problem, 0, eta)

        return Structure('none', method, p, k, 1 if ml else 0, start, cond1, one_variance)
    if problem.model in ('CM.AL', 'CM.EL'):
        cond2, start2 = conditional_kernel(problem)
        info = None
        if problem.model == 'CM.EL' and problem.family == 'binomial' and not ml:

            def info(fixed: FloatArray) -> FloatArray:
                return conditional_information(problem, fixed)

        return Structure(problem.model, method, p, k, 1 if ml else 0, start2, cond2, one_variance, information=info)
    fixed_study = problem.model == 'UM.FS'
    d = (1 if ml else 0) if fixed_study else (2 if ml else 1)
    Z = group_vectors(problem, d, method)
    base = mean_start(problem, 1)
    start3 = numpy.concatenate([numpy.zeros(p), base if fixed_study else [float(numpy.mean(base))]])

    def cond3(fixed: FloatArray, u: FloatArray) -> FloatArray:
        xb = problem.X @ fixed[:p]
        alpha = fixed[p : p + k] if fixed_study else fixed[p]
        total = numpy.zeros(u.shape[:2])
        for j in range(2):
            eta = (alpha + (xb if j == 0 else 0.0) + problem.offset[:, j])[:, None] + u @ Z[j]
            total = total + group_logpmf(problem, j, eta)
        return total

    correlated = problem.cor and problem.model == 'UM.RS' and ml
    variance = numpy.array([0.3] if d == 1 and fixed_study else ([0.5] if d == 1 else ([0.5, 0.3, 0.0] if correlated else [0.5, 0.3])))
    if d == 0:
        variance = numpy.zeros(0)
    return Structure(problem.model, method, p, k, d, start3, cond3, variance, correlated=correlated)


def loglik_function(problem: Problem, method: str, rule: str = 'agq', points: int = 7) -> tuple[Structure, Scalar]:
    """The structure and its marginal log-likelihood as a function of the parameter vector ``(fixed, variance)``."""
    if rule not in RULES:
        raise ValueError(f'unsupported rule: {rule}')
    st = structure(problem, method)
    nf = st.start.size
    modes: dict[str, FloatArray] = {}

    def loglik(theta: FloatArray) -> float:
        fixed, variance = theta[:nf], theta[nf:]
        if st.d == 0:
            return float(numpy.sum(st.cond(fixed, numpy.zeros((st.k, 1, 0)))))
        start = modes.get('mode', numpy.zeros((st.k, 1, st.d)))
        try:
            values, mode = study_integrals(st.cond, fixed, st.covariance(variance), rule, points, start, problem.model)
        except ArithmeticError:
            return -math.inf
        modes['mode'] = mode
        return float(numpy.sum(values))

    return st, loglik


def gradient_at(loglik: Scalar, theta: FloatArray, *, fine: bool = True) -> FloatArray:
    """Gradient of ``loglik`` at ``theta`` (the score): central differences, Richardson-extrapolated at steps ``5e-5`` and ``1e-4`` when ``fine``."""
    if not fine:
        return fd_gradient(loglik, theta, 1e-4)
    return (4 * fd_gradient(loglik, theta, 5e-5) - fd_gradient(loglik, theta, 1e-4)) / 3


def newton_polish(loglik: Scalar, theta: FloatArray, free: numpy.typing.NDArray[numpy.bool_]) -> FloatArray:
    """Newton steps on the ``free`` parameters until the score is below ``1e-8``."""
    theta = theta.copy()
    index = numpy.flatnonzero(free)
    for _ in range(25):
        g = gradient_at(loglik, theta)[index]
        if numpy.max(numpy.abs(g)) < 1e-8:
            break

        def sub(x: FloatArray, theta: FloatArray = theta) -> float:
            full = theta.copy()
            full[index] = x
            return loglik(full)

        try:
            step = numpy.linalg.solve(fd_hessian(sub, theta[index], 5e-3), -g)
        except numpy.linalg.LinAlgError:
            break
        current = loglik(theta)
        scale = 1.0
        candidate = theta
        while scale > 1e-6:
            candidate = theta.copy()
            candidate[index] += scale * step
            if loglik(candidate) >= current - 1e-12:
                break
            scale /= 2
        else:
            break
        theta = candidate
    return theta


def maximise(loglik: Scalar, theta0: FloatArray, st: Structure, *, free: numpy.typing.NDArray[numpy.bool_] | None = None) -> FloatArray:
    """Maximum of ``loglik``: BFGS from ``theta0``, then Newton steps, then ``scipy.optimize.root`` on the score for fixed-effects fits."""
    free = numpy.ones(theta0.size, dtype=bool) if free is None else free
    index = numpy.flatnonzero(free)

    def expand(x: FloatArray) -> FloatArray:
        full = theta0.copy()
        full[index] = x
        return full

    def negative(x: FloatArray) -> float:
        value = loglik(expand(x))
        return -value if math.isfinite(value) else 1e300

    result = scipy.optimize.minimize(  # type: ignore[call-overload]
        negative,
        theta0[index],
        jac=lambda x: numpy.nan_to_num(-gradient_at(loglik, expand(x), fine=False)[index], nan=0.0, posinf=0.0, neginf=0.0),
        method='BFGS',
        options={'gtol': 1e-7, 'maxiter': 400},
    )
    theta = newton_polish(loglik, expand(result.x), free)
    if st.d == 0:
        root = scipy.optimize.root(lambda x: gradient_at(loglik, expand(x))[index], theta[index], method='hybr', tol=1e-13)  # type: ignore[call-overload]
        if root.success and loglik(expand(root.x)) >= loglik(theta) - 1e-9:
            theta = expand(root.x)
    return theta


def observed_vcov(loglik: Scalar, theta: FloatArray, st: Structure) -> FloatArray:
    """Covariance of ``beta``: the inverse observed information over all parameters (a variance at zero is dropped); ``CM.EL`` ``FE`` is analytic."""
    nf = st.start.size
    if st.information is not None:
        return numpy.linalg.inv(st.information(theta[:nf]))[: st.p, : st.p]
    keep = numpy.ones(theta.size, dtype=bool)
    keep[nf:] = numpy.abs(theta[nf:]) >= BOUNDARY
    if st.correlated:
        keep[nf + 2] = keep[nf] and keep[nf + 1]
    index = numpy.flatnonzero(keep)

    def sub(x: FloatArray) -> float:
        full = theta.copy()
        full[index] = x
        return loglik(full)

    try:
        return numpy.linalg.inv(-fd_hessian(sub, theta[index], 5e-3))[: st.p, : st.p]
    except numpy.linalg.LinAlgError as error:
        raise ArithmeticError(f'{st.model}: singular observed information') from error


def fixed_start(problem: Problem, st: Structure) -> FloatArray:
    """Fixed parameters of the fixed-effects model of ``problem``, as a start for the mixed model."""
    if problem.model == 'UM.RS':
        return st.start
    fe_st, fe_loglik = loglik_function(problem, 'FE')
    return maximise(fe_loglik, fe_st.start, fe_st)


def fit_problem(problem: Problem, method: str, rule: str = 'agq', points: int = 7) -> GLMMReference:
    """Maximum likelihood fit of ``problem`` by the formulas: the heart of every ``glmm_fit``-style entry point."""
    st, loglik = loglik_function(problem, method, rule, points)
    nf = st.start.size
    theta0 = numpy.concatenate([st.start, st.variance_start])
    if method == 'ML' and problem.model != 'UM.RS':
        theta0[:nf] = fixed_start(problem, st)
    theta = maximise(loglik, theta0, st)
    norm = float(numpy.max(numpy.abs(gradient_at(loglik, theta))))
    if not norm <= GRADIENT_TOLERANCE:
        raise ArithmeticError(f'{st.model}: the score at the optimum is {norm:.3g}, not below {GRADIENT_TOLERANCE:g}')
    vb = observed_vcov(loglik, theta, st)
    tau2, sigma2, rho = st.parameters(theta[nf:])
    return GLMMReference(theta[: st.p].copy(), numpy.sqrt(numpy.diag(vb)), vb, tau2, sigma2, rho, loglik(theta), norm, theta, rule, points)


def glmm_fit(
    ai: FloatArray,
    bi: FloatArray,
    ci: FloatArray,
    di: FloatArray,
    model: str,
    method: str,
    rule: str = 'agq',
    points: int = 7,
    *,
    X: FloatArray | None = None,
    coding: float = 0.5,
    cor: bool = False,
    link: str = 'logit',
) -> GLMMReference:
    """The formulas' own maximum likelihood fit of the two-group binomial models: estimates, standard errors from the observed information, variances."""
    return fit_problem(glmm_problem(ai, bi, ci, di, model, X=X, coding=coding, cor=cor, link=link), method, rule, points)


def poisson_glmm_fit(
    x1i: FloatArray,
    x2i: FloatArray,
    t1i: FloatArray,
    t2i: FloatArray,
    model: str,
    method: str,
    rule: str = 'agq',
    points: int = 7,
    *,
    X: FloatArray | None = None,
    coding: float = 0.5,
    cor: bool = False,
    link: str = 'log',
) -> GLMMReference:
    """The formulas' own fit of the two-group Poisson models (``IRR``); ``CM.AL`` and ``CM.EL`` coincide (a binomial given the total)."""
    return fit_problem(poisson_problem(x1i, x2i, t1i, t2i, model, X=X, coding=coding, cor=cor, link=link), method, rule, points)


def one_group_glmm_fit(
    xi: FloatArray, ni: FloatArray, family: str, link: str, method: str, rule: str = 'agq', points: int = 7, *, X: FloatArray | None = None
) -> GLMMReference:
    """The formulas' own fit of the one-group models: ``PLO`` (binomial, logit), ``PR`` (identity), ``PLN`` (log) and ``IRLN`` (Poisson, log)."""
    return fit_problem(one_group_problem(xi, ni, family, link, X=X), method, rule, points)


def closed_form_plo(xi: FloatArray, ni: FloatArray) -> float:
    """Fixed-effects ``PLO`` without moderators: ``logit(sum xi / sum ni)`` (Hamza et al. 2008)."""
    return float(scipy.special.logit(numpy.sum(xi) / numpy.sum(ni)))


def closed_form_irln(xi: FloatArray, ti: FloatArray) -> float:
    """Fixed-effects ``IRLN`` without moderators: ``log(sum xi / sum ti)``."""
    return float(math.log(numpy.sum(xi) / numpy.sum(ti)))


def loglik_at(problem: Problem, method: str, theta: FloatArray, rule: str = 'agq', points: int = 7) -> tuple[float, FloatArray]:
    """The formulas' marginal log-likelihood at ``theta`` (``(fixed, variance)`` as in ``GLMMReference.theta``) and its score."""
    _, loglik = loglik_function(problem, method, rule, points)
    return loglik(theta), gradient_at(loglik, theta)


def parameter_vector(st: Structure, beta: FloatArray, tau2: float, sigma2: float, rho: float) -> FloatArray:
    """Parameter vector for ``beta`` and the variances of a model (nuisance parameters at their start)."""
    variance: list[float] = []
    if st.model == 'UM.RS':
        variance.append(math.sqrt(sigma2))
        if st.method == 'ML':
            variance.append(math.sqrt(tau2))
            if st.correlated:
                variance.append(math.atanh(rho))
    elif st.method == 'ML':
        variance.append(math.sqrt(tau2))
    fixed = st.start.copy()
    fixed[: st.p] = beta
    return numpy.concatenate([fixed, numpy.array(variance)])


def profile_loglik_at(
    problem: Problem, method: str, beta: FloatArray, tau2: float, sigma2: float = math.nan, rho: float = math.nan, rule: str = 'agq', points: int = 7
) -> float:
    """Log-likelihood at given ``beta`` and variances, maximised over the parameters a fit does not report (``UM.FS`` study intercepts, ``UM.RS`` intercept)."""
    st, loglik = loglik_function(problem, method, rule, points)
    theta = parameter_vector(st, beta, tau2, sigma2, rho)
    free = numpy.zeros(theta.size, dtype=bool)
    free[st.p : st.start.size] = True
    return loglik(maximise(loglik, theta, st, free=free))


def rma_glmm_rule(model: str, method: str, nagq: int) -> tuple[str, int]:
    """The approximation ``rma_glmm`` documents for a fit: Laplace for ``UM.RS`` with ``ML`` (two random effects, which ``glmer`` allows no more
    than one quadrature point for), the exact integral for ``CM.EL`` (``control`` ``int_rel_tol`` tightened in the cases) and adaptive quadrature
    with ``nAGQ`` points otherwise (Pinheiro and Bates 1995)."""
    if model == 'UM.RS' and method == 'ML':
        return 'laplace', 1
    if model == 'CM.EL':
        return 'exact', nagq
    return 'agq', nagq


def rma_glmm_counts(
    ai: FloatArray, bi: FloatArray | None, ci: FloatArray, di: FloatArray | None, n1i: FloatArray | None, n2i: FloatArray | None, *, drop00: bool = True
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """The 2x2 counts ``rma_glmm`` fits for ``OR`` from ``ai, bi, ci, di`` or ``ai, n1i, ci, n2i``: incomplete studies omitted, and with ``drop00``
    the studies without an event or without a non-event in both groups dropped (Stijnen et al. 2010; they carry no information on the odds ratio)."""
    a, c = numpy.asarray(ai, dtype=numpy.float64), numpy.asarray(ci, dtype=numpy.float64)
    b = numpy.asarray(n1i, dtype=numpy.float64) - a if bi is None else numpy.asarray(bi, dtype=numpy.float64)
    d = numpy.asarray(n2i, dtype=numpy.float64) - c if di is None else numpy.asarray(di, dtype=numpy.float64)
    keep = numpy.isfinite(a + b + c + d)
    if drop00:
        keep &= ~(((a == 0) & (c == 0)) | ((b == 0) & (d == 0)))
    return a[keep], b[keep], c[keep], d[keep]


def rma_glmm_problem(
    ai: FloatArray,
    ci: FloatArray,
    model: str,
    method: str = 'ML',
    nAGQ: int = 7,
    coding: float = 0.5,
    cor: bool = False,
    drop00: bool = True,
    bi: FloatArray | None = None,
    di: FloatArray | None = None,
    n1i: FloatArray | None = None,
    n2i: FloatArray | None = None,
) -> tuple[Problem, str, int]:
    """The problem, rule and quadrature points of a ``measure = 'OR'`` call of ``rma_glmm``, from its own arguments."""
    a, b, c, d = rma_glmm_counts(ai, bi, ci, di, n1i, n2i, drop00=drop00)
    rule, points = rma_glmm_rule(model, method, nAGQ)
    return glmm_problem(a, b, c, d, model, coding=coding, cor=cor), rule, points


FIT_CACHE: dict[tuple[typing.Any, ...], GLMMReference] = {}


def rma_glmm_fit(
    ai: FloatArray,
    ci: FloatArray,
    model: str,
    method: str = 'ML',
    nAGQ: int = 7,
    coding: float = 0.5,
    cor: bool = False,
    drop00: bool = True,
    bi: FloatArray | None = None,
    di: FloatArray | None = None,
    n1i: FloatArray | None = None,
    n2i: FloatArray | None = None,
) -> GLMMReference:
    """The formulas' own fit of an ``rma_glmm`` call, kept for the other ``rma_glmm_*`` formulas of the same case."""
    problem, rule, points = rma_glmm_problem(ai, ci, model, method, nAGQ, coding, cor, drop00, bi, di, n1i, n2i)
    key = (problem.y.tobytes(), problem.size.tobytes(), model, method, rule, points, coding, cor)
    if key not in FIT_CACHE:
        FIT_CACHE[key] = fit_problem(problem, method, rule, points)
    return FIT_CACHE[key]


def reference_field(name: str) -> collections.abc.Callable[..., typing.Any]:
    """A formula giving one field of ``rma_glmm_fit`` (``beta``, ``se``, ``tau2``, ``sigma2``), called with the same arguments."""

    @functools.wraps(rma_glmm_fit)
    def field(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        return getattr(rma_glmm_fit(*args, **kwargs), name)

    field.__doc__ = f"The ``{name}`` of the formulas' own fit of an ``rma_glmm`` call."
    return field


rma_glmm_beta = reference_field('beta')
rma_glmm_se = reference_field('se')
rma_glmm_sigma2 = reference_field('sigma2')
rma_glmm_tau2 = reference_field('tau2')


def profile_check(problem: Problem, method: str, beta: FloatArray, tau2: float, sigma2: float, rho: float, rule: str, points: int) -> tuple[float, float]:
    """The log-likelihood at given ``beta`` and variances, maximised over the parameters a fit does not report, and the largest score of the
    reported parameters there."""
    st, loglik = loglik_function(problem, method, rule, points)
    theta = parameter_vector(st, beta, tau2, sigma2, rho)
    free = numpy.zeros(theta.size, dtype=bool)
    free[st.p : st.start.size] = True
    if free.any():
        theta = maximise(loglik, theta, st, free=free)
    reported = numpy.ones(theta.size, dtype=bool)
    reported[st.p : st.start.size] = False
    return loglik(theta), float(numpy.max(numpy.abs(gradient_at(loglik, theta)[reported])))
