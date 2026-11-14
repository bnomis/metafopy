"""Independent re-derivations of the ``rma_glmm`` likelihoods and fits, coded from the defining publications.

Rule: this module imports ``numpy`` and ``scipy`` only, never ``metafopy`` (``test_coverage.py`` checks it with ``ast``), and each function cites the
equation it codes, so that an expected value never comes from metafopy's or metafor's source.

A ``derived`` case names a function here as ``glmm_formulas.<name>``, the value of an ``expect`` key; it is called with the arguments of the case that
it declares (see ``cases.formula_value``). See ``agents/specs/correctness-glmm/spec.md``.
"""

import collections.abc
import contextlib
import math
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
