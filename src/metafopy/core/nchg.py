"""Exact conditional likelihood of 2x2 tables (non-central hypergeometric distribution).

The conditional model ``model='CM.EL'`` of ``rma.glmm``
(https://github.com/wviechtb/metafor/blob/master/R/rma.glmm.r and
``misc.func.hidden.rma.glmm.r``) conditions on the margins of each table, so
that ``ai`` follows Fisher's non-central hypergeometric distribution with odds
ratio ``exp(mu_i)``; in the random-effects model ``mu_i`` is further integrated
over a normal distribution with ``integrate()``. R evaluates the distribution
with ``BiasedUrn::dFNCHypergeo`` (Fog 2008, doi:10.1080/03610910701790236);
here the exact probabilities are normalised directly, which agrees with that
function to its precision of 1e-10. The integral is evaluated by QUADPACK's
``qagi`` through SciPy, the routine behind R's ``integrate()``.

References: Liao and Rosen (2001), doi:10.1198/000313001753272547 (the
algorithm of metafor's ``.dnoncenhypergeom``); Stijnen et al. (2010),
doi:10.1002/sim.3767 (the exact-likelihood random-effects model).
"""

import dataclasses
import math
import sys

import numpy
import numpy.typing
import scipy.integrate
import scipy.special

import metafopy.errors

__all__ = ['IntegrationControl', 'Tables', 'neg_loglik', 'study_loglik']

type FloatArray = numpy.typing.NDArray[numpy.float64]

SQRT_2PI = math.sqrt(2 * math.pi)
LOG_ODDS_BOUND = 12 * math.log(10)


@dataclasses.dataclass(frozen=True, slots=True)
class IntegrationControl:
    """Arguments of R's ``integrate()`` (the ``intCtrl`` control list of ``rma.glmm``)."""

    rel_tol: float = sys.float_info.epsilon**0.25
    subdivisions: int = 100


@dataclasses.dataclass(frozen=True, slots=True)
class Tables:
    """Precomputed support of the non-central hypergeometric distribution of each table.

    ``log_base[i]`` holds ``lchoose(n1, y) + lchoose(n2, m1 - y)`` over the support ``y = lo[i] .. hi[i]``
    and ``index[i]`` the position of the observed ``ai`` in it.
    """

    support: tuple[FloatArray, ...]
    log_base: tuple[FloatArray, ...]
    index: tuple[int, ...]

    @property
    def k(self) -> int:
        return len(self.index)

    @classmethod
    def from_counts(cls, ai: FloatArray, bi: FloatArray, ci: FloatArray, di: FloatArray) -> Tables:
        """Tables with ``ai`` events in group 1 (size ``ai + bi``) and ``ci`` in group 2 (size ``ci + di``).

        Raises:
            metafopy.errors.InputError: a count is negative or not an integer.
        """
        support: list[FloatArray] = []
        base: list[FloatArray] = []
        index: list[int] = []
        for a, b, c, d in zip(ai, bi, ci, di, strict=True):
            if min(a, b, c, d) < 0 or any(v != math.floor(v) for v in (a, b, c, d)):
                raise metafopy.errors.InputError('The counts must be non-negative integers.')
            n1, n2, m1 = a + b, c + d, a + c
            lo, hi = max(0.0, m1 - n2), min(n1, m1)
            y = numpy.arange(lo, hi + 1.0)
            support.append(y)
            base.append(
                scipy.special.gammaln(n1 + 1)
                - scipy.special.gammaln(y + 1)
                - scipy.special.gammaln(n1 - y + 1)
                + scipy.special.gammaln(n2 + 1)
                - scipy.special.gammaln(m1 - y + 1)
                - scipy.special.gammaln(n2 - m1 + y + 1)
            )
            index.append(int(a - lo))
        return cls(tuple(support), tuple(base), tuple(index))


def log_probability(tables: Tables, i: int, log_or: float) -> float:
    """``log P(ai | margins, odds ratio = exp(log_or))`` for table ``i``, with ``log_or`` clamped to +-log(1e12) as metafor does."""
    lo = min(max(log_or, -LOG_ODDS_BOUND), LOG_ODDS_BOUND)
    w = tables.log_base[i] + tables.support[i] * lo
    top = float(w.max())
    return float(w[tables.index[i]] - top - math.log(float(numpy.exp(w - top).sum())))


def study_loglik(tables: Tables, i: int, mu: float, tau2: float, random: bool, control: IntegrationControl) -> float:
    """Log-likelihood contribution of table ``i`` for the fixed (``random=False``) or normal-mixed log odds ratio."""
    if not random:
        return log_probability(tables, i, mu)
    sd = math.sqrt(tau2)

    def integrand(t: float) -> float:
        lo = min(max(t, -LOG_ODDS_BOUND), LOG_ODDS_BOUND)
        return math.exp(log_probability(tables, i, lo)) * math.exp(-0.5 * ((lo - mu) / sd) ** 2) / (sd * SQRT_2PI)

    value = scipy.integrate.quad(
        integrand,
        -math.inf,
        math.inf,
        epsabs=control.rel_tol,
        epsrel=control.rel_tol,
        limit=control.subdivisions,
    )[0]
    return math.log(value) if value > 0 else -math.inf


def neg_loglik(parms: FloatArray, tables: Tables, X: FloatArray, random: bool, control: IntegrationControl) -> float:
    """Minus the log-likelihood at ``parms = (beta, log(tau2))``; the last element is ignored when ``random`` is false.

    Mirrors metafor's ``.dnchg()``.
    """
    p = X.shape[1]
    beta = parms[:p]
    tau2 = math.exp(parms[p]) if random else 0.0
    mu = X @ beta
    total = 0.0
    for i in range(tables.k):
        total += study_loglik(tables, i, float(mu[i]), tau2, random, control)
    return -total
