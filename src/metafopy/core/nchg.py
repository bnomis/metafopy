"""Exact conditional likelihood of 2x2 tables (non-central hypergeometric distribution).

The conditional model ``model='CM.EL'`` of ``rma.glmm``
(https://github.com/wviechtb/metafor/blob/master/R/rma.glmm.r and
``misc.func.hidden.rma.glmm.r``) conditions on the margins of each table, so
that ``ai`` follows Fisher's non-central hypergeometric distribution with odds
ratio ``exp(mu_i)``; in the random-effects model ``mu_i`` is further integrated
over a normal distribution with ``integrate()``. R evaluates the distribution
with ``BiasedUrn::dFNCHypergeo`` (Fog 2008, doi:10.1080/03610910701790236),
which sums the proportional function outwards from the mean only until a term
falls below ``0.1 * precision`` (``control$dnchgprec``, 1e-10 by default). That
truncation is a deterministic jump of about 1e-11 in the normalising constant;
``numDeriv::hessian()`` divides it by steps of about 1e-6, so it shows in the
standard errors of the exact-likelihood model. :func:`fisher_log_probability` ports
``CFishersNCHypergeometric::probability`` (BiasedUrn ``src/fnchyppr.cpp``) including
the truncation; ``control$dnchgcalc="dnoncenhypergeom"`` selects the untruncated
algorithm of Liao and Rosen. The integral is evaluated by QUADPACK's
``qagi`` through SciPy, the routine behind R's ``integrate()``.

References: Liao and Rosen (2001), doi:10.1198/000313001753272547 (the
algorithm of metafor's ``.dnoncenhypergeom``); Stijnen et al. (2010),
doi:10.1002/sim.4040 (the exact-likelihood random-effects model).
"""

import dataclasses
import math
import sys

import numpy
import numpy.typing
import scipy.integrate
import scipy.special

import metafopy.errors

__all__ = ['DNCHG_CALCS', 'IntegrationControl', 'Tables', 'fisher_log_probability', 'neg_loglik', 'study_loglik']

type FloatArray = numpy.typing.NDArray[numpy.float64]

SQRT_2PI = math.sqrt(2 * math.pi)
LOG_ODDS_BOUND = 12 * math.log(10)
DNCHG_CALCS = ('dFNCHypergeo', 'dnoncenhypergeom')
FAK_LEN = 1024
LN_SQRT_2PI = 0.918938533204672722


@dataclasses.dataclass(frozen=True, slots=True)
class IntegrationControl:
    """Arguments of R's ``integrate()`` (the ``intCtrl`` control list of ``rma.glmm``)."""

    rel_tol: float = sys.float_info.epsilon**0.25
    subdivisions: int = 100
    calc: str = 'dFNCHypergeo'
    precision: float = 1e-10


@dataclasses.dataclass(frozen=True, slots=True)
class Tables:
    """Precomputed support of the non-central hypergeometric distribution of each table.

    ``log_base[i]`` holds ``lchoose(n1, y) + lchoose(n2, m1 - y)`` over the support ``y = lo[i] .. hi[i]``
    and ``index[i]`` the position of the observed ``ai`` in it.
    """

    support: tuple[FloatArray, ...]
    log_base: tuple[FloatArray, ...]
    index: tuple[int, ...]
    margins: tuple[tuple[int, int, int], ...] = ()

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
        margins: list[tuple[int, int, int]] = []
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
            margins.append((int(m1), int(n1), int(n1 + n2)))
        return cls(tuple(support), tuple(base), tuple(index), tuple(margins))


def build_ln_fac_table() -> FloatArray:
    """``log(n!)`` for ``n < FAK_LEN`` as BiasedUrn's ``LnFac`` tabulates it (a running sum of logarithms)."""
    table = numpy.zeros(FAK_LEN)
    total = 0.0
    for n in range(1, FAK_LEN):
        total += math.log(n)
        table[n] = total
    return table


# auditme: accept[mutable-module-level-exported] read-only table of BiasedUrn's LnFac, built once at import
LN_FAC_TABLE = build_ln_fac_table()


def ln_fac(n: int) -> float:
    """BiasedUrn's ``LnFac``: the table below ``FAK_LEN`` and Stirling's series with the terms up to ``n^-3`` above."""
    if n < FAK_LEN:
        return float(LN_FAC_TABLE[n])
    n1 = float(n)
    r = 1.0 / n1
    return (n1 + 0.5) * math.log(n1) - n1 + LN_SQRT_2PI + r * (1.0 / 12.0 + r * r * (-1.0 / 360.0))


def full_log_factorials(draws: int, m: int, m2: int, x: int) -> float:
    """``log(x!) + log(x2!) + log((m - x)!) + log((m2 - x2)!)`` with ``x2 = draws - x``."""
    return ln_fac(x) + ln_fac(draws - x) + ln_fac(m - x) + ln_fac(m2 - (draws - x))


def step_log_factorials(draws: int, m: int, m2: int, xs: numpy.ndarray, direction: int) -> FloatArray:
    """BiasedUrn's increments of ``xFac`` when ``x`` moves by ``direction`` (+1 or -1) onto each element of ``xs``."""
    x2 = draws - xs
    if direction > 0:
        return numpy.asarray(numpy.log(xs * (m2 - x2) / ((x2 + 1.0) * (m - xs + 1))), dtype=numpy.float64)
    return numpy.asarray(numpy.log(x2 * (m - xs) / ((xs + 1.0) * (m2 - x2 + 1))), dtype=numpy.float64)


@dataclasses.dataclass(slots=True)
class Walk:
    """The state of ``CFishersNCHypergeometric::lng``: the last ``x`` it was called with and the ``xFac`` it holds."""

    last_x: int
    x_fac: float | None


def tail_sum(
    draws: int,
    m: int,
    m2: int,
    start: int,
    direction: int,
    limit: int,
    walk: Walk,
    m_fac: float,
    logodds: float,
    scale: float,
    accur: float,
) -> float:
    """Sum of ``exp(lng(x))`` from ``start`` towards ``limit`` up to and including the first term below ``accur``.

    ``walk`` carries ``xFac`` between calls and is advanced; ``walk.x_fac`` is ``None`` when ``lng`` would compute it afresh.
    """
    total = 0.0
    x = start
    while (x <= limit) if direction > 0 else (x >= limit):
        count = min(256, abs(limit - x) + 1)
        xs = x + direction * numpy.arange(count)
        inc = step_log_factorials(draws, m, m2, xs, direction)
        if walk.x_fac is None or abs(int(xs[0]) - walk.last_x) != 1:
            seq = numpy.concatenate(([full_log_factorials(draws, m, m2, int(xs[0]))], inc[1:]))
            facs = numpy.cumsum(seq)
        else:
            facs = numpy.cumsum(numpy.concatenate(([walk.x_fac], inc)))[1:]
        y = numpy.exp(m_fac - facs + xs * logodds - scale)
        stop = numpy.flatnonzero(y < accur)
        if stop.size:
            idx = int(stop[0])
            total += float(y[: idx + 1].sum())
            walk.last_x, walk.x_fac = int(xs[idx]), float(facs[idx])
            return total
        total += float(y.sum())
        walk.last_x, walk.x_fac = int(xs[-1]), float(facs[-1])
        x = int(xs[-1]) + direction
    return total


def fisher_log_probability(draws: int, m: int, total: int, x: int, odds: float, precision: float) -> float:
    """``log`` of ``BiasedUrn::dFNCHypergeo(x, m, total - m, draws, odds, precision)``.

    Port of ``CFishersNCHypergeometric::probability`` (https://github.com/cran/BiasedUrn/blob/master/src/fnchyppr.cpp): the
    proportional function is summed outwards from the (truncated) Cornfield mean until a term drops below ``0.1 * precision``,
    the reciprocal sum normalises ``x``, and the odds ``1`` case uses the exact hypergeometric formula. Above ``FAK_LEN`` the log
    factorials are updated from the previous term as BiasedUrn does, which avoids the cancellation of large log factorials.
    """
    xmin = max(0, m + draws - total)
    xmax = min(draws, m)
    if x < xmin or x > xmax:
        return -math.inf
    if draws == 0:
        return 0.0
    if odds == 1.0:
        return (
            ln_fac(m)
            - ln_fac(x)
            - ln_fac(m - x)
            + ln_fac(total - m)
            - ln_fac(draws - x)
            - ln_fac((total - m) - (draws - x))
            - (ln_fac(total) - ln_fac(draws) - ln_fac(total - draws))
        )
    accur = precision * 0.1
    a = (m + draws) * odds + (total - m - draws)
    b = a * a - 4.0 * odds * (odds - 1.0) * m * draws
    b = math.sqrt(b) if b > 0.0 else 0.0
    mean = (a - b) / (2.0 * (odds - 1.0))
    logodds = math.log(odds)
    m2 = total - m
    if m < FAK_LEN and m2 < FAK_LEN:
        return tabulated_log_probability(draws, m, total, x, logodds, int(mean), accur)
    m_fac = ln_fac(m) + ln_fac(m2)
    x1 = min(max(int(mean), xmin), xmax)
    fac1 = full_log_factorials(draws, m, m2, x1)
    scale = m_fac - fac1 + x1 * logodds
    walk = Walk(x1, fac1)
    rsum = 1.0 + tail_sum(draws, m, m2, x1 - 1, -1, xmin, walk, m_fac, logodds, scale, accur)
    rsum += tail_sum(draws, m, m2, x1 + 1, 1, xmax, walk, m_fac, logodds, scale, accur)
    step = x - walk.last_x
    if step == 0 and walk.x_fac is not None:
        fac = walk.x_fac
    elif abs(step) == 1 and walk.x_fac is not None:
        fac = walk.x_fac + float(step_log_factorials(draws, m, m2, numpy.array([x]), step)[0])
    else:
        fac = full_log_factorials(draws, m, m2, x)
    return m_fac - fac + x * logodds - scale - math.log(rsum)


def tabulated_log_probability(draws: int, m: int, total: int, x: int, logodds: float, mean: int, accur: float) -> float:
    """The part of :func:`fisher_log_probability` for ``m`` and ``total - m`` below ``FAK_LEN``, where BiasedUrn takes every log factorial from its table."""
    xmin = max(0, m + draws - total)
    xmax = min(draws, m)
    xs = numpy.arange(xmin, xmax + 1)
    lf = LN_FAC_TABLE[xs] + LN_FAC_TABLE[draws - xs] + LN_FAC_TABLE[m - xs] + LN_FAC_TABLE[(total - m) - (draws - xs)]
    lng = -lf + xs * logodds
    start = min(max(mean, xmin), xmax) - xmin
    rel = lng - lng[start]
    y = numpy.exp(rel)
    rsum = 1.0
    for tail in (y[:start][::-1], y[start + 1 :]):
        if tail.size:
            stop = numpy.flatnonzero(tail < accur)
            rsum += float(tail[: int(stop[0]) + 1].sum()) if stop.size else float(tail.sum())
    return float(rel[x - xmin] - math.log(rsum))


def log_probability(tables: Tables, i: int, log_or: float, control: IntegrationControl | None = None) -> float:
    """``log P(ai | margins, odds ratio = exp(log_or))`` for table ``i``, with ``log_or`` clamped to +-log(1e12) as metafor does.

    With ``control.calc == 'dFNCHypergeo'`` (metafor's default) the probability is that of BiasedUrn, truncated tails included;
    ``'dnoncenhypergeom'`` normalises over the whole support.
    """
    lo = min(max(log_or, -LOG_ODDS_BOUND), LOG_ODDS_BOUND)
    if control is not None and control.calc == 'dFNCHypergeo' and tables.margins:
        draws, m, total = tables.margins[i]
        x = int(tables.support[i][tables.index[i]])
        return fisher_log_probability(draws, m, total, x, math.exp(lo), control.precision)
    w = tables.log_base[i] + tables.support[i] * lo
    top = float(w.max())
    return float(w[tables.index[i]] - top - math.log(float(numpy.exp(w - top).sum())))


def study_loglik(tables: Tables, i: int, mu: float, tau2: float, random: bool, control: IntegrationControl) -> float:
    """Log-likelihood contribution of table ``i`` for the fixed (``random=False``) or normal-mixed log odds ratio."""
    if not random:
        return log_probability(tables, i, mu, control)
    sd = math.sqrt(tau2)

    def integrand(t: float) -> float:
        lo = min(max(t, -LOG_ODDS_BOUND), LOG_ODDS_BOUND)
        return math.exp(log_probability(tables, i, lo, control)) * math.exp(-0.5 * ((lo - mu) / sd) ** 2) / (sd * SQRT_2PI)

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
