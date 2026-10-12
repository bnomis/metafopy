"""Distribution of a positive quadratic form in normal variables (Farebrother's AS 204).

metafor's generalised Q-statistic estimator (``method="GENQM"``) and CIs
(``confint(type="GENQ")``) evaluate ``P(Q <= q)`` for ``Q = sum_i lambda_i
chi^2_{h_i}(delta_i)`` with ``CompQuadForm::farebrother()``. This is a line for
line port of that package's ``ruben.cpp`` (CompQuadForm 1.4.4, GPL >= 2), so
that roots found by ``uniroot()`` agree with R.

References
----------
Farebrother, R. W. (1984). Algorithm AS 204: The distribution of a positive
linear combination of chi-squared random variables. *Applied Statistics*,
33(3), 332-339. doi:10.2307/2347721

Sheil, J. & O'Muircheartaigh, I. (1977). Algorithm AS 106: The distribution of
non-negative quadratic forms in normal variables. *Applied Statistics*, 26(1),
92-98. doi:10.2307/2346884

Duchesne, P. & Lafaye de Micheaux, P. (2010). Computing the distribution of
quadratic forms. *Computational Statistics & Data Analysis*, 54(4), 858-862.
doi:10.1016/j.csda.2009.11.025
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from scipy.special import ndtr

__all__ = ['FarebrotherResult', 'farebrother']


@dataclass(frozen=True, slots=True)
class FarebrotherResult:
    """``Qq`` is ``P(Q > q)``, as ``CompQuadForm::farebrother()`` returns it."""

    dnsty: float
    ifault: int
    Qq: float


def farebrother(
    q: float,
    lam: Sequence[float],
    h: Sequence[int] | None = None,
    delta: Sequence[float] | None = None,
    maxit: int = 100000,
    eps: float = 1e-10,
    mode: float = 1.0,
) -> FarebrotherResult:
    """Upper tail probability of ``sum_i lam[i] * chi^2_{h[i]}(delta[i])`` at ``q``."""
    n = len(lam)
    mult = [1] * n if h is None else [int(v) for v in h]
    dlt = [0.0] * n if delta is None else [float(v) for v in delta]
    if any(d < 0 for d in dlt):
        raise ValueError("All non centrality parameters in 'delta' should be positive!")
    if len(mult) != n or len(dlt) != n:
        raise ValueError('lambda, h and delta should have the same length!')
    dnsty, ifault, res = _ruben([float(v) for v in lam], mult, dlt, n, float(q), float(mode), int(maxit), float(eps))
    return FarebrotherResult(dnsty=dnsty, ifault=ifault, Qq=1.0 - res)


def _ruben(lam: list[float], mult: list[int], delta: list[float], n: int, c: float, mode: float, maxit: int, eps: float) -> tuple[float, int, float]:
    """Returns ``(dnsty, ifault, res)`` with ``res = P(Q < c)``."""
    dnsty = 0.0
    if n < 1 or c <= 0 or maxit < 1 or eps <= 0.0:
        return dnsty, 2, -2.0

    tol = -200.0
    total = lam[0]
    bbeta = total
    for i in range(1, n + 1):
        hold = lam[i - 1]
        if hold <= 0.0 or mult[i - 1] < 1 or delta[i - 1] < 0.0:
            return dnsty, -i, -7.0
        if bbeta > hold:
            bbeta = hold
        if total < hold:
            total = hold

    if mode > 0.0:
        bbeta = mode * bbeta
    else:
        bbeta = 2.0 / (1.0 / bbeta + 1.0 / total)

    gamma = [0.0] * n
    theta = [0.0] * n
    a = [0.0] * maxit
    b = [0.0] * maxit

    k = 0
    total = 1.0
    sum1 = 0.0
    for i in range(1, n + 1):
        hold = bbeta / lam[i - 1]
        gamma[i - 1] = 1.0 - hold
        total = total * hold ** mult[i - 1]
        sum1 = sum1 + delta[i - 1]
        k = k + mult[i - 1]
        theta[i - 1] = 1.0

    ao = math.exp(0.5 * (_log(total) - sum1))
    if ao <= 0.0:
        return 0.0, 1, 0.0

    # probability and density of chi-squared on k df; 0.22579135264473 = ln(sqrt(pi / 2))
    z = c / bbeta
    if k % 2 == 0:
        i = 2
        lans = -0.5 * z
        dans = math.exp(lans)
        pans = 1.0 - dans
    else:
        i = 1
        lans = -0.5 * (z + math.log(z)) - 0.22579135264473
        dans = math.exp(lans)
        pans = float(ndtr(math.sqrt(z))) - float(ndtr(-math.sqrt(z)))

    k = k - 2
    for j in range(i, k + 1, 2):
        if lans < tol:
            lans = lans + math.log(z / j)
            dans = math.exp(lans)
        else:
            dans = dans * z / j
        pans = pans - dans

    # successive terms of the expansion
    prbty = pans
    dnsty = dans
    eps2 = eps / ao
    aoinv = 1.0 / ao
    total = aoinv - 1.0

    ifault = 0
    m = 1
    while m <= maxit:
        sum1 = 0.0
        for i in range(1, n + 1):
            hold = theta[i - 1]
            hold2 = hold * gamma[i - 1]
            theta[i - 1] = hold2
            sum1 = sum1 + hold2 * mult[i - 1] + m * delta[i - 1] * (hold - hold2)
        sum1 = 0.5 * sum1
        b[m - 1] = sum1
        for i in range(m - 1, 0, -1):
            sum1 = sum1 + b[i - 1] * a[m - i - 1]
        sum1 = sum1 / m
        a[m - 1] = sum1
        k = k + 2
        if lans < tol:
            lans = lans + math.log(z / k)
            dans = math.exp(lans)
        else:
            dans = dans * z / k
        pans = pans - dans
        total = total - sum1
        dnsty = dnsty + dans * sum1
        sum1 = pans * sum1
        prbty = prbty + sum1
        if prbty < -aoinv:
            return dnsty, 3, -3.0
        if abs(pans * total) < eps2 and abs(sum1) < eps2:
            ifault = 0
            break
        m += 1

    if m >= maxit:
        ifault = 4
    dnsty = ao * dnsty / (bbeta + bbeta)
    prbty = ao * prbty
    if prbty < 0.0 or prbty > 1.0:
        ifault = ifault + 5
    elif dnsty < 0.0:
        ifault = ifault + 6
    return dnsty, ifault, prbty


def _log(x: float) -> float:
    return math.log(x) if x > 0 else -math.inf
