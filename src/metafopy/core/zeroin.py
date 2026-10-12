"""Brent's root finder, ported line for line from R's ``R_zeroin2``.

metafor relies on ``uniroot()`` (default ``tol = .Machine$double.eps^0.25``)
for the Paule-Mandel estimators and Q-profile confidence intervals, so the
values it reports depend on the exact iterates of R's implementation. Using
SciPy's ``brentq`` would converge to a slightly different point inside the
same tolerance; porting R's routine reproduces metafor's numbers exactly.

Source: R ``src/library/stats/src/zeroin.c`` (GPL-2 | GPL-3), itself based on
Brent (1973), *Algorithms for Minimization without Derivatives*, ch. 4.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable
from dataclasses import dataclass

__all__ = ["RootResult", "uniroot", "zeroin2"]

EPSILON = sys.float_info.epsilon
DEFAULT_TOL = EPSILON**0.25


@dataclass(frozen=True, slots=True)
class RootResult:
    root: float
    f_root: float
    iterations: int
    estim_prec: float
    converged: bool


def zeroin2(
    f: Callable[[float], float],
    ax: float,
    bx: float,
    fa: float,
    fb: float,
    tol: float,
    maxit: int,
) -> tuple[float, int, float]:
    """Direct port of ``R_zeroin2``; returns ``(root, iterations, estim_prec)``.

    ``iterations`` is ``-1`` when the iteration limit was reached.
    """
    a, b = ax, bx
    c, fc = a, fa
    remaining = maxit + 1

    if fa == 0.0:
        return a, 0, 0.0
    if fb == 0.0:
        return b, 0, 0.0

    while remaining:
        remaining -= 1
        prev_step = b - a

        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb

        tol_act = 2 * EPSILON * abs(b) + tol / 2
        new_step = (c - b) / 2

        if abs(new_step) <= tol_act or fb == 0.0:
            return b, maxit + 1 - remaining - 1, abs(c - b)

        if abs(prev_step) >= tol_act and abs(fa) > abs(fb):
            cb = c - b
            if a == c:
                t1 = fb / fa
                p = cb * t1
                q = 1.0 - t1
            else:
                q = fa / fc
                t1 = fb / fc
                t2 = fb / fa
                p = t2 * (cb * q * (q - t1) - (b - a) * (t1 - 1.0))
                q = (q - 1.0) * (t1 - 1.0) * (t2 - 1.0)
            if p > 0:
                q = -q
            else:
                p = -p

            if p < (0.75 * cb * q - abs(tol_act * q) / 2) and p < abs(prev_step * q / 2):
                new_step = p / q

        if abs(new_step) < tol_act:
            new_step = tol_act if new_step > 0 else -tol_act

        a, fa = b, fb
        b += new_step
        fb = f(b)
        if (fb > 0 and fc > 0) or (fb < 0 and fc < 0):
            c, fc = a, fa

    return b, -1, abs(c - b)


def uniroot(
    f: Callable[[float], float],
    lower: float,
    upper: float,
    *,
    tol: float = DEFAULT_TOL,
    maxiter: int = 1000,
) -> RootResult:
    """Find a root of ``f`` in ``[lower, upper]`` as R's ``uniroot()`` does.

    Raises ``ValueError`` when ``f(lower)`` and ``f(upper)`` have the same sign
    (R: "f() values at end points not of opposite sign").
    """
    if not lower < upper:
        raise ValueError("lower < upper is not fulfilled")
    f_lower = f(lower)
    f_upper = f(upper)
    if math.isnan(f_lower) or math.isnan(f_upper):
        raise ValueError("f() value at an end point is NA")
    if f_lower * f_upper > 0:
        raise ValueError("f() values at end points not of opposite sign")
    root, iterations, prec = zeroin2(f, lower, upper, f_lower, f_upper, tol, maxiter)
    return RootResult(
        root=root,
        f_root=f(root),
        iterations=iterations,
        estim_prec=prec,
        converged=iterations >= 0,
    )
