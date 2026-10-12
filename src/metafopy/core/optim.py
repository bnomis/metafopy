"""R's one-dimensional minimiser, Nelder-Mead and numerical Hessian, ported line for line.

metafor computes the tetrachoric correlation (``escalc(measure="RTET")``) with
``optimize()`` followed by ``optim(method="Nelder-Mead", hessian=TRUE)``. The
estimate and its variance depend on where those routines stop, so they are
ported exactly rather than replaced by SciPy's optimisers.

Sources (R 4.3.3, GPL-2 | GPL-3):

- ``src/library/stats/src/optimize.c``: ``Brent_fmin``, after Brent (1973),
  *Algorithms for Minimization without Derivatives*, ch. 5.
- ``src/appl/optim.c``: ``nmmin``, after Nash (1990), *Compact Numerical
  Methods for Computers*, 2nd ed., Algorithm 19.
- ``src/library/stats/src/optim.c``: ``optimhess`` and ``fmingr`` (central
  differences with step ``ndeps``).
"""

from __future__ import annotations

import math
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ['NelderMeadResult', 'fmin', 'nelder_mead', 'optimhess']

EPSILON = sys.float_info.epsilon
DBL_MAX = sys.float_info.max
_BIG = 1.0e35

type FloatArray = NDArray[np.float64]


def fmin(f: Callable[[float], float], ax: float, bx: float, tol: float = EPSILON**0.25) -> float:
    """Minimise ``f`` on ``[ax, bx]`` as R's ``optimize()`` does; returns the minimiser.

    Non-finite function values are replaced by ``DBL_MAX``, as R's ``fcn1`` wrapper does.
    """

    def g(x: float) -> float:
        val = f(x)
        return val if math.isfinite(val) else DBL_MAX

    c = (3.0 - math.sqrt(5.0)) * 0.5
    eps = math.sqrt(EPSILON)

    a = ax
    b = bx
    v = a + c * (b - a)
    w = v
    x = v

    d = 0.0
    e = 0.0
    fx = g(x)
    fv = fx
    fw = fx
    tol3 = tol / 3.0

    while True:
        xm = (a + b) * 0.5
        tol1 = eps * abs(x) + tol3
        t2 = tol1 * 2.0

        if abs(x - xm) <= t2 - (b - a) * 0.5:
            break
        p = 0.0
        q = 0.0
        r = 0.0
        if abs(e) > tol1:
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = (q - r) * 2.0
            if q > 0.0:
                p = -p
            else:
                q = -q
            r = e
            e = d

        if abs(p) >= abs(q * 0.5 * r) or p <= q * (a - x) or p >= q * (b - x):
            e = b - x if x < xm else a - x
            d = c * e
        else:
            d = p / q
            u = x + d
            if u - a < t2 or b - u < t2:
                d = tol1
                if x >= xm:
                    d = -d

        if abs(d) >= tol1:
            u = x + d
        elif d > 0.0:
            u = x + tol1
        else:
            u = x - tol1

        fu = g(u)

        if fu <= fx:
            if u < x:
                b = x
            else:
                a = x
            v = w
            w = x
            x = u
            fv = fw
            fw = fx
            fx = fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v = w
                fv = fw
                w = u
                fw = fu
            elif fu <= fv or v in (x, w):
                v = u
                fv = fu

    return x


@dataclass(frozen=True, slots=True)
class NelderMeadResult:
    par: FloatArray
    value: float
    fncount: int
    fail: int


def nelder_mead(
    fn: Callable[[FloatArray], float],
    par: Sequence[float] | FloatArray,
    *,
    abstol: float = -math.inf,
    reltol: float = math.sqrt(EPSILON),
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 2.0,
    maxit: int = 500,
) -> NelderMeadResult:
    """R's ``optim(method="Nelder-Mead")`` with its default control settings."""
    bvec = np.array(par, dtype=np.float64)
    n = bvec.shape[0]

    def f_at(x: FloatArray) -> float:
        return float(fn(x.copy()))

    if maxit <= 0:
        return NelderMeadResult(bvec, f_at(bvec), 0, 0)

    # P has n + 1 rows (coordinates, then function value) and n + 2 columns
    # (vertices, then the centroid), indexed as in the C source.
    n1 = n + 1
    C = n + 2
    P = np.zeros((n1, n + 2))
    fail = 0
    f = f_at(bvec)
    if not math.isfinite(f):
        raise ValueError('function cannot be evaluated at initial parameters')
    funcount = 1
    convtol = reltol * (abs(f) + reltol)
    P[n1 - 1, 0] = f
    P[:n, 0] = bvec

    L = 1
    size = 0.0
    step = 0.0
    for i in range(n):
        if 0.1 * abs(bvec[i]) > step:
            step = 0.1 * abs(bvec[i])
    if step == 0.0:
        step = 0.1
    for j in range(2, n1 + 1):
        P[:n, j - 1] = bvec
        trystep = step
        while P[j - 2, j - 1] == bvec[j - 2]:
            P[j - 2, j - 1] = bvec[j - 2] + trystep
            trystep *= 10
        size += trystep
    oldsize = size
    calcvert = True

    while True:
        if calcvert:
            for j in range(n1):
                if j + 1 != L:
                    bvec = P[:n, j].copy()
                    f = f_at(bvec)
                    if not math.isfinite(f):
                        f = _BIG
                    funcount += 1
                    P[n1 - 1, j] = f
            calcvert = False

        VL = P[n1 - 1, L - 1]
        VH = VL
        H = L

        for j in range(1, n1 + 1):
            if j != L:
                f = P[n1 - 1, j - 1]
                if f < VL:
                    L = j
                    VL = f
                if f > VH:
                    H = j
                    VH = f

        if VL + convtol >= VH or abstol >= VL:
            break

        for i in range(n):
            temp = -P[i, H - 1]
            for j in range(n1):
                temp += P[i, j]
            P[i, C - 1] = temp / n
        for i in range(n):
            bvec[i] = (1.0 + alpha) * P[i, C - 1] - alpha * P[i, H - 1]
        f = f_at(bvec)
        if not math.isfinite(f):
            f = _BIG
        funcount += 1
        VR = f
        if VR < VL:
            P[n1 - 1, C - 1] = f
            for i in range(n):
                f = gamma * bvec[i] + (1 - gamma) * P[i, C - 1]
                P[i, C - 1] = bvec[i]
                bvec[i] = f
            f = f_at(bvec)
            if not math.isfinite(f):
                f = _BIG
            funcount += 1
            if f < VR:
                P[:n, H - 1] = bvec
                P[n1 - 1, H - 1] = f
            else:
                P[:n, H - 1] = P[:n, C - 1]
                P[n1 - 1, H - 1] = VR
        else:
            if VR < VH:
                P[:n, H - 1] = bvec
                P[n1 - 1, H - 1] = VR

            for i in range(n):
                bvec[i] = (1 - beta) * P[i, H - 1] + beta * P[i, C - 1]
            f = f_at(bvec)
            if not math.isfinite(f):
                f = _BIG
            funcount += 1

            if f < P[n1 - 1, H - 1]:
                P[:n, H - 1] = bvec
                P[n1 - 1, H - 1] = f
            elif VR >= VH:
                calcvert = True
                size = 0.0
                for j in range(n1):
                    if j + 1 != L:
                        for i in range(n):
                            P[i, j] = beta * (P[i, j] - P[i, L - 1]) + P[i, L - 1]
                            size += abs(P[i, j] - P[i, L - 1])
                if size < oldsize:
                    oldsize = size
                else:
                    fail = 10
                    break

        if funcount > maxit:
            break

    if funcount > maxit:
        fail = 1
    return NelderMeadResult(P[:n, L - 1].copy(), float(P[n1 - 1, L - 1]), funcount, fail)


def optimhess(fn: Callable[[FloatArray], float], par: Sequence[float] | FloatArray, ndeps: float = 1e-3) -> FloatArray:
    """Numerical Hessian as computed by R's ``optim(..., hessian=TRUE)`` without a gradient."""
    p = np.array(par, dtype=np.float64)
    n = p.shape[0]

    def grad(x: FloatArray) -> FloatArray:
        df = np.empty(n)
        for i in range(n):
            xp = x.copy()
            xp[i] = x[i] + ndeps
            val1 = float(fn(xp))
            xm = x.copy()
            xm[i] = x[i] - ndeps
            val2 = float(fn(xm))
            df[i] = (val1 - val2) / (2 * ndeps)
            if not math.isfinite(df[i]):
                raise ValueError(f'non-finite finite-difference value [{i + 1}]')
        return df

    ans = np.empty((n, n))
    dpar = p.copy()
    for i in range(n):
        dpar[i] = dpar[i] + ndeps
        df1 = grad(dpar)
        dpar[i] = dpar[i] - 2 * ndeps
        df2 = grad(dpar)
        ans[i, :] = (df1 - df2) / (2 * ndeps)
        dpar[i] = dpar[i] + ndeps
    for i in range(n):
        for j in range(i):
            tmp = 0.5 * (ans[i, j] + ans[j, i])
            ans[i, j] = ans[j, i] = tmp
    return ans
