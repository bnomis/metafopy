"""Numerical derivatives as R's numDeriv and lme4 compute them.

``rma.glmm`` obtains the covariance matrix of the conditional-model estimates
from ``numDeriv::hessian()`` (Richardson extrapolation of central differences,
https://cran.r-project.org/package=numDeriv, Gilbert and Varadhan) and lme4's
``vcov()`` of a ``glmer`` fit from the central-difference Hessian
``lme4:::deriv12()`` (https://github.com/lme4/lme4/blob/master/R/deriv.R).
Both are ported step by step so that they inherit the same truncation errors.
``conv.delta()`` differentiates a transformation with ``numDeriv::grad()``
(:func:`grad_scalar`, the same Richardson scheme for first derivatives).
"""

import collections.abc
import math
import sys

import numpy
import numpy.typing

__all__ = ['deriv12', 'grad_scalar', 'hessian']

type FloatArray = numpy.typing.NDArray[numpy.float64]
type Objective = collections.abc.Callable[[FloatArray], float]

ZERO_TOL = math.sqrt(sys.float_info.epsilon / 7e-7)


def extrapolate(approx: FloatArray, r: int) -> float:
    """Richardson extrapolation of the sequence ``approx`` (step halved each time)."""
    work = approx.copy()
    for m in range(1, r):
        for k in range(r - m):
            work[k] = (work[k + 1] * 4.0**m - work[k]) / (4.0**m - 1.0)
    return float(work[0])


def grad_scalar(func: collections.abc.Callable[[float], float], x: float, *, d: float = 1e-4, eps: float = 1e-4, r: int = 4, v: float = 2.0) -> float:
    """``numDeriv::grad(func, x)`` for a scalar ``x`` (default Richardson method).

    Raises:
        ArithmeticError: the function is not finite at one of the evaluation points (numDeriv stops with
            "function returns NA at ... distance from x").
    """
    h = abs(d * x) + eps * (abs(x) < ZERO_TOL)
    approx = numpy.zeros(r)
    for k in range(r):
        approx[k] = (func(x + h) - func(x - h)) / (2 * h)
        if math.isnan(approx[k]):
            raise ArithmeticError(f'function returns NA at {h} distance from x.')
        h = h / v
    return extrapolate(approx, r)


def hessian(func: Objective, x: FloatArray, *, d: float = 0.1, eps: float = 1e-4, r: int = 4) -> FloatArray:
    """``numDeriv::hessian(func, x, method.args=list(d=d, eps=eps, r=r))`` for the default Richardson method (``v = 2``)."""
    n = x.shape[0]
    f0 = func(x)
    h0 = numpy.abs(d * x) + eps * (numpy.abs(x) < ZERO_TOL)
    hdiag = numpy.zeros(n)
    unit = numpy.eye(n)
    for i in range(n):
        h = h0.copy()
        haprox = numpy.zeros(r)
        for k in range(r):
            f1 = func(x + unit[i] * h)
            f2 = func(x - unit[i] * h)
            haprox[k] = (f1 - 2 * f0 + f2) / h[i] ** 2
            h = h / 2.0
        hdiag[i] = extrapolate(haprox, r)
    H = numpy.zeros((n, n))
    for i in range(n):
        H[i, i] = hdiag[i]
        for j in range(i):
            h = h0.copy()
            daprox = numpy.zeros(r)
            for k in range(r):
                shift = unit[i] * h + unit[j] * h
                f1 = func(x + shift)
                f2 = func(x - shift)
                daprox[k] = (f1 - 2 * f0 + f2 - hdiag[i] * h[i] ** 2 - hdiag[j] * h[j] ** 2) / (2 * h[i] * h[j])
                h = h / 2.0
            H[i, j] = H[j, i] = extrapolate(daprox, r)
    return H


def deriv12(
    func: Objective,
    x: FloatArray,
    *,
    delta: float = 1e-4,
    lower: FloatArray | None = None,
) -> FloatArray:
    """The Hessian ``lme4:::deriv12(func, x, delta, lower=lower)`` computes by central differences.

    Parameters with a finite ``lower`` bound closer than ``delta`` use the
    distance to the bound as the lower step, as lme4 does; a parameter on its
    bound yields non-finite entries (lme4 then falls back to another estimate).
    """
    nx = x.shape[0]
    lo = numpy.full(nx, numpy.nan) if lower is None else lower
    fx = func(x)
    xadd = x + delta
    xsub = x - delta
    ldelta = numpy.full(nx, delta)
    active = ~numpy.isnan(lo) & (xsub < lo)
    ldelta[active] = x[active] - lo[active]
    xsub[active] = lo[active]
    udelta = numpy.full(nx, delta)

    def shifted(shift: dict[int, float]) -> FloatArray:
        z = x.copy()
        for pos, val in shift.items():
            z[pos] = val
        return z

    H = numpy.zeros((nx, nx))
    with numpy.errstate(divide='ignore', invalid='ignore'):
        for j in range(nx):
            fadd = func(shifted({j: xadd[j]}))
            fsub = func(shifted({j: xsub[j]}))
            H[j, j] = fadd / udelta[j] ** 2 - 2 * fx / (udelta[j] * ldelta[j]) + fsub / ldelta[j] ** 2
            for i in range(j):
                faa = func(shifted({i: xadd[i], j: xadd[j]}))
                fas = func(shifted({i: xadd[i], j: xsub[j]}))
                fsa = func(shifted({i: xsub[i], j: xadd[j]}))
                fss = func(shifted({i: xsub[i], j: xsub[j]}))
                H[i, j] = H[j, i] = (
                    faa / (udelta[i] + udelta[j]) ** 2
                    - fas / (udelta[i] + ldelta[j]) ** 2
                    - fsa / (ldelta[i] + udelta[j]) ** 2
                    + fss / (ldelta[i] + ldelta[j]) ** 2
                )
    return H
