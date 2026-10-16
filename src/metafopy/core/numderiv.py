"""Numerical derivatives as R's numDeriv and lme4 compute them.

``rma.glmm`` obtains the covariance matrix of the conditional-model estimates
from ``numDeriv::hessian()`` (Richardson extrapolation of central differences,
https://cran.r-project.org/package=numDeriv, Gilbert and Varadhan) and lme4's
``vcov()`` of a ``glmer`` fit from the central-difference Hessian
``lme4:::deriv12()`` (https://github.com/lme4/lme4/blob/master/R/deriv.R).
Both are ported step by step so that they inherit the same truncation errors.
``conv.delta()`` differentiates a transformation with ``numDeriv::grad()``
(:func:`grad_scalar`, the same Richardson scheme for first derivatives).

The other ``hesspack`` choices of metafor's ``control`` lists are ported as well:
``pracma::hessian`` (:func:`hessian_pracma`, https://cran.r-project.org/package=pracma) and
``calculus::hessian`` (:func:`hessian_calculus`, https://cran.r-project.org/package=calculus);
:func:`hessian_by_package` dispatches on the name of the package.
"""

import collections.abc
import math
import sys
import typing

import numpy
import numpy.typing

import metafopy.errors

__all__ = ['HESSPACKS', 'deriv12', 'grad_scalar', 'hessian', 'hessian_by_package', 'hessian_calculus', 'hessian_pracma']

HESSPACKS = ('calculus', 'numDeriv', 'pracma')

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


def hessian(
    func: Objective,
    x: FloatArray,
    *,
    d: float = 0.1,
    eps: float = 1e-4,
    r: int = 4,
    v: float = 2.0,
    zero_tol: float = ZERO_TOL,
) -> FloatArray:
    """``numDeriv::hessian(func, x, method.args=list(d=d, eps=eps, r=r, v=v, zero.tol=zero_tol))`` for the Richardson method."""
    n = x.shape[0]
    f0 = func(x)
    h0 = numpy.abs(d * x) + eps * (numpy.abs(x) < zero_tol)
    hdiag = numpy.zeros(n)
    unit = numpy.eye(n)
    for i in range(n):
        h = h0.copy()
        haprox = numpy.zeros(r)
        for k in range(r):
            f1 = func(x + unit[i] * h)
            f2 = func(x - unit[i] * h)
            haprox[k] = (f1 - 2 * f0 + f2) / h[i] ** 2
            h = h / v
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
                h = h / v
            H[i, j] = H[j, i] = extrapolate(daprox, r)
    return H


def hessian_pracma(func: Objective, x: FloatArray, *, h: float = sys.float_info.epsilon**0.25) -> FloatArray:
    """``pracma::hessian(func, x, h)``: central differences with the step ``h`` for all parameters.

    Source: https://github.com/cran/pracma/blob/master/R/hessian.R (a single parameter uses ``fderiv(f, x, n=2, h=h)``).
    """
    n = x.shape[0]
    if n == 1:
        return numpy.array([[(func(x + h) - 2 * func(x) + func(x - h)) / h**2]])
    out = numpy.full((n, n), numpy.nan)
    hh = numpy.eye(n) * h
    f0 = func(x)
    for i in range(n - 1):
        hi = hh[i]
        out[i, i] = (func(x - hi) - 2 * f0 + func(x + hi)) / h**2
        for j in range(i + 1, n):
            hj = hh[j]
            out[i, j] = out[j, i] = (func(x + hi + hj) - func(x + hi - hj) - func(x - hi + hj) + func(x - hi - hj)) / (4 * h**2)
    hi = hh[n - 1]
    out[n - 1, n - 1] = (func(x - hi) - 2 * f0 + func(x + hi)) / h**2
    return out


def stencil_weights(points: numpy.ndarray, order: int) -> FloatArray:
    """Weights ``w`` with ``sum_j w_j points_j^k = delta(k, order)`` (``calculus``'s ``C.i()``)."""
    count = points.shape[0]
    vander = points[None, :] ** numpy.arange(count)[:, None]
    rhs = numpy.zeros(count)
    rhs[order] = 1.0
    return numpy.asarray(numpy.linalg.solve(vander, rhs), dtype=numpy.float64)


def hessian_calculus(func: Objective, x: FloatArray, *, accuracy: int = 4, stepsize: float | None = None) -> FloatArray:
    """``calculus::hessian(func, x, accuracy=accuracy, stepsize=stepsize)``: finite differences with a stencil of ``accuracy + 1`` points.

    Source: ``hessian()`` and ``D.num()`` in https://github.com/cran/calculus/blob/master/R (``operators.R``, ``derivative.R``). The
    second derivatives use the one-dimensional stencil of order 2, the mixed ones the tensor product of two stencils of order 1.
    """
    n = x.shape[0]
    eps = sys.float_info.epsilon
    if stepsize is None:
        base = (eps ** (1 / 3)) ** (1 / 2)
        h_second_vec = h_cross_vec = numpy.abs(base * x) + base * (numpy.abs(x) < eps ** (2 / 3))
    else:
        h_second_vec = h_cross_vec = numpy.full(n, float(stepsize))
    imax2 = (2 + accuracy - 1) // 2
    pts2 = numpy.arange(-imax2, imax2 + 1, dtype=numpy.float64)
    w2 = stencil_weights(pts2, 2)
    out = numpy.zeros((n, n))
    for i in range(n):
        total = 0.0
        for off, w in zip(pts2, w2, strict=True):
            z = x.copy()
            z[i] = z[i] + off * h_second_vec[i]
            total += w * func(z)
        out[i, i] = 2.0 * total / h_second_vec[i] ** 2
    if n > 1:
        imax1 = (1 + accuracy - 1) // 2
        pts1 = numpy.arange(-imax1, imax1 + 1, dtype=numpy.float64)
        w1 = stencil_weights(pts1, 1)
        for j in range(n - 1):
            for i in range(j + 1, n):
                total = 0.0
                for oi, wi in zip(pts1, w1, strict=True):
                    for oj, wj in zip(pts1, w1, strict=True):
                        z = x.copy()
                        z[i] = z[i] + oi * h_cross_vec[i]
                        z[j] = z[j] + oj * h_cross_vec[j]
                        total += wi * wj * func(z)
                out[i, j] = out[j, i] = total / (h_cross_vec[i] * h_cross_vec[j])
    return out


def hessian_by_package(func: Objective, x: FloatArray, hesspack: str, ctrl: collections.abc.Mapping[str, typing.Any]) -> FloatArray:
    """The Hessian as metafor's ``hesspack`` and ``hessianCtrl`` control entries ask for it.

    ``numDeriv`` reads ``d``, ``eps``, ``r``, ``v`` and ``zero.tol`` from ``ctrl`` (``zero_tol`` is accepted too), ``pracma`` reads ``h`` and
    ``calculus`` reads ``accuracy`` and ``stepsize``; entries a package does not use are ignored, as R passes them on silently.

    Raises:
        metafopy.errors.InputError: ``hesspack`` is not one of :data:`HESSPACKS`.
    """
    if hesspack == 'numDeriv':
        return hessian(
            func,
            x,
            d=float(ctrl.get('d', 0.1)),
            eps=float(ctrl.get('eps', 1e-4)),
            r=int(ctrl.get('r', 4)),
            v=float(ctrl.get('v', 2.0)),
            zero_tol=float(ctrl.get('zero.tol', ctrl.get('zero_tol', ZERO_TOL))),
        )
    if hesspack == 'pracma':
        return hessian_pracma(func, x, h=float(ctrl.get('h', sys.float_info.epsilon**0.25)))
    if hesspack == 'calculus':
        step = ctrl.get('stepsize')
        return hessian_calculus(func, x, accuracy=int(ctrl.get('accuracy', 4)), stepsize=None if step is None else float(step))
    raise metafopy.errors.InputError(f"'hesspack' should be one of {', '.join(repr(h) for h in HESSPACKS)}")


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
