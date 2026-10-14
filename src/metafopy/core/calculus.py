"""Finite-difference derivatives as R's calculus package computes them for a function.

metafor's ``deltamethod()`` obtains the Jacobian, the Hessians and the third-derivative tensor of the
user's function with ``calculus::derivative()`` and ``calculus::hessian()``. For an R function both
call the internal ``D.num()``: central differences on an equally spaced grid of ``accuracy``-order
(default 4) with the weights from the Vandermonde system, and the step ``h = eps^(1/3)^(1/order)``
scaled by ``|x|`` (``h`` itself where ``|x| < eps^(2/3)``). The routine is ported step by step so
that it is evaluated at the same points (https://cran.r-project.org/package=calculus, Guidotti
2022, doi:10.18637/jss.v104.i05).
"""

import collections.abc
import itertools
import math
import sys

import numpy
import numpy.typing

__all__ = ['derivative_each', 'derivative_mixed', 'hessian']

type FloatArray = numpy.typing.NDArray[numpy.float64]
type VectorFunction = collections.abc.Callable[[FloatArray], FloatArray]

EPS = sys.float_info.epsilon


def steps(x0: FloatArray, total_order: int) -> FloatArray:
    h = (EPS ** (1 / 3)) ** (1 / total_order)
    out: FloatArray = numpy.abs(h * x0) + h * (numpy.abs(x0) < EPS ** (2 / 3))
    return out


def points(order: int, accuracy: int) -> numpy.typing.NDArray[numpy.intp]:
    i_max = int((order + accuracy - 1) / 2)
    return numpy.arange(-i_max, i_max + 1)


def weights(grid: numpy.typing.NDArray[numpy.intp], order: int) -> FloatArray:
    """The finite-difference weights of ``D.num``'s ``C.i``: ``solve(t(i^(0:(n-1))), e_{order+1})``."""
    n = grid.shape[0]
    m = numpy.power.outer(grid.astype(numpy.float64), numpy.arange(n, dtype=numpy.float64)).T
    k = numpy.zeros(n)
    k[order] = 1.0
    out: FloatArray = numpy.linalg.solve(m, k)
    return out


def derivative_each(f: VectorFunction, x0: FloatArray, order: int, accuracy: int = 4) -> FloatArray:
    """``calculus::derivative(f, var=x0, order=order, drop=FALSE)``: the pure ``order``-th derivative in each variable, shape ``(q, p)``."""
    p = x0.shape[0]
    h = steps(x0, order)
    grid = points(order, accuracy)
    c = weights(grid, order)
    cols = []
    for j in range(p):
        vals = []
        for i in grid:
            x = x0.copy()
            x[j] = x[j] + i * h[j]
            vals.append(numpy.atleast_1d(numpy.asarray(f(x), dtype=numpy.float64)))
        F = numpy.stack(vals, axis=-1)
        cols.append(math.factorial(order) / h[j] ** order * (F @ c))
    return numpy.stack(cols, axis=-1)


def derivative_mixed(f: VectorFunction, x0: FloatArray, orders: collections.abc.Sequence[int], accuracy: int = 4) -> FloatArray:
    """``calculus::derivative(f, var=x0, order=orders)``: one mixed partial derivative, ``orders[j]`` times in variable ``j``."""
    d = numpy.asarray(orders, dtype=numpy.intp)
    h = steps(x0, int(d.sum()))
    w = numpy.flatnonzero(d > 0)
    if w.shape[0] == 0:
        w = numpy.array([0])
    grids = [points(int(d[j]), accuracy) for j in w]
    cs = [weights(g, int(d[j])) for g, j in zip(grids, w, strict=True)]
    total = None
    for combo in itertools.product(*[range(g.shape[0]) for g in grids]):
        x = x0.copy()
        coef = 1.0
        for pos, j in enumerate(w):
            x[j] = x[j] + grids[pos][combo[pos]] * h[j]
            coef *= cs[pos][combo[pos]]
        val = coef * numpy.atleast_1d(numpy.asarray(f(x), dtype=numpy.float64))
        total = val if total is None else total + val
    scale = float(numpy.prod([math.factorial(int(dj)) / hj ** int(dj) for dj, hj in zip(d, h, strict=True)]))
    if total is None:
        raise ArithmeticError('no evaluation points')
    out: FloatArray = scale * total
    return out


def hessian(f: VectorFunction, x0: FloatArray, accuracy: int = 4) -> FloatArray:
    """``calculus::hessian(f, var=x0, accuracy=accuracy)`` for a vector-valued ``f``: shape ``(q, p, p)``."""
    p = x0.shape[0]
    diag = derivative_each(f, x0, 2, accuracy)
    q = diag.shape[0]
    H = numpy.zeros((q, p, p))
    for j in range(p):
        H[:, j, j] = diag[:, j]
    if p > 1:
        h = steps(x0, 2)
        grid = points(1, accuracy)
        c = weights(grid, 1)
        for j in range(p - 1):
            for i in range(j + 1, p):
                total = numpy.zeros(q)
                for b in range(grid.shape[0]):
                    for a in range(grid.shape[0]):
                        x = x0.copy()
                        x[i] = x[i] + grid[a] * h[i]
                        x[j] = x[j] + grid[b] * h[j]
                        total = total + c[a] * c[b] * numpy.atleast_1d(numpy.asarray(f(x), dtype=numpy.float64)) / (h[i] * h[j])
                H[:, i, j] = total
                H[:, j, i] = total
    return H
