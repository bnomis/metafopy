"""R's ``optim(method="BFGS")`` ported line for line.

``selmodel()`` minimises the negative log-likelihood with ``optim()``'s variable-metric (BFGS) routine and a
finite-difference gradient. Where the optimiser stops depends on every step it takes, so the routine is
ported rather than replaced by SciPy's BFGS. Sources (R 4.3.3, GPL-2 | GPL-3):

- ``src/appl/optim.c``: ``vmmin``, after Nash (1990), *Compact Numerical Methods for Computers*, 2nd ed.,
  Algorithm 21 (Fletcher's variable-metric method).
- ``src/library/stats/src/optim.c``: ``fmingr`` (central differences with step ``ndeps``).
"""

import collections.abc
import dataclasses
import math
import sys

import numpy
import numpy.typing

import metafopy.errors

__all__ = ['BFGSResult', 'optim_bfgs']

type FloatArray = numpy.typing.NDArray[numpy.float64]

STEPREDN = 0.2
ACCTOL = 0.0001
RELTEST = 10.0


@dataclasses.dataclass(frozen=True, slots=True)
class BFGSResult:
    """``par``, ``value``, the function and gradient counts and R's ``convergence`` code (0 = converged, 1 = ``maxit`` reached)."""

    par: FloatArray
    value: float
    fncount: int
    grcount: int
    convergence: int


def optim_bfgs(
    fn: collections.abc.Callable[[FloatArray], float],
    par: collections.abc.Sequence[float] | FloatArray,
    *,
    ndeps: float = 1e-3,
    maxit: int = 100,
    reltol: float = math.sqrt(sys.float_info.epsilon),
    abstol: float = -math.inf,
) -> BFGSResult:
    """Minimise ``fn`` from ``par`` as ``optim(par, fn, method="BFGS", control=list(ndeps, maxit, reltol, abstol))`` does.

    Raises:
        metafopy.errors.EstimationError: the initial value or a finite-difference gradient is not finite.
    """
    b = numpy.array(par, dtype=numpy.float64)
    n = b.shape[0]

    def fminfn(x: FloatArray) -> float:
        return float(fn(x.copy()))

    def fmingr(x: FloatArray) -> FloatArray:
        df = numpy.zeros(n)
        work = x.copy()
        for i in range(n):
            work[i] = x[i] + ndeps
            val1 = float(fn(work.copy()))
            work[i] = x[i] - ndeps
            val2 = float(fn(work.copy()))
            df[i] = (val1 - val2) / (2 * ndeps)
            if not math.isfinite(df[i]):
                raise metafopy.errors.EstimationError(f'non-finite value in the numerical gradient [{i + 1}]')
            work[i] = x[i]
        return df

    if maxit <= 0:
        return BFGSResult(b, fminfn(b), 0, 0, 0)

    g = numpy.zeros(n)
    t = numpy.zeros(n)
    X = numpy.zeros(n)
    c = numpy.zeros(n)
    B = [[0.0] * (i + 1) for i in range(n)]
    f = fminfn(b)
    if not math.isfinite(f):
        raise metafopy.errors.EstimationError("initial value in 'vmmin' is not finite")
    fmin = f
    funcount = gradcount = 1
    g = fmingr(b)
    it = 1
    ilast = gradcount
    count = 0
    steplength = 1.0

    while True:
        if ilast == gradcount:
            for i in range(n):
                for j in range(i):
                    B[i][j] = 0.0
                B[i][i] = 1.0
        X = b.copy()
        c = g.copy()
        gradproj = 0.0
        for i in range(n):
            s = 0.0
            for j in range(i + 1):
                s -= B[i][j] * g[j]
            for j in range(i + 1, n):
                s -= B[j][i] * g[j]
            t[i] = s
            gradproj += s * g[i]

        if gradproj < 0.0:
            steplength = 1.0
            accpoint = False
            while True:
                count = 0
                for i in range(n):
                    b[i] = X[i] + steplength * t[i]
                    if RELTEST + X[i] == RELTEST + b[i]:
                        count += 1
                if count < n:
                    f = fminfn(b)
                    funcount += 1
                    accpoint = math.isfinite(f) and f <= fmin + gradproj * steplength * ACCTOL
                    if not accpoint:
                        steplength *= STEPREDN
                if count == n or accpoint:
                    break
            enough = f > abstol and abs(f - fmin) > reltol * (abs(fmin) + reltol)
            if not enough:
                count = n
                fmin = f
            if count < n:
                fmin = f
                g = fmingr(b)
                gradcount += 1
                it += 1
                D1 = 0.0
                for i in range(n):
                    t[i] = steplength * t[i]
                    c[i] = g[i] - c[i]
                    D1 += t[i] * c[i]
                if D1 > 0:
                    D2 = 0.0
                    for i in range(n):
                        s = 0.0
                        for j in range(i + 1):
                            s += B[i][j] * c[j]
                        for j in range(i + 1, n):
                            s += B[j][i] * c[j]
                        X[i] = s
                        D2 += s * c[i]
                    D2 = 1.0 + D2 / D1
                    for i in range(n):
                        for j in range(i + 1):
                            B[i][j] += (D2 * t[i] * t[j] - X[i] * t[j] - t[i] * X[j]) / D1
                else:
                    ilast = gradcount
            elif ilast < gradcount:
                count = 0
                ilast = gradcount
        else:
            count = 0
            if ilast == gradcount:
                count = n
            else:
                ilast = gradcount
        if it >= maxit:
            break
        if gradcount - ilast > 2 * n:
            ilast = gradcount
        if not (count != n or ilast != gradcount):
            break

    return BFGSResult(b, fmin, funcount, gradcount, 0 if it < maxit else 1)
