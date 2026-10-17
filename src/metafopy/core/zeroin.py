"""Brent's root finder, ported line for line from R's ``R_zeroin2``.

metafor relies on ``uniroot()`` (default ``tol = .Machine$double.eps^0.25``)
for the Paule-Mandel estimators and Q-profile confidence intervals, so the
values it reports depend on the exact iterates of R's implementation. Using
SciPy's ``brentq`` would converge to a slightly different point inside the
same tolerance; porting R's routine reproduces metafor's numbers exactly.

Source: R ``src/library/stats/src/zeroin.c`` (GPL-2 | GPL-3), itself based on
Brent (1973), *Algorithms for Minimization without Derivatives*, ch. 4.
"""

import collections.abc
import dataclasses
import math
import sys
import typing
import warnings

import metafopy.errors

__all__ = ['RootResult', 'uniroot', 'zeroin2']

EPSILON = sys.float_info.epsilon
DEFAULT_TOL = EPSILON**0.25


@dataclasses.dataclass(frozen=True, slots=True)
class RootResult:
    root: float
    f_root: float
    iterations: int
    estim_prec: float
    converged: bool


def zeroin2(
    f: collections.abc.Callable[[float], float],
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
    f: collections.abc.Callable[[float], float],
    lower: float,
    upper: float,
    *,
    tol: float = DEFAULT_TOL,
    maxiter: int = 1000,
    extend_int: typing.Literal['no', 'yes', 'downX', 'upX'] = 'no',
) -> RootResult:
    """Find a root of ``f`` in ``[lower, upper]`` as R's ``uniroot()`` does.

    ``extend_int`` is R's ``extendInt``: the interval is widened (by steps that double
    in size) until ``f`` changes sign, in any direction (``'yes'``) or given that ``f``
    is increasing (``'upX'``) or decreasing (``'downX'``).

    Raises ``ValueError`` when ``f(lower)`` and ``f(upper)`` have the same sign
    (R: "f() values at end points not of opposite sign").

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    if not lower < upper:
        raise metafopy.errors.InputError('lower < upper is not fulfilled')
    f_lower = f(lower)
    f_upper = f(upper)
    if math.isnan(f_lower):
        raise metafopy.errors.InputError('f.lower = f(lower) is NA')
    if math.isnan(f_upper):
        raise metafopy.errors.InputError('f.upper = f(upper) is NA')
    sig = {'yes': None, 'downX': -1, 'no': 0, 'upX': 1}[extend_int]
    big = sys.float_info.max
    low_t = max(min(f_lower, big), -big)
    upp_t = max(min(f_upper, big), -big)
    do_x = low_t * upp_t > 0 if sig is None else (sig * low_t > 0 or sig * upp_t < 0)
    it = 0
    if do_x:
        if sig is None:
            delta = [0.01 * max(1e-4, abs(lower)), 0.01 * max(1e-4, abs(upper))]
            while f_lower * f_upper > 0 and (math.isfinite(lower) or math.isfinite(upper)):
                it += 1
                if it > maxiter:
                    raise metafopy.errors.InputError(f'no sign change found in {it - 1} iterations')
                if math.isfinite(lower):
                    old_lower, old_f = lower, f_lower
                    lower -= delta[0]
                    f_lower = f(lower)
                    if math.isnan(f_lower):
                        lower, f_lower = old_lower, old_f
                        delta[0] /= 4
                if math.isfinite(upper):
                    old_upper, old_f = upper, f_upper
                    upper += delta[1]
                    f_upper = f(upper)
                    if math.isnan(f_upper):
                        upper, f_upper = old_upper, old_f
                        delta[1] /= 4
                delta = [2 * d for d in delta]
        else:
            step = 0.01 * max(1e-4, abs(lower))
            while sig * f_lower > 0:
                it += 1
                if it > maxiter:
                    raise metafopy.errors.InputError(f'no sign change found in {it - 1} iterations')
                lower -= step
                f_lower = f(lower)
                step *= 2
            step = 0.01 * max(1e-4, abs(upper))
            while sig * f_upper < 0:
                it += 1
                if it > maxiter:
                    raise metafopy.errors.InputError(f'no sign change found in {it - 1} iterations')
                upper += step
                f_upper = f(upper)
                step *= 2
    if sign(f_lower) * sign(f_upper) > 0:
        raise metafopy.errors.InputError(
            'did not succeed extending the interval endpoints for f(lower) * f(upper) <= 0' if do_x else 'f() values at end points not of opposite sign'
        )
    root, iterations, prec = zeroin2(f, lower, upper, max(min(f_lower, big), -big), max(min(f_upper, big), -big), tol, maxiter)
    converged = iterations >= 0
    if not converged:
        warnings.warn(f'_NOT_ converged in {maxiter} iteration{"" if maxiter == 1 else "s"}', stacklevel=2)
        iterations = maxiter
    return RootResult(
        root=root,
        f_root=f(root),
        iterations=iterations + it if do_x else iterations,
        estim_prec=prec,
        converged=converged,
    )


def sign(x: float) -> int:
    return int(x > 0) - int(x < 0)
