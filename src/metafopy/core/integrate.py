"""R's ``integrate()`` on top of SciPy's QUADPACK.

R's ``integrate()`` calls the QUADPACK routines ``dqags`` (finite range) and ``dqagi`` (infinite range) with a
21-point (respectively 15-point) Gauss-Kronrod rule and the epsilon algorithm. SciPy's ``quad`` wraps the same
Fortran routines (``qags``/``qagi``), so with the same tolerances both return the same value. Like R, any
QUADPACK error code (and a non-finite integrand) is an error instead of a warning.
"""

import collections.abc
import math
import sys
import typing

import scipy.integrate

import metafopy.errors

__all__ = ['DEFAULT_REL_TOL', 'integrate']

DEFAULT_REL_TOL = sys.float_info.epsilon**0.25


def integrate(
    f: collections.abc.Callable[..., float],
    lower: float,
    upper: float,
    args: tuple[typing.Any, ...] = (),
    *,
    subdivisions: int = 100,
    rel_tol: float = DEFAULT_REL_TOL,
    abs_tol: float | None = None,
) -> float:
    """The value of ``integrate(f, lower, upper, subdivisions=, rel.tol=, abs.tol=)`` (``abs.tol`` defaults to ``rel.tol``, as in R).

    Raises:
        metafopy.errors.EstimationError: QUADPACK reported an error or the integral is not finite.
    """
    out = scipy.integrate.quad(f, lower, upper, args=args, epsabs=rel_tol if abs_tol is None else abs_tol, epsrel=rel_tol, limit=subdivisions, full_output=1)
    if len(out) > 3:
        raise metafopy.errors.EstimationError(str(out[3]))
    value = float(out[0])
    if not math.isfinite(value):
        raise metafopy.errors.EstimationError('non-finite function value')
    return value
