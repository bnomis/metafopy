"""Stopping test for SciPy's BFGS on flat likelihoods.

PORT's ``nlminb`` (metafor's default optimiser) stops once the relative change of the objective it still predicts is below
``rel.tol = 1e-10`` (Gay, 1983, *ACM Trans. Math. Softw.* 9, 503-524; R's ``?nlminb``). SciPy's BFGS tests the largest gradient entry
instead, which fails on a ridge or a boundary (a correlation of -1, a log-variance heading to minus infinity): the line search then
ends with 'precision loss' although no step could improve the objective by more than rounding error.
"""

import math

import numpy
import scipy.optimize

__all__ = ['REL_TOL', 'newton_decrement']

REL_TOL = 1e-10


def newton_decrement(res: scipy.optimize.OptimizeResult) -> float:
    """The decrease ``g' H^-1 g / 2`` a Newton step with the inverse Hessian of a BFGS run would still bring (infinite when unavailable)."""
    hess_inv = getattr(res, 'hess_inv', None)
    jac = getattr(res, 'jac', None)
    if hess_inv is None or jac is None:
        return math.inf
    g = numpy.asarray(jac, dtype=numpy.float64)
    dense = hess_inv.todense() if hasattr(hess_inv, 'todense') else hess_inv
    return float(0.5 * g @ numpy.asarray(dense, dtype=numpy.float64) @ g)
