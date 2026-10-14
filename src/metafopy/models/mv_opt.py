"""Maximisation of the (restricted) log-likelihood of :func:`metafopy.rma_mv`.

metafor minimises ``-ll`` with ``nlminb`` (a derivative-free-start quasi-Newton routine from the PORT library). Here the
BFGS algorithm of SciPy is used with the analytic score of :func:`metafopy.models.mv_lik.gradient`, which reaches the same
optimum to well within the tolerance of the parity tests (Nocedal and Wright, 2006, *Numerical Optimization*, ch. 6).
"""

import collections.abc
import dataclasses
import math

import numpy
import numpy.typing
import scipy.optimize

import metafopy.errors
import metafopy.models.mv_lik

__all__ = ['OptResult', 'maximise']

type FloatArray = numpy.typing.NDArray[numpy.float64]

HUGE = 1e100
SLACK_GRADIENT = 1e-4


@dataclasses.dataclass(frozen=True, slots=True)
class OptResult:
    par: FloatArray
    value: float
    iterations: int


def maximise(
    problem: metafopy.models.mv_lik.Problem,
    par0: FloatArray,
    free: numpy.typing.NDArray[numpy.intp],
    reml: bool,
    gtol: float = 1e-7,
    maxiter: int = 5000,
    nonsmooth: bool = False,
    log: collections.abc.Callable[[FloatArray, float], None] | None = None,
) -> OptResult:
    """Maximise the log-likelihood over the entries ``free`` of the parameter vector, starting from ``par0``.

    ``nonsmooth`` (a lasso penalty with ``optbeta``) accepts the last iterate when the line search can no longer improve it,
    since the gradient does not vanish at a kink. ``log`` is called with the parameters and the log-likelihood of each evaluation.

    Raises:
        metafopy.errors.EstimationError: the optimiser did not converge.
    """
    base = par0.copy()

    def expand(x: FloatArray) -> FloatArray:
        par = base.copy()
        par[free] = x
        return par

    def objective(x: FloatArray) -> tuple[float, FloatArray]:
        par = expand(x)
        val = metafopy.models.mv_lik.negative_loglik(problem, par, reml)
        if log is not None and math.isfinite(val):
            # auditme: accept[redundant-sink-caller-plus-callee] the check of the initial values is an evaluation of the trace too, as nlminb's first one in R
            log(par, -val)
        if not math.isfinite(val):
            return HUGE, numpy.zeros(x.shape[0])
        grad = metafopy.models.mv_lik.gradient(problem, par, reml, free)
        if not numpy.all(numpy.isfinite(grad)):
            return HUGE, numpy.zeros(x.shape[0])
        return val, grad

    x0 = par0[free].copy()
    if objective(x0)[0] >= HUGE:
        raise metafopy.errors.EstimationError(
            'Cannot evaluate the log-likelihood at the initial values (the marginal variance-covariance matrix is not positive definite).'
        )

    res = scipy.optimize.minimize(objective, x0, jac=True, method='BFGS', options={'gtol': gtol, 'maxiter': maxiter})
    x = res.x
    val = float(res.fun)
    jac = numpy.asarray(res.jac, dtype=numpy.float64)
    gnorm = float(numpy.max(numpy.abs(jac))) if jac.size else 0.0
    if not res.success and gnorm > SLACK_GRADIENT:
        refined = scipy.optimize.minimize(objective, x, jac=True, method='L-BFGS-B', options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': gtol})
        if refined.fun <= val:
            x, val = refined.x, float(refined.fun)
            rjac = numpy.asarray(refined.jac, dtype=numpy.float64)
            gnorm = float(numpy.max(numpy.abs(rjac))) if rjac.size else 0.0
        if gnorm > SLACK_GRADIENT and not nonsmooth:
            raise metafopy.errors.EstimationError(f'Optimizer (BFGS) did not achieve convergence ({res.message}).')
    return OptResult(par=expand(x), value=-val, iterations=int(res.nit))
