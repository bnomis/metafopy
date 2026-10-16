"""Optimisers for location-scale models.

metafor minimises the negative (restricted) log-likelihood with ``nlminb`` (PORT library, numerical gradient) when
the scale model has a log link and with ``constrOptim`` (an adaptive logarithmic barrier around ``optim``'s
Nelder-Mead) when it has an identity link, which needs ``tau^2_i >= 0``. ``constrOptim`` is ported here
step by step on top of the exact Nelder-Mead of :mod:`metafopy.core.optim`; ``nlminb`` is replaced by SciPy's BFGS
followed by a Nelder-Mead polish, which reaches the same optimum to well within the parity tolerance.

Source of ``constrOptim``: R ``src/library/stats/R/constrOptim.R`` (GPL-2 | GPL-3).
"""

import collections.abc
import dataclasses
import functools
import math
import typing

import numpy
import numpy.typing
import scipy.optimize

import metafopy.core.optim
import metafopy.core.stationarity
import metafopy.errors

__all__ = ['OptResult', 'constr_optim', 'minimise']

type FloatArray = numpy.typing.NDArray[numpy.float64]
type Objective = collections.abc.Callable[[FloatArray], float]

HUGE = 1e100


@dataclasses.dataclass(frozen=True, slots=True)
class OptResult:
    par: FloatArray
    value: float
    iterations: int


def constr_optim(
    theta: FloatArray,
    f: Objective,
    ui: FloatArray,
    ci: FloatArray,
    *,
    mu: float = 1e-4,
    outer_iterations: int = 100,
    outer_eps: float = 1e-5,
    maxit: int = 500,
) -> OptResult:
    """R's ``constrOptim(theta, f, grad=NULL, ui, ci)``: minimise ``f`` subject to ``ui %*% theta - ci >= 0``.

    Raises:
        metafopy.errors.EstimationError: the starting value is not in the interior of the feasible region.
    """
    if numpy.any(ui @ theta - ci <= 0):
        raise metafopy.errors.EstimationError('initial value is not in the interior of the feasible region')

    def barrier(x: FloatArray, theta_old: FloatArray) -> float:
        ui_theta = ui @ x
        gi = ui_theta - ci
        if numpy.any(gi < 0):
            return math.nan
        gi_old = ui @ theta_old - ci
        with numpy.errstate(divide='ignore', invalid='ignore'):
            # auditme: accept[redundant-sink-caller-plus-callee] port of the constrOptim barrier, which evaluates the log twice by definition
            bar = float(numpy.sum(gi_old * numpy.log(gi) - ui_theta))
        if not math.isfinite(bar):
            bar = -math.inf
        return f(x) - mu * bar

    obj = f(theta)
    r = barrier(theta, theta)
    total = 0
    last = metafopy.core.optim.NelderMeadResult(theta, r, 0, 0)
    obj_old = obj
    for _ in range(outer_iterations):
        obj_old = obj
        r_old = r
        theta_old = theta
        last = metafopy.core.optim.nelder_mead(functools.partial(barrier, theta_old=theta_old), theta_old, maxit=maxit)
        r = last.value
        if math.isfinite(r) and math.isfinite(r_old) and abs(r - r_old) < (0.001 + abs(r)) * outer_eps:
            break
        theta = last.par
        total += last.fncount
        obj = f(theta)
        if obj > obj_old:
            break
    return OptResult(par=last.par, value=f(last.par), iterations=total)


def minimise(f: Objective, x0: FloatArray, *, method: str = 'BFGS', maxiter: int = 5000, gtol: float = 1e-9, polish_converged: bool = True) -> OptResult:
    """Minimise ``f`` with a gradient-free start: SciPy's quasi-Newton method, then a Nelder-Mead polish.

    Non-finite values of ``f`` count as a very large value.

    Raises:
        metafopy.errors.EstimationError: ``f`` cannot be evaluated at the starting value or no run converged.
    """

    def safe(x: FloatArray) -> float:
        val = f(x)
        return val if math.isfinite(val) else HUGE

    if safe(x0) >= HUGE:
        raise metafopy.errors.EstimationError('Cannot evaluate the log-likelihood at the initial values.')
    if x0.shape[0] == 0:
        return OptResult(par=x0.copy(), value=safe(x0), iterations=0)
    options: dict[str, float | int] = {'maxiter': maxiter}
    if method == 'BFGS':
        options['gtol'] = gtol
    if method in ('BFGS', 'L-BFGS-B', 'CG', 'Nelder-Mead'):
        first = scipy.optimize.minimize(safe, x0, method=typing.cast(typing.Any, method), options=typing.cast(typing.Any, options))
    else:
        raise metafopy.errors.InputError(f'unknown optimizer {method!r}')
    best_x, best_val = numpy.asarray(first.x, dtype=numpy.float64), float(first.fun)
    nit = int(first.nit)
    # a final simplex search guards against a premature stop of the quasi-Newton iterations; with polish_converged=False it is skipped after a
    # converged run or a stop with a negligible predicted decrease, as it would only walk further down a likelihood that is flat towards a
    # boundary (a log-variance heading to minus infinity), where nlminb stops once the relative change of the objective is below 1e-10
    decrement = metafopy.core.stationarity.newton_decrement(first)
    if polish_converged or (not first.success and decrement > metafopy.core.stationarity.REL_TOL * (1 + abs(best_val))):
        polish = scipy.optimize.minimize(
            safe,
            best_x,
            method='Nelder-Mead',
            options={'maxiter': maxiter * 4, 'xatol': 1e-10, 'fatol': 1e-13, 'adaptive': True},
        )
        if float(polish.fun) < best_val:
            best_x, best_val = numpy.asarray(polish.x, dtype=numpy.float64), float(polish.fun)
            nit += int(polish.nit)
    if best_val >= HUGE:
        raise metafopy.errors.EstimationError('Optimizer did not achieve convergence.')
    return OptResult(par=best_x, value=best_val, iterations=nit)
