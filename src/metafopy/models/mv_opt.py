"""Maximisation of the (restricted) log-likelihood of :func:`metafopy.rma_mv`.

metafor minimises ``-ll`` with ``nlminb`` (a derivative-free-start quasi-Newton routine from the PORT library). Here the
BFGS algorithm of SciPy is used with the analytic score of :func:`metafopy.models.mv_lik.gradient`, which reaches the same
optimum to well within the tolerance of the parity tests (Nocedal and Wright, 2006, *Numerical Optimization*, ch. 6). BFGS is
followed by a Newton polish on the score (:func:`polish_score`; Nocedal and Wright, 2006, ch. 3, Newton's method with a step
test), because the gradient slack accepts BFGS stops ("precision loss") that vary by platform and by run, up to 1.1e-06 of the
score's scale on CI Windows (measured 2026-11-17; see ``agents/specs/rma-mv-reml-stop/spec.md``). The other
optimisers of metafor's ``control$optimizer`` (``optim`` with ``optmethod``, ``nlm``, ``bobyqa``, ``hjk``, ...) are run through
:func:`metafopy.models.sel_opt.minimize`, which ports ``optim``'s BFGS and Nelder-Mead and maps the others to SciPy methods;
``mfmaxit`` limits the number of likelihood evaluations as in metafor; those optimisers are not polished.
"""

import collections.abc
import dataclasses
import math
import typing

import numpy
import numpy.typing
import scipy.optimize

import metafopy.core.stationarity
import metafopy.errors
import metafopy.models.mv_lik
import metafopy.models.sel_opt

__all__ = ['POLISH_GRADIENT', 'POLISH_MAX_STEP', 'POLISH_STEPS', 'OptResult', 'maximise', 'polish_score', 'score_jacobian']

type FloatArray = numpy.typing.NDArray[numpy.float64]

HUGE = 1e100
SLACK_GRADIENT = 1e-4
POLISH_GRADIENT = 1e-10  # stop when the largest |gradient| entry of -ll is at most this
POLISH_MAX_STEP = 1e-3  # a Newton step with a larger entry (transformed scale) means a ridge or a boundary: stop there
POLISH_STEPS = 8


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
    choice: metafopy.models.sel_opt.OptimizerChoice | None = None,
    opt_ctrl: collections.abc.Mapping[str, typing.Any] | None = None,
    mfmaxit: float = math.inf,
) -> OptResult:
    """Maximise the log-likelihood over the entries ``free`` of the parameter vector, starting from ``par0``.

    ``nonsmooth`` (a lasso penalty with ``optbeta``) accepts the last iterate when the line search can no longer improve it,
    since the gradient does not vanish at a kink. ``log`` is called with the parameters and the log-likelihood of each evaluation.

    ``choice`` names an optimiser of metafor other than the default ``nlminb`` and ``opt_ctrl`` its settings; ``mfmaxit`` is the
    most evaluations of the log-likelihood allowed.

    Raises:
        metafopy.errors.EstimationError: the optimiser did not converge or ``mfmaxit`` was reached.
        metafopy.errors.NotSupportedError: the optimiser has no counterpart here.
    """
    base = par0.copy()
    evaluations = 0

    def count() -> None:
        nonlocal evaluations
        if evaluations > mfmaxit:
            raise metafopy.errors.EstimationError(
                f'Error during the optimization (maximum number of iterations, mfmaxit={mfmaxit}, reached). Use verbose=True to see the progress.'
            )
        evaluations += 1

    def expand(x: FloatArray) -> FloatArray:
        par = base.copy()
        par[free] = x
        return par

    def objective(x: FloatArray) -> tuple[float, FloatArray]:
        count()
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
    if choice is not None:
        return maximise_other(problem, par0, free, reml, choice, opt_ctrl or {}, expand, count, log)
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
        decrement = metafopy.core.stationarity.newton_decrement(res)
        refined = scipy.optimize.minimize(objective, x, jac=True, method='L-BFGS-B', options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': gtol})
        improved = refined.fun < val - metafopy.core.stationarity.REL_TOL * (1 + abs(val))
        if refined.fun <= val:
            x, val = refined.x, float(refined.fun)
            rjac = numpy.asarray(refined.jac, dtype=numpy.float64)
            gnorm = float(numpy.max(numpy.abs(rjac))) if rjac.size else 0.0
        if gnorm > SLACK_GRADIENT and not nonsmooth and (improved or decrement > metafopy.core.stationarity.REL_TOL * (1 + abs(val))):
            raise metafopy.errors.EstimationError(f'Optimizer (BFGS) did not achieve convergence ({res.message}).')
    par = expand(x)
    if not nonsmooth and free.size:
        par, val = polish_score(problem, par, free, reml)
    return OptResult(par=par, value=-val, iterations=int(res.nit))


def score_jacobian(
    problem: metafopy.models.mv_lik.Problem,
    par: FloatArray,
    free: numpy.typing.NDArray[numpy.intp],
    reml: bool,
) -> FloatArray:
    """Hessian of ``-ll`` in the free entries of the transformed parameters: central differences of
    :func:`metafopy.models.mv_lik.gradient` (step ``1e-5 * max(1, |par_i|)``), symmetrised. ``len(free) x len(free)``."""
    hessian = numpy.empty((free.size, free.size))
    for column, index in enumerate(free):
        step = 1e-5 * max(1.0, abs(float(par[index])))
        up = par.copy()
        down = par.copy()
        up[index] += step
        down[index] -= step
        hessian[:, column] = (metafopy.models.mv_lik.gradient(problem, up, reml, free) - metafopy.models.mv_lik.gradient(problem, down, reml, free)) / (
            2 * step
        )
    return (hessian + hessian.T) / 2


def polish_score(
    problem: metafopy.models.mv_lik.Problem,
    par: FloatArray,
    free: numpy.typing.NDArray[numpy.intp],
    reml: bool,
) -> tuple[FloatArray, float]:
    """Newton steps on the score from ``par`` (the BFGS stop) until the largest gradient entry is at most ``POLISH_GRADIENT``; returns the parameters
    and ``-ll`` there. A step is taken only while the Hessian is positive definite, no entry of the step exceeds ``POLISH_MAX_STEP``, ``-ll`` does not
    rise by more than ``metafopy.core.stationarity.REL_TOL * (1 + |-ll|)`` and the largest gradient entry falls; otherwise ``par`` as reached so far is
    returned. At most ``POLISH_STEPS`` steps.

    The objective test is loose on purpose: near the optimum ``-ll`` changes by less than its own rounding, so a tight test rejects every useful
    step and the fall of the gradient is what accepts a step, as the score is what the tests check (Nocedal and Wright, 2006, ch. 3).
    """
    val = metafopy.models.mv_lik.negative_loglik(problem, par, reml)
    grad = metafopy.models.mv_lik.gradient(problem, par, reml, free)
    for _ in range(POLISH_STEPS):
        gnorm = float(numpy.max(numpy.abs(grad)))
        if gnorm <= POLISH_GRADIENT:
            break
        try:
            factor = numpy.linalg.cholesky(score_jacobian(problem, par, free, reml))
        except numpy.linalg.LinAlgError:
            break
        step = -numpy.linalg.solve(factor.T, numpy.linalg.solve(factor, grad))
        if not numpy.all(numpy.isfinite(step)) or float(numpy.max(numpy.abs(step))) > POLISH_MAX_STEP:
            break
        trial = par.copy()
        trial[free] += step
        trial_val = metafopy.models.mv_lik.negative_loglik(problem, trial, reml)
        if not math.isfinite(trial_val) or trial_val > val + metafopy.core.stationarity.REL_TOL * (1 + abs(val)):
            break
        trial_grad = metafopy.models.mv_lik.gradient(problem, trial, reml, free)
        if not numpy.all(numpy.isfinite(trial_grad)) or float(numpy.max(numpy.abs(trial_grad))) >= gnorm:
            break
        par, val, grad = trial, trial_val, trial_grad
    return par, val


def maximise_other(
    problem: metafopy.models.mv_lik.Problem,
    par0: FloatArray,
    free: numpy.typing.NDArray[numpy.intp],
    reml: bool,
    choice: metafopy.models.sel_opt.OptimizerChoice,
    opt_ctrl: collections.abc.Mapping[str, typing.Any],
    expand: collections.abc.Callable[[FloatArray], FloatArray],
    count: collections.abc.Callable[[], None],
    log: collections.abc.Callable[[FloatArray, float], None] | None,
) -> OptResult:
    """Maximise with an optimiser of metafor that has no analytic-gradient counterpart: it sees only the negative log-likelihood.

    Raises:
        metafopy.errors.EstimationError: the log-likelihood cannot be evaluated at the initial values or the optimiser failed.
        metafopy.errors.NotSupportedError: the optimiser has no counterpart here.
    """

    def objective(x: FloatArray) -> float:
        count()
        par = expand(x)
        val = metafopy.models.mv_lik.negative_loglik(problem, par, reml)
        if log is not None and math.isfinite(val):
            # auditme: accept[redundant-sink-caller-plus-callee] objective is the one place that logs, and it is called once per evaluation
            log(par, -val)
        return val

    x0 = par0[free].copy()
    if not math.isfinite(objective(x0)):
        raise metafopy.errors.EstimationError(
            'Cannot evaluate the log-likelihood at the initial values (the marginal variance-covariance matrix is not positive definite).'
        )
    try:
        x = metafopy.models.sel_opt.minimize(objective, x0, choice, opt_ctrl)
    except metafopy.errors.EstimationError:
        raise
    except (ValueError, ArithmeticError, numpy.linalg.LinAlgError) as err:
        # auditme: accept[except-interpolation] metafor reports the optimiser message
        raise metafopy.errors.EstimationError(f'Error during the optimization: {err}') from err
    val = objective(x)
    return OptResult(par=expand(x), value=-val, iterations=0)
