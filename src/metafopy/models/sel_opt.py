"""Optimisers for :func:`metafopy.selmodel`.

metafor minimises the negative log-likelihood with ``optim()`` (BFGS with a finite-difference gradient by default) or
with one of many other R optimisers. The defaults ``optimizer="optim"`` with ``optmethod="BFGS"`` or ``"Nelder-Mead"``
are ported exactly (:func:`metafopy.core.bfgs.optim_bfgs`, :func:`metafopy.core.optim.nelder_mead`). The other names
metafor accepts are mapped to the closest SciPy method; they find the same optimum but not through the same
sequence of steps, so results agree with metafor only up to the convergence tolerance:

=================================================================  ==========================
metafor optimizer                                                  SciPy method
=================================================================  ==========================
``nlminb``, ``nlm``, ``L-BFGS-B``, ``ucminf``, ``lbfgsb3c``,      ``L-BFGS-B``
``Rcgmin``, ``Rvmmin``, ``optimParallel``
``CG``                                                             ``CG``
``uobyqa``, ``newuoa``, ``bobyqa``, ``nloptr``                     ``Powell``
``hjk``, ``nmk``, ``mads``, ``subplex``, ``BBoptim``              ``Nelder-Mead``
``solnp``, ``constrOptim.nl``, ``auglag``, ``alabama`` (stepcon)   ``SLSQP``
=================================================================  ==========================
"""

import collections.abc
import dataclasses
import math
import typing
import warnings

import numpy
import numpy.typing
import scipy.optimize

import metafopy.core.bfgs
import metafopy.core.optim
import metafopy.errors

__all__ = ['OPTIMIZERS', 'OPTIMIZERS_MV', 'OptimizerChoice', 'choose_optimizer', 'minimize']

type FloatArray = numpy.typing.NDArray[numpy.float64]

HUGE = 1e100

# auditme: accept[non-alphabetical-literal] the order follows metafor's documented option order
OPTIMIZERS = (
    'optim',
    'nlminb',
    'uobyqa',
    'newuoa',
    'bobyqa',
    'nloptr',
    'nlm',
    'hjk',
    'nmk',
    'mads',
    'ucminf',
    'lbfgsb3c',
    'subplex',
    'BBoptim',
    'optimParallel',
    'solnp',
    'alabama',
    'constrOptim.nl',
    'auglag',
    'Nelder-Mead',
    'BFGS',
    'CG',
    'L-BFGS-B',
    'SANN',
    'Brent',
    'Rcgmin',
    'Rvmmin',
)
# the optimisers of rma.mv: those of selmodel without the constrained ones
OPTIMIZERS_MV = tuple(name for name in OPTIMIZERS if name not in ('alabama', 'auglag', 'constrOptim.nl', 'solnp'))
# auditme: accept[non-alphabetical-literal] the order is that of optim()'s documentation
OPTMETHODS = ('Nelder-Mead', 'BFGS', 'CG', 'L-BFGS-B', 'SANN', 'Brent')
# auditme: accept[non-alphabetical-dict-keys] grouped by the SciPy method each optimiser maps to
SCIPY = {
    'nlminb': 'L-BFGS-B',
    'nlm': 'L-BFGS-B',
    'L-BFGS-B': 'L-BFGS-B',
    'ucminf': 'L-BFGS-B',
    'lbfgsb3c': 'L-BFGS-B',
    'Rcgmin': 'L-BFGS-B',
    'Rvmmin': 'L-BFGS-B',
    'optimParallel': 'L-BFGS-B',
    'CG': 'CG',
    'uobyqa': 'Powell',
    'newuoa': 'Powell',
    'bobyqa': 'Powell',
    'nloptr': 'Powell',
    'hjk': 'Nelder-Mead',
    'nmk': 'Nelder-Mead',
    'mads': 'Nelder-Mead',
    'subplex': 'Nelder-Mead',
    'BBoptim': 'Nelder-Mead',
}
# SciPy's default tolerances (1e-4 for Nelder-Mead and Powell) stop 0.1 to 1 % short of the optimum on a log-likelihood; metafor's optimisers
# (dfoptim's hjk and nmk, minqa's bobyqa, ucminf, nlm, ...) converge to about 1e-6 or better, so the stand-ins are run to a tight tolerance
TIGHT: dict[str, dict[str, float]] = {
    'L-BFGS-B': {'ftol': 1e-14, 'gtol': 1e-9},
    'Nelder-Mead': {'fatol': 1e-12, 'xatol': 1e-9},
    'Powell': {'ftol': 1e-13, 'xtol': 1e-9},
    'SLSQP': {'ftol': 1e-10},
}
CONSTRAINED = ('auglag', 'constrOptim.nl', 'nloptr', 'solnp')


@dataclasses.dataclass(frozen=True, slots=True)
class OptimizerChoice:
    optimizer: str
    optmethod: str


def partial_match(value: str, options: collections.abc.Sequence[str], what: str) -> str:
    """R's ``match.arg``: an exact match or a unique prefix.

    Raises:
        metafopy.errors.InputError: the value is not (the start of) exactly one option.
    """
    if value in options:
        return value
    hits = [o for o in options if o.startswith(value)]
    if len(hits) == 1:
        return hits[0]
    raise metafopy.errors.InputError(f"'{what}' should be one of {', '.join(repr(o) for o in options)}")


def choose_optimizer(optimizer: str, optmethod: str, type_: str, ncpus: int = 1) -> OptimizerChoice:
    """metafor's optimiser bookkeeping: shortcuts for ``optim`` methods and the constrained optimiser for ``type="stepcon"``."""
    opt = partial_match(optimizer, OPTIMIZERS, 'optimizer')
    method = partial_match(optmethod, OPTMETHODS, 'optmethod')
    if opt in OPTMETHODS:
        method = opt
        opt = 'optim'
    if ncpus > 1:
        opt = 'optimParallel'
    if opt == 'alabama':
        opt = 'constrOptim.nl'
    if type_ == 'stepcon':
        if opt == 'optim' and method == 'BFGS':
            opt = 'solnp'
        elif opt not in CONSTRAINED:
            opt = 'solnp'
            warnings.warn(
                "Can only use optimizers 'solnp', 'nloptr', 'constrOptim.nl', or 'auglag' when type='stepcon' (resetting to 'solnp').",
                stacklevel=4,
            )
    if type_ != 'stepcon' and opt == 'constrOptim.nl':
        opt = 'optim'
        warnings.warn("Cannot use 'constrOptim.nl' optimizer to fit this model (resetting to 'optim').", stacklevel=4)
    return OptimizerChoice(opt, method)


def minimize(
    fn: collections.abc.Callable[[FloatArray], float],
    par0: FloatArray,
    choice: OptimizerChoice,
    control: collections.abc.Mapping[str, typing.Any],
    ineq: collections.abc.Callable[[FloatArray], FloatArray] | None = None,
) -> FloatArray:
    """The parameter vector minimising ``fn``.

    ``control`` holds the optimiser settings (``maxit``, ``reltol``, ``abstol``, ``ndeps`` for the ported ``optim`` methods and
    ``maxit``/``tol`` for the SciPy ones). ``ineq`` returns the values that must be non-negative (``type="stepcon"``).

    Raises:
        metafopy.errors.EstimationError: the optimiser did not converge.
        metafopy.errors.NotSupportedError: the optimiser has no counterpart here.
    """
    opt = choice.optimizer
    if opt == 'optim' and choice.optmethod == 'BFGS':
        res = metafopy.core.bfgs.optim_bfgs(
            fn,
            par0,
            ndeps=float(control.get('ndeps', 1e-3)),
            maxit=int(control.get('maxit', 100)),
            reltol=float(control.get('reltol', math.sqrt(2.220446049250313e-16))),
            abstol=float(control.get('abstol', -math.inf)),
        )
        if res.convergence != 0:
            raise metafopy.errors.EstimationError(f'Optimizer (optim) did not achieve convergence (convergence = {res.convergence}).')
        return res.par
    if opt == 'optim' and choice.optmethod == 'Nelder-Mead':
        nm = metafopy.core.optim.nelder_mead(
            fn,
            par0,
            maxit=int(control.get('maxit', 500)),
            reltol=float(control.get('reltol', math.sqrt(2.220446049250313e-16))),
            abstol=float(control.get('abstol', -math.inf)),
        )
        if nm.fail != 0:
            raise metafopy.errors.EstimationError(f'Optimizer (optim) did not achieve convergence (convergence = {nm.fail}).')
        return nm.par
    method = 'SLSQP' if opt in CONSTRAINED else SCIPY.get(opt) or SCIPY.get(choice.optmethod)
    if method is None:
        raise metafopy.errors.NotSupportedError(f'optimizer {opt!r} (method {choice.optmethod!r}) is not available; see help(metafopy.models.sel_opt)')

    def objective(x: FloatArray) -> float:
        val = float(fn(x))
        return val if math.isfinite(val) else HUGE

    constraints: list[dict[str, typing.Any]] = []
    if method == 'SLSQP' and ineq is not None:
        constraints.append({'type': 'ineq', 'fun': ineq})
    options: dict[str, typing.Any] = {'maxiter': int(control.get('maxit', 1000))}
    options.update(TIGHT.get(method, {}))
    if method == 'Nelder-Mead':
        options['maxfev'] = 20 * options['maxiter']
        options['maxiter'] = options['maxfev']
    sp_kwargs: dict[str, typing.Any] = {'method': method, 'options': options, 'tol': control.get('tol', None)}
    if constraints:
        sp_kwargs['constraints'] = constraints
    res_sp = scipy.optimize.minimize(objective, par0, **sp_kwargs)
    if not res_sp.success and method != 'Powell' and TIGHT.get(method):
        loose = dict(sp_kwargs)
        loose['options'] = {'maxiter': options['maxiter']} if method != 'Nelder-Mead' else {'maxfev': options['maxfev'], 'maxiter': options['maxiter']}
        res_sp = scipy.optimize.minimize(objective, res_sp.x, **loose)
    if not res_sp.success and method != 'Powell':
        raise metafopy.errors.EstimationError(f'Optimizer ({opt}) did not achieve convergence ({res_sp.message}).')
    return numpy.asarray(res_sp.x, dtype=numpy.float64)
