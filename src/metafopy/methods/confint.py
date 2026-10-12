"""Confidence intervals for model coefficients and variance components.

For ``rma.uni`` fits, the CI for tau^2 uses the Q-profile method (Viechtbauer
2007, doi:10.1002/sim.2514) by default, with I^2, H^2 and tau intervals derived
from its bounds, exactly as metafor's ``confint.rma.uni()`` does.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import singledispatch
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

from metafopy.core.linalg import invcalc
from metafopy.core.zeroin import DEFAULT_TOL, uniroot
from metafopy.models._tau2 import FIXED_EFFECTS
from metafopy.models.rma_uni import RMAResult, level_to_alpha

__all__ = ['ConfintResult', 'confint']

FloatArray = NDArray[np.float64]
CIType = Literal['qp', 'wald', 'wald.log', 'wald.sqrt']


@dataclass(frozen=True, slots=True)
class ConfintResult:
    """``fixed`` and/or ``random`` tables with ``estimate``, ``ci.lb``, ``ci.ub`` columns."""

    fixed: pd.DataFrame | None
    random: pd.DataFrame | None
    lb_sign: str = ''
    ub_sign: str = ''

    def __str__(self) -> str:
        parts = [t.round(4).to_string() for t in (self.fixed, self.random) if t is not None]
        return '\n\n'.join(parts)


@singledispatch
def confint(model: Any, *args: Any, **kwargs: Any) -> ConfintResult:
    """Confidence intervals for a fitted model."""
    raise TypeError(f'confint() not implemented for {type(model).__name__}')


def _qe(tau2: float, y: FloatArray, vi: FloatArray, X: FloatArray) -> float:
    if np.any(tau2 + vi < 0):
        raise ValueError('Some marginal variances are negative.')
    w = 1.0 / (vi + tau2)
    WX = w[:, None] * X
    P = np.diag(w) - WX @ invcalc(X, w) @ WX.T
    return float(y @ P @ y)


@confint.register
def _confint_rma_uni(
    model: RMAResult,
    *,
    fixed: bool = False,
    random: bool = True,
    type: CIType = 'qp',
    level: float | None = None,
    tau2_min: float | None = None,
    tau2_max: float | None = None,
    tol: float = DEFAULT_TOL,
    maxiter: int = 1000,
) -> ConfintResult:
    """CIs for an ``rma.uni`` fit (metafor's ``confint.rma.uni``)."""
    x = model
    if not fixed and not random:
        raise ValueError("at least one of 'fixed' and 'random' must be True")
    alpha = x.level if level is None else level_to_alpha(level)

    res_random: pd.DataFrame | None = None
    lb_sign = ub_sign = ''
    if random:
        k, p = x.k, x.p
        if k == 1:
            raise ValueError('stopped because k = 1')
        if x.method in FIXED_EFFECTS:
            raise ValueError('model does not contain a random-effects component')
        if x.tau2_fix:
            raise ValueError('model does not contain an estimated random-effects component')
        if x.method == 'GENQ':
            raise NotImplementedError('generalised Q-statistic CIs are not implemented yet')

        lo = min(0.0, x.tau2) if tau2_min is None else tau2_min
        hi = max(100.0, x.tau2 * 10, lo * 10) if tau2_max is None else tau2_max
        tau2_lb = tau2_ub = math.nan

        if type == 'qp':
            if not x.allvipos:
                raise ValueError('cannot compute CI for tau^2 when there are non-positive sampling variances')
            y, vi, X = x.yi, x.vi, x.X
            crit_u = float(stats.chi2.isf(alpha / 2, k - p))
            crit_l = float(stats.chi2.ppf(alpha / 2, k - p))
            qe_max = _qe(hi, y, vi, X)
            qe_min = _qe(lo, y, vi, X)

            if qe_min < crit_l:
                tau2_lb = tau2_ub = lo
                lb_sign = ub_sign = '<'
            elif qe_max > crit_l:
                tau2_ub = hi
                ub_sign = '>'
            else:
                tau2_ub = _root(lambda t: _qe(t, y, vi, X) - crit_l, lo, hi, tol, maxiter)

            if qe_max > crit_u:
                tau2_lb = tau2_ub = hi
                lb_sign = ub_sign = '>'
            elif qe_min < crit_u:
                tau2_lb = lo
                if lo > 0:
                    lb_sign = '<'
            else:
                tau2_lb = _root(lambda t: _qe(t, y, vi, X) - crit_u, lo, hi, tol, maxiter)
        else:
            crit = float(stats.norm.isf(alpha / 2))
            if type == 'wald':
                tau2_lb = max(0.0 if tau2_min is None else tau2_min, x.tau2 - crit * x.se_tau2)
                tau2_ub = x.tau2 + crit * x.se_tau2
            elif type == 'wald.log':
                if x.tau2 >= 0:
                    tau2_lb = math.exp(math.log(x.tau2) - crit * x.se_tau2 / x.tau2)
                    tau2_ub = max(x.tau2, math.exp(math.log(x.tau2) + crit * x.se_tau2 / x.tau2))
            elif type == 'wald.sqrt':
                if x.tau2 >= 0:
                    st = math.sqrt(x.tau2)
                    tau2_lb = max(0.0, st - crit * x.se_tau2 / (2 * st)) ** 2
                    tau2_ub = (st + crit * x.se_tau2 / (2 * st)) ** 2
            else:
                raise ValueError(f'unknown CI type {type!r}')

        vt = x.vt

        def sq(v: float) -> float:
            return math.sqrt(v) if v >= 0 else math.nan

        res_random = pd.DataFrame(
            {
                'estimate': [x.tau2, sq(x.tau2), x.I2, x.H2],
                'ci.lb': [tau2_lb, sq(tau2_lb), 100 * tau2_lb / (vt + tau2_lb), tau2_lb / vt + 1],
                'ci.ub': [tau2_ub, sq(tau2_ub), 100 * tau2_ub / (vt + tau2_ub), tau2_ub / vt + 1],
            },
            index=['tau^2', 'tau', 'I^2(%)', 'H^2'],
        )

    res_fixed: pd.DataFrame | None = None
    if fixed:
        if x.ddf is not None:
            crit = float(stats.t.isf(alpha / 2, x.ddf)) if x.ddf > 0 else math.nan
        else:
            crit = float(stats.norm.isf(alpha / 2))
        res_fixed = pd.DataFrame(
            {'estimate': x.beta, 'ci.lb': x.beta - crit * x.se, 'ci.ub': x.beta + crit * x.se},
            index=list(x.coef_names),
        )

    return ConfintResult(fixed=res_fixed, random=res_random, lb_sign=lb_sign, ub_sign=ub_sign)


def _root(f: Callable[[float], float], lo: float, hi: float, tol: float, maxiter: int) -> float:
    try:
        return uniroot(f, lo, hi, tol=tol, maxiter=maxiter).root
    except ValueError:
        return math.nan
