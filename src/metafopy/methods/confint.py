"""Confidence intervals for model coefficients and variance components.

For ``rma.uni`` fits, the CI for tau^2 uses the Q-profile method (Viechtbauer
2007, doi:10.1002/sim.2514) by default, with I^2, H^2 and tau intervals derived
from its bounds, exactly as metafor's ``confint.rma.uni()`` does.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import singledispatch
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

from metafopy._data import MetaData
from metafopy.core.linalg import invcalc
from metafopy.core.zeroin import DEFAULT_TOL, uniroot
from metafopy.models._tau2 import FIXED_EFFECTS, genq_func
from metafopy.models.rma_uni import RMAResult, fit_rma_uni, level_to_alpha

__all__ = ['ConfintResult', 'confint']

FloatArray = NDArray[np.float64]
CIType = Literal['qp', 'genq', 'pl', 'ht', 'wald', 'wald.log', 'wald.sqrt']


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
    type: CIType | None = None,
    level: float | None = None,
    transf: Callable[..., Any] | None = None,
    targs: Any = None,
    tau2_min: float | None = None,
    tau2_max: float | None = None,
    tol: float = DEFAULT_TOL,
    maxiter: int = 1000,
) -> ConfintResult:
    """CIs for an ``rma.uni`` fit (metafor's ``confint.rma.uni``).

    ``type`` is ``'qp'`` (Q-profile; Viechtbauer, 2007, doi:10.1002/sim.2514),
    ``'genq'`` (generalised Q; Jackson, 2013, doi:10.1002/jrsm.1081; the default
    for ``method='GENQ'``/``'GENQM'`` fits), ``'pl'`` (profile likelihood),
    ``'ht'`` (Higgins & Thompson, 2002, doi:10.1002/sim.1186) or one of the
    Wald types. ``transf`` (with optional ``targs``) is applied to the fixed
    effects and their bounds.
    """
    x = model
    if type is None:
        type = 'genq' if x.method in ('GENQ', 'GENQM') else 'qp'
    else:
        type = cast(CIType, type.lower())
        if type not in ('qp', 'genq', 'pl', 'ht', 'wald', 'wald.log', 'wald.sqrt'):
            raise ValueError(f'unknown CI type {type!r}')
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
        if type == 'genq' and x.method not in ('GENQ', 'GENQM'):
            raise ValueError('model must be fitted with method="GENQ" or method="GENQM" to use this option')

        lo = min(0.0, x.tau2) if tau2_min is None else tau2_min
        hi = max(100.0, x.tau2 * 10, lo * 10) if tau2_max is None else tau2_max
        tau2_lb = tau2_ub = math.nan
        lb_conv = ub_conv = False

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
                lb_conv = ub_conv = True
            elif qe_max > crit_l:
                tau2_ub = hi
                ub_sign = '>'
                ub_conv = True
            else:
                tau2_ub, ub_conv = _root(lambda t: _qe(t, y, vi, X) - crit_l, lo, hi, tol, maxiter)

            if qe_max > crit_u:
                tau2_lb = tau2_ub = hi
                lb_sign = ub_sign = '>'
                lb_conv = ub_conv = True
            elif qe_min < crit_u:
                tau2_lb = lo
                lb_conv = True
                if lo > 0:
                    lb_sign = '<'
            else:
                tau2_lb, lb_conv = _root(lambda t: _qe(t, y, vi, X) - crit_u, lo, hi, tol, maxiter)
        elif type == 'genq':
            if x.weights is None:
                raise ValueError('cannot extract weights from the model object')
            y, vi, X = x.yi, x.vi, x.X
            A = x.weights
            AX = A[:, None] * X
            P = np.diag(A) - AX @ invcalc(X, A) @ AX.T
            Q = float(y @ P @ y)

            def g(t: float, level_: float, getlower: bool) -> float:
                return genq_func(t, P, vi, Q, level_, k, p, getlower=getlower)

            g_max = g(hi, 0.0, True)
            g_min = g(lo, 0.0, True)
            if g_min > 1 - alpha / 2:
                tau2_lb = tau2_ub = lo
                lb_sign = ub_sign = '<'
                lb_conv = ub_conv = True
            elif g_max < 1 - alpha / 2:
                tau2_ub = hi
                ub_sign = '>'
                ub_conv = True
            else:
                # metafor calls uniroot() here with its default tol and maxiter
                tau2_ub, ub_conv = _root(lambda t: g(t, alpha / 2, False), lo, hi, DEFAULT_TOL, 1000)

            if g_max < alpha / 2:
                tau2_lb = tau2_ub = hi
                lb_sign = ub_sign = '>'
                lb_conv = ub_conv = True
            elif g_min > alpha / 2:
                tau2_lb = lo
                lb_conv = True
                if lo > 0:
                    lb_sign = '<'
            else:
                tau2_lb, lb_conv = _root(lambda t: g(t, alpha / 2, True), lo, hi, DEFAULT_TOL, 1000)
        elif type == 'pl':
            if lo > x.tau2:
                raise ValueError('lower bound of interval to be searched must be <= actual value of component')
            if hi < x.tau2:
                raise ValueError('upper bound of interval to be searched must be >= actual value of component')
            objective = float(stats.chi2.ppf(1 - alpha, 1))

            def prof(t: float) -> float:
                return _profile_lrt(x, t) - objective

            val = _try(prof, lo)
            if not math.isnan(val):
                if val < 0:
                    tau2_lb = lo
                    lb_conv = True
                    if lo > 0:
                        lb_sign = '<'
                else:
                    tau2_lb, lb_conv = _root(prof, lo, x.tau2, tol, maxiter, check_conv=True)
            val = _try(prof, hi)
            if not math.isnan(val):
                if val < 0:
                    tau2_ub = hi
                    ub_conv = True
                    ub_sign = '>'
                else:
                    tau2_ub, ub_conv = _root(prof, x.tau2, hi, tol, maxiter, check_conv=True)
        elif type == 'ht':
            if not x.int_only:
                raise ValueError('method only applicable to models without moderators')
            if k <= 2:
                raise ValueError('method only applicable when k > 2')
            if k < x.QE:
                se_lnH = 0.5 * (math.log(x.QE) - math.log(k - 1)) / (math.sqrt(2 * x.QE) - math.sqrt(2 * k - 3))
            else:
                se_lnH = math.sqrt(1 / (2 * (k - 2)) * (1 - 1 / (3 * (k - 2) ** 2)))
            crit = float(stats.norm.isf(alpha / 2))
            lb_conv = ub_conv = True
            H2_lb = math.exp(math.log(x.H2) - crit * 2 * se_lnH)
            H2_ub = math.exp(math.log(x.H2) + crit * 2 * se_lnH)
            I2_lb = (H2_lb - 1) / H2_lb
            I2_ub = (H2_ub - 1) / H2_ub
            tau2_lb = max(0.0, I2_lb * x.vt / (1 - I2_lb))
            tau2_ub = I2_ub * x.vt / (1 - I2_ub)
        else:
            crit = float(stats.norm.isf(alpha / 2))
            lb_conv = ub_conv = True
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

        if not lb_conv:
            warnings.warn('Error in iterative search for the lower bound.', stacklevel=3)
        if not ub_conv:
            warnings.warn('Error in iterative search for the upper bound.', stacklevel=3)

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
        est = x.beta.copy()
        lb = x.beta - crit * x.se
        ub = x.beta + crit * x.se
        if transf is not None:

            def tf(v: float) -> float:
                return float(transf(v) if targs is None else transf(v, targs))

            est = np.array([tf(v) for v in est])
            lb = np.array([tf(v) for v in lb])
            ub = np.array([tf(v) for v in ub])
        lb, ub = np.minimum(lb, ub), np.maximum(lb, ub)
        res_fixed = pd.DataFrame({'estimate': est, 'ci.lb': lb, 'ci.ub': ub}, index=list(x.coef_names))

    return ConfintResult(fixed=res_fixed, random=res_random, lb_sign=lb_sign, ub_sign=ub_sign)


def _root(f: Callable[[float], float], lo: float, hi: float, tol: float, maxiter: int, *, check_conv: bool = False) -> tuple[float, bool]:
    """``try(uniroot(...)$root)``: ``(root, converged)``, or ``(nan, False)`` on error."""
    try:
        res = uniroot(f, lo, hi, tol=tol, maxiter=maxiter)
    except ValueError, ArithmeticError, np.linalg.LinAlgError:
        return math.nan, False
    if check_conv and not res.converged:
        return math.nan, False
    return res.root, True


def _try(f: Callable[[float], float], v: float) -> float:
    try:
        return f(v)
    except ValueError, ArithmeticError, np.linalg.LinAlgError:
        return math.nan


def _loglik(res: RMAResult) -> float:
    return res.fit_stats_reml.ll if res.method == 'REML' else res.fit_stats_ml.ll


def _profile_lrt(x: RMAResult, tau2: float) -> float:
    """``-2 (ll(tau2) - ll(tau2_hat))`` from a refit with tau^2 fixed (metafor's ``.profile.rma.uni``)."""
    md = MetaData.from_inputs(x.yi, x.vi, weights=x.weights, mods=x.X, intercept=False)
    md = replace(md, X_names=tuple(x.coef_names))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = fit_rma_uni(
            md,
            method=x.method,
            weighted=x.weighted,
            test=x.test,
            level=1 - x.level,
            control=x.control,
            tau2=tau2,
            _skip_r2=True,
        )
    return -2 * (_loglik(res) - _loglik(x))
