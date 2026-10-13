"""Tetrachoric correlation and its sampling variance (metafor's hidden ``.rtet``).

The correlation and the two thresholds are estimated by maximum likelihood
under a bivariate normal latent model (Pearson, 1900; see Kirk, 1973,
doi:10.1007/BF02291424). As in metafor, ``rho`` is first optimised with the
thresholds fixed at the sample margins, then all three parameters jointly by
Nelder-Mead, and the variance is the ``[0, 0]`` element of the inverse
numerical Hessian.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.special import ndtri

from metafopy.core.bvn import bvn_rect
from metafopy.core.optim import fmin, nelder_mead, optimhess
from metafopy.effects._na import NA

__all__ = ['rtet']

_INF = math.inf


def _negll(par: NDArray[np.float64], ai: float, bi: float, ci: float, di: float, maxcor: float, fixcut: bool) -> float:
    rho = float(par[0])
    cut_row = float(par[1])
    cut_col = float(par[2])
    ni = ai + bi + ci + di
    if abs(rho) > maxcor:
        rho = math.copysign(maxcor, rho)
    if fixcut:
        cut_row = float(ndtri((ai + bi) / ni))
        cut_col = float(ndtri((ai + ci) / ni))
    p_ai = bvn_rect((-_INF, -_INF), (cut_col, cut_row), rho)
    p_bi = bvn_rect((cut_col, -_INF), (_INF, cut_row), rho)
    p_ci = bvn_rect((-_INF, cut_row), (cut_col, _INF), rho)
    p_di = bvn_rect((cut_col, cut_row), (_INF, _INF), rho)
    if p_ai <= 0 or p_bi <= 0 or p_ci <= 0 or p_di <= 0:
        return _INF
    return -(_xlogy(ai, p_ai) + _xlogy(bi, p_bi) + _xlogy(ci, p_ci) + _xlogy(di, p_di))


def _xlogy(x: float, p: float) -> float:
    # R evaluates 0 * log(p) as 0 for finite log(p); p > 0 is guaranteed here
    return x * math.log(p)


def rtet(ai: float, bi: float, ci: float, di: float, maxcor: float = 0.9999) -> tuple[float, float]:
    """Return ``(yi, vi)`` for one 2x2 table, as metafor's ``.rtet``."""
    ni = ai + bi + ci + di

    if (ai + bi) == 0 or (ci + di) == 0 or (ai + ci) == 0 or (bi + di) == 0:
        return 0.0, _INF
    if bi == 0 and ci == 0:
        return 1.0, 0.0
    if ai == 0 and di == 0:
        return -1.0, 0.0

    def f1(rho: float) -> float:
        return _negll(np.array([rho, 0.0, 0.0]), ai, bi, ci, di, maxcor, True)

    try:
        start = fmin(f1, -1.0, 1.0)
    except ValueError, ArithmeticError:
        warnings.warn('Could not estimate tetrachoric correlation coefficient.', stacklevel=3)
        return NA, NA

    def fn(par: NDArray[np.float64]) -> float:
        return _negll(par, ai, bi, ci, di, maxcor, False)

    par0 = [start, float(ndtri((ai + bi) / ni)), float(ndtri((ai + ci) / ni))]
    try:
        res = nelder_mead(fn, par0)
        hess = optimhess(fn, res.par)
    except ValueError, ArithmeticError:
        warnings.warn('Could not estimate tetrachoric correlation coefficient.', stacklevel=3)
        return NA, NA

    try:
        chol = np.linalg.cholesky(hess)
        inv_chol = np.linalg.inv(chol)
        vi = float((inv_chol.T @ inv_chol)[0, 0])
    except np.linalg.LinAlgError:
        warnings.warn('Could not estimate sampling variance of tetrachoric correlation coefficient.', stacklevel=3)
        vi = NA

    yi = float(res.par[0])
    if bi == 0 or ci == 0:
        yi = 1.0
    if ai == 0 or di == 0:
        yi = -1.0
    return yi, vi
