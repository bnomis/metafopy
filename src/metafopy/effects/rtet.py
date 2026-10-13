"""Tetrachoric correlation and its sampling variance (metafor's hidden ``.rtet``).

The correlation and the two thresholds are estimated by maximum likelihood
under a bivariate normal latent model (Pearson, 1900; see Kirk, 1973,
doi:10.1007/BF02291424). As in metafor, ``rho`` is first optimised with the
thresholds fixed at the sample margins, then all three parameters jointly by
Nelder-Mead, and the variance is the ``[0, 0]`` element of the inverse
numerical Hessian.
"""

import math
import warnings

import numpy
import numpy.typing
import scipy.special

import metafopy.core.bvn
import metafopy.core.optim
import metafopy.effects.na

__all__ = ['rtet']

INF = math.inf


def negll(par: numpy.typing.NDArray[numpy.float64], ai: float, bi: float, ci: float, di: float, maxcor: float, fixcut: bool) -> float:
    rho = float(par[0])
    cut_row = float(par[1])
    cut_col = float(par[2])
    ni = ai + bi + ci + di
    if abs(rho) > maxcor:
        rho = math.copysign(maxcor, rho)
    if fixcut:
        cut_row = float(scipy.special.ndtri((ai + bi) / ni))
        cut_col = float(scipy.special.ndtri((ai + ci) / ni))
    p_ai = metafopy.core.bvn.bvn_rect((-INF, -INF), (cut_col, cut_row), rho)
    p_bi = metafopy.core.bvn.bvn_rect((cut_col, -INF), (INF, cut_row), rho)
    p_ci = metafopy.core.bvn.bvn_rect((-INF, cut_row), (cut_col, INF), rho)
    p_di = metafopy.core.bvn.bvn_rect((cut_col, cut_row), (INF, INF), rho)
    if p_ai <= 0 or p_bi <= 0 or p_ci <= 0 or p_di <= 0:
        return INF
    return -(xlogy(ai, p_ai) + xlogy(bi, p_bi) + xlogy(ci, p_ci) + xlogy(di, p_di))


def xlogy(x: float, p: float) -> float:
    # R evaluates 0 * log(p) as 0 for finite log(p); p > 0 is guaranteed here
    return x * math.log(p)


def rtet(ai: float, bi: float, ci: float, di: float, maxcor: float = 0.9999) -> tuple[float, float]:
    """Return ``(yi, vi)`` for one 2x2 table, as metafor's ``.rtet``."""
    ni = ai + bi + ci + di

    if (ai + bi) == 0 or (ci + di) == 0 or (ai + ci) == 0 or (bi + di) == 0:
        return 0.0, INF
    if bi == 0 and ci == 0:
        return 1.0, 0.0
    if ai == 0 and di == 0:
        return -1.0, 0.0

    def f1(rho: float) -> float:
        return negll(numpy.array([rho, 0.0, 0.0]), ai, bi, ci, di, maxcor, True)

    try:
        start = metafopy.core.optim.fmin(f1, -1.0, 1.0)
    except ValueError, ArithmeticError:
        warnings.warn('Could not estimate tetrachoric correlation coefficient.', stacklevel=3)
        return metafopy.effects.na.NA, metafopy.effects.na.NA

    def fn(par: numpy.typing.NDArray[numpy.float64]) -> float:
        return negll(par, ai, bi, ci, di, maxcor, False)

    par0 = [start, float(scipy.special.ndtri((ai + bi) / ni)), float(scipy.special.ndtri((ai + ci) / ni))]
    try:
        res = metafopy.core.optim.nelder_mead(fn, par0)
        hess = metafopy.core.optim.optimhess(fn, res.par)
    except ValueError, ArithmeticError:
        warnings.warn('Could not estimate tetrachoric correlation coefficient.', stacklevel=3)
        return metafopy.effects.na.NA, metafopy.effects.na.NA

    try:
        chol = numpy.linalg.cholesky(hess)
        inv_chol = numpy.linalg.inv(chol)
        vi = float((inv_chol.T @ inv_chol)[0, 0])
    except numpy.linalg.LinAlgError:
        warnings.warn('Could not estimate sampling variance of tetrachoric correlation coefficient.', stacklevel=3)
        vi = metafopy.effects.na.NA

    yi = float(res.par[0])
    if bi == 0 or ci == 0:
        yi = 1.0
    if ai == 0 or di == 0:
        yi = -1.0
    return yi, vi
