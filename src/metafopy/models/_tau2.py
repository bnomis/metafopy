"""Heterogeneity (tau^2) estimators for the normal-normal random-effects model.

Each estimator follows metafor's ``rma.uni()`` code path exactly, including its
quirks: estimation works on mean-centred effect sizes, ML/REML/EB use Fisher
scoring started from the Hedges estimate with step-halving at ``tau2_min``,
and PM/PMM search with R's ``uniroot()`` (ported in :mod:`metafopy.core.zeroin`).

References
----------
DerSimonian & Laird (1986), doi:10.1016/0197-2456(86)90046-2;
Viechtbauer (2005), doi:10.3102/10769986030003261;
Veroniki et al. (2016), doi:10.1002/jrsm.1164.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from metafopy.core.linalg import invcalc, logdet
from metafopy.core.zeroin import uniroot

FloatArray = NDArray[np.float64]

ESTIMATORS = (
    'EE',
    'FE',
    'CE',
    'DL',
    'DLIT',
    'HE',
    'HS',
    'HSk',
    'SJ',
    'SJIT',
    'ML',
    'REML',
    'EB',
    'PM',
    'MP',
    'PMM',
    'GENQ',
)
FIXED_EFFECTS = ('EE', 'FE', 'CE')


class EstimationError(RuntimeError):
    """An estimator failed; ``rma_uni`` falls back to the next method if one was given."""


@dataclass(frozen=True, slots=True)
class Control:
    """Control settings for ``rma_uni`` (names follow metafor with ``.`` -> ``_``)."""

    tau2_init: float | None = None
    tau2_min: float = 0.0
    tau2_max: float = 100.0
    threshold: float = 1e-5
    tol: float = np.finfo(float).eps ** 0.25
    ll0check: bool = True
    maxiter: int = 100
    stepadj: float = 1.0
    evtol: float = 1e-7
    REMLf: bool = True


@dataclass(frozen=True, slots=True)
class Tau2Fit:
    tau2: float
    se_tau2: float
    iterations: int


def _P(X: FloatArray, w: FloatArray) -> FloatArray:
    """``W - W X (X'WX)^-1 X' W`` for ``W = diag(w)``."""
    WX = w[:, None] * X
    return np.diag(w) - WX @ invcalc(X, w) @ WX.T


def _qe(tau2: float, Y: FloatArray, vi: FloatArray, X: FloatArray) -> float:
    if np.any(tau2 + vi < 0):
        raise EstimationError('Some marginal variances are negative.')
    P = _P(X, 1.0 / (vi + tau2))
    return float(Y @ P @ Y)


def estimate_tau2(
    method: str,
    yi: FloatArray,
    vi: FloatArray,
    X: FloatArray,
    *,
    weights: FloatArray | None,
    tau2_fixed: float | None,
    allvipos: bool,
    con: Control,
) -> Tau2Fit:
    """Estimate tau^2 (and its SE) with one of metafor's estimators."""
    k, p = X.shape
    ymc = yi - yi.mean()
    fix = tau2_fixed is not None

    def pick(value: float) -> float:
        return float(tau2_fixed) if fix else float(value)  # type: ignore[arg-type]

    if k == 1:
        return Tau2Fit(tau2=pick(0.0), se_tau2=math.nan, iterations=0)

    iterations = 0
    tau2 = math.nan
    tau2_0 = math.nan
    P = PV = None
    trP = trPV = math.nan
    wi = np.empty(0)

    if method in ('HS', 'HSk'):
        if not allvipos:
            raise EstimationError(f'{method} estimator cannot be used when there are non-positive sampling variances.')
        wi = 1.0 / vi
        P = _P(X, wi)
        RSS = float(ymc @ P @ ymc)
        if method == 'HS':
            tau2 = pick((RSS - k) / wi.sum())
        else:
            tau2 = pick((k / (k - p) * RSS - k) / wi.sum())

    if method in ('HE', 'ML', 'REML', 'EB'):
        P = np.eye(k) - X @ invcalc(X, np.ones(k)) @ X.T
        RSS = float(ymc @ P @ ymc)
        PV = P * vi[None, :]
        trPV = float(np.trace(PV))
        tau2 = pick((RSS - trPV) / (k - p))

    if method == 'DL':
        if not allvipos:
            raise EstimationError('DL estimator cannot be used when there are non-positive sampling variances.')
        wi = 1.0 / vi
        P = _P(X, wi)
        RSS = float(ymc @ P @ ymc)
        trP = float(np.trace(P))
        tau2 = pick((RSS - (k - p)) / trP)

    if method == 'DLIT':
        tau2 = con.tau2_init if con.tau2_init is not None else 0.0
        change = con.threshold + 1
        while change > con.threshold:
            iterations += 1
            old = tau2
            if np.any(tau2 + vi < 0):
                raise EstimationError('Some marginal variances are negative.')
            wi = 1.0 / (vi + tau2)
            P = _P(X, wi)
            trP = float(np.trace(P))
            trPV = float(np.sum(np.diag(P) * vi))
            RSS = float(ymc @ P @ ymc)
            tau2 = pick((RSS - trPV) / trP)
            tau2 = max(tau2, con.tau2_min)
            change = abs(old - tau2)
            if iterations > con.maxiter:
                raise EstimationError('Iterative DL estimator did not converge.')

    if method == 'GENQ':
        if weights is None:
            raise EstimationError("Must specify the 'weights' argument when method='GENQ'.")
        P = _P(X, weights)
        PV = P * vi[None, :]
        trP = float(np.trace(P))
        trPV = float(np.trace(PV))
        RSS = float(ymc @ P @ ymc)
        tau2 = pick((RSS - trPV) / trP)

    if method == 'SJ':
        tau2_0 = con.tau2_init if con.tau2_init is not None else float(np.mean(ymc**2))
        wi = 1.0 / (vi + tau2_0)
        P = _P(X, wi)
        RSS = float(ymc @ P @ ymc)
        PV = P * vi[None, :]
        tau2 = pick(tau2_0 * RSS / (k - p))

    if method == 'SJIT':
        tau2 = con.tau2_init if con.tau2_init is not None else float(np.mean(ymc**2))
        change = con.threshold + 1
        while change > con.threshold:
            iterations += 1
            old = tau2
            wi = 1.0 / (vi + tau2)
            P = _P(X, wi)
            RSS = float(ymc @ P @ ymc)
            tau2 = pick(tau2 * RSS / (k - p))
            change = abs(old - tau2)
            if iterations > con.maxiter:
                raise EstimationError('Iterative SJ estimator did not converge.')

    if method in ('PM', 'MP', 'PMM'):
        if not allvipos:
            raise EstimationError(f'{method} estimator cannot be used when there are non-positive sampling variances.')
        target = float(stats.chi2.ppf(0.5, k - p)) if method == 'PMM' else float(k - p)
        if fix:
            tau2 = float(tau2_fixed)  # type: ignore[arg-type]
        elif _qe(con.tau2_min, ymc, vi, X) < target:
            tau2 = con.tau2_min
        elif _qe(con.tau2_max, ymc, vi, X) > target:
            raise EstimationError("Value of 'tau2_max' too low. Try increasing 'tau2_max' or switch to another method.")
        else:
            try:
                res = uniroot(
                    lambda t: _qe(t, ymc, vi, X) - target,
                    con.tau2_min,
                    con.tau2_max,
                    tol=con.tol,
                    maxiter=con.maxiter,
                )
            except ValueError as exc:
                raise EstimationError('Error in iterative search for tau^2 using uniroot().') from exc
            tau2 = res.root
            iterations = res.iterations

    if method in ('ML', 'REML', 'EB'):
        tau2 = con.tau2_init if con.tau2_init is not None else max(0.0, tau2, con.tau2_min)
        change = con.threshold + 1
        while change > con.threshold:
            iterations += 1
            old = tau2
            if np.any(tau2 + vi < 0):
                raise EstimationError('Some marginal variances are negative.')
            wi = 1.0 / (vi + tau2)
            if np.any(np.isinf(wi)):
                raise EstimationError('Division by zero when computing the inverse variance weights.')
            P = _P(X, wi)
            if method == 'ML':
                PP = P @ P
                adj = (float(ymc @ PP @ ymc) - float(wi.sum())) / float(np.sum(wi**2))
            elif method == 'REML':
                PP = P @ P
                adj = (float(ymc @ PP @ ymc) - float(np.trace(P))) / float(np.trace(PP))
            else:
                adj = (float(ymc @ P @ ymc) * k / (k - p) - k) / float(wi.sum())
            adj *= con.stepadj
            if math.isnan(adj):
                adj = 0.0
            while tau2 + adj < con.tau2_min:
                adj /= 2
            tau2 = pick(tau2 + adj)
            change = abs(old - tau2)
            if iterations > con.maxiter:
                raise EstimationError('Fisher scoring algorithm did not converge. See the rma_uni docs for remedies.')

        if method in ('ML', 'REML') and con.ll0check and allvipos and not fix:
            ll0 = _ll(method, ymc, vi, X, 0.0)
            ll = _ll(method, ymc, vi, X, tau2)
            if ll0 - ll > con.tol and tau2 > con.threshold:
                # metafor warns: Fisher scoring may have gotten stuck at a local maximum
                tau2 = 0.0

        wi = 1.0 / (vi + tau2)
        if np.any(np.isinf(wi)):
            raise EstimationError('Division by zero when computing the inverse variance weights.')
        P = _P(X, wi)

    tau2 = max(con.tau2_min, tau2)
    if np.any(tau2 + vi < 0):
        raise EstimationError('Some marginal variances are negative.')

    t = max(tau2, 0.0)
    se = math.nan
    if method in ('HS', 'HSk'):
        assert P is not None
        se = math.sqrt(1 / wi.sum() ** 2 * (2 * (k - p) + 4 * t * np.trace(P) + 2 * t**2 * np.sum(P * P)))
        if method == 'HSk':
            se *= k / (k - p)
    elif method == 'HE':
        assert PV is not None
        se = math.sqrt(1 / (k - p) ** 2 * (2 * np.sum(PV * PV.T) + 4 * t * trPV + 2 * t**2 * (k - p)))
    elif method == 'DL':
        assert P is not None
        se = math.sqrt(1 / trP**2 * (2 * (k - p) + 4 * t * trP + 2 * t**2 * np.sum(P * P)))
    elif method == 'GENQ':
        assert P is not None
        assert PV is not None
        se = math.sqrt(1 / trP**2 * (2 * np.sum(PV * PV.T) + 4 * t * np.sum(PV * P) + 2 * t**2 * np.sum(P * P)))
    elif method == 'SJ':
        assert P is not None
        assert PV is not None
        se = math.sqrt(tau2_0**2 / (k - p) ** 2 * (2 * np.sum(PV * PV.T) + 4 * t * np.sum(PV * P) + 2 * t**2 * np.sum(P * P)))
    elif method == 'ML':
        se = math.sqrt(2 / np.sum(wi**2))
    elif method == 'REML':
        assert P is not None
        se = math.sqrt(2 / np.sum(P * P))
    elif method in ('EB', 'PM', 'MP', 'PMM', 'DLIT', 'SJIT'):
        wi = 1.0 / (vi + tau2)
        se = math.sqrt(2 * k**2 / (k - p) / wi.sum() ** 2)

    return Tau2Fit(tau2=float(tau2), se_tau2=float(se), iterations=iterations)


def _ll(method: str, y: FloatArray, vi: FloatArray, X: FloatArray, tau2: float) -> float:
    """ML or REML log-likelihood used by metafor's ``ll0check``."""
    k, p = int(X.shape[0]), int(X.shape[1])
    wi = 1.0 / (vi + tau2)
    beta = invcalc(X, wi) @ (X.T @ (wi * y))
    RSS = float(np.sum(wi * (y - X @ beta) ** 2))
    if method == 'ML':
        return -0.5 * k * math.log(2 * math.pi) - 0.5 * float(np.sum(np.log(vi + tau2))) - 0.5 * RSS
    return -0.5 * (k - p) * math.log(2 * math.pi) - 0.5 * float(np.sum(np.log(vi + tau2))) - 0.5 * logdet((X.T * wi) @ X) - 0.5 * RSS
