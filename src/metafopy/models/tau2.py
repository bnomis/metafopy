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

import dataclasses
import math

import numpy
import numpy.typing
import scipy.stats

import metafopy.core.farebrother
import metafopy.core.linalg
import metafopy.core.zeroin
import metafopy.errors

FloatArray = numpy.typing.NDArray[numpy.float64]

# auditme: accept[non-alphabetical-literal] order is metafor's documented order and is meaningful
ESTIMATORS = (
    'EE',
    'FE',
    'CE',
    'DL',
    'DLIT',
    'HE',
    'CO',
    'VC',
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
    'GENQM',
)
# auditme: accept[non-alphabetical-literal] order is metafor's documented order and is meaningful
FIXED_EFFECTS = ('EE', 'FE', 'CE')


@dataclasses.dataclass(frozen=True, slots=True)
class Control:
    """Control settings for ``rma_uni`` (names follow metafor with ``.`` -> ``_``)."""

    tau2_init: float | None = None
    tau2_min: float = 0.0
    tau2_max: float = 100.0
    threshold: float = 1e-5
    tol: float = numpy.finfo(float).eps ** 0.25
    ll0check: bool = True
    maxiter: int = 100
    stepadj: float = 1.0
    evtol: float = 1e-7
    REMLf: bool = True


@dataclasses.dataclass(frozen=True, slots=True)
class Tau2Fit:
    tau2: float
    se_tau2: float
    iterations: int


def projection(X: FloatArray, w: FloatArray) -> FloatArray:
    """``W - W X (X'WX)^-1 X' W`` for ``W = diag(w)``."""
    WX = w[:, None] * X
    return numpy.diag(w) - WX @ metafopy.core.linalg.invcalc(X, w) @ WX.T


def qe(tau2: float, Y: FloatArray, vi: FloatArray, X: FloatArray) -> float:
    if numpy.any(tau2 + vi < 0):
        raise metafopy.errors.EstimationError('Some marginal variances are negative.')
    P = projection(X, 1.0 / (vi + tau2))
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
    """Estimate tau^2 (and its SE) with one of metafor's estimators.

    Raises:
        metafopy.errors.EstimationError: the estimator cannot be used for these data or failed to converge.
    """
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
    wi = numpy.empty(0)

    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    if method in ('HS', 'HSk'):
        if not allvipos:
            raise metafopy.errors.EstimationError(f'{method} estimator cannot be used when there are non-positive sampling variances.')
        wi = 1.0 / vi
        P = projection(X, wi)
        RSS = float(ymc @ P @ ymc)
        if method == 'HS':
            tau2 = pick((RSS - k) / wi.sum())
        else:
            tau2 = pick((k / (k - p) * RSS - k) / wi.sum())

    if method in ('HE', 'CO', 'VC', 'ML', 'REML', 'EB'):
        P = numpy.eye(k) - X @ metafopy.core.linalg.invcalc(X, numpy.ones(k)) @ X.T
        RSS = float(ymc @ P @ ymc)
        PV = P * vi[None, :]
        trPV = float(numpy.trace(PV))
        tau2 = pick((RSS - trPV) / (k - p))

    if method == 'DL':
        if not allvipos:
            raise metafopy.errors.EstimationError('DL estimator cannot be used when there are non-positive sampling variances.')
        wi = 1.0 / vi
        P = projection(X, wi)
        RSS = float(ymc @ P @ ymc)
        trP = float(numpy.trace(P))
        tau2 = pick((RSS - (k - p)) / trP)

    if method == 'DLIT':
        tau2 = con.tau2_init if con.tau2_init is not None else 0.0
        change = con.threshold + 1
        while change > con.threshold:
            iterations += 1
            old = tau2
            if numpy.any(tau2 + vi < 0):
                raise metafopy.errors.EstimationError('Some marginal variances are negative.')
            wi = 1.0 / (vi + tau2)
            P = projection(X, wi)
            trP = float(numpy.trace(P))
            trPV = float(numpy.sum(numpy.diag(P) * vi))
            RSS = float(ymc @ P @ ymc)
            tau2 = pick((RSS - trPV) / trP)
            tau2 = max(tau2, con.tau2_min)
            change = abs(old - tau2)
            if iterations > con.maxiter:
                raise metafopy.errors.EstimationError('Iterative DL estimator did not converge.')

    if method == 'GENQ':
        if weights is None:
            raise metafopy.errors.EstimationError("Must specify the 'weights' argument when method='GENQ'.")
        P = projection(X, weights)
        PV = P * vi[None, :]
        trP = float(numpy.trace(P))
        trPV = float(numpy.trace(PV))
        RSS = float(ymc @ P @ ymc)
        tau2 = pick((RSS - trPV) / trP)

    if method == 'GENQM':
        if weights is None:
            raise metafopy.errors.EstimationError("Must specify the 'weights' argument when method='GENQM'.")
        Pg = projection(X, weights)
        P = Pg
        PV = Pg * vi[None, :]
        trP = float(numpy.trace(Pg))
        if fix:
            tau2 = float(tau2_fixed)  # type: ignore[arg-type]
        else:
            RSS = float(ymc @ P @ ymc)
            if genq_func(con.tau2_min, Pg, vi, RSS, 0.0, k, p, getlower=True) > 0.5:
                tau2 = con.tau2_min
            elif genq_func(con.tau2_max, Pg, vi, RSS, 0.0, k, p, getlower=True) < 0.5:
                raise metafopy.errors.EstimationError("Value of 'tau2_max' too low. Try increasing 'tau2_max' or switch to another method.")
            else:
                try:
                    res = metafopy.core.zeroin.uniroot(
                        lambda t: genq_func(t, Pg, vi, RSS, 0.5, k, p, getlower=False),
                        con.tau2_min,
                        con.tau2_max,
                        tol=con.tol,
                        maxiter=con.maxiter,
                    )
                except ValueError as exc:
                    raise metafopy.errors.EstimationError('Error in iterative search for tau^2 using uniroot().') from exc
                tau2 = res.root
                iterations = res.iterations

    if method == 'SJ':
        tau2_0 = con.tau2_init if con.tau2_init is not None else float(numpy.mean(ymc**2))
        wi = 1.0 / (vi + tau2_0)
        P = projection(X, wi)
        RSS = float(ymc @ P @ ymc)
        PV = P * vi[None, :]
        tau2 = pick(tau2_0 * RSS / (k - p))

    if method == 'SJIT':
        tau2 = con.tau2_init if con.tau2_init is not None else float(numpy.mean(ymc**2))
        change = con.threshold + 1
        while change > con.threshold:
            iterations += 1
            old = tau2
            wi = 1.0 / (vi + tau2)
            P = projection(X, wi)
            RSS = float(ymc @ P @ ymc)
            tau2 = pick(tau2 * RSS / (k - p))
            change = abs(old - tau2)
            if iterations > con.maxiter:
                raise metafopy.errors.EstimationError('Iterative SJ estimator did not converge.')

    if method in ('PM', 'MP', 'PMM'):
        if not allvipos:
            raise metafopy.errors.EstimationError(f'{method} estimator cannot be used when there are non-positive sampling variances.')
        target = float(scipy.stats.chi2.ppf(0.5, k - p)) if method == 'PMM' else float(k - p)
        if fix:
            tau2 = float(tau2_fixed)  # type: ignore[arg-type]
        elif qe(con.tau2_min, ymc, vi, X) < target:
            tau2 = con.tau2_min
        elif qe(con.tau2_max, ymc, vi, X) > target:
            raise metafopy.errors.EstimationError("Value of 'tau2_max' too low. Try increasing 'tau2_max' or switch to another method.")
        else:
            try:
                res = metafopy.core.zeroin.uniroot(
                    lambda t: qe(t, ymc, vi, X) - target,
                    con.tau2_min,
                    con.tau2_max,
                    tol=con.tol,
                    maxiter=con.maxiter,
                )
            except ValueError as exc:
                raise metafopy.errors.EstimationError('Error in iterative search for tau^2 using uniroot().') from exc
            tau2 = res.root
            iterations = res.iterations

    if method in ('ML', 'REML', 'EB'):
        tau2 = con.tau2_init if con.tau2_init is not None else max(0.0, tau2, con.tau2_min)
        change = con.threshold + 1
        while change > con.threshold:
            iterations += 1
            old = tau2
            if numpy.any(tau2 + vi < 0):
                raise metafopy.errors.EstimationError('Some marginal variances are negative.')
            wi = 1.0 / (vi + tau2)
            if numpy.any(numpy.isinf(wi)):
                raise metafopy.errors.EstimationError('Division by zero when computing the inverse variance weights.')
            P = projection(X, wi)
            if method == 'ML':
                PP = P @ P
                adj = (float(ymc @ PP @ ymc) - float(wi.sum())) / float(numpy.sum(wi**2))
            elif method == 'REML':
                PP = P @ P
                adj = (float(ymc @ PP @ ymc) - float(numpy.trace(P))) / float(numpy.trace(PP))
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
                raise metafopy.errors.EstimationError('Fisher scoring algorithm did not converge. See the rma_uni docs for remedies.')

        if method in ('ML', 'REML') and con.ll0check and allvipos and not fix:
            ll0 = log_likelihood(method, ymc, vi, X, 0.0)
            ll = log_likelihood(method, ymc, vi, X, tau2)
            if ll0 - ll > con.tol and tau2 > con.threshold:
                # metafor warns: Fisher scoring may have gotten stuck at a local maximum
                tau2 = 0.0

        wi = 1.0 / (vi + tau2)
        if numpy.any(numpy.isinf(wi)):
            raise metafopy.errors.EstimationError('Division by zero when computing the inverse variance weights.')
        P = projection(X, wi)

    tau2 = max(con.tau2_min, tau2)
    if numpy.any(tau2 + vi < 0):
        raise metafopy.errors.EstimationError('Some marginal variances are negative.')

    t = max(tau2, 0.0)
    se = math.nan
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    if method in ('HS', 'HSk'):
        if P is None:
            raise metafopy.errors.InternalError('P must not be None')
        se = math.sqrt(1 / wi.sum() ** 2 * (2 * (k - p) + 4 * t * numpy.trace(P) + 2 * t**2 * numpy.sum(P * P)))
        if method == 'HSk':
            se *= k / (k - p)
    elif method in ('HE', 'CO', 'VC'):
        if PV is None:
            raise metafopy.errors.InternalError('PV must not be None')
        se = math.sqrt(1 / (k - p) ** 2 * (2 * numpy.sum(PV * PV.T) + 4 * t * trPV + 2 * t**2 * (k - p)))
    elif method == 'DL':
        if P is None:
            raise metafopy.errors.InternalError('P must not be None')
        se = math.sqrt(1 / trP**2 * (2 * (k - p) + 4 * t * trP + 2 * t**2 * numpy.sum(P * P)))
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    elif method in ('GENQ', 'GENQM'):
        if P is None:
            raise metafopy.errors.InternalError('P must not be None')
        if PV is None:
            raise metafopy.errors.InternalError('PV must not be None')
        se = math.sqrt(1 / trP**2 * (2 * numpy.sum(PV * PV.T) + 4 * t * numpy.sum(PV * P) + 2 * t**2 * numpy.sum(P * P)))
    elif method == 'SJ':
        if P is None:
            raise metafopy.errors.InternalError('P must not be None')
        if PV is None:
            raise metafopy.errors.InternalError('PV must not be None')
        se = math.sqrt(tau2_0**2 / (k - p) ** 2 * (2 * numpy.sum(PV * PV.T) + 4 * t * numpy.sum(PV * P) + 2 * t**2 * numpy.sum(P * P)))
    elif method == 'ML':
        se = math.sqrt(2 / numpy.sum(wi**2))
    elif method == 'REML':
        if P is None:
            raise metafopy.errors.InternalError('P must not be None')
        se = math.sqrt(2 / numpy.sum(P * P))
    elif method in ('EB', 'PM', 'MP', 'PMM', 'DLIT', 'SJIT'):
        wi = 1.0 / (vi + tau2)
        se = math.sqrt(2 * k**2 / (k - p) / wi.sum() ** 2)

    return Tau2Fit(tau2=float(tau2), se_tau2=float(se), iterations=iterations)


def genq_func(tau2: float, P: FloatArray, vi: FloatArray, Q: float, level: float, k: int, p: int, *, getlower: bool) -> float:
    """metafor's ``.GENQ.func``: ``P(Q_gen > Q | tau^2) - level`` (or ``1 - P(...) - level``).

    The generalised Q-statistic is a quadratic form whose null distribution is
    a weighted sum of chi^2_1 variables with weights given by the non-zero
    eigenvalues of ``S P S``, ``S = diag(sqrt(v_i + tau^2))`` (Jackson, 2013,
    doi:10.1002/jrsm.1081; Jackson et al., 2014, doi:10.1002/sim.6131).
    """
    s = numpy.sqrt(vi + tau2)
    lam = numpy.linalg.eigvalsh(s[:, None] * P * s[None, :])[::-1]
    Qq = metafopy.core.farebrother.farebrother(Q, lam[: k - p].tolist()).Qq
    return Qq - level if getlower else (1 - Qq) - level


def log_likelihood(method: str, y: FloatArray, vi: FloatArray, X: FloatArray, tau2: float) -> float:
    """ML or REML log-likelihood used by metafor's ``ll0check``."""
    k, p = int(X.shape[0]), int(X.shape[1])
    wi = 1.0 / (vi + tau2)
    beta = metafopy.core.linalg.invcalc(X, wi) @ (X.T @ (wi * y))
    RSS = float(numpy.sum(wi * (y - X @ beta) ** 2))
    if method == 'ML':
        return -0.5 * k * math.log(2 * math.pi) - 0.5 * float(numpy.sum(numpy.log(vi + tau2))) - 0.5 * RSS
    return -0.5 * (k - p) * math.log(2 * math.pi) - 0.5 * float(numpy.sum(numpy.log(vi + tau2))) - 0.5 * metafopy.core.linalg.logdet((X.T * wi) @ X) - 0.5 * RSS
