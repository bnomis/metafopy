"""Equal-, fixed- and random/mixed-effects meta-analysis models (metafor's ``rma.uni``).

The model is ``y_i = x_i' beta + u_i + e_i`` with ``u_i ~ N(0, tau^2)`` and
``e_i ~ N(0, v_i)`` for known sampling variances ``v_i``. ``tau^2`` is
estimated by one of the estimators in :mod:`metafopy.models._tau2`; ``beta``
by weighted least squares given ``tau^2``. Inference follows Viechtbauer
(2010), doi:10.18637/jss.v036.i03, including the Knapp & Hartung (2003)
adjustment (doi:10.1002/sim.1482).
"""

from __future__ import annotations

import math
import re
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from metafopy._data import Column, MetaData, NAAction
from metafopy._print import fmt_num, fmt_pval, fmt_table, signif_stars
from metafopy.core.linalg import (
    drop_redundant,
    invcalc,
    is_intercept,
    is_nested,
    is_positive_definite,
    logdet,
)
from metafopy.models._tau2 import (
    ESTIMATORS,
    FIXED_EFFECTS,
    Control,
    EstimationError,
    estimate_tau2,
)

__all__ = ["Control", "FitStats", "RMAResult", "rma", "rma_uni"]

type FloatArray = NDArray[np.float64]
type Test = Literal["z", "t", "knha", "adhoc"]
type Btt = int | Sequence[int] | str | None

_DBL_EPS = np.finfo(float).eps


@dataclass(frozen=True, slots=True)
class FitStats:
    """Log-likelihood, deviance and information criteria (metafor's ``fit.stats``)."""

    ll: float
    dev: float
    AIC: float
    BIC: float
    AICc: float

    def as_tuple(self) -> tuple[float, float, float, float, float]:
        return (self.ll, self.dev, self.AIC, self.BIC, self.AICc)


@dataclass(frozen=True, slots=True, eq=False)
class RMAResult:
    """A fitted ``rma.uni`` model.

    Array attributes follow metafor's names with ``.`` replaced by ``_``
    (``ci_lb``, ``se_tau2``). ``btt`` holds 0-based coefficient indices.
    """

    beta: FloatArray
    se: FloatArray
    zval: FloatArray
    pval: FloatArray
    ci_lb: FloatArray
    ci_ub: FloatArray
    vb: FloatArray
    coef_names: tuple[str, ...]
    tau2: float
    se_tau2: float
    k: int
    p: int
    m: int
    btt: tuple[int, ...]
    QE: float
    QEp: float
    QM: float
    QMp: float
    QMdf: tuple[int, int | None]
    I2: float
    H2: float
    R2: float | None
    vt: float
    fit_stats_ml: FitStats
    fit_stats_reml: FitStats
    parms: int
    method: str
    test: Test
    level: float
    ddf: int | None
    s2w: float
    weighted: bool
    weights: FloatArray | None
    tau2_fix: bool
    int_only: bool
    int_incl: bool
    allvipos: bool
    yi: FloatArray
    vi: FloatArray
    X: FloatArray
    slab: tuple[str, ...]
    data: MetaData = field(repr=False)
    control: Control = field(repr=False)
    iterations: int = 0

    # -- accessors ---------------------------------------------------------

    def coef(self) -> pd.Series:
        """Estimated coefficients, indexed by name."""
        return pd.Series(self.beta, index=list(self.coef_names), name="estimate")

    def vcov(self) -> pd.DataFrame:
        """Variance-covariance matrix of the coefficients."""
        names = list(self.coef_names)
        return pd.DataFrame(self.vb, index=names, columns=names)

    def fitted(self) -> FloatArray:
        """Fitted values ``X beta``."""
        return self.X @ self.beta

    def residuals(self) -> FloatArray:
        """Raw residuals ``y - X beta``."""
        return self.yi - self.fitted()

    def logLik(self, REML: bool | None = None) -> float:
        """Log-likelihood; REML unless the model was fitted with ML (as in metafor)."""
        return self._stats(REML).ll

    def deviance(self, REML: bool | None = None) -> float:
        return self._stats(REML).dev

    def AIC(self, REML: bool | None = None) -> float:
        return self._stats(REML).AIC

    def BIC(self, REML: bool | None = None) -> float:
        return self._stats(REML).BIC

    def AICc(self, REML: bool | None = None) -> float:
        return self._stats(REML).AICc

    def _stats(self, REML: bool | None) -> FitStats:
        if REML is None:
            REML = self.method == "REML"
        return self.fit_stats_reml if REML else self.fit_stats_ml

    def fitstats(self, REML: bool | None = None) -> pd.Series:
        """``logLik``, ``deviance``, ``AIC``, ``BIC``, ``AICc`` as a Series."""
        st = self._stats(REML)
        return pd.Series(
            st.as_tuple(),
            index=["logLik", "deviance", "AIC", "BIC", "AICc"],
            name="REML" if (REML if REML is not None else self.method == "REML") else "ML",
        )

    def weights_(self) -> FloatArray:
        """Percentage weight of each estimate in the model fit (metafor ``weights()``)."""
        if self.weighted:
            w = 1.0 / (self.vi + self.tau2) if self.weights is None else self.weights
        else:
            w = np.full(self.k, 1.0 / self.k)
        return w / w.sum() * 100

    def hatvalues(self) -> FloatArray:
        """Diagonal of the hat matrix."""
        X = self.X
        if self.weighted:
            w = 1.0 / (self.vi + self.tau2) if self.weights is None else self.weights
            H = X @ invcalc(X, w) @ (X.T * w)
        else:
            H = X @ invcalc(X, np.ones(self.k)) @ X.T
        h = np.diag(H).copy()
        h[h > 1 - 10 * _DBL_EPS] = 1.0
        return h

    # -- delegated methods (defined in metafopy.methods) ----------------------

    def predict(self, *args: Any, **kwargs: Any) -> Any:
        from metafopy.methods.predict import predict

        return predict(self, *args, **kwargs)

    def confint(self, *args: Any, **kwargs: Any) -> Any:
        from metafopy.methods.confint import confint

        return confint(self, *args, **kwargs)

    def anova(self, *args: Any, **kwargs: Any) -> Any:
        from metafopy.methods.anova import anova

        return anova(self, *args, **kwargs)

    # -- printing ----------------------------------------------------------

    def summary(self, digits: int = 4) -> str:
        return _format(self, digits)

    def __str__(self) -> str:
        return _format(self, 4)

    def __repr__(self) -> str:
        return f"<RMAResult {self.method} k={self.k} p={self.p}>"


def _control(control: Control | Mapping[str, Any] | None) -> Control:
    if control is None:
        return Control()
    if isinstance(control, Control):
        return control
    names = {f.name for f in fields(Control)}
    kwargs: dict[str, Any] = {}
    for key, value in control.items():
        name = key.replace(".", "_")
        if name not in names:
            raise TypeError(f"unknown control argument {key!r}")
        kwargs[name] = value
    return Control(**kwargs)


def set_btt(btt: Btt, p: int, int_incl: bool, names: Sequence[str]) -> tuple[int, ...]:
    if btt is None:
        if p > 1:
            return tuple(range(1, p)) if int_incl else tuple(range(p))
        return (0,)
    if isinstance(btt, str):
        found = tuple(i for i, n in enumerate(names) if re.search(btt, n))
        if not found:
            raise ValueError("cannot identify coefficient(s) matching the 'btt' pattern")
        return found
    idx = [btt] if isinstance(btt, int) else list(btt)
    out = sorted({int(i) for i in idx})
    if any(i < 0 or i >= p for i in out):
        raise ValueError(f"'btt' indices must lie in 0..{p - 1}")
    return tuple(out)


def level_to_alpha(level: float) -> float:
    """metafor's ``.level()``: 95 (or 0.95) -> 0.05."""
    return 1 - level / 100 if level > 1 else 1 - level


def _mad(x: FloatArray) -> float:
    med = float(np.median(x))
    return 1.4826 * float(np.median(np.abs(x - med)))


def rma_uni(
    yi: Column | None = None,
    vi: Column | None = None,
    *,
    sei: Column | None = None,
    weights: Column | None = None,
    ni: Column | None = None,
    mods: str | ArrayLike | pd.DataFrame | None = None,
    data: pd.DataFrame | None = None,
    slab: Column | Sequence[str] | None = None,
    subset: ArrayLike | str | None = None,
    intercept: bool = True,
    method: str | Sequence[str] = "REML",
    weighted: bool = True,
    test: Test = "z",
    level: float = 95,
    btt: Btt = None,
    tau2: float | None = None,
    control: Control | Mapping[str, Any] | None = None,
    na_action: NAAction = "omit",
) -> RMAResult:
    """Fit an equal-, fixed-, random- or mixed-effects meta-analytic model.

    Mirrors metafor's ``rma.uni()`` (alias ``rma``). ``method`` may be a single
    estimator or a sequence tried in order until one succeeds. ``btt`` takes
    0-based coefficient indices (or a regex matched against coefficient names);
    printed output reports them 1-based, as metafor does.

    Examples
    --------
    >>> from metafopy import datasets, rma
    >>> dat = datasets.load("dat.bcg")  # doctest: +SKIP
    """
    md = MetaData.from_inputs(
        yi,
        vi,
        sei=sei,
        weights=weights,
        ni=ni,
        mods=mods,
        intercept=intercept,
        data=data,
        slab=slab,
        subset=subset,
        na_action=na_action,
    )
    return fit_rma_uni(
        md,
        method=method,
        weighted=weighted,
        test=test,
        level=level,
        btt=btt,
        tau2=tau2,
        control=control,
    )


rma = rma_uni


def fit_rma_uni(
    md: MetaData,
    *,
    method: str | Sequence[str] = "REML",
    weighted: bool = True,
    test: Test = "z",
    level: float = 95,
    btt: Btt = None,
    tau2: float | None = None,
    control: Control | Mapping[str, Any] | None = None,
    _skip_r2: bool = False,
) -> RMAResult:
    """Fit ``rma.uni`` to already normalised inputs."""
    methods = [method] if isinstance(method, str) else list(method)
    for m_ in methods:
        if m_ not in ESTIMATORS:
            raise ValueError(f"unknown tau^2 estimator {m_!r}")
    if test not in ("z", "t", "knha", "adhoc"):
        raise ValueError(f"unknown test {test!r}")

    yi = md.yi.copy()
    vi = md.vi.copy()
    X = md.X.copy()
    names = list(md.X_names)
    k = md.k
    user_w = md.weights
    if k < 1:
        raise ValueError("processing terminated since k = 0")

    if np.any(vi <= 0):
        allvipos = False
        if np.any(vi < 0):
            warnings.warn("Negative sampling variances constrained to 0.", stacklevel=2)
            vi[vi < 0] = 0.0
    else:
        allvipos = True

    if k == 1 and test != "z":
        warnings.warn('Setting argument test="z" since k=1.', stacklevel=2)
        test = "z"

    drop = drop_redundant(X)
    if drop.any():
        warnings.warn("Redundant predictors dropped from the model.", stacklevel=2)
        X = X[:, ~drop]
        names = [n for n, d in zip(names, drop, strict=True) if not d]

    is_int = np.array([is_intercept(X[:, j]) for j in range(X.shape[1])])
    int_incl = bool(is_int.any())
    if int_incl:
        X = np.column_stack([np.ones(k), X[:, ~is_int]])
        names = ["intrcpt", *[n for n, i in zip(names, is_int, strict=True) if not i]]
    p = X.shape[1]
    int_only = p == 1 and is_intercept(X[:, 0])

    tau2_fix = tau2 is not None and not math.isnan(tau2) and methods[0] not in FIXED_EFFECTS
    if not (int_only and k == 1):
        need = p if (methods[0] in FIXED_EFFECTS or tau2_fix) else p + 1
        if need > k:
            raise ValueError(
                "Number of parameters to be estimated is larger than the number of observations."
            )

    btt_idx = set_btt(btt, p, int_incl, names)
    m = len(btt_idx)

    con = _control(control)
    if control is None or "tau2_max" not in {
        key.replace(".", "_") for key in (control if isinstance(control, Mapping) else {})
    }:
        con = replace(con, tau2_max=max(con.tau2_max, 10 * _mad(yi) ** 2))

    if not is_positive_definite(X.T @ X, tol=con.evtol):
        raise ValueError("Model matrix not of full rank. Cannot fit model.")

    # -- heterogeneity ---------------------------------------------------
    used = methods[0]
    tau2_hat = 0.0
    se_tau2 = math.nan
    iterations = 0
    if used not in FIXED_EFFECTS:
        errors: list[str] = []
        for cand in methods:
            used = cand
            if cand in FIXED_EFFECTS:
                break
            try:
                fit = estimate_tau2(
                    cand,
                    yi,
                    vi,
                    X,
                    weights=user_w,
                    tau2_fixed=float(tau2) if tau2_fix else None,  # type: ignore[arg-type]
                    allvipos=allvipos,
                    con=con,
                )
            except EstimationError as exc:
                errors.append(str(exc))
                continue
            tau2_hat, se_tau2, iterations = fit.tau2, fit.se_tau2, fit.iterations
            break
        else:
            raise EstimationError(errors[-1])
    if used in FIXED_EFFECTS:
        tau2_hat = 0.0
        se_tau2 = math.nan

    # -- fixed effects -----------------------------------------------------
    alpha = level_to_alpha(level)
    wi = 1.0 / (vi + tau2_hat)
    beta, vb, RSS_f, s2w = _fixed_effects(yi, vi, X, tau2_hat, user_w, weighted, test)

    tdist = test in ("knha", "adhoc", "t")
    ddf = k - p if tdist else None

    sub = np.array(btt_idx)
    try:
        vsub = vb[np.ix_(sub, sub)]
        L = np.linalg.cholesky(vsub)
        b = beta[sub]
        z = np.linalg.solve(L, b)
        QM = float(z @ z)
    except np.linalg.LinAlgError:
        QM = math.nan

    se = np.sqrt(np.diag(vb))
    zval = beta / se
    if tdist:
        assert ddf is not None
        QM = QM / m
        QMdf: tuple[int, int | None] = (m, ddf)
        QMp = float(stats.f.sf(QM, m, ddf)) if ddf > 0 else math.nan
        pval = 2 * stats.t.sf(np.abs(zval), ddf) if ddf > 0 else np.full(p, math.nan)
        crit = float(stats.t.isf(alpha / 2, ddf)) if ddf > 0 else math.nan
    else:
        QMdf = (m, None)
        QMp = float(stats.chi2.sf(QM, m))
        pval = 2 * stats.norm.sf(np.abs(zval))
        crit = float(stats.norm.isf(alpha / 2))
    ci_lb = beta - crit * se
    ci_ub = beta + crit * se

    # -- heterogeneity statistics ----------------------------------------------
    QE = QEp = I2 = H2 = vt = math.nan
    if allvipos:
        if k > p:
            w_fe = 1.0 / vi
            WX = w_fe[:, None] * X
            P = np.diag(w_fe) - WX @ invcalc(X, w_fe) @ WX.T
            ymc = yi - yi.mean()
            QE = max(0.0, float(ymc @ P @ ymc))
            QEp = float(stats.chi2.sf(QE, k - p))
            vt = (k - p) / float(np.trace(P))
            if used in FIXED_EFFECTS:
                I2 = max(0.0, 100 * (QE - (k - p)) / QE)
                H2 = QE / (k - p)
            else:
                I2 = 100 * tau2_hat / (vt + tau2_hat)
                H2 = tau2_hat / vt + 1
        else:
            QE, QEp, I2, H2, vt = 0.0, 1.0, 0.0, 1.0, 0.0

    # -- fit statistics -----------------------------------------------------
    parms = p + (0 if (used in FIXED_EFFECTS or tau2_fix) else 1)
    lv = np.log(vi + tau2_hat)
    ll_ml = -0.5 * k * math.log(2 * math.pi) - 0.5 * float(lv.sum()) - 0.5 * RSS_f
    ll_reml = (
        -0.5 * (k - p) * math.log(2 * math.pi)
        + (0.5 * logdet(X.T @ X) if con.REMLf else 0.0)
        - 0.5 * float(lv.sum())
        - 0.5 * logdet((X.T * wi) @ X)
        - 0.5 * RSS_f
    )
    if k > p:
        if allvipos:
            dev_ml = -2 * (ll_ml - float(np.sum(-0.5 * np.log(2 * math.pi * vi))))
        else:
            dev_ml = -2 * ll_ml
    else:
        dev_ml = 0.0
    fs_ml = FitStats(
        ll=ll_ml,
        dev=dev_ml,
        AIC=-2 * ll_ml + 2 * parms,
        BIC=-2 * ll_ml + parms * math.log(k),
        AICc=_aicc(ll_ml, parms, k),
    )
    fs_reml = FitStats(
        ll=ll_reml,
        dev=-2 * ll_reml,
        AIC=-2 * ll_reml + 2 * parms,
        BIC=-2 * ll_reml + parms * math.log(k - p) if k > p else math.nan,
        AICc=_aicc(ll_reml, parms, k - p),
    )

    # -- pseudo R^2 ----------------------------------------------------------
    R2: float | None = None
    if not int_only and is_nested(X, np.ones((k, 1))) and not _skip_r2:
        R2 = _pseudo_r2(
            md,
            yi=yi,
            X=X,
            wi=wi,
            user_w=user_w,
            weighted=weighted,
            used=used,
            methods=methods,
            test=test,
            con=con,
            tau2_hat=tau2_hat,
        )

    return RMAResult(
        beta=beta,
        se=se,
        zval=zval,
        pval=np.asarray(pval, dtype=np.float64),
        ci_lb=ci_lb,
        ci_ub=ci_ub,
        vb=vb,
        coef_names=tuple(names),
        tau2=tau2_hat,
        se_tau2=se_tau2,
        k=k,
        p=p,
        m=m,
        btt=btt_idx,
        QE=QE,
        QEp=QEp,
        QM=QM,
        QMp=QMp,
        QMdf=QMdf,
        I2=I2,
        H2=H2,
        R2=R2,
        vt=vt,
        fit_stats_ml=fs_ml,
        fit_stats_reml=fs_reml,
        parms=parms,
        method=used,
        test=test,
        level=alpha,
        ddf=ddf,
        s2w=s2w,
        weighted=weighted,
        weights=user_w,
        tau2_fix=tau2_fix,
        int_only=int_only,
        int_incl=int_incl,
        allvipos=allvipos,
        yi=yi,
        vi=vi,
        X=X,
        slab=md.slab,
        data=md,
        control=con,
        iterations=iterations,
    )


def _fixed_effects(
    yi: FloatArray,
    vi: FloatArray,
    X: FloatArray,
    tau2: float,
    user_w: FloatArray | None,
    weighted: bool,
    test: Test,
) -> tuple[FloatArray, FloatArray, float, float]:
    """``beta``, its (Knapp-Hartung scaled) covariance, ``RSS_f`` and ``s2w``.

    Follows metafor: with user weights the estimate uses them but RSS (and the
    Knapp-Hartung factor) still use inverse-variance weights.
    """
    k, p = X.shape
    wi = 1.0 / (vi + tau2)
    knha = test in ("knha", "adhoc")
    s2w = 1.0
    if weighted:
        if np.any(np.isinf(wi)) and (user_w is None or knha):
            raise ValueError("Division by zero when computing the inverse variance weights.")
        vb = invcalc(X, wi)
        beta = vb @ (X.T @ (wi * yi))
        RSS_knha = float(np.sum(wi * (yi - X @ beta) ** 2))
        if user_w is not None:
            stXAX = invcalc(X, user_w)
            beta = stXAX @ (X.T @ (user_w * yi))
            XA = X.T * user_w
            vb = stXAX @ XA @ np.diag(vi + tau2) @ XA.T @ stXAX
        RSS_f = float(np.sum(wi * (yi - X @ beta) ** 2))
    else:
        stXX = invcalc(X, np.ones(k))
        beta = stXX @ (X.T @ yi)
        vb = stXX @ X.T @ np.diag(vi + tau2) @ X @ stXX
        RSS_f = float(np.sum(wi * (yi - X @ beta) ** 2))
        RSS_knha = math.nan
        if knha:
            if np.any(np.isinf(wi)):
                raise ValueError("Division by zero when computing the inverse variance weights.")
            b_k = invcalc(X, wi) @ (X.T @ (wi * yi))
            RSS_knha = float(np.sum(wi * (yi - X @ b_k) ** 2))
    if knha:
        s2w = 0.0 if RSS_knha <= _DBL_EPS else RSS_knha / (k - p)
    if test == "adhoc":
        s2w = max(s2w, 1.0)
    return beta, s2w * vb, RSS_f, s2w


def _aicc(ll: float, parms: int, n: int) -> float:
    big = max(n, parms + 2)
    return -2 * ll + 2 * parms * big / (big - parms - 1)


def _pseudo_r2(
    md: MetaData,
    *,
    yi: FloatArray,
    X: FloatArray,
    wi: FloatArray,
    user_w: FloatArray | None,
    weighted: bool,
    used: str,
    methods: list[str],
    test: Test,
    con: Control,
    tau2_hat: float,
) -> float | None:
    k = yi.shape[0]
    if used in FIXED_EFFECTS:
        if np.var(yi) == 0:
            return 0.0
        w = (wi if user_w is None else user_w) if weighted else np.ones(k)
        return max(0.0, 100 * _adj_r2(yi, X, w))
    md0 = replace(md, yi=yi, X=np.ones((k, 1)), X_names=("intrcpt",), intercept=True)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res0 = fit_rma_uni(
                md0,
                method=methods,
                weighted=weighted,
                test=test,
                control=con,
                _skip_r2=True,
            )
    except EstimationError, ValueError:
        return math.nan
    tau2_re = res0.tau2
    if tau2_re == 0:
        return 0.0
    return max(0.0, 100 * (tau2_re - tau2_hat) / tau2_re)


def _adj_r2(y: FloatArray, X: FloatArray, w: FloatArray) -> float:
    """Adjusted R^2 of ``lm(y ~ X, weights=w)`` (with R's added intercept)."""
    n = y.shape[0]
    Xi = np.column_stack([np.ones(n), X])
    keep = ~drop_redundant(Xi)
    Xi = Xi[:, keep]
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(Xi * sw[:, None], y * sw, rcond=None)
    resid = y - Xi @ coef
    rss = float(np.sum(w * resid**2))
    mean = float(np.sum(w * y) / np.sum(w))
    tss = float(np.sum(w * (y - mean) ** 2))
    r2 = 1 - rss / tss
    rdf = n - Xi.shape[1]
    return float(1 - (1 - r2) * ((n - 1) / rdf))


# -- printing -----------------------------------------------------------------


def _format(res: RMAResult, digits: int) -> str:
    out: list[str] = [""]
    fe = res.method in FIXED_EFFECTS
    if fe:
        kind = (
            "Equal-Effects"
            if res.method == "EE"
            else ("Fixed-Effects" if res.method == "FE" else "Common-Effects")
        )
        out.append(f"{kind} Model (k = {res.k})")
    else:
        kind = "Random-Effects" if res.int_only else "Mixed-Effects"
        out.append(f"{kind} Model (k = {res.k}; tau^2 estimator: {res.method})")
    out.append("")

    if not fe:
        het = "total" if res.int_only else "residual"
        tau2_line = fmt_num(res.tau2, digits)
        if not math.isnan(res.se_tau2):
            tau2_line += f" (SE = {fmt_num(res.se_tau2, digits)})"
        rows = [
            (f"tau^2 (estimated amount of {het} heterogeneity):", tau2_line),
            ("tau (square root of estimated tau^2 value):", fmt_num(math.sqrt(res.tau2), digits)),
        ]
        if not math.isnan(res.I2):
            if res.int_only:
                rows += [
                    ("I^2 (total heterogeneity / total variability):", f"{res.I2:.2f}%"),
                    ("H^2 (total variability / sampling variability):", f"{res.H2:.2f}"),
                ]
            else:
                rows += [
                    ("I^2 (residual heterogeneity / unaccounted variability):", f"{res.I2:.2f}%"),
                    ("H^2 (unaccounted variability / sampling variability):", f"{res.H2:.2f}"),
                ]
        if res.R2 is not None:
            rows.append(("R^2 (amount of heterogeneity accounted for):", f"{res.R2:.2f}%"))
        width = max(len(r[0]) for r in rows)
        out += [f"{label.ljust(width)} {value}" for label, value in rows]
        out.append("")
    elif not math.isnan(res.I2):
        rows = [
            ("I^2 (total heterogeneity / total variability):", f"{res.I2:.2f}%"),
            ("H^2 (total variability / sampling variability):", f"{res.H2:.2f}"),
        ]
        width = max(len(r[0]) for r in rows)
        out += [f"{label.ljust(width)} {value}" for label, value in rows]
        out.append("")

    if not math.isnan(res.QE):
        name, label = ("Q", "Heterogeneity") if res.int_only else ("QE", "Residual Heterogeneity")
        out.append(f"Test for {label}:")
        out.append(
            f"{name}(df = {res.k - res.p}) = {fmt_num(res.QE, digits)}, "
            f"p-val {_pfmt(res.QEp, digits)}"
        )
        out.append("")

    if not res.int_only:
        coefs = coef_span(res.btt)
        out.append(f"Test of Moderators (coefficient{'s' if res.m > 1 else ''} {coefs}):")
        if res.ddf is not None:
            out.append(
                f"F(df1 = {res.m}, df2 = {res.ddf}) = {fmt_num(res.QM, digits)}, "
                f"p-val {_pfmt(res.QMp, digits)}"
            )
        else:
            out.append(
                f"QM(df = {res.m}) = {fmt_num(res.QM, digits)}, p-val {_pfmt(res.QMp, digits)}"
            )
        out.append("")

    out.append("Model Results:")
    out.append("")
    stat = "tval" if res.ddf is not None else "zval"
    header = ["estimate", "se", stat]
    if res.ddf is not None:
        header.append("df")
    header += ["pval", "ci.lb", "ci.ub", ""]
    rows_: list[list[str]] = []
    for j in range(res.p):
        row = [
            fmt_num(res.beta[j], digits),
            fmt_num(res.se[j], digits),
            fmt_num(res.zval[j], digits),
        ]
        if res.ddf is not None:
            row.append(str(res.ddf))
        row += [
            fmt_pval(res.pval[j], digits),
            fmt_num(res.ci_lb[j], digits),
            fmt_num(res.ci_ub[j], digits),
            signif_stars(res.pval[j]),
        ]
        rows_.append(row)
    out.append(fmt_table(header, rows_, row_names=None if res.int_only else list(res.coef_names)))
    out.append("")
    out.append("---")
    out.append("Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
    out.append("")
    return "\n".join(out)


def _pfmt(p: float, digits: int) -> str:
    s = fmt_pval(p, digits)
    return f"< {s[1:]}" if s.startswith("<") else f"= {s}"


def coef_span(idx: Sequence[int]) -> str:
    one = [i + 1 for i in idx]
    if len(one) > 1 and one == list(range(one[0], one[-1] + 1)):
        return f"{one[0]}:{one[-1]}"
    return ",".join(str(i) for i in one)
