"""Wald tests of model coefficients and likelihood-ratio tests between models."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from functools import singledispatch
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from metafopy._print import fmt_num, fmt_pval
from metafopy.core.linalg import drop_redundant
from metafopy.models._tau2 import FIXED_EFFECTS
from metafopy.models.rma_uni import Btt, RMAResult, coef_span, set_btt

__all__ = ['LRTResult', 'WaldResult', 'anova']

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class WaldResult:
    """Omnibus Wald test (``QM``) and, for linear combinations, individual tests."""

    QM: float
    QMdf: tuple[int, int | None]
    QMp: float
    btt: tuple[int, ...] | None = None
    Xb: FloatArray | None = None
    se: FloatArray | None = None
    zval: FloatArray | None = None
    pval: FloatArray | None = None

    def __str__(self) -> str:
        m, ddf = self.QMdf
        head = (
            f'Test of Moderators (coefficient{"s" if m > 1 else ""} {coef_span(self.btt)}):'
            if self.btt is not None
            else 'Omnibus Test of Hypotheses:'
        )
        p = fmt_pval(self.QMp)
        p = f'< {p[1:]}' if p.startswith('<') else f'= {p}'
        if ddf is None:
            body = f'QM(df = {m}) = {fmt_num(self.QM)}, p-val {p}'
        else:
            body = f'F(df1 = {m}, df2 = {ddf}) = {fmt_num(self.QM)}, p-val {p}'
        return f'{head}\n{body}'


@dataclass(frozen=True, slots=True)
class LRTResult:
    """Likelihood-ratio test of a full against a reduced model."""

    df_full: int
    df_reduced: int
    LRT: float
    pval: float
    fit_stats_full: tuple[float, float, float, float, float]
    fit_stats_reduced: tuple[float, float, float, float, float]
    QE_full: float
    QE_reduced: float
    tau2_full: float
    tau2_reduced: float
    R2: float

    def to_frame(self) -> pd.DataFrame:
        cols = ['df', 'AIC', 'BIC', 'AICc', 'logLik', 'LRT', 'pval', 'QE', 'tau^2', 'R^2']
        full = [
            self.df_full,
            self.fit_stats_full[2],
            self.fit_stats_full[3],
            self.fit_stats_full[4],
            self.fit_stats_full[0],
            math.nan,
            math.nan,
            self.QE_full,
            self.tau2_full,
            math.nan,
        ]
        red = [
            self.df_reduced,
            self.fit_stats_reduced[2],
            self.fit_stats_reduced[3],
            self.fit_stats_reduced[4],
            self.fit_stats_reduced[0],
            self.LRT,
            self.pval,
            self.QE_reduced,
            self.tau2_reduced,
            self.R2,
        ]
        return pd.DataFrame([full, red], index=['Full', 'Reduced'], columns=cols)

    def __str__(self) -> str:
        return self.to_frame().round(4).to_string()


@singledispatch
def anova(model: Any, *args: Any, **kwargs: Any) -> WaldResult | LRTResult:
    """Wald or likelihood-ratio tests for fitted models."""
    raise TypeError(f'anova() not implemented for {type(model).__name__}')


@anova.register
def _anova_rma_uni(
    model: RMAResult,
    model2: RMAResult | None = None,
    *,
    btt: Btt = None,
    X: ArrayLike | None = None,
    rhs: ArrayLike | float = 0.0,
) -> WaldResult | LRTResult:
    """Test coefficients of one model, or compare two nested models.

    ``anova(res, btt=...)`` tests a subset of coefficients (0-based indices);
    ``anova(res, X=L)`` tests linear combinations ``L beta = rhs``;
    ``anova(full, reduced)`` gives a likelihood-ratio test.
    """
    x = model
    tdist = x.ddf is not None
    if model2 is not None:
        return _lrt(x, model2)

    if X is not None:
        L = np.atleast_2d(np.asarray(X, dtype=np.float64))
        if L.shape[1] != x.p:
            raise ValueError(f'X must have {x.p} columns')
        m = L.shape[0]
        Xb = L @ x.beta - np.broadcast_to(np.asarray(rhs, dtype=np.float64), (m,))
        vXb = L @ x.vb @ L.T
        se = np.sqrt(np.diag(vXb))
        zval = Xb / se
        if tdist:
            assert x.ddf is not None
            pval = 2 * stats.t.sf(np.abs(zval), x.ddf)
        else:
            pval = 2 * stats.norm.sf(np.abs(zval))
        QM = QMp = math.nan
        if np.linalg.matrix_rank(L) == m:
            try:
                QM = float(Xb @ np.linalg.solve(vXb, Xb))
            except np.linalg.LinAlgError:
                QM = math.nan
            if tdist:
                assert x.ddf is not None
                QM /= m
                QMp = float(stats.f.sf(QM, m, x.ddf))
            else:
                QMp = float(stats.chi2.sf(QM, m))
        return WaldResult(
            QM=QM,
            QMdf=(m, x.ddf),
            QMp=QMp,
            Xb=Xb,
            se=se,
            zval=zval,
            pval=np.asarray(pval, dtype=np.float64),
        )

    idx = set_btt(btt, x.p, x.int_incl, x.coef_names) if btt is not None else x.btt
    sub = np.array(idx)
    m = len(idx)
    b = x.beta[sub]
    try:
        QM = float(b @ np.linalg.solve(x.vb[np.ix_(sub, sub)], b))
    except np.linalg.LinAlgError:
        QM = math.nan
    if tdist:
        assert x.ddf is not None
        QM /= m
        QMp = float(stats.f.sf(QM, m, x.ddf)) if x.ddf > 0 else math.nan
    else:
        QMp = float(stats.chi2.sf(QM, m))
    return WaldResult(QM=QM, QMdf=(m, x.ddf), QMp=QMp, btt=idx)


def _lrt(a: RMAResult, b: RMAResult) -> LRTResult:
    if a.parms == b.parms:
        raise ValueError('models do not differ in the number of parameters')
    full, red = (a, b) if a.parms > b.parms else (b, a)
    if full.k != red.k or not np.allclose(full.yi, red.yi):
        raise ValueError('models are not fitted to the same data')
    same_X = full.X.shape == red.X.shape and np.allclose(full.X, red.X)
    if full.method == 'REML' and not same_X:
        warnings.warn(
            'REML comparisons not meaningful for models with different fixed effects '
            "(refit both models with method='ML').",
            stacklevel=3,
        )
    use_reml = full.method == 'REML'
    fs_f = full.fit_stats_reml if use_reml else full.fit_stats_ml
    fs_r = red.fit_stats_reml if use_reml else red.fit_stats_ml
    LRT = max(0.0, fs_r.dev - fs_f.dev)
    df = full.parms - red.parms
    pval = float(stats.chi2.sf(LRT, df))

    if full.method in FIXED_EFFECTS and red.method in FIXED_EFFECTS:
        s2f = _sigma2(full)
        s2r = _sigma2(red)
        R2 = 100 * max(0.0, (s2r - s2f) / s2r)
    elif red.tau2 == 0:
        R2 = 0.0
    else:
        R2 = 100 * max(0.0, (red.tau2 - full.tau2) / red.tau2)

    return LRTResult(
        df_full=full.parms,
        df_reduced=red.parms,
        LRT=LRT,
        pval=pval,
        fit_stats_full=fs_f.as_tuple(),
        fit_stats_reduced=fs_r.as_tuple(),
        QE_full=full.QE,
        QE_reduced=red.QE,
        tau2_full=full.tau2,
        tau2_reduced=red.tau2,
        R2=R2,
    )


def _sigma2(res: RMAResult) -> float:
    """Residual variance of ``lm(yi ~ X, weights)`` as used for EE model R^2 in metafor."""
    n = res.k
    w = (1.0 / res.vi if res.weights is None else res.weights) if res.weighted else np.ones(n)
    Xi = np.column_stack([np.ones(n), res.X])
    Xi = Xi[:, ~drop_redundant(Xi)]
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(Xi * sw[:, None], res.yi * sw, rcond=None)
    e = res.yi - Xi @ coef
    return float(np.sum(w * e**2) / (n - Xi.shape[1]))
