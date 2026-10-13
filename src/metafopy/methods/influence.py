"""Outlier and influential-case diagnostics (metafor's ``influence.rma.uni`` and relatives).

For each study ``i`` the model is refitted without it; the externally
standardised residual (``rstudent``), ``DFFITS``, Cook's distance, the
covariance ratio, ``DFBETAS``, the leave-one-out ``tau^2`` and ``QE`` follow
Viechtbauer & Cheung (2010), doi:10.1002/jrsm.11. Also provided are the
internally standardised residuals (:func:`rstandard`), and the shortcuts
:func:`rstudent`, :func:`cooks_distance` and :func:`dfbetas`.
See https://wviechtb.github.io/metafor/reference/influence.rma.uni.html.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import singledispatch
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

from metafopy._print import fmt_num
from metafopy.core.linalg import invcalc
from metafopy.methods._common import NAAction, check_na_action, pad_to_supplied, refit, tzero
from metafopy.methods.blup import format_r_frame
from metafopy.models.rma_uni import Btt, RMAResult, set_btt

__all__ = ['InfluenceResult', 'ResidualResult', 'cooks_distance', 'dfbetas', 'influence', 'rstandard', 'rstudent']

type FloatArray = NDArray[np.float64]

_EPS = float(np.finfo(float).eps)


@dataclass(frozen=True, slots=True)
class ResidualResult:
    """Residuals with standard errors and z-values (metafor's ``list.rma`` of ``rstandard``/``rstudent``)."""

    resid: FloatArray
    se: FloatArray
    z: FloatArray
    slab: tuple[str, ...]
    digits: int = 4

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({'resid': self.resid, 'se': self.se, 'z': self.z}, index=list(self.slab))

    def __str__(self) -> str:
        cols = [(n, [fmt_num(float(v), self.digits) for v in a]) for n, a in (('resid', self.resid), ('se', self.se), ('z', self.z))]
        return '\n'.join(['', *format_r_frame(cols, self.slab), ''])


@dataclass(frozen=True, slots=True)
class InfluenceResult:
    """Case-deletion diagnostics: one entry per study.

    ``inf`` holds the studies' influence table (``rstudent``, ``dffits``,
    ``cook_d``, ``cov_r``, ``tau2_del``, ``QE_del``, ``hat``, ``weight``),
    ``dfbs`` the ``DFBETAS`` (one column per coefficient) and ``is_infl`` marks
    the studies metafor flags with an asterisk (|DFFITS| > 3 sqrt(p/(k-p)),
    Cook's D beyond the median of its chi-square reference distribution,
    hat > 3p/k or any |DFBETAS| > 1).
    """

    rstudent: FloatArray
    dffits: FloatArray
    cook_d: FloatArray
    cov_r: FloatArray
    tau2_del: FloatArray
    QE_del: FloatArray
    hat: FloatArray
    weight: FloatArray
    dfbs: FloatArray
    is_infl: NDArray[np.bool_]
    slab: tuple[str, ...]
    coef_names: tuple[str, ...]
    tau2: float
    QE: float
    k: int
    p: int
    m: int
    digits: int = 4

    @property
    def inf(self) -> tuple[str, ...]:
        """``'*'`` for each influential study, ``''`` otherwise."""
        return tuple('*' if v else '' for v in self.is_infl)

    def to_frame(self) -> pd.DataFrame:
        """The influence table, one row per study."""
        return pd.DataFrame(
            {
                'rstudent': self.rstudent,
                'dffits': self.dffits,
                'cook.d': self.cook_d,
                'cov.r': self.cov_r,
                'tau2.del': self.tau2_del,
                'QE.del': self.QE_del,
                'hat': self.hat,
                'weight': self.weight,
                'inf': self.inf,
            },
            index=list(self.slab),
        )

    def dfbs_frame(self) -> pd.DataFrame:
        """The DFBETAS, one column per coefficient."""
        return pd.DataFrame(self.dfbs, columns=list(self.coef_names), index=list(self.slab))

    def summary(self, infonly: bool = False) -> str:
        return _format_influence(self, infonly)

    def __str__(self) -> str:
        return _format_influence(self, False)


def _format_influence(res: InfluenceResult, infonly: bool) -> str:
    sel = res.is_infl if infonly else np.ones(res.slab.__len__(), dtype=np.bool_)
    if not sel.any():
        return '$inf\n\n$dfbs\n' if res.p > 1 else ''
    slab = [s for s, keep in zip(res.slab, sel, strict=True) if keep]

    def col(name: str, vals: FloatArray) -> tuple[str, list[str]]:
        return name, [fmt_num(float(v), res.digits) for v, keep in zip(vals, sel, strict=True) if keep]

    inf = [m for m, keep in zip(res.inf, sel, strict=True) if keep]
    base = [
        col('rstudent', res.rstudent),
        col('dffits', res.dffits),
        col('cook.d', res.cook_d),
        col('cov.r', res.cov_r),
        col('tau2.del', res.tau2_del),
        col('QE.del', res.QE_del),
        col('hat', res.hat),
        col('weight', res.weight),
    ]
    if res.p == 1:
        return '\n'.join(['', *format_r_frame([*base, col('dfbs', res.dfbs[:, 0]), ('inf', inf)], slab), ''])
    dfbs = [col(name, res.dfbs[:, j]) for j, name in enumerate(res.coef_names)]
    first = format_r_frame([*base, ('inf', inf)], slab)
    second = format_r_frame(dfbs, slab)
    return '\n'.join(['$inf', '', *first, '', '$dfbs', '', *second, ''])


@singledispatch
def influence(model: Any, *args: Any, **kwargs: Any) -> InfluenceResult:
    """Case-deletion diagnostics for a fitted model; see :func:`_influence_rma_uni`."""
    raise TypeError(f'influence() not implemented for {type(model).__name__}')


@influence.register
def _influence_rma_uni(model: RMAResult, digits: int = 4, *, btt: Btt = None, na_action: NAAction = 'omit') -> InfluenceResult:
    """Leave-one-out diagnostics for an ``rma.uni`` fit (metafor's ``influence()``).

    ``btt`` (0-based coefficient indices, default all coefficients) selects the
    coefficients Cook's distance and the covariance ratio refer to. With
    ``na_action='exclude'`` or ``'pass'`` the result has one row per supplied
    study. Studies whose refit fails (or loses a coefficient) get NaN.
    """
    check_na_action(na_action)
    x = model
    if x.k == 1:
        raise ValueError('Stopped because k = 1.')
    core = _influence_core(x, btt)
    inf = core.inf
    if na_action == 'fail' and not x.data.keep.all():
        raise ValueError('Missing values in results.')
    slab = x.slab
    if na_action in ('exclude', 'pass'):
        (rs, df, ck, cr, td, qd, ht, wt, dfbs), slab = pad_to_supplied(
            x, inf['rstudent'], inf['dffits'], inf['cook.d'], inf['cov.r'], inf['tau2.del'], inf['QE.del'], inf['hat'], inf['weight'], core.dfbs
        )
        is_infl = _flag(x, df, ck, ht, dfbs, core.m)
    else:
        rs, df, ck, cr, td, qd, ht, wt, dfbs = (
            inf['rstudent'],
            inf['dffits'],
            inf['cook.d'],
            inf['cov.r'],
            inf['tau2.del'],
            inf['QE.del'],
            inf['hat'],
            inf['weight'],
            core.dfbs,
        )
        is_infl = _flag(x, df, ck, ht, dfbs, core.m)
    return InfluenceResult(
        rstudent=rs,
        dffits=df,
        cook_d=ck,
        cov_r=cr,
        tau2_del=td,
        QE_del=qd,
        hat=ht,
        weight=wt,
        dfbs=dfbs,
        is_infl=is_infl,
        slab=slab,
        coef_names=x.coef_names,
        tau2=x.tau2,
        QE=x.QE,
        k=x.k,
        p=x.p,
        m=core.m,
        digits=digits,
    )


def _flag(x: RMAResult, dffits: FloatArray, cook_d: FloatArray, hat: FloatArray, dfbs: FloatArray, m: int) -> NDArray[np.bool_]:
    with np.errstate(invalid='ignore'):
        return np.asarray(
            (np.abs(dffits) > 3 * np.sqrt(x.p / (x.k - x.p))) | (stats.chi2.cdf(cook_d, df=m) > 0.5) | (hat > 3 * x.p / x.k) | np.any(np.abs(dfbs) > 1, axis=1),
            dtype=np.bool_,
        )


@dataclass(frozen=True, slots=True)
class _Core:
    inf: dict[str, FloatArray]
    dfbs: FloatArray
    m: int
    resid: FloatArray
    seresid: FloatArray


def _influence_core(x: RMAResult, btt: Btt) -> _Core:
    k, p = x.k, x.p
    sub = np.array(set_btt(btt, p, False, x.coef_names), dtype=np.intp)
    m = len(sub)
    X = x.X
    pred_full = X @ x.beta
    svb = np.linalg.inv(x.vb[np.ix_(sub, sub)])
    H: FloatArray | None = None
    A: FloatArray | None = None
    stXAX: FloatArray | None = None
    stXX: FloatArray | None = None
    if x.weighted:
        if x.weights is not None:
            A = np.diag(x.weights)
            stXAX = invcalc(X, x.weights)
            H = X @ stXAX @ X.T @ A
    else:
        stXX = invcalc(X, np.ones(k))
        H = X @ stXX @ X.T
    hat = x.hatvalues()

    tau2_del = np.full(k, np.nan)
    delpred = np.full(k, np.nan)
    vdelpred = np.full(k, np.nan)
    s2w_del = np.full(k, np.nan)
    QE_del = np.full(k, np.nan)
    dffits = np.full(k, np.nan)
    dfbs = np.full((k, p), np.nan)
    cook_d = np.full(k, np.nan)
    cov_r = np.full(k, np.nan)

    all_idx = np.arange(k)
    for i in range(k):
        res = refit(x, all_idx != i)
        if res is None:
            continue
        tau2_del[i] = res.tau2
        QE_del[i] = res.QE
        Xi = X[i]
        delpred[i] = Xi @ res.beta
        vdelpred[i] = Xi @ res.vb @ Xi
        s2w_del[i] = res.s2w
        if x.weighted and x.weights is None:
            dffits[i] = (pred_full[i] - delpred[i]) / np.sqrt(res.s2w * hat[i] * (tau2_del[i] + x.vi[i]))
        else:
            assert H is not None
            d = np.diag(H @ np.diag(tau2_del[i] + x.vi) @ H.T)
            dffits[i] = (pred_full[i] - delpred[i]) / np.sqrt(res.s2w * d)[i]
        if x.weighted:
            if x.weights is None:
                vb_del = invcalc(X, 1.0 / (x.vi + tau2_del[i]))
            else:
                assert stXAX is not None
                assert A is not None
                vb_del = stXAX @ X.T @ A @ np.diag(x.vi + tau2_del[i]) @ A @ X @ stXAX
        else:
            assert stXX is not None
            vb_del = stXX @ X.T @ np.diag(x.vi + tau2_del[i]) @ X @ stXX
        dfb = x.beta - res.beta
        dfbs[i] = dfb / np.sqrt(res.s2w * np.diag(vb_del))
        dfb_s = dfb[sub]
        cook_d[i] = float(dfb_s @ svb @ dfb_s)
        cov_r[i] = np.linalg.det(res.vb[np.ix_(sub, sub)]) / np.linalg.det(x.vb[np.ix_(sub, sub)])

    resid = tzero(x.yi - delpred)
    seresid = np.sqrt(x.vi * s2w_del + vdelpred + tau2_del * s2w_del)
    stresid = resid / seresid
    inf = {
        'rstudent': stresid,
        'dffits': dffits,
        'cook.d': cook_d,
        'cov.r': cov_r,
        'tau2.del': tau2_del,
        'QE.del': QE_del,
        'hat': hat,
        'weight': x.weights_(),
    }
    return _Core(inf=inf, dfbs=dfbs, m=m, resid=resid, seresid=seresid)


@singledispatch
def rstudent(model: Any, *args: Any, **kwargs: Any) -> ResidualResult:
    """Externally standardised (studentised deleted) residuals; see :func:`_rstudent_rma_uni`."""
    raise TypeError(f'rstudent() not implemented for {type(model).__name__}')


@rstudent.register
def _rstudent_rma_uni(model: RMAResult, digits: int = 4, *, na_action: NAAction = 'omit') -> ResidualResult:
    """Deleted residuals ``y_i - x_i' beta_(-i)`` with their standard errors and z-values."""
    check_na_action(na_action)
    if model.k == 1:
        raise ValueError('Stopped because k = 1.')
    core = _influence_core(model, None)
    if na_action == 'fail' and not model.data.keep.all():
        raise ValueError('Missing values in results.')
    z = core.inf['rstudent']
    slab = model.slab
    if na_action in ('exclude', 'pass'):
        (resid, se, z), slab = pad_to_supplied(model, core.resid, core.seresid, z)
    else:
        resid, se = core.resid, core.seresid
    return ResidualResult(resid=resid, se=se, z=z, slab=slab, digits=digits)


@singledispatch
def cooks_distance(model: Any, *args: Any, **kwargs: Any) -> pd.Series:
    """Cook's distances; see :func:`_cooks_rma_uni`."""
    raise TypeError(f'cooks_distance() not implemented for {type(model).__name__}')


@cooks_distance.register
def _cooks_rma_uni(model: RMAResult, *, btt: Btt = None, na_action: NAAction = 'omit') -> pd.Series:
    """Cook's distance of each study (metafor's ``cooks.distance``), indexed by study label."""
    res = _influence_rma_uni(model, btt=btt, na_action=na_action)
    return pd.Series(res.cook_d, index=list(res.slab), name='cook.d')


@singledispatch
def dfbetas(model: Any, *args: Any, **kwargs: Any) -> pd.DataFrame:
    """DFBETAS; see :func:`_dfbetas_rma_uni`."""
    raise TypeError(f'dfbetas() not implemented for {type(model).__name__}')


@dfbetas.register
def _dfbetas_rma_uni(model: RMAResult, *, na_action: NAAction = 'omit') -> pd.DataFrame:
    """Standardised change in each coefficient when a study is left out (one column per coefficient)."""
    return _influence_rma_uni(model, na_action=na_action).dfbs_frame()


@singledispatch
def rstandard(model: Any, *args: Any, **kwargs: Any) -> ResidualResult:
    """Internally standardised residuals; see :func:`_rstandard_rma_uni`."""
    raise TypeError(f'rstandard() not implemented for {type(model).__name__}')


@rstandard.register
def _rstandard_rma_uni(
    model: RMAResult,
    digits: int = 4,
    type: Literal['marginal', 'conditional'] = 'marginal',
    *,
    na_action: NAAction = 'omit',
) -> ResidualResult:
    """Standardised residuals; ``type='marginal'`` uses the marginal residuals, ``'conditional'`` the conditional ones."""
    check_na_action(na_action)
    x = model
    if type not in ('marginal', 'conditional'):
        raise ValueError("type must be 'marginal' or 'conditional'")
    if type == 'conditional' and (x.weights is not None or not x.weighted):
        raise ValueError('Extraction of conditional residuals not available for models with non-standard weights.')
    k = x.k
    H = _hat_matrix(x)
    ImH = np.eye(k) - H
    if type == 'marginal':
        ei = tzero(x.yi - x.X @ x.beta)
        M = np.diag(x.vi + x.tau2)
        sei = np.sqrt(np.diag(ImH @ M @ ImH.T))
    else:
        li = x.tau2 / (x.tau2 + x.vi)
        pred = li * x.yi + (1 - li) * (x.X @ x.beta)
        ei = x.yi - pred
        sei = np.sqrt(x.vi**2 / (x.vi + x.tau2) * (1 - np.diag(H)))
    if na_action == 'fail' and not x.data.keep.all():
        raise ValueError('Missing values in results.')
    z = ei / sei
    slab = x.slab
    if na_action in ('exclude', 'pass'):
        (ei, sei, z), slab = pad_to_supplied(x, ei, sei, z)
    return ResidualResult(resid=ei, se=sei, z=z, slab=slab, digits=digits)


def _hat_matrix(x: RMAResult) -> FloatArray:
    """The full hat matrix (metafor's ``hatvalues(x, type="matrix")``)."""
    X = x.X
    if x.weighted:
        if x.weights is None:
            w = 1.0 / (x.vi + x.tau2)
            return X @ invcalc(X, w) @ (X.T * w)
        return X @ invcalc(X, x.weights) @ (X.T * x.weights)
    return X @ invcalc(X, np.ones(x.k)) @ X.T


def hatvalues(model: RMAResult, type: Literal['diagonal', 'matrix'] = 'diagonal') -> FloatArray:
    """Hat values (diagonal) or the full hat matrix of an ``rma.uni`` fit."""
    if type == 'diagonal':
        return model.hatvalues()
    if type == 'matrix':
        return _hat_matrix(model)
    raise ValueError("type must be 'diagonal' or 'matrix'")
