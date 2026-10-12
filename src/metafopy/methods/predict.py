"""Predicted values, confidence and prediction intervals (metafor's ``predict.rma``)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import singledispatch
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from metafopy.models._tau2 import FIXED_EFFECTS
from metafopy.models.rma_uni import RMAResult, level_to_alpha
from metafopy.transf import apply_transf, is_tau2_transf

__all__ = ['PredictResult', 'predict']

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class PredictResult:
    """Predictions with confidence (``ci``) and prediction (``pi``) intervals.

    ``se`` is NaN when a transformation was applied, as in metafor.
    """

    pred: FloatArray
    se: FloatArray
    ci_lb: FloatArray
    ci_ub: FloatArray
    pi_lb: FloatArray
    pi_ub: FloatArray
    slab: tuple[str, ...] | None = None

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                'pred': self.pred,
                'se': self.se,
                'ci.lb': self.ci_lb,
                'ci.ub': self.ci_ub,
                'pi.lb': self.pi_lb,
                'pi.ub': self.pi_ub,
            },
            index=list(self.slab) if self.slab is not None else None,
        )

    def __str__(self) -> str:
        frame = self.to_frame()
        for col in ('se', 'pi.lb', 'pi.ub'):
            if frame[col].isna().all():
                frame = frame.drop(columns=col)
        return frame.round(4).to_string()


@singledispatch
def predict(model: Any, *args: Any, **kwargs: Any) -> PredictResult:
    """Predicted values for a fitted model; see :func:`_predict_rma_uni`."""
    raise TypeError(f'predict() not implemented for {type(model).__name__}')


@predict.register
def _predict_rma_uni(
    model: RMAResult,
    newmods: ArrayLike | None = None,
    *,
    intercept: bool | None = None,
    transf: Callable[..., Any] | None = None,
    targs: Any = None,
    level: float | None = None,
) -> PredictResult:
    """Predict from an ``rma.uni`` fit.

    Without ``newmods`` an intercept-only model gives the pooled estimate and a
    meta-regression gives the fitted value of each study. ``newmods`` holds
    moderator values without the intercept column (it is added unless
    ``intercept=False``) or the full set of ``p`` columns. ``transf`` is
    applied to each value (with ``targs`` if given; the integral and mode
    transformations in :mod:`metafopy.transf` default to ``targs={'tau2': tau2}``).
    """
    x = model
    p = x.p
    slab: tuple[str, ...] | None = None
    if newmods is None:
        X_new = np.ones((1, 1)) if x.int_only else x.X
        if not x.int_only:
            slab = x.slab
    else:
        arr = np.asarray(newmods, dtype=np.float64)
        n_mods = p - 1 if x.int_incl else p
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        elif arr.ndim == 1:
            arr = arr[:, None] if n_mods == 1 else arr[None, :]
        if x.int_incl and arr.shape[1] == p - 1:
            col = 1.0 if intercept is None or intercept else 0.0
            arr = np.column_stack([np.full(arr.shape[0], col), arr])
        if arr.shape[1] != p:
            raise ValueError(f'dimensions of newmods ({arr.shape[1]} columns) do not match the model ({p})')
        X_new = arr

    alpha = x.level if level is None else level_to_alpha(level)

    pred = X_new @ x.beta
    vpred = np.einsum('ij,jk,ik->i', X_new, x.vb, X_new)
    vpred = np.where(vpred < 0, np.nan, vpred)
    se = np.sqrt(vpred)
    if x.ddf is not None:
        crit = float(stats.t.isf(alpha / 2, x.ddf)) if x.ddf > 0 else np.nan
    else:
        crit = float(stats.norm.isf(alpha / 2))
    ci_lb = pred - crit * se
    ci_ub = pred + crit * se
    pi_se = np.sqrt(vpred + x.tau2)
    pi_lb = pred - crit * pi_se
    pi_ub = pred + crit * pi_se
    if x.method in FIXED_EFFECTS:
        # metafor drops prediction intervals for equal/fixed/common-effects models
        pi_lb = np.full_like(pred, np.nan)
        pi_ub = np.full_like(pred, np.nan)

    if transf is not None:
        pred, ci_lb, ci_ub, pi_lb, pi_ub = (apply_transf(a, transf, targs, x.tau2) for a in (pred, ci_lb, ci_ub, pi_lb, pi_ub))
        se = np.full_like(pred, np.nan)
        ci_lb, ci_ub = np.minimum(ci_lb, ci_ub), np.maximum(ci_lb, ci_ub)
        pi_lb, pi_ub = np.minimum(pi_lb, pi_ub), np.maximum(pi_lb, pi_ub)
        if is_tau2_transf(transf):
            # metafor drops the prediction interval after an integral/mode transformation
            pi_lb = np.full_like(pred, np.nan)
            pi_ub = np.full_like(pred, np.nan)

    return PredictResult(pred=pred, se=se, ci_lb=ci_lb, ci_ub=ci_ub, pi_lb=pi_lb, pi_ub=pi_ub, slab=slab)
