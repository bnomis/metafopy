"""Helpers shared by the diagnostics: refitting a model on a subset of its studies, padding results for NA rows."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import asdict, replace
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from metafopy._data import MetaData
from metafopy.models._tau2 import EstimationError
from metafopy.models.rma_uni import RMAResult, fit_rma_uni

type FloatArray = NDArray[np.float64]
type BoolArray = NDArray[np.bool_]
type NAAction = Literal['omit', 'exclude', 'pass', 'fail']

_EPS = float(np.finfo(float).eps)

# what a failed refit looks like to callers (metafor wraps the refit in try() and records NA)
REFIT_ERRORS = (EstimationError, ValueError, ArithmeticError, np.linalg.LinAlgError)


def check_na_action(na_action: str) -> None:
    if na_action not in ('omit', 'exclude', 'pass', 'fail'):
        raise ValueError(f'unknown na_action {na_action!r}')


def subset_metadata(res: RMAResult, idx: NDArray[np.intp] | BoolArray) -> MetaData:
    """The model inputs of ``res`` restricted to the studies in ``idx`` (positions among the ``k`` studies used)."""
    md = res.data
    sel = np.asarray(idx)
    keep = np.zeros_like(md.keep)
    keep[np.flatnonzero(md.keep)[sel]] = True
    return replace(
        md,
        yi=res.yi[sel],
        vi=res.vi[sel],
        X=res.X[sel],
        X_names=tuple(res.coef_names),
        slab=tuple(np.asarray(res.slab, dtype=object)[sel].tolist()),
        keep=keep,
        weights=None if res.weights is None else res.weights[sel],
        ni=None if md.ni is None else md.ni[sel],
        intercept=res.int_incl,
    )


def refit(
    res: RMAResult,
    idx: NDArray[np.intp] | BoolArray,
    *,
    skip_r2: bool = True,
    method: str | None = None,
) -> RMAResult | None:
    """Refit ``res`` on the studies in ``idx`` with the same settings, as metafor's diagnostics do.

    Returns ``None`` when the refit fails or drops a coefficient (R records ``NA``
    in both cases). The control settings, including ``tau2_max``, are those of the
    original fit.
    """
    md = subset_metadata(res, idx)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = fit_rma_uni(
                md,
                method=res.method if method is None else method,
                weighted=res.weighted,
                test=res.test,
                level=1 - res.level,
                tau2=res.tau2 if res.tau2_fix else None,
                control=asdict(res.control),
                _skip_r2=skip_r2,
            )
    except REFIT_ERRORS:
        return None
    if out.p != res.p:
        return None
    return out


def refit_arrays(
    res: RMAResult,
    *,
    yi: FloatArray | None = None,
    X: FloatArray | None = None,
    btt: Any = None,
    tau2: float | Literal['same'] | None = 'same',
    method: str | None = None,
    names: tuple[str, ...] | None = None,
    intercept: bool | None = None,
    skip_r2: bool = True,
) -> RMAResult | None:
    """Refit ``res`` with replacement ``yi`` and/or model matrix ``X`` (same studies, same settings); ``None`` if the refit fails.

    ``tau2='same'`` keeps ``res``'s fixed ``tau^2`` (if any), a number fixes ``tau^2`` at that value
    and ``None`` forces estimation.
    """
    md = replace(
        res.data,
        yi=res.yi if yi is None else yi,
        X=res.X if X is None else X,
        X_names=tuple(res.coef_names) if names is None else names,
        intercept=res.int_incl if intercept is None else intercept,
    )
    if tau2 == 'same':
        tau2_arg: float | None = res.tau2 if res.tau2_fix else None
    else:
        tau2_arg = tau2
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return fit_rma_uni(
                md,
                method=res.method if method is None else method,
                weighted=res.weighted,
                test=res.test,
                level=1 - res.level,
                btt=btt,
                tau2=tau2_arg,
                control=asdict(res.control),
                _skip_r2=skip_r2,
            )
    except REFIT_ERRORS:
        return None


def pad_to_supplied(res: RMAResult, *arrays: FloatArray) -> tuple[list[FloatArray], tuple[str, ...]]:
    """Pad per-study results to one row per supplied study (NaN where left out), as ``na.exclude``/``na.pass`` do."""
    keep = res.data.keep
    idx = np.flatnonzero(keep)
    out: list[FloatArray] = []
    for arr in arrays:
        full = np.full((keep.shape[0], *arr.shape[1:]), np.nan)
        full[idx] = arr
        out.append(full)
    labels = [str(i + 1) for i in range(keep.shape[0])]
    for pos, i in enumerate(idx):
        labels[i] = res.slab[pos]
    return out, tuple(labels)


def psort(lb: FloatArray, ub: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Order interval bounds elementwise (metafor's ``.psort``)."""
    with np.errstate(invalid='ignore'):
        flip = lb > ub
    return np.where(flip, ub, lb), np.where(flip, lb, ub)


def tzero(x: FloatArray) -> FloatArray:
    """Snap tiny residuals to zero as metafor does (``abs(x) < 100 * .Machine$double.eps``)."""
    out = x.copy()
    out[np.abs(out) < 100 * _EPS] = 0.0
    return out


def apply_each(values: FloatArray, transf: Callable[..., Any], targs: Any) -> FloatArray:
    """``sapply(values, transf[, targs])``."""
    vals = np.atleast_1d(np.asarray(values, dtype=np.float64))
    out = [transf(v) if targs is None else transf(v, targs) for v in vals]
    return np.array([float(np.asarray(o, dtype=np.float64).reshape(-1)[0]) for o in out], dtype=np.float64).reshape(vals.shape)
