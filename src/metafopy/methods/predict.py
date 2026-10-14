"""Predicted values, confidence and prediction intervals (metafor's ``predict.rma``)."""

import collections.abc
import dataclasses
import functools
import math
import typing
import warnings

import numpy
import numpy.typing
import pandas
import scipy.stats

import metafopy.errors
import metafopy.models.glmm_result
import metafopy.models.mv_result
import metafopy.models.rma_uni
import metafopy.models.tau2
import metafopy.transf

__all__ = ['PredictResult', 'predict']

FloatArray = numpy.typing.NDArray[numpy.float64]


@dataclasses.dataclass(frozen=True, slots=True)
class PredictResult:
    """Predictions with confidence (``ci``) and prediction (``pi``) intervals.

    ``se`` is NaN when a transformation was applied, as in metafor. ``pi_se`` is the standard error of the
    prediction (before any transformation), ``pi_dist`` the distribution (``'norm'`` or ``'t'``) of the prediction
    and ``pi_ddf`` its degrees of freedom (one per prediction) (used by ``forest(predstyle='shade')`` and ``'dist'``).
    ``prob`` holds the probabilities requested via ``prob``, ``X`` the model matrix of the predictions (``addx=True``)
    and ``vcov`` their variance-covariance matrix (``vcov=True`` without ``transf``).
    """

    pred: FloatArray
    se: FloatArray
    ci_lb: FloatArray
    ci_ub: FloatArray
    pi_lb: FloatArray
    pi_ub: FloatArray
    slab: tuple[str, ...] | None = None
    pi_se: FloatArray | None = None
    pi_dist: str | None = None
    pi_ddf: float | FloatArray | None = None
    prob: FloatArray | None = None
    X: FloatArray | None = None
    vcov: FloatArray | None = None

    def to_frame(self) -> pandas.DataFrame:
        cols: dict[str, typing.Any] = {
            'pred': self.pred,
            'se': self.se,
            'ci.lb': self.ci_lb,
            'ci.ub': self.ci_ub,
            'pi.lb': self.pi_lb,
            'pi.ub': self.pi_ub,
        }
        if self.prob is not None:
            cols['prob'] = self.prob
        return pandas.DataFrame(cols, index=list(self.slab) if self.slab is not None else None)

    def __str__(self) -> str:
        frame = self.to_frame()
        for col in ('se', 'pi.lb', 'pi.ub'):
            if frame[col].isna().all():
                frame = frame.drop(columns=col)
        return frame.round(4).to_string()


def prob_values(prob: str | float, pred: FloatArray, pi_se: FloatArray | None, pi_dist: str, pi_ddf: float | FloatArray | None) -> FloatArray | None:
    """Probabilities that a true outcome is below/above a value (``prob='>0.2'``, ``'<0'``; a number means ``'>'``).

    ``None`` when the value cannot be parsed (metafor then drops ``prob``).
    """
    text = str(prob)
    lower_tail = '<' in text
    try:
        value = float(text.replace('<', '').replace('>', '').replace('=', ''))
    except ValueError:
        return None
    if pi_se is None:
        return numpy.full(pred.shape, math.nan)
    z = (value - pred) / pi_se
    if pi_dist == 'norm':
        return numpy.asarray(scipy.stats.norm.cdf(z) if lower_tail else scipy.stats.norm.sf(z), dtype=numpy.float64)
    df = numpy.asarray(math.nan if pi_ddf is None else pi_ddf, dtype=numpy.float64)
    return numpy.asarray(scipy.stats.t.cdf(z, df) if lower_tail else scipy.stats.t.sf(z, df), dtype=numpy.float64)


def expand_hetvar(hetvar: numpy.typing.ArrayLike, k_new: int) -> FloatArray:
    """``hetvar`` recycled to ``k_new`` values (metafor's ``.expand1``).

    Raises:
        metafopy.errors.InputError: the length of ``hetvar`` does not match.
    """
    try:
        hv = numpy.atleast_1d(numpy.asarray(hetvar, dtype=numpy.float64))
    except (TypeError, ValueError) as err:
        raise metafopy.errors.InputError("Argument 'hetvar' must be a numeric vector.") from err
    if hv.size == 1:
        hv = numpy.full(k_new, float(hv[0]))
    if hv.size != k_new:
        raise metafopy.errors.InputError(f"Length of the 'hetvar' argument ({hv.size}) does not match the number of predicted values ({k_new}).")
    return hv


def check_prob_adjust(prob: str | float | None, adjust: bool) -> None:
    if prob is not None and adjust:
        raise metafopy.errors.InputError("Cannot use 'prob' argument when adjust=True.")


@functools.singledispatch
def predict(model: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> PredictResult:
    """Predicted values for a fitted model; see :func:`_predict_rma_uni`.

    Raises:
        metafopy.errors.ArgumentError: an argument has the wrong type or is not accepted.
    """
    raise metafopy.errors.ArgumentError(f'predict() not implemented for {type(model).__name__}')


@predict.register
def predict_rma_uni(
    model: metafopy.models.rma_uni.RMAResult,
    newmods: numpy.typing.ArrayLike | None = None,
    *,
    intercept: bool | None = None,
    tau2_levels: collections.abc.Sequence[str | int] | str | int | None = None,
    gamma2_levels: collections.abc.Sequence[str | int] | str | int | None = None,
    hetvar: numpy.typing.ArrayLike | None = None,
    prob: str | float | None = None,
    addx: bool = False,
    transf: collections.abc.Callable[..., typing.Any] | None = None,
    targs: typing.Any = None,
    level: float | None = None,
    adjust: bool = False,
    vcov: bool = False,
) -> PredictResult:
    """Predict from an ``rma.uni`` fit.

    Without ``newmods`` an intercept-only model gives the pooled estimate and a
    meta-regression gives the fitted value of each study. ``newmods`` holds
    moderator values without the intercept column (it is added unless
    ``intercept=False``) or the full set of ``p`` columns. ``transf`` is
    applied to each value (with ``targs`` if given; the integral and mode
    transformations in :mod:`metafopy.transf` default to ``targs={'tau2': tau2}``).

    ``hetvar`` replaces ``tau2`` in the prediction interval; ``prob`` (such as
    ``'>0.2'`` or ``'<0'``) adds the probability that a true outcome is above or
    below the value; ``addx`` adds the model matrix; ``adjust`` applies a
    Bonferroni correction to the intervals; ``vcov`` adds the
    variance-covariance matrix of the predictions. ``tau2_levels`` and
    ``gamma2_levels`` are accepted, as in metafor, and have no effect.

    R source: https://github.com/wviechtb/metafor/blob/master/R/predict.rma.r;
    reference page: https://wviechtb.github.io/metafor/reference/predict.rma.html.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    x = model
    if x.robust is not None and x.robust.robumethod == 'clubSandwich':
        raise metafopy.errors.InputError('predict() is not supported for fits made robust with club_sandwich=True.')
    return location_prediction(
        LocationModel(
            k=x.k,
            p=x.p,
            int_only=x.int_only,
            int_incl=x.int_incl,
            X=x.X,
            slab=x.slab,
            beta=x.beta,
            vb=x.vb,
            ddf=x.ddf,
            alpha=x.level,
            tau2=x.tau2,
            fixed=x.method in metafopy.models.tau2.FIXED_EFFECTS,
        ),
        newmods,
        intercept,
        transf,
        targs,
        level,
        hetvar=hetvar,
        prob=prob,
        addx=addx,
        adjust=adjust,
        vcov=vcov,
    )


@predict.register
def predict_rma_glmm(
    model: metafopy.models.glmm_result.GLMMResult,
    newmods: numpy.typing.ArrayLike | None = None,
    *,
    intercept: bool | None = None,
    tau2_levels: collections.abc.Sequence[str | int] | str | int | None = None,
    gamma2_levels: collections.abc.Sequence[str | int] | str | int | None = None,
    hetvar: numpy.typing.ArrayLike | None = None,
    prob: str | float | None = None,
    addx: bool = False,
    transf: collections.abc.Callable[..., typing.Any] | None = None,
    targs: typing.Any = None,
    level: float | None = None,
    adjust: bool = False,
    vcov: bool = False,
) -> PredictResult:
    """Predict from an ``rma.glmm`` fit (``predict.rma``; arguments as in :func:`predict_rma_uni`).

    The coefficients are on the scale of the model's link (log odds ratios, log risk ratios, risk
    differences, log odds, proportions, log proportions or log rates). Without ``newmods`` a
    meta-regression gives the fitted value of each study that entered the model. Prediction
    intervals add ``tau2`` and are absent for the fixed-effects methods.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    x = model
    labels = tuple(s for s, ok in zip(x.slab, x.tables.not_na, strict=True) if ok)
    return location_prediction(
        LocationModel(
            k=x.k,
            p=x.p,
            int_only=x.int_only,
            int_incl=x.int_incl,
            X=x.X,
            slab=labels,
            beta=x.beta,
            vb=x.vb,
            ddf=x.ddf,
            alpha=x.level,
            tau2=x.tau2,
            fixed=x.method in metafopy.models.tau2.FIXED_EFFECTS,
        ),
        newmods,
        intercept,
        transf,
        targs,
        level,
        hetvar=hetvar,
        prob=prob,
        addx=addx,
        adjust=adjust,
        vcov=vcov,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class LocationModel:
    """What ``predict.rma`` needs from a model with a single ``tau2`` (``rma.uni`` or ``rma.glmm``)."""

    k: int
    p: int
    int_only: bool
    int_incl: bool
    X: FloatArray
    slab: tuple[str, ...]
    beta: FloatArray
    vb: FloatArray
    ddf: int | None
    alpha: float
    tau2: float
    fixed: bool


def location_prediction(
    x: LocationModel,
    newmods: numpy.typing.ArrayLike | None,
    intercept: bool | None,
    transf: collections.abc.Callable[..., typing.Any] | None,
    targs: typing.Any,
    level: float | None,
    *,
    hetvar: numpy.typing.ArrayLike | None = None,
    prob: str | float | None = None,
    addx: bool = False,
    adjust: bool = False,
    vcov: bool = False,
) -> PredictResult:
    """``predict.rma`` for ``rma.uni`` and ``rma.glmm`` models.

    Raises:
        metafopy.errors.InputError: ``newmods`` does not match the model.
    """
    p = x.p
    slab: tuple[str, ...] | None = None
    if newmods is None:
        X_new = numpy.ones((1, 1)) if x.int_only else x.X
        if not x.int_only:
            slab = x.slab
    else:
        arr = numpy.asarray(newmods, dtype=numpy.float64)
        n_mods = p - 1 if x.int_incl else p
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        elif arr.ndim == 1:
            arr = arr[:, None] if n_mods == 1 else arr[None, :]
        if x.int_incl and arr.shape[1] == p - 1:
            col = 1.0 if intercept is None or intercept else 0.0
            arr = numpy.column_stack([numpy.full(arr.shape[0], col), arr])
        if arr.shape[1] != p:
            raise metafopy.errors.InputError(f'dimensions of newmods ({arr.shape[1]} columns) do not match the model ({p})')
        X_new = arr

    check_prob_adjust(prob, adjust)
    alpha = x.alpha if level is None else metafopy.models.rma_uni.level_to_alpha(level)
    k_new = X_new.shape[0]
    if adjust:
        alpha = alpha / k_new

    pred = X_new @ x.beta
    vpred = numpy.einsum('ij,jk,ik->i', X_new, x.vb, X_new)
    vpred = numpy.where(vpred < 0, numpy.nan, vpred)
    se = numpy.sqrt(vpred)
    if x.ddf is not None:
        crit = float(scipy.stats.t.isf(alpha / 2, x.ddf)) if x.ddf > 0 else numpy.nan
    else:
        crit = float(scipy.stats.norm.isf(alpha / 2))
    ci_lb = pred - crit * se
    ci_ub = pred + crit * se
    het = x.tau2 if hetvar is None else expand_hetvar(hetvar, k_new)
    pi_se = numpy.sqrt(vpred + het)
    pi_ddf = numpy.full(pred.shape, float(x.k - p if x.ddf is None else x.ddf))
    pi_dist = 'norm' if x.ddf is None else 't'
    prob_out = None if prob is None else prob_values(prob, pred, pi_se, pi_dist, pi_ddf)
    vcov_pred = X_new @ x.vb @ X_new.T if vcov else None
    if vcov_pred is not None:
        vcov_pred = (vcov_pred + vcov_pred.T) / 2
    pi_lb = pred - crit * pi_se
    pi_ub = pred + crit * pi_se
    if x.fixed:
        # metafor drops prediction intervals for equal/fixed/common-effects models
        pi_lb = numpy.full_like(pred, numpy.nan)
        pi_ub = numpy.full_like(pred, numpy.nan)

    if transf is not None:
        pred, ci_lb, ci_ub, pi_lb, pi_ub = (metafopy.transf.apply_transf(a, transf, targs, x.tau2) for a in (pred, ci_lb, ci_ub, pi_lb, pi_ub))
        se = numpy.full_like(pred, numpy.nan)
        ci_lb, ci_ub = numpy.minimum(ci_lb, ci_ub), numpy.maximum(ci_lb, ci_ub)
        pi_lb, pi_ub = numpy.minimum(pi_lb, pi_ub), numpy.maximum(pi_lb, pi_ub)
        if metafopy.transf.is_tau2_transf(transf):
            # metafor drops the prediction interval after an integral/mode transformation
            pi_lb = numpy.full_like(pred, numpy.nan)
            pi_ub = numpy.full_like(pred, numpy.nan)

    return PredictResult(
        pred=pred,
        se=se,
        ci_lb=ci_lb,
        ci_ub=ci_ub,
        pi_lb=pi_lb,
        pi_ub=pi_ub,
        slab=slab,
        pi_se=pi_se,
        pi_dist=pi_dist,
        pi_ddf=pi_ddf,
        prob=prob_out,
        X=X_new.copy() if addx else None,
        vcov=vcov_pred if transf is None else None,
    )


def level_indices(
    levels: collections.abc.Sequence[str | int] | str | int,
    labels: collections.abc.Sequence[str],
    k_new: int,
    what: str,
) -> list[int]:
    """0-based variance-component positions from labels or 0-based integers, recycled to ``k_new``.

    Raises:
        metafopy.errors.InputError: a level does not exist or the number of levels does not match.
    """
    seq = [levels] if isinstance(levels, str | int) else list(levels)
    out: list[int] = []
    for v in seq:
        if isinstance(v, str):
            if v not in labels:
                raise metafopy.errors.InputError(f"Non-existing levels specified via '{what}' argument.")
            out.append(list(labels).index(v))
        else:
            if not 0 <= int(v) < len(labels):
                raise metafopy.errors.InputError(f"Non-existing values specified via '{what}' argument.")
            out.append(int(v))
    if len(out) == 1:
        out = out * k_new
    if len(out) != k_new:
        raise metafopy.errors.InputError(f"Length of the '{what}' argument ({len(out)}) does not match the number of predicted values ({k_new}).")
    return out


@predict.register
def predict_rma_mv(
    model: metafopy.models.mv_result.RMAMVResult,
    newmods: numpy.typing.ArrayLike | None = None,
    *,
    intercept: bool | None = None,
    tau2_levels: collections.abc.Sequence[str | int] | str | int | None = None,
    gamma2_levels: collections.abc.Sequence[str | int] | str | int | None = None,
    hetvar: numpy.typing.ArrayLike | None = None,
    prob: str | float | None = None,
    addx: bool = False,
    transf: collections.abc.Callable[..., typing.Any] | None = None,
    targs: typing.Any = None,
    level: float | None = None,
    adjust: bool = False,
    vcov: bool = False,
) -> PredictResult:
    """Predict from an ``rma.mv`` fit (``predict.rma`` for ``rma.mv`` objects).

    Prediction intervals add the sum of the ``sigma2`` values and, when present, the ``tau2``
    and ``gamma2`` values selected via ``tau2_levels`` and ``gamma2_levels`` (labels of the
    inner factor or 0-based positions). They are NaN when several ``tau2``/``gamma2`` values exist
    and no levels are given. ``hetvar`` replaces the sum of the variance components (the levels
    are then ignored, with a warning); ``prob``, ``addx`` and ``vcov`` are as in :func:`predict_rma_uni`.

    R source: https://github.com/wviechtb/metafor/blob/master/R/predict.rma.r;
    reference page: https://wviechtb.github.io/metafor/reference/predict.rma.html.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    x = model
    if x.robust is not None and x.robust.robumethod == 'clubSandwich':
        raise metafopy.errors.InputError('predict() is not supported for fits made robust with club_sandwich=True.')
    check_prob_adjust(prob, adjust)
    if hetvar is not None and (tau2_levels is not None or gamma2_levels is not None):
        tau2_levels = gamma2_levels = None
        warnings.warn("Arguments 'tau2.levels' and 'gamma2.levels' ignored when specifying the 'hetvar' argument.", stacklevel=2)
    p = x.p
    slab: tuple[str, ...] | None = None
    general = any(v in ('GEN', 'GDIAG') for v in x.struct)
    g_labels = list(x.group_g.levels_f[0]) if x.group_g is not None else []
    h_labels = list(x.group_h.levels_f[0]) if x.group_h is not None else []
    tau2_idx: list[int] | None = None
    gamma2_idx: list[int] | None = None
    if newmods is None:
        if x.int_only and not general:
            counts = [
                len(v) if isinstance(v, collections.abc.Sequence) and not isinstance(v, str) else 1 for v in (tau2_levels, gamma2_levels) if v is not None
            ]
            X_new = numpy.ones((max(counts, default=1), 1))
            k_new = X_new.shape[0]
            if x.with_g:
                tau2_idx = [0] * k_new if x.tau2s == 1 else (None if tau2_levels is None else level_indices(tau2_levels, g_labels, k_new, 'tau2_levels'))
            if x.with_h:
                gamma2_idx = (
                    [0] * k_new if x.gamma2s == 1 else (None if gamma2_levels is None else level_indices(gamma2_levels, h_labels, k_new, 'gamma2_levels'))
                )
        else:
            X_new = numpy.ones((1, 1)) if x.int_only else x.X
            if not x.int_only:
                slab = x.slab
            k_new = X_new.shape[0]
            if x.with_g and x.group_g is not None:
                factor = x.group_g.inner_factor
                tau2_idx = [0] * k_new if x.tau2s == 1 or factor is None else [int(c) for c in factor.codes]
            if x.with_h and x.group_h is not None:
                factor = x.group_h.inner_factor
                gamma2_idx = [0] * k_new if x.gamma2s == 1 or factor is None else [int(c) for c in factor.codes]
    else:
        arr = numpy.asarray(newmods, dtype=numpy.float64)
        n_mods = p - 1 if x.int_incl else p
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        elif arr.ndim == 1:
            arr = arr[:, None] if n_mods == 1 else arr[None, :]
        if x.int_incl and arr.shape[1] == p - 1:
            col = 1.0 if intercept is None or intercept else 0.0
            arr = numpy.column_stack([numpy.full(arr.shape[0], col), arr])
        if arr.shape[1] != p:
            raise metafopy.errors.InputError(f'dimensions of newmods ({arr.shape[1]} columns) do not match the model ({p})')
        X_new = arr
        k_new = X_new.shape[0]
        if x.with_g:
            tau2_idx = [0] * k_new if x.tau2s == 1 else (None if tau2_levels is None else level_indices(tau2_levels, g_labels, k_new, 'tau2_levels'))
        if x.with_h:
            gamma2_idx = [0] * k_new if x.gamma2s == 1 else (None if gamma2_levels is None else level_indices(gamma2_levels, h_labels, k_new, 'gamma2_levels'))

    k_new = X_new.shape[0]
    alpha = metafopy.models.rma_uni.level_to_alpha(x.level if level is None else level)
    if adjust:
        alpha = alpha / k_new
    pred = X_new @ x.beta
    vpred = numpy.einsum('ij,jk,ik->i', X_new, x.vb, X_new)
    vpred = numpy.where(vpred < 0, numpy.nan, vpred)
    se = numpy.sqrt(vpred)
    ddf_row: FloatArray | None = None
    if x.test == 'z':
        crit = numpy.full(k_new, float(scipy.stats.norm.isf(alpha / 2)))
    else:
        ddf_coef = numpy.where(numpy.isnan(x.ddf), x.k - p, x.ddf)
        ddf_row = numpy.array([float(numpy.min(ddf_coef[X_new[i] != 0])) if numpy.any(X_new[i] != 0) else math.nan for i in range(k_new)])
        crit = numpy.array([float(scipy.stats.t.isf(alpha / 2, d)) if d > 0 else math.nan for d in ddf_row])
    ci_lb = pred - crit * se
    ci_ub = pred + crit * se

    total = vpred + float(numpy.sum(x.sigma2)) if x.sigma2 is not None else vpred.copy()
    pi_ok = True
    if x.with_g:
        if tau2_idx is None:
            pi_ok = False
        else:
            total = total + numpy.asarray(x.tau2, dtype=numpy.float64)[tau2_idx]
    if x.with_h:
        if gamma2_idx is None:
            pi_ok = False
        else:
            total = total + numpy.asarray(x.gamma2, dtype=numpy.float64)[gamma2_idx]
    if hetvar is not None:
        total = vpred + expand_hetvar(hetvar, k_new)
        pi_ok = True
    pi_se_all: FloatArray | None = None
    if pi_ok:
        pi_se = numpy.sqrt(total)
        pi_se_all = pi_se
        pi_lb = pred - crit * pi_se
        pi_ub = pred + crit * pi_se
    else:
        pi_lb = numpy.full(k_new, numpy.nan)
        pi_ub = numpy.full(k_new, numpy.nan)

    pi_dist = 'norm' if x.test == 'z' else 't'
    prob_out = None
    if prob is not None:
        pi_ddf_prob = ddf_row if ddf_row is not None else numpy.full(k_new, float(x.k - p))
        prob_out = prob_values(prob, pred, pi_se_all, pi_dist, pi_ddf_prob)
    vcov_pred = X_new @ x.vb @ X_new.T if vcov else None
    if vcov_pred is not None:
        vcov_pred = (vcov_pred + vcov_pred.T) / 2

    if transf is not None:
        tau2_value = float(numpy.sum(x.tau2)) if x.tau2 is not None else 0.0
        pred, ci_lb, ci_ub, pi_lb, pi_ub = (metafopy.transf.apply_transf(a, transf, targs, tau2_value) for a in (pred, ci_lb, ci_ub, pi_lb, pi_ub))
        se = numpy.full_like(pred, numpy.nan)
        ci_lb, ci_ub = numpy.minimum(ci_lb, ci_ub), numpy.maximum(ci_lb, ci_ub)
        pi_lb, pi_ub = numpy.minimum(pi_lb, pi_ub), numpy.maximum(pi_lb, pi_ub)

    return PredictResult(
        pred=pred,
        se=se,
        ci_lb=ci_lb,
        ci_ub=ci_ub,
        pi_lb=pi_lb,
        pi_ub=pi_ub,
        slab=slab,
        pi_se=pi_se_all,
        pi_dist=pi_dist,
        pi_ddf=ddf_row,
        prob=prob_out,
        X=X_new.copy() if addx else None,
        vcov=vcov_pred if transf is None else None,
    )
