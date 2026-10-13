"""Predicted values, confidence and prediction intervals (metafor's ``predict.rma``)."""

import collections.abc
import dataclasses
import functools
import typing

import numpy
import numpy.typing
import pandas
import scipy.stats

import metafopy.errors
import metafopy.models.rma_uni
import metafopy.models.tau2
import metafopy.transf

__all__ = ['PredictResult', 'predict']

FloatArray = numpy.typing.NDArray[numpy.float64]


@dataclasses.dataclass(frozen=True, slots=True)
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

    def to_frame(self) -> pandas.DataFrame:
        return pandas.DataFrame(
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
    transf: collections.abc.Callable[..., typing.Any] | None = None,
    targs: typing.Any = None,
    level: float | None = None,
) -> PredictResult:
    """Predict from an ``rma.uni`` fit.

    Without ``newmods`` an intercept-only model gives the pooled estimate and a
    meta-regression gives the fitted value of each study. ``newmods`` holds
    moderator values without the intercept column (it is added unless
    ``intercept=False``) or the full set of ``p`` columns. ``transf`` is
    applied to each value (with ``targs`` if given; the integral and mode
    transformations in :mod:`metafopy.transf` default to ``targs={'tau2': tau2}``).

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    x = model
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

    alpha = x.level if level is None else metafopy.models.rma_uni.level_to_alpha(level)

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
    pi_se = numpy.sqrt(vpred + x.tau2)
    pi_lb = pred - crit * pi_se
    pi_ub = pred + crit * pi_se
    if x.method in metafopy.models.tau2.FIXED_EFFECTS:
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

    return PredictResult(pred=pred, se=se, ci_lb=ci_lb, ci_ub=ci_ub, pi_lb=pi_lb, pi_ub=pi_ub, slab=slab)
