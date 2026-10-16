"""Predicted values for location-scale models (metafor's ``predict.rma.ls``).

Four situations arise: neither ``newmods`` nor ``newscale`` (fitted values of the studies), only ``newmods`` (predicted
average true effects; no prediction interval, since ``tau^2`` cannot be predicted), only ``newscale`` (predicted
``log(tau^2)``, or ``tau^2`` with the identity link; use ``transf=numpy.exp`` for ``tau^2`` under the log link) and both
(predicted average true effects with confidence and prediction intervals).
"""

import collections.abc
import dataclasses
import math
import typing
import warnings

import numpy
import numpy.typing
import scipy.stats

import metafopy.errors
import metafopy.methods.common
import metafopy.methods.fitted
import metafopy.methods.predict
import metafopy.models.rma_ls_result
import metafopy.models.rma_uni
import metafopy.models.tau2
import metafopy.transf

__all__ = ['LSPredictResult', 'predict_rma_ls']

type FloatArray = numpy.typing.NDArray[numpy.float64]


@dataclasses.dataclass(frozen=True, slots=True)
class LSPredictResult(metafopy.methods.predict.PredictResult):
    """Predictions of a location-scale model.

    ``pred_type`` is ``'location'`` or ``'scale'``. For location predictions ``pi_se`` is the standard error used for the
    prediction interval and ``pi_dist`` its distribution (``'norm'`` or ``'t'`` with ``pi_ddf`` degrees of freedom).
    ``prob`` holds the probabilities requested via ``prob``; ``X`` and ``Z`` the model matrices (``addx``, ``addz``);
    ``vcov`` the variance-covariance matrix of the predictions (``vcov=True`` without ``transf``).
    """

    pred_type: str = 'location'
    ddf: float | None = None
    Z: FloatArray | None = None


def new_matrix(
    newvals: numpy.typing.ArrayLike,
    *,
    p: int,
    int_incl: bool,
    int_only: bool,
    intercept: bool,
    intercept_specified: bool,
    force_intercept: bool,
    what: str,
    model_part: str,
    coef_names: collections.abc.Sequence[str] | None = None,
) -> FloatArray:
    """The model matrix for new moderator (or scale) values, with the intercept column added where metafor adds it.

    Raises:
        metafopy.errors.InputError: the dimensions do not match the model.
    """
    values, names = metafopy.methods.predict.named_newmods(newvals)
    try:
        arr = numpy.asarray(values, dtype=numpy.float64)
    except (TypeError, ValueError) as err:
        raise metafopy.errors.InputError(f"Argument '{what}' should only contain numeric variables.") from err
    n_vars = p - 1 if int_incl else p
    if names is not None and coef_names is not None and n_vars > 1 and arr.ndim <= 2 and len(names) == (arr.shape[-1] if arr.ndim else 1) and all(names):
        var_names = list(coef_names)[1:] if int_incl else list(coef_names)
        order = numpy.argsort(metafopy.methods.predict.match_names(names, var_names), kind='stable')
        arr = arr[..., order]
    if arr.ndim > 2:
        raise metafopy.errors.InputError(f"Argument '{what}' should be a vector or matrix.")
    ncol1 = arr.ndim < 2 or arr.shape[1] == 1
    single = ncol1 and ((not int_incl and p == 1) or (int_incl and p == 2))
    if single:
        X = arr.reshape(-1, 1)
    elif arr.ndim < 2:
        X = arr.reshape(1, -1)
    else:
        X = arr
    if not single and X.shape[1] == p:
        if intercept_specified:
            warnings.warn(f"Arguments 'intercept' ignored when '{what}' includes '{p}' columns.", stacklevel=4)
    elif int_incl and not (int_only and X.shape == (1, 1) and X[0, 0] == 1):
        col = 1.0 if (force_intercept or intercept) else 0.0
        X = numpy.column_stack([numpy.full(X.shape[0], col), X])
    if X.shape[1] != p:
        raise metafopy.errors.InputError(f"Dimensions of '{what}' ({X.shape[1]}) do not match the dimensions of the {model_part} ({p}).")
    return X


def is_one(values: numpy.typing.ArrayLike) -> bool:
    """Whether ``values`` is the single value 1 (metafor's ``identical(newmods, 1)``)."""
    arr = numpy.asarray(values, dtype=numpy.float64)
    return arr.size == 1 and float(arr.reshape(-1)[0]) == 1.0


def quad_rows(M: FloatArray, V: FloatArray) -> FloatArray:
    """``diag(M V M')`` with negative values set to NaN."""
    v = numpy.einsum('ij,jk,ik->i', M, V, M)
    return numpy.where(v < 0, numpy.nan, v)


@metafopy.methods.predict.predict.register
def predict_rma_ls(
    model: metafopy.models.rma_ls_result.RMALSResult,
    newmods: numpy.typing.ArrayLike | None = None,
    *,
    intercept: bool | None = None,
    addx: bool = False,
    newscale: numpy.typing.ArrayLike | None = None,
    addz: bool = False,
    prob: str | None = None,
    level: float | None = None,
    adjust: bool = False,
    transf: collections.abc.Callable[..., typing.Any] | None = None,
    targs: typing.Any = None,
    vcov: bool = False,
    predtype: str | None = None,
    pi_type: str | None = None,
    newvi: numpy.typing.ArrayLike | None = None,
    na_action: metafopy.methods.common.NAAction = 'omit',
    digits: int | collections.abc.Mapping[str, int] | None = None,
) -> LSPredictResult:
    """Predict from a location-scale model (``predict.rma.ls``).

    ``newmods`` and ``newscale`` give the moderator and scale variable values without the intercept column (the
    intercept is added unless ``intercept=False``, or the full ``p`` or ``q`` columns). With the identity link negative
    predicted ``tau^2`` values are constrained to 0. ``predtype`` is ``'default'``, ``'simple'``, ``'riley'`` or ``'t'``
    (``pi_type`` is its old spelling); ``newvi`` adds sampling variances of new studies to the prediction interval; ``prob``
    (``'<0.2'`` or ``'>0.2'``) adds the probability that a true outcome falls on that side of the value. ``na_action`` stands for R's
    ``na.action`` option: for the fitted values (no ``newmods`` and ``newscale``) ``'omit'`` returns the studies used in fitting, ``'exclude'`` one
    row per study with NaN for those left out and ``'pass'`` one row per study computed from the values it has; predictions for ``newmods`` or
    ``newscale`` drop (``'omit'``) or keep (the others) the rows that are NaN.

    Raises:
        metafopy.errors.InputError: an argument is invalid, ``predtype`` is unknown or the dimensions do not match the model.
    """
    x = model
    metafopy.methods.common.check_na_action(na_action)
    ptype = metafopy.methods.predict.resolve_predtype(predtype, pi_type)
    int_spec = intercept is not None
    icpt = True if intercept is None else intercept
    crit_level = x.level if level is None else metafopy.models.rma_uni.level_to_alpha(level)

    if newmods is not None and x.int_only and not is_one(newmods):
        raise metafopy.errors.InputError('Cannot specify new moderator values for models without moderators.')
    if newscale is not None and x.Z_int_only and not is_one(newscale):
        raise metafopy.errors.InputError('Cannot specify new scale values for models without scale variables.')

    X_new: FloatArray | None = None
    Z_new: FloatArray | None = None
    if newmods is not None:
        X_new = new_matrix(
            newmods,
            p=x.p,
            int_incl=x.int_incl,
            int_only=x.int_only,
            intercept=icpt,
            intercept_specified=int_spec,
            force_intercept=False,
            what='newmods',
            model_part='model',
            coef_names=x.coef_names,
        )
    if newscale is not None:
        Z_new = new_matrix(
            newscale,
            p=x.q,
            int_incl=x.Z_int_incl,
            int_only=x.Z_int_only,
            intercept=icpt,
            intercept_specified=int_spec,
            force_intercept=newmods is not None,
            what='newscale',
            model_part='scale model',
            coef_names=x.alpha_names,
        )

    pred_mui = True
    rows = metafopy.methods.fitted.rows_of(x)
    if newmods is None and newscale is None:
        X_use, Z_use = rows.X_f, metafopy.methods.fitted.scale_rows(x)
        k_new = X_use.shape[0]
        tau2_f = numpy.full(k_new, math.nan)
        tau2_f[rows.not_na] = x.tau2
    elif X_new is None and Z_new is not None:
        Z_use = Z_new
        X_use = numpy.empty((0, x.p))
        tau2_f = numpy.full(Z_use.shape[0], math.nan)
        k_new = Z_use.shape[0]
        addx = False
        pred_mui = False
    elif X_new is not None and Z_new is None:
        X_use = X_new
        k_new = X_use.shape[0]
        Z_use = numpy.full((k_new, x.q), math.nan)
        tau2_f = numpy.full(k_new, math.nan)
        addz = False
    elif X_new is not None and Z_new is not None:
        X_use, Z_use = X_new, Z_new
        k_new = X_use.shape[0]
        tau2_f = Z_use @ x.alpha
        if x.link == 'log':
            tau2_f = numpy.exp(tau2_f)
        elif numpy.any(tau2_f < 0):
            warnings.warn("Negative predicted 'tau2' values constrained to 0.", stacklevel=2)
            tau2_f = numpy.where(tau2_f < 0, 0.0, tau2_f)
        if k_new == 1 and Z_use.shape[0] > 1:
            X_use = X_use[numpy.zeros(Z_use.shape[0], dtype=int)]
            k_new = Z_use.shape[0]
        if tau2_f.shape[0] == 1 and k_new > 1:
            Z_use = Z_use[numpy.zeros(k_new, dtype=int)]
            tau2_f = numpy.full(k_new, tau2_f[0])
        if tau2_f.shape[0] != k_new:
            raise metafopy.errors.InputError(f"Dimensions of 'newmods' ({k_new}) do not match the dimensions of 'newscale' ({tau2_f.shape[0]}).")
    else:  # pragma: no cover - the cases above are exhaustive
        raise metafopy.errors.InputError("Arguments 'newmods' and 'newscale' are inconsistent.")

    crit_alpha = crit_level / (k_new if adjust else 1)
    # auditme: accept[literal-duplication] metafor option values and column labels repeated as in the R source
    t_test = x.test in ('knha', 'adhoc', 't')
    pi_dist: str | None = None
    pi_ddf: float | None = None
    pi_se: FloatArray | None = None
    prob_out: FloatArray | None = None
    if pred_mui:
        ddf = float(x.k - x.p) if x.ddf is None else float(x.ddf)
        pred = X_use @ x.beta
        vpred = quad_rows(X_use, x.vb)
        if t_test:
            crit = float(scipy.stats.t.isf(crit_alpha / 2, ddf)) if ddf > 0 else math.nan
            pi_dist = 't'
        else:
            crit = float(scipy.stats.norm.isf(crit_alpha / 2))
            pi_dist = 'norm'
    else:
        ddf = float(x.k - x.q) if x.ddf_alpha is None else float(x.ddf_alpha)
        pred = Z_use @ x.alpha
        vpred = quad_rows(Z_use, x.va)
        if t_test:
            crit = float(scipy.stats.t.isf(crit_alpha / 2, ddf)) if ddf > 0 else math.nan
        else:
            crit = float(scipy.stats.norm.isf(crit_alpha / 2))
    se = numpy.sqrt(vpred)
    ci_lb = pred - crit * se
    ci_ub = pred + crit * se

    if pred_mui:
        vcov_pred = X_use @ x.vb @ X_use.T if vcov else None
        vpred_pi = vpred
        if ptype == 'simple':
            crit = float(scipy.stats.norm.isf(crit_alpha / 2))
            vpred_pi = numpy.zeros_like(vpred)
            pi_dist = 'norm'
        pi_ddf = ddf
        if ptype in ('riley', 't'):
            pi_ddf = float(x.k - x.p - x.q) if ptype == 'riley' else float(x.k - x.p)
            pi_ddf = max(pi_ddf, 1.0)
            crit = float(scipy.stats.t.isf(crit_alpha / 2, pi_ddf))
            pi_dist = 't'
        nv = metafopy.methods.predict.expand_newvi(newvi, k_new)
        pi_se = numpy.sqrt(vpred_pi + tau2_f + nv)
        pi_lb = pred - crit * pi_se
        pi_ub = pred + crit * pi_se
        if prob is not None:
            if adjust:
                raise metafopy.errors.InputError("Cannot use 'prob' argument when adjust=True.")
            prob_out = metafopy.methods.predict.prob_values(prob, pred, pi_se, pi_dist or 'norm', pi_ddf)
    else:
        vcov_pred = Z_use @ x.va @ Z_use.T if vcov else None
        pi_lb = numpy.full(k_new, math.nan)
        pi_ub = numpy.full(k_new, math.nan)

    do_transf = transf is not None
    if transf is not None:
        pred, ci_lb, ci_ub, pi_lb, pi_ub = (metafopy.transf.apply_transf(a, transf, targs) for a in (pred, ci_lb, ci_ub, pi_lb, pi_ub))
        se = numpy.full(k_new, math.nan)
    ci_lb, ci_ub = numpy.minimum(ci_lb, ci_ub), numpy.maximum(ci_lb, ci_ub)
    pi_lb, pi_ub = numpy.minimum(pi_lb, pi_ub), numpy.maximum(pi_lb, pi_ub)

    if not pred_mui and x.link == 'identity' and not do_transf:
        if numpy.any(pred < 0):
            warnings.warn("Negative predicted 'tau2' values constrained to 0.", stacklevel=2)
        pred = numpy.where(pred < 0, 0.0, pred)
        ci_lb = numpy.where(ci_lb < 0, 0.0, ci_lb)
        ci_ub = numpy.where(ci_ub < 0, 0.0, ci_ub)

    slab: tuple[str, ...]
    if (pred_mui and newmods is None) or (not pred_mui and newscale is None):
        slab = rows.slab
    else:
        slab = tuple(str(i + 1) for i in range(k_new))
    if k_new == 1 and not ((pred_mui and newmods is None) or (not pred_mui and newscale is None)):
        # auditme: accept[literal-duplication] metafor option values and column labels repeated as in the R source
        slab = ('',)

    if x.method in metafopy.models.tau2.FIXED_EFFECTS or not pred_mui:
        pi_lb = numpy.full(k_new, math.nan)
        pi_ub = numpy.full(k_new, math.nan)

    out = LSPredictResult(
        pred=pred,
        se=se,
        ci_lb=ci_lb,
        ci_ub=ci_ub,
        pi_lb=pi_lb,
        pi_ub=pi_ub,
        slab=slab,
        pred_type='location' if pred_mui else 'scale',
        ddf=ddf if x.test != 'z' else None,
        pi_se=pi_se,
        pi_dist=pi_dist,
        pi_ddf=pi_ddf if pi_dist == 't' else None,
        prob=prob_out,
        X=X_use if addx else None,
        x_names=tuple(x.coef_names) if addx and X_use is not None and X_use.shape[1] == len(x.coef_names) else None,
        Z=Z_use if addz else None,
        vcov=vcov_pred if (vcov and not do_transf) else None,
    )
    fitted_rows = newmods is None and newscale is None
    keep, blank = metafopy.methods.predict.rows_to_keep(pred, rows.not_na, fitted_rows, na_action)
    result = out if keep.all() and blank is None else metafopy.methods.predict.restrict_rows(out, keep, blank)
    return metafopy.methods.predict.with_digits(typing.cast(LSPredictResult, result), digits, model.digits)
