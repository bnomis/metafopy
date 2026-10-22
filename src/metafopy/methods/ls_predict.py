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

import metafopy.core.labelled
import metafopy.errors
import metafopy.methods.common
import metafopy.methods.fitted
import metafopy.methods.pairmat
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
    z_names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        """As :class:`~metafopy.methods.predict.PredictResult`, and the scale model matrix gets its column names."""
        metafopy.methods.predict.PredictResult.__post_init__(self)
        if isinstance(self.Z, numpy.ndarray) and self.z_names is not None and self.Z.ndim == 2 and len(self.z_names) == self.Z.shape[1]:
            object.__setattr__(self, 'Z', metafopy.core.labelled.LabelledMatrix(self.Z, None, self.z_names))


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
    newmods: numpy.typing.ArrayLike | metafopy.methods.pairmat.PairmatSpec | None = None,
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
    verbose: bool = False,
) -> LSPredictResult:
    """Predicted values for a location-scale model, fitted by [rma_uni][metafopy.rma_uni] with ``scale`` (metafor's ``predict.rma.ls``).

    What is predicted depends on the arguments. Without ``newmods`` and ``newscale`` the result holds the fitted values of the studies, with prediction
    intervals that use the ``tau^2`` of each study. With ``newmods`` only it holds the predicted effects with confidence intervals; the prediction
    intervals are NaN, since ``tau^2`` is not known for the new studies. With both it holds the predicted effects with confidence and prediction
    intervals, using the ``tau^2`` predicted from ``newscale``. With ``newscale`` only, ``pred`` holds the predicted ``log(tau^2)`` (log link) or
    ``tau^2`` (identity link), ``pred_type`` is ``'scale'`` and the prediction intervals are NaN (use ``transf=numpy.exp`` for ``tau^2`` under the
    log link). With the identity link and no ``transf``, negative predicted ``tau^2`` values are set to 0 with a warning.

    R source: <https://github.com/wviechtb/metafor/blob/master/R/predict.rma.r>; reference page:
    <https://wviechtb.github.io/metafor/reference/predict.rma.html>.

    Args:
        model: Location-scale model fitted by [rma_uni][metafopy.rma_uni] with ``scale``.
        newmods: Moderator values of new studies to predict for, without the intercept column (it is added, see ``intercept``) or with all ``p``
            columns (then ``intercept`` is ignored, with a warning if given). A number or 1-d array gives one prediction per value if the model has a
            single moderator and one prediction otherwise; a 2-d array gives one prediction per row. A dict, pandas Series or DataFrame with string
            names is matched by name to the moderators of the model when it has more than one moderator (a name may be shortened to any subsequence
            of the moderator name, as in metafor). A [pairmat][metafopy.pairmat] specification is also accepted. Factors must
            be given as the contrast variables of the model matrix. A model without moderators accepts only ``newmods=1``. Default: none (the fitted
            values of the studies if ``newscale`` is not given either).
        intercept: Whether to include the intercept when predicting for ``newmods`` (and ``newscale`` if ``newmods`` is not given): if False, the
            intercept column is set to 0. Default: the column of ones is added if the model has an intercept. Ignored when the matrix has all its
            columns; the intercept column of ``newscale`` is always 1 when ``newmods`` is given too.
        addx: Add the model matrix of the predictions to the result (``X``).
        newscale: Values of the scale variables of the new studies, without the intercept column (added, see ``intercept``) or with all ``q`` columns;
            same forms as ``newmods``. A model without scale variables accepts only ``newscale=1``. If both are given, they must have the same number
            of rows, except that a single row of either is repeated. Default: none (see the description).
        addz: Add the model matrix of the scale variables of the predictions to the result (``Z``).
        prob: Probability to add to the result, as a string such as ``'>0.2'`` or ``'<0'``: the probability that a true outcome is above (``>``) or
            below (``<``) the cutoff, from the predictive distribution (normal or t, see ``predtype``). The cutoff may be an arithmetic expression as
            for [predict][metafopy.predict]; a string that is not one gives no ``prob``. Not computed for scale predictions (``newscale`` only).
            Cannot be combined with ``adjust=True``. Default: none.
        level: Confidence and prediction interval level in percent (``95`` is 95%). As in metafor, a value below 1 is read as the confidence level if it is
            above 0.5 (``0.95``) and as the significance level otherwise (``0.05``). Default: the level of the model.
        adjust: Apply a Bonferroni correction: the significance level is divided by the number of predictions, which widens the intervals.
        transf: Function applied to each predicted value and interval bound separately (for example ``numpy.exp``); ``se`` is then NaN and the lower
            and upper bounds are put in order. Default: no transformation.
        targs: Second argument passed to ``transf`` (``transf(value, targs)``). Default: none; unlike for [predict][metafopy.predict] of an
            ``rma_uni`` fit, the integral and mode transformations get no default ``tau2``.
        vcov: Add the variance-covariance matrix of the predictions to the result (``vcov``; metafor returns it as a second list element). Ignored with
            ``transf``.
        predtype: Type of prediction interval: ``'default'`` (the normal distribution, or the t distribution with ``k - p`` degrees of freedom if the
            test is ``'t'``, ``'knha'`` or ``'adhoc'``), ``'simple'`` (ignore the uncertainty of the predicted value, normal distribution),
            ``'riley'`` (t distribution with ``k - p - q`` degrees of freedom, Riley et al., 2011,
            [doi:10.1136/bmj.d549](https://doi.org/10.1136/bmj.d549)) or ``'t'`` (t distribution with ``k - p`` degrees of freedom); degrees of
            freedom below 1 are set to 1. Case is ignored and a unique prefix is accepted. Default: ``pi_type``, else ``'default'``.
        pi_type: Old spelling of ``predtype``, used when ``predtype`` is None.
        newvi: Sampling variances of the new studies, added to the variance of the prediction interval: a number or one value per prediction. Default: 0.
        na_action: What to do with studies left out of the fit because of missing values: ``'omit'``, ``'exclude'``, ``'pass'`` or ``'fail'``
            (metafor takes this from the ``na.action`` option). For the fitted values (no ``newmods`` or ``newscale``), ``'omit'`` returns the
            studies used in fitting, ``'exclude'`` one row per study with NaN for those left out and ``'pass'`` one row per study computed from the
            values it has. ``'fail'`` raises if any study was left out of the fit, also when ``newmods`` or ``newscale`` is given. Predictions for
            ``newmods`` or ``newscale`` drop (``'omit'``) or keep (``'exclude'``, ``'pass'``) the rows that are NaN.
        digits: Number of decimals used when printing, or a mapping with any of the keys ``est``, ``se``, ``test``, ``pval``, ``ci``, ``var``,
            ``sevar``, ``fit`` and ``het`` (a unique prefix of a key is accepted). It is stored in the result and does not change the values.
            Default: the ``digits`` option, else that of the model.
        verbose: Accepted for compatibility with metafor; it has no effect here.

    Returns:
        The predictions, an ``LSPredictResult``: a [PredictResult][metafopy.PredictResult] with the extra fields ``Z`` (the scale model matrix, with
        ``addz=True``; None otherwise) and ``z_names``. ``pred_type`` is ``'location'`` or ``'scale'``; the predicted ``tau^2`` of a scale prediction is
        in ``pred`` (there is no ``tau2`` field). ``ddf`` is a single number, None if the test is ``'z'``. Unlike metafor, ``vcov`` is a field of the
        result, not a second list element.

    Raises:
        metafopy.errors.InputError: an argument is invalid (for example ``prob`` with ``adjust=True``, an unknown ``predtype`` or ``na_action``, a
            ``newvi`` of the wrong length, moderator or scale values for a model without them), or the dimensions of ``newmods`` or ``newscale`` do
            not match the model or each other.
    """
    newmods = metafopy.methods.pairmat.resolve(newmods, model)
    x = model
    metafopy.methods.predict.check_prob_adjust(prob, adjust, vcov)
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
    elif pred_mui:
        slab = metafopy.methods.predict.prediction_slab(newmods, k_new, x.p, x.int_incl)
    else:
        slab = ('',) if k_new == 1 else tuple(str(i + 1) for i in range(k_new))

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
        z_names=tuple(x.alpha_names) if addz and Z_use is not None and Z_use.shape[1] == len(x.alpha_names) else None,
        vcov=vcov_pred if (vcov and not do_transf) else None,
        method=x.method,
        transf=do_transf,
        has_pi=pred_mui and x.method not in metafopy.models.tau2.FIXED_EFFECTS,
    )
    fitted_rows = newmods is None and newscale is None
    keep, blank = metafopy.methods.predict.rows_to_keep(pred, rows.not_na, fitted_rows, na_action)
    result = out if keep.all() and blank is None else metafopy.methods.predict.restrict_rows(out, keep, blank)
    return metafopy.methods.predict.with_digits(typing.cast(LSPredictResult, result), digits, model.digits)
