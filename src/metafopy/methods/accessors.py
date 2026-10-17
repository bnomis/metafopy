"""Generic accessors of fitted models: ``se``, ``nobs``, ``df_residual``, ``formula``, ``model_matrix`` and ``update``.

Ports of metafor's ``se.rma``, ``nobs.rma``, ``df.residual.rma``, ``formula.rma``,
``model.matrix.rma`` and ``update.rma`` for ``rma_uni``, ``rma_mv``, ``rma_mh``,
``rma_peto``, ``rma_glmm``, location-scale and ``selmodel`` fits. Each is a
function here and, except ``se`` (the result classes keep the standard errors
in their ``se`` attribute), a method of the result classes (``res.nobs()``,
...); ``formula`` is only a method (``res.formula()``), because a top-level
``metafopy.formula`` would hide the :mod:`metafopy.formula` module.

R sources: https://github.com/wviechtb/metafor/blob/master/R/se.r,
https://github.com/wviechtb/metafor/blob/master/R/nobs.rma.r,
https://github.com/wviechtb/metafor/blob/master/R/df.residual.rma.r,
https://github.com/wviechtb/metafor/blob/master/R/formula.rma.r,
https://github.com/wviechtb/metafor/blob/master/R/model.matrix.rma.r,
https://github.com/wviechtb/metafor/blob/master/R/update.rma.r;
reference pages: https://wviechtb.github.io/metafor/reference/se.html,
https://wviechtb.github.io/metafor/reference/model.matrix.rma.html,
https://wviechtb.github.io/metafor/reference/update.rma.html.
"""

import dataclasses
import re
import typing

import numpy
import numpy.typing
import pandas

import metafopy.errors
import metafopy.methods.common
import metafopy.methods.fitted
import metafopy.models.binary
import metafopy.models.call
import metafopy.models.glmm_result
import metafopy.models.mv_result
import metafopy.models.rma_ls_result
import metafopy.models.rma_uni
import metafopy.models.sel_result

__all__ = ['NobsAll', 'df_residual', 'formula_of', 'model_matrix', 'nobs', 'se', 'update']

type FloatArray = numpy.typing.NDArray[numpy.float64]
type Fit = (
    metafopy.models.rma_uni.RMAResult
    | metafopy.models.mv_result.RMAMVResult
    | metafopy.models.binary.FixedBinaryResult
    | metafopy.models.glmm_result.GLMMResult
    | metafopy.models.rma_ls_result.RMALSResult
    | metafopy.models.sel_result.SelmodelResult
)


@dataclasses.dataclass(frozen=True, slots=True)
class NobsAll:
    """``nobs(all=True)``: the numbers of studies, supplied rows, rows in the subset, rows without missing values,
    effective observations and residual degrees of freedom."""

    studies: int
    data: int
    subset: int
    not_na: int
    effective: int
    df_residual: int

    def to_series(self) -> pandas.Series:
        return pandas.Series(dataclasses.asdict(self)).rename({'not_na': 'not.na', 'df_residual': 'df.residual'})


def check_fit(obj: typing.Any, what: str) -> None:
    if not isinstance(
        obj,
        metafopy.models.rma_uni.RMAResult
        | metafopy.models.mv_result.RMAMVResult
        | metafopy.models.binary.FixedBinaryResult
        | metafopy.models.glmm_result.GLMMResult
        | metafopy.models.rma_ls_result.RMALSResult
        | metafopy.models.sel_result.SelmodelResult,
    ):
        raise metafopy.errors.ArgumentError(f'{what}() not implemented for {type(obj).__name__}')


def effective(obj: Fit) -> tuple[int, int]:
    """metafor's ``k.eff`` and ``p.eff``."""
    if isinstance(obj, metafopy.models.glmm_result.GLMMResult):
        return obj.k_eff, obj.p_eff
    if isinstance(obj, metafopy.models.binary.FixedBinaryResult):
        return obj.k, 1
    if isinstance(obj, metafopy.models.mv_result.RMAMVResult):
        return obj.k, obj.p_eff if obj.p_eff > 0 else obj.p
    return obj.k, obj.p


def call_arguments(obj: Fit) -> dict[str, typing.Any]:
    return {} if obj.call is None else obj.call.arguments


def se(obj: typing.Any, type: str | None = None) -> typing.Any:
    """Standard errors of the coefficients, as metafor's ``se()``.

    A location-scale model gives ``{'beta': ..., 'alpha': ...}`` and a selection
    model ``{'beta': ..., 'delta': ...}`` unless ``type`` picks one of them;
    other objects with a ``vcov()`` method give the square roots of its diagonal.

    Raises:
        metafopy.errors.ArgumentError: the object has no standard errors.
    """
    if isinstance(obj, metafopy.models.sel_result.SelmodelResult):
        beta = pandas.Series(obj.se, index=list(obj.coef_names))
        out = {'beta': beta, 'delta': pandas.Series(obj.se_delta, index=list(obj.delta_names))}
        return out.get(type or '', out)
    if isinstance(obj, metafopy.models.rma_ls_result.RMALSResult):
        beta = pandas.Series(obj.se, index=list(obj.coef_names))
        out = {'beta': beta, 'alpha': pandas.Series(obj.se_alpha, index=list(obj.alpha_names))}
        return out.get(type or '', out)
    if hasattr(obj, 'se') and hasattr(obj, 'coef_names') and isinstance(getattr(obj, 'se', None), numpy.ndarray):
        return pandas.Series(numpy.asarray(obj.se, dtype=numpy.float64), index=list(obj.coef_names))
    vcov = getattr(obj, 'vcov', None)
    if callable(vcov):
        vb = vcov()
        arr = numpy.asarray(vb, dtype=numpy.float64)
        if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
            index = list(vb.index) if isinstance(vb, pandas.DataFrame) else None
            return pandas.Series(numpy.sqrt(numpy.diag(arr)), index=index)
    raise metafopy.errors.ArgumentError('Default method for extracting the standard errors does not work for such model objects.')


def nobs(obj: Fit, all: bool = False) -> int | NobsAll:
    """The number of studies (``k``), or with ``all=True`` the breakdown of metafor's ``nobs(..., all=TRUE)``.

    Raises:
        metafopy.errors.ArgumentError: the object is not a model fit.
    """
    check_fit(obj, 'nobs')
    if not all:
        return obj.k
    k_eff, p_eff = effective(obj)
    has_subset = call_arguments(obj).get('subset') is not None
    if isinstance(obj, metafopy.models.rma_uni.RMAResult | metafopy.models.rma_ls_result.RMALSResult):
        md = obj.data
        k_all = md.k_all
        n_na = int(numpy.sum(md.na_rows)) if md.na_rows is not None else 0
        subset = obj.k + n_na if has_subset else 0
    elif isinstance(obj, metafopy.models.sel_result.SelmodelResult):
        k_all, subset = obj.k, 0
    else:
        k_all = obj.k_all
        subset = obj.k_f if has_subset else 0
    return NobsAll(studies=obj.k, data=k_all, subset=subset, not_na=obj.k, effective=k_eff, df_residual=k_eff - p_eff)


def df_residual(obj: Fit) -> int:
    """Residual degrees of freedom ``k.eff - p.eff``, as metafor's ``df.residual()``.

    Raises:
        metafopy.errors.ArgumentError: the object is not a model fit.
    """
    check_fit(obj, 'df_residual')
    k_eff, p_eff = effective(obj)
    return k_eff - p_eff


def formula_of(obj: Fit, type: str = 'mods') -> str | None:
    """The model formula, as metafor's ``formula.rma``: ``type='mods'`` (``'~x'``), ``'yi'`` (``'yi ~ x'``) or ``'scale'``.

    ``None`` when the model was not specified by a formula (or for ``rma_mh`` and ``rma_peto`` fits).

    Raises:
        metafopy.errors.InputError: ``type`` is unknown, or ``'scale'`` for a model that is not a location-scale model.
    """
    check_fit(obj, 'formula')
    if type not in ('mods', 'yi', 'scale'):
        raise metafopy.errors.InputError("'type' must be one of 'mods', 'yi' or 'scale'.")
    if type == 'scale':
        if not isinstance(obj, metafopy.models.rma_ls_result.RMALSResult):
            raise metafopy.errors.InputError("Can only use type='scale' for location-scale models.")
        scale = call_arguments(obj).get('scale')
        return scale if isinstance(scale, str) else obj.formula_scale
    if isinstance(obj, metafopy.models.sel_result.SelmodelResult):
        return formula_of(obj.base, type)
    if isinstance(obj, metafopy.models.binary.FixedBinaryResult):
        return None
    args = call_arguments(obj)
    yi = args.get('yi')
    yi_formula = yi if isinstance(yi, str) and '~' in yi else None
    if type == 'yi':
        return yi_formula
    if yi_formula is not None:
        return deparse_rhs(yi_formula.split('~', 1)[1])
    mods = args.get('mods')
    return deparse_rhs(mods.split('~', 1)[1]) if isinstance(mods, str) and '~' in mods else None


def deparse_rhs(rhs: str) -> str:
    """R's deparse of a one-sided formula: ``~ablat + year`` (no space after the tilde, one space around ``+`` and ``*``)."""
    text = re.sub(r'\s*([+*])\s*', r' \1 ', rhs.strip())
    return '~' + re.sub(r'\s+', ' ', text)


def model_matrix(obj: Fit, asdf: bool = False, *, na_action: metafopy.methods.common.NAAction = 'omit') -> typing.Any:
    """The model matrix of the studies used in fitting, as metafor's ``model.matrix.rma``.

    A ``numpy`` array, or with ``asdf=True`` a data frame with the coefficient names.
    A location-scale model gives ``{'location': X, 'scale': Z}``. ``na_action`` stands for R's ``options(na.action=)``: ``'omit'`` (default) gives the rows
    of the studies used in fitting, ``'exclude'`` and ``'pass'`` one row for every study of the subset (NaN where a value is missing), ``'fail'``
    stops when studies were left out.

    Raises:
        metafopy.errors.ArgumentError: the object is not a model fit.
        metafopy.errors.InputError: ``na_action`` is unknown, or ``'fail'`` and studies were left out.
    """
    check_fit(obj, 'model_matrix')
    metafopy.methods.common.check_na_action(na_action)
    rows = metafopy.methods.fitted.rows_of(obj)
    if na_action == 'fail' and not rows.not_na.all():
        raise metafopy.errors.InputError('Missing values in results.')
    full = na_action in ('exclude', 'pass')
    if isinstance(obj, metafopy.models.binary.FixedBinaryResult):
        X = rows.X_f if full else numpy.ones((obj.k, 1))
    else:
        X = numpy.asarray(rows.X_f if full else obj.X, dtype=numpy.float64)
    names = list(obj.coef_names)

    def wrap(M: FloatArray, cols: list[str]) -> typing.Any:
        return pandas.DataFrame(M, columns=cols) if asdf else M.copy()

    if isinstance(obj, metafopy.models.rma_ls_result.RMALSResult):
        Z = metafopy.methods.fitted.scale_rows(obj) if full else numpy.asarray(obj.Z, dtype=numpy.float64)
        return {'location': wrap(X, names), 'scale': wrap(Z, list(obj.alpha_names))}
    return wrap(X, names)


def update(obj: Fit, formula: str | None = None, evaluate: bool = True, **kwargs: typing.Any) -> typing.Any:
    """Refit a model with changed arguments, as metafor's ``update.rma(object, formula., ..., evaluate=TRUE)``.

    ``formula`` updates the ``mods`` formula (or ``yi`` when the model was specified by
    ``yi='y ~ x'``) with R's ``update.formula`` rules (``'~ . + x'``, ``'~ . - x'``).
    With ``evaluate=False`` the updated :class:`~metafopy.models.call.Call` is returned.

    Raises:
        metafopy.errors.InputError: the object has no recorded call, was made robust, or its ``mods`` is not a formula.
    """
    check_fit(obj, 'update')
    if getattr(obj, 'robust', None) is not None:
        raise metafopy.errors.InputError('Method not available for objects made robust with robust().')
    call = obj.call
    if call is None:
        raise metafopy.errors.InputError('Need an object with call component.')
    args = dict(call.arguments)
    if formula is not None:
        yi = args.get('yi')
        if isinstance(yi, str) and '~' in yi:
            args['yi'] = metafopy.models.call.update_formula(yi, formula)
        else:
            mods = args.get('mods')
            if mods is not None and not (isinstance(mods, str) and '~' in mods):
                raise metafopy.errors.InputError("The 'mods' argument in 'object' must be a formula for updating to work.")
            args['mods'] = metafopy.models.call.update_formula(mods, formula)
    args.update(kwargs)
    new_call = metafopy.models.call.Call(function=call.function, arguments=args)
    return new_call() if evaluate else new_call
