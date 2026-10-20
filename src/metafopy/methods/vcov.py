"""Variance-covariance matrices of fitted models (metafor's ``vcov.rma``).

``type='fixed'`` (the default) gives the matrix of the model coefficients (a dict with the matrices of the location and scale coefficients for
location-scale models and of the coefficients and selection parameters for selection models), ``'beta'``, ``'alpha'`` and ``'delta'`` pick one of
those, ``'obs'`` (or ``'marginal'``) the marginal variance-covariance matrix of the observed effects (``M``, available for fits that metafor stores
``M`` in: ``rma_uni``, ``rma_mv``, location-scale and selection models, ``trimfill`` and robust fits), ``'fitted'`` the matrix of the fitted values
``X vb X'`` (all fits) and ``'resid'`` that of the residuals ``(I - H) M (I - H)'`` (``rma_uni``, ``rma_mv`` and location-scale fits, not for robust
fits). The matrices of the last three types have one row per study of the subset, labelled with the study labels; ``na_action`` stands for R's
``na.action`` option and decides what happens to the studies not used in fitting.

R source: https://github.com/wviechtb/metafor/blob/master/R/vcov.rma.r; reference page: https://wviechtb.github.io/metafor/reference/vcov.rma.html.
"""

import dataclasses
import typing

import numpy
import numpy.typing
import pandas
import scipy.sparse

import metafopy.core.labelled
import metafopy.errors
import metafopy.methods.common
import metafopy.methods.deltamethod
import metafopy.methods.fitview
import metafopy.methods.matreg
import metafopy.models.binary
import metafopy.models.glmm_result
import metafopy.models.mv_result
import metafopy.models.rma_ls_result
import metafopy.models.rma_uni
import metafopy.models.sel_result

__all__ = ['VCOV_TYPES', 'vcov', 'vcov_frame']

type FloatArray = numpy.typing.NDArray[numpy.float64]
type BoolArray = numpy.typing.NDArray[numpy.bool_]

# auditme: accept[non-alphabetical-literal] order is metafor's documented order and is meaningful
VCOV_TYPES = ('fixed', 'beta', 'alpha', 'delta', 'obs', 'marginal', 'fitted', 'resid')
"""The values of ``type`` (partial matching is allowed, as in R)."""


@dataclasses.dataclass(frozen=True, slots=True)
class Rows:
    """The rows of the subset a fit was made on: their labels, which of them were used in fitting (``not.na``) and their model matrix (``X.f``)."""

    labels: tuple[str, ...]
    not_na: BoolArray
    X_f: FloatArray

    def shown(self, na_action: metafopy.methods.common.NAAction) -> tuple[str, ...]:
        """The labels of the rows left after the ``na.action`` step (the studies used in fitting for ``'omit'``)."""
        if na_action == 'omit':
            return tuple(numpy.asarray(self.labels, dtype=numpy.object_)[self.not_na])
        return self.labels


def pad_rows(X: FloatArray, not_na: BoolArray) -> FloatArray:
    full = numpy.full((not_na.shape[0], X.shape[1]), numpy.nan)
    full[not_na] = X
    return full


def uni_rows(x: metafopy.models.rma_uni.RMAResult | metafopy.models.rma_ls_result.RMALSResult) -> Rows:
    md = x.data
    not_na = metafopy.methods.common.uni_not_na(x)
    subset = md.keep if md.na_rows is None else md.keep | md.na_rows
    padded, labels = metafopy.methods.common.pad_to_supplied(typing.cast(typing.Any, x), x.X)
    if md.X_all is not None and md.X_all.shape[1] == x.X.shape[1]:
        X_f = numpy.asarray(md.X_all[subset], dtype=numpy.float64)
    else:
        X_f = padded[0]
    return Rows(labels=labels, not_na=not_na, X_f=X_f)


def rows_of(x: typing.Any) -> Rows:
    if isinstance(x, metafopy.models.sel_result.SelmodelResult):
        # R refits the model without the studies with missing values, which numbers the studies afresh
        dropped = not bool(metafopy.methods.common.uni_not_na(x.base).all())
        labels = tuple(str(i + 1) for i in range(x.k)) if dropped else tuple(x.slab)
        return Rows(labels=labels, not_na=numpy.ones(x.k, dtype=numpy.bool_), X_f=numpy.asarray(x.X, dtype=numpy.float64))
    if isinstance(x, metafopy.models.rma_uni.RMAResult | metafopy.models.rma_ls_result.RMALSResult):
        return uni_rows(x)
    if isinstance(x, metafopy.models.mv_result.RMAMVResult):
        labels = x.slab_f if len(x.slab_f) == x.not_na.shape[0] else tuple(str(i + 1) for i in range(x.not_na.shape[0]))
        X_f = x.X_f if x.X_f is not None else pad_rows(x.X, x.not_na)
        return Rows(labels=labels, not_na=x.not_na, X_f=X_f)
    t = x.tables
    X_f = x.X_f if isinstance(x, metafopy.models.glmm_result.GLMMResult) else numpy.ones((t.k_f, 1))
    return Rows(labels=tuple(t.slab), not_na=t.not_na, X_f=numpy.asarray(X_f, dtype=numpy.float64))


def marginal(x: typing.Any) -> FloatArray:
    """``M`` as a dense matrix over the studies used in fitting."""
    M = x.M
    return numpy.asarray(typing.cast(typing.Any, M).toarray() if scipy.sparse.issparse(M) else M, dtype=numpy.float64)


def embed(mat: FloatArray, not_na: BoolArray) -> FloatArray:
    """``mat`` (over the studies used in fitting) placed in a ``k.f`` x ``k.f`` matrix of NaN."""
    full = numpy.full((not_na.shape[0], not_na.shape[0]), numpy.nan)
    full[numpy.ix_(not_na, not_na)] = mat
    return full


def frame(mat: FloatArray, labels: typing.Sequence[str]) -> metafopy.core.labelled.RFrame:
    return metafopy.core.labelled.RFrame(mat, index=list(labels), columns=list(labels))


def coef_frame(mat: FloatArray, names: typing.Sequence[str]) -> metafopy.core.labelled.RFrame:
    return metafopy.core.labelled.RFrame(mat, index=list(names), columns=list(names))


def select(full: FloatArray, rows: Rows, na_action: metafopy.methods.common.NAAction) -> FloatArray:
    """The ``na.action`` step for a matrix over all ``k.f`` rows with NaN in the rows not used in fitting."""
    if na_action == 'omit':
        return full[numpy.ix_(rows.not_na, rows.not_na)]
    return full


@typing.overload
def vcov(
    object: typing.Any,
    type: typing.Literal['beta', 'alpha', 'delta', 'obs', 'marginal', 'fitted', 'resid'],
    *,
    na_action: metafopy.methods.common.NAAction = 'omit',
) -> pandas.DataFrame: ...


@typing.overload
def vcov(
    object: typing.Any,
    type: metafopy.models.rma_uni.VcovType = 'fixed',
    *,
    na_action: metafopy.methods.common.NAAction = 'omit',
) -> pandas.DataFrame | dict[str, pandas.DataFrame]: ...


def vcov(
    object: typing.Any,
    type: metafopy.models.rma_uni.VcovType = 'fixed',
    *,
    na_action: metafopy.methods.common.NAAction = 'omit',
) -> pandas.DataFrame | dict[str, pandas.DataFrame]:
    """Variance-covariance matrices of a fitted model, as metafor's ``vcov()``.

    ``type`` is ``'fixed'``, ``'beta'``, ``'alpha'``, ``'delta'``, ``'obs'`` (or ``'marginal'``), ``'fitted'`` or ``'resid'`` (partial matching is
    allowed); see the module description for what each gives and which fits allow it. For ``'obs'``, ``'fitted'`` and ``'resid'`` the frame has one
    row and column per study of the subset: ``na_action='omit'`` keeps the studies used in fitting, ``'exclude'`` and ``'pass'`` all of them (NaN
    for those not used) and ``'fail'`` stops if any study was left out (``'fitted'`` gives the whole matrix, as in R).

    A ``deltamethod`` or ``matreg`` result gives its own variance-covariance matrix (``vcov.deltamethod``, ``vcov.matreg``).

    Raises:
        metafopy.errors.ArgumentError: ``object`` is not a fitted model, or ``type='resid'`` for a fit without ``hatvalues()`` (``rma_mh``,
            ``rma_peto``, ``rma_glmm``).
        metafopy.errors.InputError: ``type`` or ``na_action`` is not valid, ``'alpha'`` for a model without scale part, ``'delta'`` for a
            model that is not a selection model, or studies are missing and ``na_action='fail'`` for ``'obs'``.
        metafopy.errors.NotSupportedError: ``'obs'`` for ``rma_mh``, ``rma_peto`` and ``rma_glmm``; ``'resid'`` for robust and selection-model fits.
    """
    if isinstance(object, (metafopy.methods.deltamethod.DeltaMethodResult, metafopy.methods.matreg.MatregResult)):
        return object.vcov()
    metafopy.methods.fitview.check_class(object)
    metafopy.methods.common.check_na_action(na_action)
    kind = metafopy.methods.common.match_arg(type, VCOV_TYPES)
    is_ls = isinstance(object, metafopy.models.rma_ls_result.RMALSResult)
    is_sel = isinstance(object, metafopy.models.sel_result.SelmodelResult)
    names = list(object.coef_names)
    beta = coef_frame(object.vb, names)
    if kind == 'fixed':
        if is_ls:
            return metafopy.core.labelled.NamedList({'beta': beta, 'alpha': coef_frame(object.va, object.alpha_names)})
        if is_sel:
            return metafopy.core.labelled.NamedList({'beta': beta, 'delta': coef_frame(object.vd, object.delta_names)})
        return beta
    if kind == 'beta':
        return beta
    if kind == 'alpha':
        if not is_ls:
            raise metafopy.errors.InputError('Can only extract var-cov matrix of alpha coefficients for location-scale models.')
        return coef_frame(object.va, object.alpha_names)
    if kind == 'delta':
        if not is_sel:
            raise metafopy.errors.InputError('Can only extract var-cov matrix of delta coefficients for selection models.')
        return coef_frame(object.vd, object.delta_names)
    rows = rows_of(object)
    if kind in ('obs', 'marginal'):
        if not isinstance(object, metafopy.models.mv_result.RMAMVResult) and 'rma.uni' not in metafopy.methods.fitview.r_class(object):
            raise metafopy.errors.NotSupportedError('Extraction of marginal var-cov matrix not available for objects of this class.')
        full = embed(marginal(object), rows.not_na)
        if na_action == 'fail' and not rows.not_na.all():
            raise metafopy.errors.InputError('Missing values in data.')
        return frame(select(full, rows, na_action), rows.shown(na_action))
    if kind == 'fitted':
        out = rows.X_f @ object.vb @ rows.X_f.T
        if na_action in ('exclude', 'pass'):
            out = out.copy()
            out[~rows.not_na, :] = numpy.nan
            out[:, ~rows.not_na] = numpy.nan
        return frame(select(out, rows, na_action), rows.shown(na_action))
    return residual_vcov(object, rows, na_action)


def residual_vcov(object: typing.Any, rows: Rows, na_action: metafopy.methods.common.NAAction) -> pandas.DataFrame:
    """``vcov(type='resid')``: ``(I - H) M (I - H)'`` with the hat matrix ``H`` of the fit."""
    if getattr(object, 'robust', None) is not None:
        raise metafopy.errors.NotSupportedError('Extraction of var-cov matrix of the residuals not available for objects of this type.')
    metafopy.methods.fitview.check_method(object, 'hatvalues', ('rma.uni', 'rma.mv'))
    metafopy.methods.fitview.check_class(object, notav=('rma.uni.selmodel', 'rma.gen'))
    if na_action == 'fail':
        # R's code never defines the result for this setting and stops with "object 'out' not found"
        raise metafopy.errors.InputError("vcov(type='resid') cannot be used with na_action='fail'.")
    H = numpy.asarray(object.hat_matrix(), dtype=numpy.float64)
    ImH = numpy.eye(H.shape[0]) - H
    ve = ImH @ marginal(object) @ ImH.T
    return frame(ve if na_action == 'omit' else embed(ve, rows.not_na), rows.shown(na_action))


def vcov_frame(object: typing.Any, type: metafopy.models.rma_uni.VcovType, na_action: metafopy.methods.common.NAAction) -> pandas.DataFrame:
    """:func:`vcov` for the fit classes whose ``'fixed'`` matrix is a single frame (never a dict)."""
    out = vcov(object, type, na_action=na_action)
    if isinstance(out, dict):
        raise metafopy.errors.InternalError('vcov() returned several matrices for a model with a single coefficient part.')
    return out
