"""Residuals of fitted models (metafor's ``residuals.rma``).

``type='response'`` gives the raw residuals ``y - X b``, ``'pearson'`` those divided by the square roots of the diagonal of the marginal
variance-covariance matrix and ``'cholesky'`` the residuals multiplied by the transposed Cholesky factor of the inverse of that matrix (they
sum, squared, to ``QE`` for models with a non-diagonal matrix). ``'rstandard'`` and ``'rstudent'`` return the ``z`` values of
:func:`metafopy.rstandard` and :func:`metafopy.rstudent`. For ``rma_mh`` and ``rma_peto`` fits ``'pearson'`` is ``'rstandard'``.

R source: https://github.com/wviechtb/metafor/blob/master/R/residuals.rma.r; reference page:
https://wviechtb.github.io/metafor/reference/residuals.rma.html.
"""

import collections.abc
import dataclasses
import typing

import numpy
import numpy.typing
import scipy.linalg
import scipy.sparse

import metafopy.errors
import metafopy.methods.common
import metafopy.methods.fitview
import metafopy.methods.influence
import metafopy.models.binary
import metafopy.models.glmm_result
import metafopy.models.mv_result
import metafopy.models.rma_ls_result
import metafopy.models.rma_uni
import metafopy.models.sel_result

__all__ = ['RESIDUAL_TYPES', 'residuals']

type FloatArray = numpy.typing.NDArray[numpy.float64]
type BoolArray = numpy.typing.NDArray[numpy.bool_]

RESIDUAL_TYPES = ('response', 'rstandard', 'rstudent', 'pearson', 'cholesky')
"""The values of ``type`` (partial matching is allowed, as in R)."""

EPS = float(numpy.finfo(float).eps)
UNI_CLASSES = ('rma.uni', 'rma.mv', 'rma.mh', 'rma.peto')
BINARY_ONLY = 'Extraction of Cholesky residuals not available for objects of class "rma.mh", "rma.peto", or "rma.glmm".'


@dataclasses.dataclass(frozen=True, slots=True)
class Layout:
    """What the response, Pearson and Cholesky residuals of a fit are computed from.

    ``response_f`` is ``yi.f - X.f b`` over the ``k.f`` rows of the subset, ``not_na`` flags the rows used in fitting, ``used`` is
    ``yi - X b`` over those rows, ``var`` the diagonal of the marginal variance-covariance matrix ``M`` over them and ``M`` a function
    returning the full matrix (``None`` when ``M`` is diagonal, so the Cholesky factor of its inverse is ``diag(1 / sqrt(var))``).
    """

    response_f: FloatArray
    not_na: BoolArray
    used: FloatArray
    var: FloatArray | None
    M: collections.abc.Callable[[], FloatArray] | None = None


def uni_layout(x: metafopy.models.rma_uni.RMAResult | metafopy.models.rma_ls_result.RMALSResult, beta: FloatArray, tau2: FloatArray | float) -> Layout:
    md = x.data
    not_na = metafopy.methods.common.uni_not_na(x)
    subset = md.keep if md.na_rows is None else md.keep | md.na_rows
    used = x.yi - x.X @ beta
    if md.yi_all is not None and md.X_all is not None and md.X_all.shape[1] == x.X.shape[1]:
        response_f = md.yi_all[subset] - md.X_all[subset] @ beta
    else:
        response_f = numpy.full(not_na.shape[0], numpy.nan)
        response_f[not_na] = used
    return Layout(response_f=response_f, not_na=not_na, used=used, var=x.vi + tau2)


def mv_layout(x: metafopy.models.mv_result.RMAMVResult) -> Layout:
    used = numpy.asarray(x.yi - x.X @ x.beta, dtype=numpy.float64)
    response_f = numpy.full(x.not_na.shape[0], numpy.nan)
    response_f[x.not_na] = used

    def dense() -> FloatArray:
        return numpy.asarray(x.M.toarray() if scipy.sparse.issparse(x.M) else x.M, dtype=numpy.float64)

    var = numpy.asarray(x.M.diagonal() if scipy.sparse.issparse(x.M) else numpy.diag(x.M), dtype=numpy.float64)
    return Layout(response_f=response_f, not_na=x.not_na, used=used, var=var, M=dense)


def table_layout(x: metafopy.models.binary.FixedBinaryResult | metafopy.models.glmm_result.GLMMResult) -> Layout:
    t = x.tables
    X_f = x.X_f if isinstance(x, metafopy.models.glmm_result.GLMMResult) else numpy.ones((t.k_f, 1))
    response_f = numpy.asarray(t.yi_f - X_f @ x.beta, dtype=numpy.float64)
    return Layout(response_f=response_f, not_na=t.not_na, used=response_f[t.not_na], var=None)


def layout_of(x: typing.Any) -> Layout:
    if isinstance(x, metafopy.models.sel_result.SelmodelResult):
        lay = uni_layout(x.base, x.beta, x.tau2)
        return dataclasses.replace(lay, response_f=lay.used, not_na=numpy.ones(lay.used.shape[0], dtype=numpy.bool_))
    if isinstance(x, metafopy.models.rma_uni.RMAResult | metafopy.models.rma_ls_result.RMALSResult):
        return uni_layout(x, x.beta, x.tau2)
    if isinstance(x, metafopy.models.mv_result.RMAMVResult):
        return mv_layout(x)
    return table_layout(x)


def standardised(x: typing.Any, kind: str, na_action: metafopy.methods.common.NAAction) -> FloatArray:
    """The ``z`` values of ``rstandard(x)`` or ``rstudent(x)``, with R's refusals for the classes those are not available for."""
    metafopy.methods.fitview.check_method(x, kind, UNI_CLASSES)
    if 'rma.mv' in metafopy.methods.fitview.r_class(x):
        notav: tuple[str, ...] = metafopy.methods.fitview.NOT_ROBUST
    elif kind == 'rstandard':
        notav = ('robust.rma', 'rma.gen', 'rma.uni.selmodel')
    else:
        notav = ('robust.rma', 'rma.ls', 'rma.gen', 'rma.uni.selmodel')
    metafopy.methods.fitview.check_class(x, notav=notav)
    fn = metafopy.methods.influence.rstandard if kind == 'rstandard' else metafopy.methods.influence.rstudent
    return numpy.asarray(fn(x, na_action=na_action).z, dtype=numpy.float64)


def residuals(x: typing.Any, type: metafopy.models.rma_uni.ResidualType = 'response', *, na_action: metafopy.methods.common.NAAction = 'omit') -> FloatArray:
    """Residuals of a fitted model (metafor's ``residuals.rma``; the ``na_action`` argument stands for R's ``na.action`` option).

    ``type`` is ``'response'`` (``y - X b``), ``'rstandard'``, ``'rstudent'``, ``'pearson'`` or ``'cholesky'``; partial matching is allowed.
    ``na_action`` is ``'omit'`` (only the studies used in fitting), ``'exclude'`` or ``'pass'`` (one value per study of the subset, NaN for
    those not used) or ``'fail'`` (stop if studies were left out). Which types are available follows R. ``'rstandard'`` and ``'rstudent'``
    exist for ``rma_uni``, ``rma_mv``, ``rma_mh`` and ``rma_peto`` fits, but not for selection models and robust fits, and ``'rstudent'`` not
    for location-scale models; ``rma_glmm`` fits have neither. ``'pearson'`` is available for all fits but ``rma_glmm`` (for ``rma_mh`` and
    ``rma_peto`` it is ``'rstandard'``) and ``'cholesky'`` for all but ``rma_mh``, ``rma_peto`` and ``rma_glmm``. For ``rma_mv`` fits the
    ``'pass'`` residuals of studies not used are NaN, as for ``'exclude'``.

    Raises:
        metafopy.errors.ArgumentError: ``x`` is not a fitted model, or ``rstandard``/``rstudent`` has no method for it.
        metafopy.errors.NotSupportedError: the type is not available for the class of ``x``.
        metafopy.errors.InputError: ``type`` or ``na_action`` is not valid, studies are missing and ``na_action='fail'``, or the marginal
            variance-covariance matrix is not positive definite (``'cholesky'``).
    """
    metafopy.methods.fitview.check_class(x)
    metafopy.methods.common.check_na_action(na_action)
    kind = metafopy.methods.common.match_arg(type, RESIDUAL_TYPES)
    binary = isinstance(x, metafopy.models.binary.FixedBinaryResult)
    glmm = isinstance(x, metafopy.models.glmm_result.GLMMResult)
    if binary and kind == 'pearson':
        kind = 'rstandard'
    if kind in ('rstandard', 'rstudent'):
        return standardised(x, kind, na_action)
    if kind == 'pearson' and glmm:
        raise metafopy.errors.NotSupportedError('Extraction of Pearson residuals not available for objects of class "rma.glmm".')
    if kind == 'cholesky' and (binary or glmm):
        raise metafopy.errors.NotSupportedError(BINARY_ONLY)
    lay = layout_of(x)
    if kind == 'response':
        full = metafopy.methods.common.tzero(lay.response_f)
    elif kind == 'pearson':
        se = numpy.full(lay.not_na.shape[0], numpy.nan)
        se[lay.not_na] = numpy.sqrt(typing.cast(FloatArray, lay.var))
        full = metafopy.methods.common.tzero(lay.response_f) / se
    else:
        full = numpy.full(lay.not_na.shape[0], numpy.nan)
        full[lay.not_na] = cholesky_residuals(metafopy.methods.common.tzero(lay.used), lay)
    return metafopy.methods.common.apply_na_action(full, lay.not_na, na_action)


def cholesky_residuals(e: FloatArray, lay: Layout) -> FloatArray:
    """``L e`` where ``L'L`` is the inverse of the marginal variance-covariance matrix and ``L`` is upper triangular."""
    if lay.M is None:
        return e / numpy.sqrt(typing.cast(FloatArray, lay.var))
    try:
        W = scipy.linalg.cho_solve(scipy.linalg.cho_factor(lay.M(), lower=True), numpy.eye(e.shape[0]))
        L = numpy.linalg.cholesky((W + W.T) / 2).T
    except (numpy.linalg.LinAlgError, scipy.linalg.LinAlgError) as err:
        raise metafopy.errors.InputError('Could not take Cholesky decomposition of the marginal var-cov matrix.') from err
    return numpy.asarray(L @ e, dtype=numpy.float64)
