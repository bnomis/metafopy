"""The rows of a fit before studies with missing values were dropped, and ``fitted()`` and ``hatvalues()`` over them.

Ports of metafor's ``fitted.rma``, ``hatvalues.rma.uni`` and ``hatvalues.rma.mv``: ``na_action`` stands for R's ``na.action`` option (``'omit'`` is
``na.omit``, ``'exclude'`` is ``na.exclude``, ``'pass'`` is ``na.pass`` and ``'fail'`` is ``na.fail``). Under ``'omit'`` only the studies used in
fitting are returned, under ``'exclude'`` one value per study of the subset with NaN for the studies left out and under ``'pass'`` one value per
study of the subset computed from whatever data the study has (a study with a missing outcome but complete moderators gets its fitted value;
one with a missing moderator value gets NaN).

R sources: https://github.com/wviechtb/metafor/blob/master/R/fitted.rma.r, https://github.com/wviechtb/metafor/blob/master/R/hatvalues.rma.uni.r,
https://github.com/wviechtb/metafor/blob/master/R/hatvalues.rma.mv.r; reference pages: https://wviechtb.github.io/metafor/reference/fitted.rma.html,
https://wviechtb.github.io/metafor/reference/hatvalues.rma.html.
"""

import dataclasses
import typing

import numpy
import numpy.typing

import metafopy.errors
import metafopy.methods.common
import metafopy.models.binary
import metafopy.models.glmm_result
import metafopy.models.mv_result
import metafopy.models.rma_ls_result
import metafopy.models.rma_uni
import metafopy.models.sel_result

__all__ = ['Rows', 'fitted_values', 'hat_values', 'rows_of']

type FloatArray = numpy.typing.NDArray[numpy.float64]
type BoolArray = numpy.typing.NDArray[numpy.bool_]


@dataclasses.dataclass(frozen=True, slots=True)
class Rows:
    """The ``k.f`` rows of the subset of a fit: the model matrix ``X.f`` (NaN where a moderator value is missing), the flags ``not.na`` of the rows
    used in fitting and the study labels."""

    X_f: FloatArray
    not_na: BoolArray
    slab: tuple[str, ...]


def uni_rows(x: metafopy.models.rma_uni.RMAResult | metafopy.models.rma_ls_result.RMALSResult) -> Rows:
    md = x.data
    not_na = metafopy.methods.common.uni_not_na(x)
    subset = md.keep if md.na_rows is None else md.keep | md.na_rows
    X_f = numpy.full((not_na.shape[0], x.X.shape[1]), numpy.nan)
    X_f[not_na] = x.X
    if md.X_all is not None and md.X_all.shape[1] == x.X.shape[1] and md.X_all[md.keep].shape == x.X.shape and numpy.allclose(md.X_all[md.keep], x.X):
        X_f = numpy.asarray(md.X_all[subset], dtype=numpy.float64)
    labels = [str(i + 1) for i in numpy.flatnonzero(subset)] if md.labels_all is None else [md.labels_all[i] for i in numpy.flatnonzero(subset)]
    for pos, i in enumerate(numpy.flatnonzero(not_na)):
        labels[i] = x.slab[pos]
    return Rows(X_f=X_f, not_na=not_na, slab=tuple(labels))


def rows_of(x: typing.Any) -> Rows:
    """The rows of the subset of a fit (``X.f``, ``not.na`` and the labels), as metafor stores them."""
    if isinstance(x, metafopy.models.sel_result.SelmodelResult):
        return Rows(X_f=x.X, not_na=numpy.ones(x.X.shape[0], dtype=numpy.bool_), slab=tuple(x.base.slab))
    if isinstance(x, metafopy.models.rma_uni.RMAResult | metafopy.models.rma_ls_result.RMALSResult):
        return uni_rows(x)
    if isinstance(x, metafopy.models.mv_result.RMAMVResult):
        X_f = x.X_f
        if X_f is None or X_f.shape[0] != x.not_na.shape[0]:
            X_f = numpy.full((x.not_na.shape[0], x.X.shape[1]), numpy.nan)
            X_f[x.not_na] = x.X
        return Rows(X_f=X_f, not_na=x.not_na, slab=x.slab_f if len(x.slab_f) == x.not_na.shape[0] else tuple(str(i + 1) for i in range(x.not_na.shape[0])))
    if isinstance(x, metafopy.models.glmm_result.GLMMResult):
        return Rows(X_f=x.X_f, not_na=x.tables.not_na, slab=tuple(x.slab))
    if isinstance(x, metafopy.models.binary.FixedBinaryResult):
        return Rows(X_f=numpy.ones((x.tables.not_na.shape[0], 1)), not_na=x.tables.not_na, slab=tuple(x.slab))
    raise metafopy.errors.ArgumentError(f'{type(x).__name__} is not a fitted model')


def fitted_values(x: typing.Any, na_action: metafopy.methods.common.NAAction = 'omit') -> FloatArray:
    """Fitted values ``X.f beta`` over the rows of the subset, handled as ``na_action`` says (``'omit'``, ``'exclude'``, ``'pass'`` or ``'fail'``).

    Raises:
        metafopy.errors.InputError: ``na_action`` is unknown, or ``'fail'`` and studies were left out.
    """
    metafopy.methods.common.check_na_action(na_action)
    rows = rows_of(x)
    full = numpy.asarray(rows.X_f @ x.beta, dtype=numpy.float64)
    return metafopy.methods.common.apply_na_action(full, rows.not_na, na_action)


def scale_rows(x: metafopy.models.rma_ls_result.RMALSResult) -> FloatArray:
    """The scale model matrix ``Z.f`` of a location-scale fit over the rows of its subset (NaN where a scale value is missing)."""
    not_na = metafopy.methods.common.uni_not_na(x)
    Z_f = x.Z_f
    if Z_f is None or Z_f.shape[0] != not_na.shape[0]:
        Z_f = numpy.full((not_na.shape[0], x.Z.shape[1]), numpy.nan)
        Z_f[not_na] = x.Z
    return numpy.asarray(Z_f, dtype=numpy.float64)


def scale_values(x: metafopy.models.rma_ls_result.RMALSResult, na_action: metafopy.methods.common.NAAction = 'omit') -> FloatArray:
    """Fitted scale values ``Z.f alpha`` of a location-scale fit over the rows of the subset (as :func:`fitted_values`)."""
    metafopy.methods.common.check_na_action(na_action)
    not_na = metafopy.methods.common.uni_not_na(x)
    return metafopy.methods.common.apply_na_action(numpy.asarray(scale_rows(x) @ x.alpha, dtype=numpy.float64), not_na, na_action)


def hat_values(
    x: metafopy.models.rma_uni.RMAResult | metafopy.models.rma_ls_result.RMALSResult,
    hat: FloatArray,
    na_action: metafopy.methods.common.NAAction = 'omit',
) -> FloatArray:
    """The hat values (a vector) or hat matrix ``hat`` of the studies used in fitting spread over the rows of the subset, as ``na_action`` says.

    Under ``'exclude'`` and ``'pass'`` the rows of the studies left out are NaN, as in metafor.

    Raises:
        metafopy.errors.InputError: ``na_action`` is unknown, or ``'fail'`` and studies were left out.
    """
    metafopy.methods.common.check_na_action(na_action)
    not_na = metafopy.methods.common.uni_not_na(x)
    if hat.ndim == 1:
        full = numpy.full(not_na.shape[0], numpy.nan)
        full[not_na] = hat
    else:
        full = numpy.full((not_na.shape[0], not_na.shape[0]), numpy.nan)
        full[numpy.ix_(not_na, not_na)] = hat
    return metafopy.methods.common.apply_na_action(full, not_na, na_action)
