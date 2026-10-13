"""Helpers shared by the diagnostics: refitting a model on a subset of its studies, padding results for NA rows."""

import collections.abc
import dataclasses
import typing
import warnings

import numpy
import numpy.typing

import metafopy.errors
import metafopy.metadata
import metafopy.models.rma_uni
import metafopy.models.tau2

type FloatArray = numpy.typing.NDArray[numpy.float64]
type BoolArray = numpy.typing.NDArray[numpy.bool_]
# auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
type NAAction = typing.Literal['omit', 'exclude', 'pass', 'fail']

EPS = float(numpy.finfo(float).eps)

# what a failed refit looks like to callers (metafor wraps the refit in try() and records NA)
REFIT_ERRORS = (metafopy.errors.EstimationError, ValueError, ArithmeticError, numpy.linalg.LinAlgError)


def check_na_action(na_action: str) -> None:
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    if na_action not in ('omit', 'exclude', 'pass', 'fail'):
        raise metafopy.errors.InputError(f'unknown na_action {na_action!r}')


def subset_metadata(res: metafopy.models.rma_uni.RMAResult, idx: numpy.typing.NDArray[numpy.intp] | BoolArray) -> metafopy.metadata.MetaData:
    """The model inputs of ``res`` restricted to the studies in ``idx`` (positions among the ``k`` studies used)."""
    md = res.data
    sel = numpy.asarray(idx)
    keep = numpy.zeros_like(md.keep)
    keep[numpy.flatnonzero(md.keep)[sel]] = True
    return dataclasses.replace(
        md,
        yi=res.yi[sel],
        vi=res.vi[sel],
        X=res.X[sel],
        X_names=tuple(res.coef_names),
        slab=tuple(numpy.asarray(res.slab, dtype=object)[sel].tolist()),
        keep=keep,
        weights=None if res.weights is None else res.weights[sel],
        ni=None if md.ni is None else md.ni[sel],
        intercept=res.int_incl,
    )


def refit(
    res: metafopy.models.rma_uni.RMAResult,
    idx: numpy.typing.NDArray[numpy.intp] | BoolArray,
    *,
    skip_r2: bool = True,
    method: str | None = None,
) -> metafopy.models.rma_uni.RMAResult | None:
    """Refit ``res`` on the studies in ``idx`` with the same settings, as metafor's diagnostics do.

    Returns ``None`` when the refit fails or drops a coefficient (R records ``NA``
    in both cases). The control settings, including ``tau2_max``, are those of the
    original fit.
    """
    md = subset_metadata(res, idx)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = metafopy.models.rma_uni.fit_rma_uni(
                md,
                method=res.method if method is None else method,
                weighted=res.weighted,
                test=res.test,
                level=1 - res.level,
                tau2=res.tau2 if res.tau2_fix else None,
                control=dataclasses.asdict(res.control),
                skip_r2=skip_r2,
            )
    except REFIT_ERRORS:
        return None
    if out.p != res.p:
        return None
    return out


def refit_arrays(
    res: metafopy.models.rma_uni.RMAResult,
    *,
    yi: FloatArray | None = None,
    X: FloatArray | None = None,
    btt: typing.Any = None,
    tau2: float | typing.Literal['same'] | None = 'same',
    method: str | None = None,
    names: tuple[str, ...] | None = None,
    intercept: bool | None = None,
    skip_r2: bool = True,
) -> metafopy.models.rma_uni.RMAResult | None:
    """Refit ``res`` with replacement ``yi`` and/or model matrix ``X`` (same studies, same settings); ``None`` if the refit fails.

    ``tau2='same'`` keeps ``res``'s fixed ``tau^2`` (if any), a number fixes ``tau^2`` at that value
    and ``None`` forces estimation.
    """
    md = dataclasses.replace(
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
            return metafopy.models.rma_uni.fit_rma_uni(
                md,
                method=res.method if method is None else method,
                weighted=res.weighted,
                test=res.test,
                level=1 - res.level,
                btt=btt,
                tau2=tau2_arg,
                control=dataclasses.asdict(res.control),
                skip_r2=skip_r2,
            )
    except REFIT_ERRORS:
        return None


def pad_to_supplied(res: metafopy.models.rma_uni.RMAResult, *arrays: FloatArray) -> tuple[list[FloatArray], tuple[str, ...]]:
    """Pad per-study results to one row per supplied study (NaN where left out), as ``na.exclude``/``na.pass`` do."""
    keep = res.data.keep
    idx = numpy.flatnonzero(keep)
    out: list[FloatArray] = []
    for arr in arrays:
        full = numpy.full((keep.shape[0], *arr.shape[1:]), numpy.nan)
        full[idx] = arr
        out.append(full)
    labels = [str(i + 1) for i in range(keep.shape[0])]
    for pos, i in enumerate(idx):
        labels[i] = res.slab[pos]
    return out, tuple(labels)


def psort(lb: FloatArray, ub: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Order interval bounds elementwise (metafor's ``.psort``)."""
    with numpy.errstate(invalid='ignore'):
        flip = lb > ub
    return numpy.where(flip, ub, lb), numpy.where(flip, lb, ub)


def tzero(x: FloatArray) -> FloatArray:
    """Snap tiny residuals to zero as metafor does (``abs(x) < 100 * .Machine$double.eps``)."""
    out = x.copy()
    out[numpy.abs(out) < 100 * EPS] = 0.0
    return out


def apply_each(values: FloatArray, transf: collections.abc.Callable[..., typing.Any], targs: typing.Any) -> FloatArray:
    """``sapply(values, transf[, targs])``."""
    vals = numpy.atleast_1d(numpy.asarray(values, dtype=numpy.float64))
    out = [transf(v) if targs is None else transf(v, targs) for v in vals]
    return numpy.array([float(numpy.asarray(o, dtype=numpy.float64).reshape(-1)[0]) for o in out], dtype=numpy.float64).reshape(vals.shape)
