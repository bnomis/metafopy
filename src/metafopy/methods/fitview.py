"""A common view of the data and estimates of any fitted model, for the functions that metafor writes once for all ``rma`` objects.

metafor's ``forest.rma``, ``funnel.rma``, ``ranktest``, ``radial.rma``, ``labbe.rma`` and others read the same
elements (``yi.f``, ``vi.f``, ``X.f``, ``not.na``, ``slab``, ``ids``, ``k.all``, ``beta``, ``level``, ...) from every class of
model object. :func:`fit_view` collects them from a fitted :class:`~metafopy.RMAResult`, :class:`~metafopy.RMAMVResult`,
:class:`~metafopy.MHResult`, :class:`~metafopy.PetoResult`, :class:`~metafopy.RMALSResult`,
:class:`~metafopy.SelmodelResult` or :class:`~metafopy.GLMMResult`; :func:`check_class` raises the error metafor's
``.chkclass`` raises for a class a method is not available for. See
https://github.com/wviechtb/metafor/blob/master/R/misc.func.hidden.r.
"""

import dataclasses
import math
import typing

import numpy
import numpy.typing

import metafopy.errors
import metafopy.models.binary
import metafopy.models.glmm_result
import metafopy.models.mv_result
import metafopy.models.rma_ls_result
import metafopy.models.rma_mh
import metafopy.models.rma_peto
import metafopy.models.rma_uni
import metafopy.models.sel_result
import metafopy.models.tables

__all__ = ['FIXED_UNI_ONLY', 'Fit', 'FitView', 'check_class', 'check_method', 'fit_view', 'is_rma', 'r_class']

type FloatArray = numpy.typing.NDArray[numpy.float64]
type Fit = (
    metafopy.models.rma_uni.RMAResult
    | metafopy.models.mv_result.RMAMVResult
    | metafopy.models.binary.FixedBinaryResult
    | metafopy.models.rma_ls_result.RMALSResult
    | metafopy.models.sel_result.SelmodelResult
    | metafopy.models.glmm_result.GLMMResult
)
# the class metafor's rstandard(), influence(), leave1out(), permutest() and friends refuse
NOT_ROBUST = ('robust.rma',)
FIXED_UNI_ONLY = ('rma.gen', 'rma.glmm', 'rma.ls', 'rma.mv', 'rma.uni.selmodel', 'robust.rma')
"""The classes the functions written for plain ``rma.uni``, ``rma.mh`` and ``rma.peto`` fits are not available for."""
type BoolArray = numpy.typing.NDArray[numpy.bool_]
type IntArray = numpy.typing.NDArray[numpy.intp]


def r_class(x: typing.Any) -> tuple[str, ...]:
    """The class vector metafor gives the corresponding R object (empty for anything that is not a fitted model)."""
    robust = NOT_ROBUST if getattr(x, 'robust', None) is not None else ()
    if isinstance(x, metafopy.models.rma_uni.RMAResult):
        return (*robust, 'rma.uni', 'rma')
    if isinstance(x, metafopy.models.mv_result.RMAMVResult):
        return (*robust, 'rma.mv', 'rma')
    if isinstance(x, metafopy.models.rma_mh.MHResult):
        return ('rma.mh', 'rma')
    if isinstance(x, metafopy.models.rma_peto.PetoResult):
        return ('rma.peto', 'rma')
    if isinstance(x, metafopy.models.rma_ls_result.RMALSResult):
        return ('rma.ls', 'rma.uni', 'rma')
    if isinstance(x, metafopy.models.sel_result.SelmodelResult):
        return ('rma.uni.selmodel', 'rma.uni', 'rma')
    if isinstance(x, metafopy.models.glmm_result.GLMMResult):
        return ('rma.glmm', 'rma')
    return ()


def is_rma(x: typing.Any) -> bool:
    """Whether ``x`` is a fitted model (an R object of class ``rma``)."""
    return 'rma' in r_class(x)


def check_class(x: typing.Any, notav: tuple[str, ...] = (), must: str = 'rma', type: str = 'Method') -> None:
    """metafor's ``.chkclass(class(x), must=, notav=)``.

    Raises:
        metafopy.errors.NotSupportedError: ``x`` is of a class in ``notav`` (metafor stops with "Method not available ...").
        metafopy.errors.ArgumentError: ``x`` is not of class ``must``.
    """
    cls = r_class(x)
    if must not in cls:
        raise metafopy.errors.ArgumentError(f'Argument \'x\' must be an object of class "{must}".')
    if any(c in notav for c in cls):
        raise metafopy.errors.NotSupportedError(f'{type} not available for objects of class "{cls[0]}".')


def check_method(x: typing.Any, name: str, classes: tuple[str, ...] = ('rma.uni',)) -> None:
    """R's S3 dispatch of ``name`` when only methods for ``classes`` exist (and no default method).

    Raises:
        metafopy.errors.ArgumentError: ``x`` is a fitted model of none of the ``classes`` (R: "no applicable method").
    """
    cls = r_class(x)
    if cls and not any(c in classes for c in cls):
        shown = ', '.join(f"'{c}'" for c in cls)
        raise metafopy.errors.ArgumentError(f'no applicable method for \'{name}\' applied to an object of class "c({shown})"')


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class FitView:
    """The data and estimates of a fitted model as metafor's class-independent functions see them.

    ``yi``, ``vi``, ``X``, ``slab``, ``ni`` and ``weights`` hold the rows with an estimate, variance and moderators
    (the rows of ``yi.f``/``vi.f``/``X.f`` that are not ``NA``), ``ids`` their 0-based positions among the ``k_all``
    supplied rows and ``keep`` the same as a mask. ``na_rows`` flags the rows of the subset left out for missing
    values, whose estimates, variances, design rows and labels are in ``yi_all``, ``vi_all``, ``X_all`` and
    ``labels_all`` (``None`` where the model does not keep them). ``weights`` are the percentage weights
    (``None`` for models without a ``weights()`` method), ``tau2`` the (first) heterogeneity variance.
    ``pred_newmods`` tells that the fitted values of the rows must be computed by passing ``X`` to ``predict``
    (the model's own fitted values are for a different set of rows).
    """

    model: typing.Any
    yi: FloatArray
    vi: FloatArray
    X: FloatArray
    slab: tuple[str, ...]
    ids: IntArray
    keep: BoolArray
    k_all: int
    beta: FloatArray
    level: float
    method: str
    measure: str
    int_only: bool
    int_incl: bool
    tau2: float
    slab_null: bool
    ni: FloatArray | None = None
    weights: FloatArray | None = None
    na_rows: BoolArray | None = None
    yi_all: FloatArray | None = None
    vi_all: FloatArray | None = None
    X_all: FloatArray | None = None
    labels_all: tuple[str, ...] | None = None
    pred_newmods: bool = False

    @property
    def k(self) -> int:
        return int(self.yi.shape[0])

    @property
    def fitted(self) -> FloatArray:
        out: FloatArray = self.X @ self.beta
        return out


def default_slab(labels: tuple[str, ...], ids: IntArray) -> bool:
    """Whether the labels are the default ones (metafor's ``slab.null``): the 1-based row numbers."""
    return all(s == str(int(i) + 1) for s, i in zip(labels, ids, strict=True))


def uni_view(x: metafopy.models.rma_uni.RMAResult | metafopy.models.rma_ls_result.RMALSResult, model: typing.Any) -> FitView:
    md = x.data
    ids = numpy.flatnonzero(md.keep).astype(numpy.intp)
    ni = md.ni
    if ni is not None and ni.shape[0] == md.k_all and md.k_all != x.k:
        ni = ni[md.keep]
    tau2 = float(numpy.atleast_1d(x.tau2)[0]) if numpy.ndim(x.tau2) else float(x.tau2)
    return FitView(
        model=model,
        yi=x.yi,
        vi=x.vi,
        X=x.X,
        slab=tuple(x.slab),
        ids=ids,
        keep=md.keep,
        k_all=md.k_all,
        beta=x.beta,
        level=x.level,
        method=x.method,
        measure=x.measure,
        int_only=x.int_only,
        int_incl=x.int_incl,
        tau2=tau2,
        slab_null=default_slab(tuple(x.slab), ids),
        ni=None if ni is None else numpy.asarray(ni, dtype=numpy.float64),
        weights=x.weights_(),
        na_rows=md.na_rows,
        yi_all=md.yi_all,
        vi_all=md.vi_all,
        X_all=md.X_all,
        labels_all=md.labels_all,
    )


def table_view(
    x: metafopy.models.binary.FixedBinaryResult | metafopy.models.glmm_result.GLMMResult,
    t: metafopy.models.tables.TableSet,
    X_f: FloatArray,
    weights_f: FloatArray | None,
    tau2: float,
    int_only: bool,
    int_incl: bool,
    pred_newmods: bool,
) -> FitView:
    rows = t.not_na_yivi
    ids = t.ids[rows]
    keep = numpy.zeros(t.k_all, dtype=numpy.bool_)
    keep[ids] = True
    na_rows = numpy.zeros(t.k_all, dtype=numpy.bool_)
    na_rows[t.ids[~rows]] = True
    yi_all = numpy.full(t.k_all, numpy.nan)
    vi_all = numpy.full(t.k_all, numpy.nan)
    X_all = numpy.full((t.k_all, X_f.shape[1]), numpy.nan)
    yi_all[t.ids], vi_all[t.ids], X_all[t.ids] = t.yi_f, t.vi_f, X_f
    labels = [str(i + 1) for i in range(t.k_all)]
    for i, s in zip(t.ids, t.slab, strict=True):
        labels[int(i)] = s
    return FitView(
        model=x,
        yi=t.yi_f[rows],
        vi=t.vi_f[rows],
        X=X_f[rows],
        slab=tuple(s for s, ok in zip(t.slab, rows, strict=True) if ok),
        ids=ids,
        keep=keep,
        k_all=t.k_all,
        beta=x.beta,
        level=x.level,
        method=x.method,
        measure=x.measure,
        int_only=int_only,
        int_incl=int_incl,
        tau2=tau2,
        slab_null=t.slab_null,
        ni=t.ni_f[rows],
        weights=None if weights_f is None else weights_f[rows],
        na_rows=na_rows,
        yi_all=yi_all,
        vi_all=vi_all,
        X_all=X_all,
        labels_all=tuple(labels),
        pred_newmods=pred_newmods,
    )


def mv_view(x: metafopy.models.mv_result.RMAMVResult) -> FitView:
    ids = numpy.asarray(x.ids, dtype=numpy.intp)[x.not_na]
    keep = numpy.zeros(x.k_all, dtype=numpy.bool_)
    keep[ids] = True
    tau2 = 0.0 if x.tau2 is None or numpy.size(x.tau2) == 0 else float(numpy.atleast_1d(x.tau2)[0])
    return FitView(
        model=x,
        yi=x.yi,
        vi=x.vi,
        X=x.X,
        slab=tuple(x.slab),
        ids=ids,
        keep=keep,
        k_all=x.k_all,
        beta=x.beta,
        level=metafopy.models.rma_uni.level_to_alpha(x.level),  # rma_mv keeps the level in percent
        method=x.method,
        measure=x.measure,
        int_only=x.int_only,
        int_incl=x.int_incl,
        tau2=tau2,
        slab_null=default_slab(tuple(x.slab), ids),
        weights=x.weights_(),
    )


def fit_view(x: typing.Any) -> FitView:
    """The :class:`FitView` of a fitted model.

    Raises:
        metafopy.errors.ArgumentError: ``x`` is not a fitted model.
    """
    if isinstance(x, metafopy.models.rma_uni.RMAResult | metafopy.models.rma_ls_result.RMALSResult):
        return uni_view(x, x)
    if isinstance(x, metafopy.models.sel_result.SelmodelResult):
        base = uni_view(x.base, x)
        # metafor's selmodel() objects have slab.null = FALSE: the default labels are the row numbers and get the room of real labels
        return dataclasses.replace(base, beta=x.beta, tau2=float(x.tau2), weights=None, pred_newmods=not x.int_only, slab_null=False)
    if isinstance(x, metafopy.models.mv_result.RMAMVResult):
        return mv_view(x)
    if isinstance(x, metafopy.models.binary.FixedBinaryResult):
        t = x.tables
        return table_view(x, t, numpy.ones((t.k_f, 1)), x.weights_(na_action='pass'), 0.0, True, True, False)
    if isinstance(x, metafopy.models.glmm_result.GLMMResult):
        tau2 = 0.0 if math.isnan(x.tau2) else float(x.tau2)
        return table_view(x, x.tables, x.X_f, None, tau2, x.int_only, x.int_incl, not x.int_only)
    raise metafopy.errors.ArgumentError(f'Argument \'x\' must be an object of class "rma" (got {type(x).__name__}).')
