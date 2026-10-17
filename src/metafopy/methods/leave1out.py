"""Leave-one-out and cumulative meta-analyses (metafor's ``leave1out.rma.uni`` and ``cumul.rma.uni``).

``leave1out`` refits an intercept-only model with each study (or cluster of
studies) left out in turn; ``cumul`` adds the studies one at a time in a given
order (Lau et al. 1992, doi:10.1056/NEJM199207023270206 popularised the
cumulative approach). See
https://wviechtb.github.io/metafor/reference/leave1out.rma.uni.html and
https://wviechtb.github.io/metafor/reference/cumul.rma.uni.html.
"""

import collections.abc
import dataclasses
import functools
import typing

import numpy
import numpy.typing
import pandas

import metafopy.core.parallel
import metafopy.errors
import metafopy.methods.blup
import metafopy.methods.common
import metafopy.methods.fitview
import metafopy.mfopt
import metafopy.models.rma_uni
import metafopy.models.tau2
import metafopy.printing
import metafopy.transf

__all__ = ['CumulResult', 'Leave1outResult', 'cumul', 'leave1out']

type FloatArray = numpy.typing.NDArray[numpy.float64]


@dataclasses.dataclass(frozen=True, slots=True)
class Leave1outResult:
    """One row per left-out study with the re-estimated pooled effect and heterogeneity statistics.

    ``se`` is NaN when a transformation was applied, as in metafor. ``tau2`` is
    ``None`` for equal/fixed/common-effects models. ``measure``, ``level`` (the alpha of the
    confidence intervals) and ``slab_null`` (the studies carry no labels of their own) are used by
    the plots of cumulative results.
    """

    estimate: FloatArray
    se: FloatArray
    stat: FloatArray
    pval: FloatArray
    ci_lb: FloatArray
    ci_ub: FloatArray
    Q: FloatArray
    Qp: FloatArray
    tau2: FloatArray | None
    I2: FloatArray
    H2: FloatArray
    slab: tuple[str, ...]
    stat_name: str = 'zval'
    digits: int = 4
    transf: bool = False
    k: FloatArray | None = None
    order: numpy.typing.NDArray[typing.Any] | None = None
    measure: str = 'GEN'
    level: float = 0.05
    slab_null: bool = False

    def to_frame(self) -> pandas.DataFrame:
        """Table with metafor's column names (``estimate``, ``se``, ``zval``/``tval``, ``pval``, ``ci.lb``, ...)."""
        cols: dict[str, typing.Any] = {}
        if self.k is not None:
            cols['k'] = self.k
        cols.update(
            {
                'estimate': self.estimate,
                'se': self.se,
                self.stat_name: self.stat,
                'pval': self.pval,
                'ci.lb': self.ci_lb,
                'ci.ub': self.ci_ub,
                'Q': self.Q,
                'Qp': self.Qp,
            }
        )
        if self.tau2 is not None:
            cols['tau2'] = self.tau2
        cols['I2'] = self.I2
        cols['H2'] = self.H2
        return pandas.DataFrame(cols, index=list(self.slab))

    def __str__(self) -> str:
        d = self.digits
        spec: list[tuple[str, FloatArray, int]] = []
        if self.k is not None:
            spec.append(('k', self.k, 0))
        spec.append(('estimate', self.estimate, d))
        if not self.transf:
            spec.append(('se', self.se, d))
        spec += [
            (self.stat_name, self.stat, d),
            ('pval', self.pval, d),
            ('ci.lb', self.ci_lb, d),
            ('ci.ub', self.ci_ub, d),
            ('Q', self.Q, d),
            ('Qp', self.Qp, d),
        ]
        if self.tau2 is not None:
            spec.append(('tau2', self.tau2, d))
        spec += [('I2', self.I2, d), ('H2', self.H2, d)]
        cols = [(name, [metafopy.printing.fmt_num(float(v), dd) for v in vals]) for name, vals, dd in spec]
        return '\n'.join(['', *metafopy.methods.blup.format_r_frame(cols, self.slab), ''])


@dataclasses.dataclass(frozen=True, slots=True)
class CumulResult(Leave1outResult):
    """Cumulative meta-analysis results (:class:`Leave1outResult` plus ``k`` and the ordering variable)."""


@functools.singledispatch
def leave1out(x: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> Leave1outResult:
    """Leave-one-out analysis of a fitted model; see :func:`_leave1out_rma_uni`.

    Raises:
        metafopy.errors.ArgumentError: an argument has the wrong type or is not accepted.
    """
    raise metafopy.errors.ArgumentError(f'leave1out() not implemented for {type(x).__name__}')


def collect(
    fits: list[metafopy.models.rma_uni.RMAResult | None],
    x: metafopy.models.rma_uni.RMAResult,
    transf: collections.abc.Callable[..., typing.Any] | None,
    targs: typing.Any,
) -> dict[str, FloatArray]:
    n = len(fits)

    def get(attr: str) -> FloatArray:
        return numpy.array([numpy.nan if f is None else float(getattr(f, attr)) for f in fits], dtype=numpy.float64)

    out = {
        'estimate': numpy.array([numpy.nan if f is None else float(f.beta[0]) for f in fits]),
        'se': numpy.array([numpy.nan if f is None else float(f.se[0]) for f in fits]),
        'stat': numpy.array([numpy.nan if f is None else float(f.zval[0]) for f in fits]),
        'pval': numpy.array([numpy.nan if f is None else float(f.pval[0]) for f in fits]),
        'ci_lb': numpy.array([numpy.nan if f is None else float(f.ci_lb[0]) for f in fits]),
        'ci_ub': numpy.array([numpy.nan if f is None else float(f.ci_ub[0]) for f in fits]),
        'Q': get('QE'),
        'Qp': get('QEp'),
        'tau2': get('tau2'),
        'I2': get('I2'),
        'H2': get('H2'),
        'k': numpy.array([numpy.nan if f is None else float(f.k) for f in fits]),
    }
    if transf is not None:
        out['estimate'] = metafopy.transf.apply_transf(out['estimate'], transf, targs, x.tau2)
        out['se'] = numpy.full(n, numpy.nan)
        out['ci_lb'] = metafopy.transf.apply_transf(out['ci_lb'], transf, targs, x.tau2)
        out['ci_ub'] = metafopy.transf.apply_transf(out['ci_ub'], transf, targs, x.tau2)
    out['ci_lb'], out['ci_ub'] = metafopy.methods.common.psort(out['ci_lb'], out['ci_ub'])
    return out


@leave1out.register
def leave1out_rma_uni(
    x: metafopy.models.rma_uni.RMAResult,
    cluster: numpy.typing.ArrayLike | str | None = None,
    digits: int | None = None,
    transf: collections.abc.Callable[..., typing.Any] | None = None,
    targs: typing.Any = None,
    *,
    na_action: metafopy.methods.common.NAAction = 'omit',
    progbar: bool = False,
) -> Leave1outResult:
    """Refit an intercept-only ``rma.uni`` model leaving out each study (or each ``cluster`` of studies).

    ``transf`` is applied to the estimates and CI bounds (the integral and mode
    transformations default to the model's ``tau2``); the standard errors are
    then NaN. ``cluster`` gives one value per study supplied (or a column name
    in the data the model was fitted to) and leaves out all studies of one
    cluster at a time. With ``na_action='exclude'``/``'pass'`` (and no
    ``cluster``) the output has one row per supplied study. ``digits`` defaults to the ``digits`` option, else to that of the model;
    ``progbar=True`` prints a progress bar to stderr while the refits are made.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    metafopy.methods.common.check_na_action(na_action)
    metafopy.methods.fitview.check_class(x, notav=metafopy.methods.fitview.NOT_ROBUST)
    if not x.int_only:
        raise metafopy.errors.InputError('Method only applicable to models without moderators.')
    if x.k == 1:
        raise metafopy.errors.InputError('Stopped because k = 1.')
    miss_cluster = cluster is None
    if miss_cluster:
        labels_k = numpy.arange(x.k)
    else:
        labels_k = cluster_values(x, cluster)
    ids = list(dict.fromkeys(labels_k.tolist()))
    if not miss_cluster:
        ids = sorted(ids)
    fits = [metafopy.methods.common.refit(x, labels_k != cid) for cid in metafopy.core.parallel.progress(ids, len(ids), progbar)]
    col = collect(fits, x, transf, targs)
    slab: tuple[str, ...]
    if miss_cluster:
        slab = tuple(f'-{s}' for s in x.slab)
    else:
        slab = tuple(f'-{cid}' for cid in ids)
    if na_action == 'fail' and not x.data.keep.all():
        raise metafopy.errors.InputError('Missing values in results.')
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    if miss_cluster and na_action in ('exclude', 'pass'):
        keys = ['estimate', 'se', 'stat', 'pval', 'ci_lb', 'ci_ub', 'Q', 'Qp', 'tau2', 'I2', 'H2']
        padded, full_slab = metafopy.methods.common.pad_to_supplied(x, *[col[key] for key in keys])
        col.update(dict(zip(keys, padded, strict=True)))
        slab = tuple(f'-{s}' for s in full_slab)
    fe = x.method in metafopy.models.tau2.FIXED_EFFECTS
    return Leave1outResult(
        estimate=col['estimate'],
        se=col['se'],
        stat=col['stat'],
        pval=col['pval'],
        ci_lb=col['ci_lb'],
        ci_ub=col['ci_ub'],
        Q=col['Q'],
        Qp=col['Qp'],
        tau2=None if fe else col['tau2'],
        I2=col['I2'],
        H2=col['H2'],
        slab=slab,
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        stat_name='tval' if x.test in ('knha', 'adhoc', 't') else 'zval',
        digits=metafopy.mfopt.option_digits(digits, x.digits),
        transf=transf is not None,
    )


def cluster_values(x: metafopy.models.rma_uni.RMAResult, values: numpy.typing.ArrayLike | str | None) -> numpy.typing.NDArray[typing.Any]:
    """Resolve a per-study variable (array over the supplied rows or a column of the model's data) to the studies used.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    if isinstance(values, str):
        if x.data.data is None or values not in x.data.data.columns:
            raise metafopy.errors.InputError(f'cannot find the variable {values!r} in the data')
        arr = x.data.data[values].to_numpy()
    else:
        arr = numpy.asarray(values)
    keep = x.data.keep
    if arr.shape[0] != keep.shape[0]:
        raise metafopy.errors.InputError(f'Length of the variable ({arr.shape[0]}) does not match the length of the data ({keep.shape[0]}).')
    return arr[keep]


@functools.singledispatch
def cumul(x: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> CumulResult:
    """Cumulative meta-analysis of a fitted model; see :func:`_cumul_rma_uni`.

    Raises:
        metafopy.errors.ArgumentError: an argument has the wrong type or is not accepted.
    """
    raise metafopy.errors.ArgumentError(f'cumul() not implemented for {type(x).__name__}')


@cumul.register
def cumul_rma_uni(
    x: metafopy.models.rma_uni.RMAResult,
    order: numpy.typing.ArrayLike | str | None = None,
    digits: int | None = None,
    transf: collections.abc.Callable[..., typing.Any] | None = None,
    targs: typing.Any = None,
    collapse: bool = False,
    *,
    decreasing: bool = False,
    progbar: bool = False,
) -> CumulResult:
    """Fit the intercept-only model to the first ``i`` studies for ``i = 1, ..., k``.

    ``order`` is a variable (one value per supplied study, or a column name of
    the model's data) giving the order in which studies are added; without it
    the original order is used. With ``collapse=True`` studies sharing the same
    value of ``order`` are added together. Studies dropped for missing values
    in the fit are skipped (the ``na.omit`` behaviour). ``digits`` defaults to the ``digits`` option, else to that of the model; ``progbar=True``
    prints a progress bar to stderr while the refits are made.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    metafopy.methods.fitview.check_class(x, notav=metafopy.methods.fitview.NOT_ROBUST)
    if not x.int_only:
        raise metafopy.errors.InputError('Method only applicable to models without moderators.')
    k = x.k
    if order is None:
        # metafor orders by the position among all rows supplied, so `order` holds the positions of the studies used
        orvar = pandas.Series(numpy.flatnonzero(x.data.keep))
        collapse = False
    else:
        orvar = pandas.Series(cluster_values(x, order))
    # R: order(orvar, decreasing=, na.last=NA); the radix sort is stable in both directions
    sub = orvar.dropna().sort_values(ascending=not decreasing, kind='stable')
    srt = sub.index.to_numpy()
    ov = orvar.iloc[srt].to_numpy()
    uor = numpy.array(list(dict.fromkeys(ov.tolist())), dtype=ov.dtype) if collapse else ov
    n = len(uor)
    fits: list[metafopy.models.rma_uni.RMAResult | None] = []
    for i in metafopy.core.parallel.progress(range(n), n, progbar):
        incl = srt[numpy.isin(ov, uor[: i + 1])] if collapse else srt[: i + 1]
        mask = numpy.zeros(k, dtype=numpy.bool_)
        mask[incl] = True
        fits.append(metafopy.methods.common.refit(x, mask))
    col = collect(fits, x, transf, targs)
    slab = tuple(label(u) for u in uor) if collapse else tuple(numpy.asarray(x.slab, dtype=object)[srt].tolist())
    fe = x.method in metafopy.models.tau2.FIXED_EFFECTS
    return CumulResult(
        estimate=col['estimate'],
        se=col['se'],
        stat=col['stat'],
        pval=col['pval'],
        ci_lb=col['ci_lb'],
        ci_ub=col['ci_ub'],
        Q=col['Q'],
        Qp=col['Qp'],
        tau2=None if fe else col['tau2'],
        I2=col['I2'],
        H2=col['H2'],
        slab=slab,
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        stat_name='tval' if x.test in ('knha', 'adhoc', 't') else 'zval',
        digits=metafopy.mfopt.option_digits(digits, x.digits),
        transf=transf is not None,
        k=col['k'],
        order=uor,
        measure=x.measure,
        level=x.level,
        slab_null=bool(numpy.all([lab == str(i + 1) for lab, i in zip(x.slab, numpy.flatnonzero(x.data.keep), strict=True)])) and not collapse,
    )


def label(value: typing.Any) -> str:
    """An ordering value as R's ``as.character`` would print it (``1948`` rather than ``1948.0``)."""
    if isinstance(value, float | numpy.floating) and float(value).is_integer():
        return str(int(value))
    return str(value)
