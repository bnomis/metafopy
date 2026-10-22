"""Leave-one-out and cumulative meta-analyses (metafor's ``leave1out.rma.uni`` and ``cumul.rma.uni``).

``leave1out`` refits an intercept-only model with each study (or cluster of
studies) left out in turn; ``cumul`` adds the studies one at a time in a given
order (Chalmers & Lau 1993, doi:10.1177/096228029300200204; Lau, Schmid &
Chalmers 1995, doi:10.1016/0895-4356(94)00106-Z). See
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
import metafopy.core.timing
import metafopy.errors
import metafopy.methods.blup
import metafopy.methods.common
import metafopy.methods.fitview
import metafopy.methods.listrma
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

    Attributes:
        estimate: The pooled estimates, one per left-out study.
        se: Their standard errors (NaN after a transformation).
        stat: The test statistics (z or t values) of the estimates.
        pval: Their p-values.
        ci_lb: Lower bounds of the confidence intervals.
        ci_ub: Upper bounds of the confidence intervals.
        Q: Test statistics for the test of heterogeneity.
        Qp: Their p-values.
        tau2: Estimates of ``tau^2``, or ``None`` for equal-, fixed- and common-effects models.
        I2: Values of ``I^2``.
        H2: Values of ``H^2``.
        slab: Row labels (``'-<study>'`` for a left-out study).
        stat_name: Name of the test statistic column, ``'zval'`` or ``'tval'``.
        digits: Number of decimals used when the result is printed.
        digit_map: Per-quantity decimals when ``digits`` was given as a mapping, else ``None``.
        transf: Whether a transformation was applied.
        k: Number of studies in each fit (cumulative results; ``None`` for a leave-one-out result).
        order: Values of the ordering variable (cumulative results; ``None`` for a leave-one-out result).
        measure: Effect size measure of the model (set for cumulative results; ``'GEN'`` for a leave-one-out result).
        level: Alpha of the confidence intervals (set for cumulative results; 0.05 for a leave-one-out result).
        slab_null: Whether the studies carry no labels of their own.
        ids: 0-based positions of the studies among those supplied, if known.
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
    digit_map: collections.abc.Mapping[str, int] | None = None
    transf: bool = False
    k: FloatArray | None = None
    order: numpy.typing.NDArray[typing.Any] | None = None
    measure: str = 'GEN'
    level: float = 0.05
    slab_null: bool = False
    ids: numpy.typing.NDArray[numpy.intp] | None = None

    @property
    def zval(self) -> FloatArray:
        """The test statistics of the pooled effects, metafor's ``zval`` (``tval`` for the t-based tests, see :attr:`stat_name`)."""
        return self.stat

    @property
    def tval(self) -> FloatArray:
        """The test statistics of the pooled effects, metafor's ``tval`` for the t-based tests (the same array as :attr:`zval`)."""
        return self.stat

    @property
    def test(self) -> str:
        """``'t'`` for the t-based tests and ``'z'`` otherwise (metafor stores the model's ``test``, so ``'knha'`` and ``'adhoc'`` read ``'t'`` here)."""
        return 't' if self.stat_name == 'tval' else 'z'

    def __getitem__(self, index: typing.Any) -> typing.Self:
        """The rows ``index`` (metafor's ``[.list.rma``; a NumPy index, 0-based)."""
        return metafopy.methods.listrma.rows(self, index, self.estimate.shape[0])

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
        by_kind = metafopy.mfopt.digits_map(self.digit_map, self.digits)
        spec: list[tuple[str, FloatArray, int]] = []
        if self.k is not None:
            spec.append(('k', self.k, 0))
        spec.append(('estimate', self.estimate, by_kind['est']))
        if not self.transf:
            spec.append(('se', self.se, by_kind['se']))
        spec += [
            (self.stat_name, self.stat, by_kind['test']),
            ('pval', self.pval, by_kind['pval']),
            ('ci.lb', self.ci_lb, by_kind['ci']),
            ('ci.ub', self.ci_ub, by_kind['ci']),
            ('Q', self.Q, by_kind['test']),
            ('Qp', self.Qp, by_kind['pval']),
        ]
        if self.tau2 is not None:
            spec.append(('tau2', self.tau2, by_kind['var']))
        spec += [('I2', self.I2, by_kind['het']), ('H2', self.H2, by_kind['het'])]
        cols = [(name, [metafopy.printing.fmt_num(float(v), dd) for v in vals]) for name, vals, dd in spec]
        return '\n'.join(['', *metafopy.methods.blup.format_r_frame(cols, self.slab), ''])

    def with_digits(self, digits: typing.Any) -> typing.Self:
        """The result printed with ``digits`` (a number, or a mapping with any of the keys of :data:`metafopy.mfopt.DIGIT_KEYS`)."""
        return metafopy.mfopt.with_digits(self, digits, self.digit_map or self.digits)


@dataclasses.dataclass(frozen=True, slots=True)
class CumulResult(Leave1outResult):
    """Cumulative meta-analysis results ([Leave1outResult][metafopy.Leave1outResult] plus ``k`` and the ordering variable).

    Each row holds the result of the model fitted to the studies added so far, in the order the studies were added. ``k`` is the number of
    studies included and ``order`` the values of the ordering variable. The other attributes are those of
    [Leave1outResult][metafopy.Leave1outResult].
    """


@functools.singledispatch
def leave1out(x: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> Leave1outResult:
    """Leave-one-out analysis of a fitted model: the model is refitted leaving out each study (or cluster) in turn.

    This is a single-dispatch function: the implementation is chosen from the class of ``x``. See
    [leave1out_rma_uni][metafopy.methods.leave1out.leave1out_rma_uni] for [rma_uni][metafopy.rma_uni] fits and
    [leave1out_fixed_binary][metafopy.methods.binary.leave1out_fixed_binary] for [rma_mh][metafopy.rma_mh] and [rma_peto][metafopy.rma_peto]
    fits.

    Args:
        x: A fitted model object.
        *args: Positional arguments passed to the implementation chosen by the class of ``x``.
        **kwargs: Keyword arguments passed to the implementation chosen by the class of ``x``.

    Returns:
        What the implementation returns, a [Leave1outResult][metafopy.Leave1outResult].

    Raises:
        metafopy.errors.ArgumentError: an argument has the wrong type or is not accepted.

    Metafor reference:
        [leave1out](https://wviechtb.github.io/metafor/reference/leave1out.html)
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
@metafopy.core.timing.timed
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
    ``cluster``) the output has one row per supplied study.

    Args:
        x: A model without moderators (an equal- or random-effects model) fitted by [rma_uni][metafopy.rma_uni]; not allowed for a model
            passed through [robust][metafopy.robust].
        cluster: Clustering variable: a column name of the data the model was fitted to, or an array with one value per supplied study. All
            studies of one cluster are left out at a time and the rows are labelled by the sorted cluster ids. ``None`` (the default) leaves out
            one study at a time. Unlike metafor, missing values in ``cluster`` are not rejected.
        digits: Number of decimals used when the result is printed. ``None`` uses the ``digits`` option, else the ``digits`` of the model.
        transf: A function applied to the estimates and confidence interval bounds (for example ``numpy.exp``). The integral and mode
            transformations of `metafopy.transf` use the ``tau2`` of the model. The standard errors are then NaN.
            ``None`` (the default) applies no transformation.
        targs: Extra argument passed to ``transf`` as ``transf(x, targs)``. ``None`` (the default) calls ``transf(x)``.
        na_action: How studies dropped for missing values are handled: ``'omit'`` (the default; no row for them), ``'exclude'`` or ``'pass'``
            (a row of NaN for each, only without ``cluster``) or ``'fail'`` (which, as in metafor, does not raise an error here and gives the
            same result as ``'omit'``).
        progbar: Print a progress bar to stderr while the refits are made (default ``False``).

    Returns:
        A [Leave1outResult][metafopy.Leave1outResult] with one row per left-out study or cluster (labelled ``-<label>``) holding the
        re-estimated ``estimate``, ``se``, ``stat`` (``zval``, or ``tval`` for t, knha and adhoc tests), ``pval``, ``ci_lb``, ``ci_ub``, ``Q``, ``Qp``, ``I2``
        and ``H2`` (and ``tau2`` for random-effects models). A row is NaN when its refit fails.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid, the model has moderators or there is a single study.
        metafopy.errors.NotSupportedError: the model was passed through [robust][metafopy.robust].

    Metafor reference:
        [leave1out](https://wviechtb.github.io/metafor/reference/leave1out.html)
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
    # na.fail does not stop metafor here (the fit has already dropped the studies with missing values); the labels are those of 'omit'
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
        digit_map=metafopy.mfopt.digit_map_of(digits, x.digits),
        transf=transf is not None,
        ids=x.ids[x.not_na] if len(slab) == int(x.not_na.sum()) else x.ids,
    )


def cluster_values(
    x: metafopy.models.rma_uni.RMAResult, values: numpy.typing.ArrayLike | str | None, rows: numpy.typing.NDArray[numpy.bool_] | None = None
) -> numpy.typing.NDArray[typing.Any]:
    """Resolve a per-study variable (array over the supplied rows or a column of the model's data) to the studies used, or to ``rows``.

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
    return arr[keep if rows is None else rows]


@functools.singledispatch
def cumul(x: typing.Any, *args: typing.Any, **kwargs: typing.Any) -> CumulResult:
    """Cumulative meta-analysis of a fitted model: the model is fitted to the first ``i`` studies for ``i = 1, ..., k``.

    This is a single-dispatch function: the implementation is chosen from the class of ``x``. See
    [cumul_rma_uni][metafopy.methods.leave1out.cumul_rma_uni] for [rma_uni][metafopy.rma_uni] fits and
    [cumul_fixed_binary][metafopy.methods.binary.cumul_fixed_binary] for [rma_mh][metafopy.rma_mh] and [rma_peto][metafopy.rma_peto] fits.

    Args:
        x: A fitted model object.
        *args: Positional arguments passed to the implementation chosen by the class of ``x``.
        **kwargs: Keyword arguments passed to the implementation chosen by the class of ``x``.

    Returns:
        What the implementation returns, a [CumulResult][metafopy.CumulResult].

    Raises:
        metafopy.errors.ArgumentError: an argument has the wrong type or is not accepted.

    Metafor reference:
        [cumul](https://wviechtb.github.io/metafor/reference/cumul.html)
    """
    raise metafopy.errors.ArgumentError(f'cumul() not implemented for {type(x).__name__}')


@cumul.register
@metafopy.core.timing.timed
def cumul_rma_uni(
    x: metafopy.models.rma_uni.RMAResult,
    order: numpy.typing.ArrayLike | str | None = None,
    digits: int | None = None,
    transf: collections.abc.Callable[..., typing.Any] | None = None,
    targs: typing.Any = None,
    collapse: bool = False,
    *,
    decreasing: bool = False,
    na_action: metafopy.methods.common.NAAction = 'omit',
    progbar: bool = False,
) -> CumulResult:
    """Fit the intercept-only model to the first ``i`` studies for ``i = 1, ..., k`` (metafor's ``cumul.rma.uni``).

    A cumulative meta-analysis describes the accumulation of evidence as studies are added, typically in chronological order (Chalmers & Lau,
    1993, [doi:10.1177/096228029300200204](https://doi.org/10.1177/096228029300200204); Lau, Schmid & Chalmers, 1995,
    [doi:10.1016/0895-4356(94)00106-Z](https://doi.org/10.1016/0895-4356(94)00106-Z)). Studies dropped for missing values in the fit are skipped.
    ``na_action`` stands for R's ``options(na.action=)``, as in metafor (see
    <https://github.com/wviechtb/metafor/blob/master/R/cumul.rma.uni.r>).

    Args:
        x: A model without moderators (an equal- or random-effects model) fitted by [rma_uni][metafopy.rma_uni]; not allowed for a model
            passed through [robust][metafopy.robust].
        order: Variable giving the order in which the studies are added, ascending: a column name of the data the model was fitted to, or an
            array with one value per supplied study. Studies with a missing value are not added. ``None`` (the default) uses the original order
            and ignores ``collapse``.
        digits: Number of decimals used when the result is printed. ``None`` uses the ``digits`` option, else the ``digits`` of the model.
        transf: A function applied to the estimates and confidence interval bounds (for example ``numpy.exp``). The integral and mode
            transformations of `metafopy.transf` use the ``tau2`` of the model. The standard errors are then NaN. ``None`` (the default)
            applies no transformation.
        targs: Extra argument passed to ``transf`` as ``transf(x, targs)``. ``None`` (the default) calls ``transf(x)``.
        collapse: Add studies with the same value of ``order`` together instead of one at a time (default ``False``).
        decreasing: Add the studies in descending order of ``order`` (default ``False``). metafor takes this as an undocumented extra argument.
        na_action: How studies dropped for missing values are handled: ``'omit'`` (the default; no row for them), ``'exclude'`` or ``'pass'``
            (a row of missing values for each) or ``'fail'`` (raise an error).
        progbar: Print a progress bar to stderr while the refits are made (default ``False``).

    Returns:
        A [CumulResult][metafopy.CumulResult] with one row per added study (or group of studies) holding the number of studies included ``k``,
        the values of ``order`` (the 0-based positions of the studies if ``order`` is ``None``) and the re-estimated ``estimate``, ``se``,
        ``stat`` (``zval``, or ``tval`` for t, knha and adhoc tests), ``pval``, ``ci_lb``, ``ci_ub``, ``Q``, ``Qp``, ``I2`` and ``H2``
        (and ``tau2`` for random-effects models).

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid, the model has moderators, or ``na_action='fail'`` and values are missing.
        metafopy.errors.NotSupportedError: the model was passed through [robust][metafopy.robust].

    Metafor reference:
        [cumul](https://wviechtb.github.io/metafor/reference/cumul.html)
    """
    metafopy.methods.fitview.check_class(x, notav=metafopy.methods.fitview.NOT_ROBUST)
    metafopy.methods.common.check_na_action(na_action)
    if na_action == 'fail' and not metafopy.methods.common.uni_not_na(x).all():
        raise metafopy.errors.InputError('Missing values in data.')
    if not x.int_only:
        raise metafopy.errors.InputError('Method only applicable to models without moderators.')
    k = x.k
    na_rows = x.data.na_rows
    # na.exclude and na.pass keep a row of missing values for each study left out of the fit (the rows `sel` are those of the subset)
    fill = na_action in ('exclude', 'pass') and na_rows is not None and bool(na_rows.any())
    sel = (x.data.keep | na_rows) if fill and na_rows is not None else x.data.keep
    not_na = x.data.keep[sel]
    used_pos = numpy.cumsum(not_na) - 1
    if order is None:
        # metafor orders by the position among all rows supplied, so `order` holds the positions of the studies used
        orvar = pandas.Series(numpy.flatnonzero(sel))
        collapse = False
    else:
        orvar = pandas.Series(cluster_values(x, order, sel))
    # R: order(orvar, decreasing=, na.last=NA); the radix sort is stable in both directions
    sub = orvar.dropna().sort_values(ascending=not decreasing, kind='stable')
    srt = sub.index.to_numpy()
    ov = orvar.iloc[srt].to_numpy()
    uor = numpy.array(list(dict.fromkeys(ov.tolist())), dtype=ov.dtype) if collapse else ov
    n = len(uor)
    fits: list[metafopy.models.rma_uni.RMAResult | None] = []
    for i in metafopy.core.parallel.progress(range(n), n, progbar):
        added = srt[numpy.isin(ov, uor[i : i + 1])] if collapse else srt[i : i + 1]
        if not not_na[added].any():
            fits.append(None)  # a study (or all studies of a group) without data is not added, its row stays missing
            continue
        incl = srt[numpy.isin(ov, uor[: i + 1])] if collapse else srt[: i + 1]
        incl = incl[not_na[incl]]
        mask = numpy.zeros(k, dtype=numpy.bool_)
        mask[used_pos[incl]] = True
        fits.append(metafopy.methods.common.refit(x, mask))
    col = collect(fits, x, transf, targs)
    if fill:
        labels_all = x.data.labels_all
        labels = (
            numpy.asarray(labels_all, dtype=object)[sel] if labels_all is not None else numpy.array([str(i + 1) for i in numpy.flatnonzero(sel)], dtype=object)
        )
    else:
        labels = numpy.asarray(x.slab, dtype=object)
    slab = tuple(label(u) for u in uor) if collapse else tuple(labels[srt].tolist())
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
        digit_map=metafopy.mfopt.digit_map_of(digits, x.digits),
        transf=transf is not None,
        k=col['k'],
        order=uor,
        measure=x.measure,
        level=x.level,
        ids=None if collapse else (x.ids if fill else x.ids[x.not_na])[srt],
        slab_null=bool(numpy.all([lab == str(i + 1) for lab, i in zip(x.slab, numpy.flatnonzero(x.data.keep), strict=True)])) and not collapse,
    )


def label(value: typing.Any) -> str:
    """An ordering value as R's ``as.character`` would print it (``1948`` rather than ``1948.0``)."""
    if isinstance(value, float | numpy.floating) and float(value).is_integer():
        return str(int(value))
    return str(value)
