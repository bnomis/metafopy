"""Plot of the likelihood functions of the individual studies (metafor's ``llplot``).

For ``measure='GEN'`` each study contributes the normal likelihood of its effect size ``yi`` with sampling variance ``vi``; for
``measure='OR'`` the likelihood of the (log) odds ratio is the conditional one given the margins of the 2x2 table, which follows Fisher's
non-central hypergeometric distribution (Fog 2008, doi:10.1080/03610910701790236). With ``scale=True`` every likelihood is rescaled to area
one, so that the curves show how much information each study has about the effect. See
https://wviechtb.github.io/metafor/reference/llplot.html and Stijnen et al. (2010), doi:10.1002/sim.3767.
"""

import collections.abc
import dataclasses
import math
import typing
import warnings

import matplotlib.axes
import numpy
import numpy.typing
import scipy.stats

import metafopy.core.nchg
import metafopy.effects
import metafopy.errors
import metafopy.metadata
import metafopy.plot
import metafopy.plot.draw
import metafopy.plot.plain
import metafopy.plot.rutils

__all__ = ['LlplotResult', 'llplot']

type FloatArray = numpy.typing.NDArray[numpy.float64]
type Column = metafopy.metadata.Column


@dataclasses.dataclass(frozen=True, slots=True)
class LlplotResult:
    """The likelihoods drawn (what metafor's ``llplot()`` returns invisibly): ``lls[i, j]`` is the likelihood of study ``i`` at ``xs[j]``.

    Values outside of the (very wide) confidence interval of a study are NaN and not drawn.
    """

    ax: matplotlib.axes.Axes
    xs: FloatArray
    lls: FloatArray


def odds_ratio_likelihood(a: FloatArray, b: FloatArray, c: FloatArray, d: FloatArray, xs: FloatArray) -> FloatArray:
    """The conditional likelihood of each table at the log odds ratios ``xs`` (rows are studies)."""
    tables = metafopy.core.nchg.Tables.from_counts(a, b, c, d)
    out = numpy.empty((a.shape[0], xs.shape[0]))
    for i in range(a.shape[0]):
        lo = numpy.clip(xs, -metafopy.core.nchg.LOG_ODDS_BOUND, metafopy.core.nchg.LOG_ODDS_BOUND)
        w = tables.log_base[i][:, None] + tables.support[i][:, None] * lo[None, :]
        top = w.max(axis=0)
        out[i] = numpy.exp(w[tables.index[i]] - top) / numpy.exp(w - top).sum(axis=0)
    return out


def r_rank(x: FloatArray) -> numpy.typing.NDArray[numpy.intp]:
    """``rank(x)`` as an integer index (average ranks of ties are truncated when used to index a vector, as R does)."""
    return numpy.asarray(scipy.stats.rankdata(x, method='average'), dtype=float).astype(numpy.intp) - 1


def llplot(
    measure: str = 'GEN',
    yi: Column | None = None,
    vi: Column | None = None,
    sei: Column | None = None,
    ai: Column | None = None,
    bi: Column | None = None,
    ci: Column | None = None,
    di: Column | None = None,
    n1i: Column | None = None,
    n2i: Column | None = None,
    data: typing.Any = None,
    subset: numpy.typing.ArrayLike | None = None,
    drop00: bool = True,
    xvals: int = 1000,
    xlim: collections.abc.Sequence[float] | None = None,
    ylim: collections.abc.Sequence[float] | None = None,
    xlab: str | None = None,
    ylab: str | None = None,
    scale: bool = True,
    lty: str | collections.abc.Sequence[str] | None = None,
    lwd: float | collections.abc.Sequence[float] | None = None,
    col: str | collections.abc.Sequence[str] | None = None,
    level: float = 99.99,
    refline: float | None = 0,
    *,
    onlyo1: bool = False,
    addyi: bool = True,
    addvi: bool = True,
    ax: matplotlib.axes.Axes | None = None,
) -> LlplotResult:
    """Plot the likelihood function of the true effect given the data of each study (``llplot``).

    Mirrors metafor's ``llplot()``, see <https://wviechtb.github.io/metafor/reference/llplot.html>.

    ``measure`` is ``'GEN'`` (give ``yi`` with ``vi`` or ``sei``) or ``'OR'`` (give the counts ``ai``, ``bi``, ``ci``, ``di``, or ``ai``,
    ``n1i``, ``ci``, ``n2i``). Studies with two zero cells in a column (``drop00``) are left out for ``'OR'``; tables with a zero cell are drawn
    dashed and those with two dotted, as in metafor. ``xvals`` is the number of points of each curve, ``level`` the confidence level of the
    interval outside which the curves are not drawn, ``scale`` rescales each likelihood to area one, and ``lty``, ``lwd`` and ``col``
    are per study (by default the more precise studies get thinner and darker lines). ``refline`` draws a vertical line.

    Args:
        measure: ``'GEN'`` (the default; give ``yi`` with ``vi`` or ``sei``) or ``'OR'`` (give the counts ``ai``, ``bi``, ``ci``, ``di``, or ``ai``, ``n1i``,
            ``ci``, ``n2i``).
        yi: Observed effect sizes (array or column name of ``data``), for ``measure='GEN'``.
        vi: Sampling variances (array or column name), for ``measure='GEN'``.
        sei: Standard errors (array or column name), for ``measure='GEN'``; an alternative to ``vi``.
        ai: 2x2 table counts, upper left cell (array or column name), for ``measure='OR'``.
        bi: 2x2 table counts, upper right cell.
        ci: 2x2 table counts, lower left cell.
        di: 2x2 table counts, lower right cell.
        n1i: Size of group 1; an alternative to ``bi``.
        n2i: Size of group 2; an alternative to ``di``.
        data: Data frame holding the variables named in the other arguments.
        subset: Which studies to use: a boolean array or an array of 0-based indices.
        drop00: Drop studies with two zero cells in a column (``measure='OR'``). Default ``True``.
        xvals: Number of points of each likelihood curve.
        xlim: Limits of the horizontal axis as (low, high). Default: derived from the data.
        ylim: Limits of the vertical axis as (low, high). Default: derived from the data.
        xlab: Label of the horizontal axis. Default: derived from the model.
        ylab: Label of the vertical axis. Default: derived from the model.
        scale: Rescale each likelihood to area one.
        lty: Line types per study (by default tables with a zero cell are dashed and those with two dotted).
        lwd: Line widths per study (by default the more precise studies get thinner lines).
        col: Colours per study (by default the more precise studies get darker lines).
        level: Confidence level in percent of the interval outside which the curves are not drawn.
        refline: Position of a vertical reference line, or ``None`` for none.
        onlyo1: Passed to [escalc][metafopy.escalc]: only the first outcome of the 2x2 tables is used.
        addyi: Passed to [escalc][metafopy.escalc]: apply ``add`` to the effect size.
        addvi: Passed to [escalc][metafopy.escalc]: apply ``add`` to the sampling variance.
        ax: Matplotlib axes to draw into. Default: the current axes.

    Returns:
        The points and the likelihoods of the studies, and the axes.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    if measure not in ('GEN', 'OR'):
        raise metafopy.errors.InputError('Currently only measure="GEN" or measure="OR" can be specified.')
    axx = metafopy.plot.rutils.get_ax(ax)
    alpha = metafopy.plot.rutils.level_to_alpha(level)
    mask = None if subset is None else numpy.asarray(subset)
    id0: numpy.typing.NDArray[numpy.bool_] | None = None
    id00: numpy.typing.NDArray[numpy.bool_] | None = None
    counts: tuple[FloatArray, FloatArray, FloatArray, FloatArray] | None = None
    if measure == 'GEN':
        yv = metafopy.metadata.column(yi, data, 'yi')
        vv = metafopy.metadata.column(vi, data, 'vi')
        sv = metafopy.metadata.column(sei, data, 'sei')
        if vv is None:
            if sv is None:
                raise metafopy.errors.InputError("Must specify the 'vi' or 'sei' argument.")
            vv = sv**2
        if yv is None:
            raise metafopy.errors.InputError(
                'Cannot construct plot. Check that all of the required information is specified\n  via the appropriate arguments (i.e., yi, vi).'
            )
        if yv.shape[0] != vv.shape[0]:
            raise metafopy.errors.InputError('Supplied data vectors are not all of the same length.')
        k_all = int(yv.shape[0])
        ids = numpy.arange(k_all)
        yv, vv = yv.copy(), vv.copy()
    else:
        a, b, c, d = (metafopy.metadata.column(v, data, name) for v, name in ((ai, 'ai'), (bi, 'bi'), (ci, 'ci'), (di, 'di')))
        n1 = metafopy.metadata.column(n1i, data, 'n1i')
        n2 = metafopy.metadata.column(n2i, data, 'n2i')
        if b is None and n1 is not None and a is not None:
            b = n1 - a
        if d is None and n2 is not None and c is not None:
            d = n2 - c
        if a is None or b is None or c is None or d is None:
            raise metafopy.errors.InputError(
                'Cannot construct plot. Check that all of the required information is specified\n'
                '  via the appropriate arguments (i.e., ai, bi, ci, di or ai, n1i, ci, n2i).'
            )
        a, b, c, d = a.copy(), b.copy(), c.copy(), d.copy()
        if min(a.shape[0], b.shape[0], c.shape[0], d.shape[0]) != max(a.shape[0], b.shape[0], c.shape[0], d.shape[0]):
            raise metafopy.errors.InputError('Supplied data vectors are not all of the same length.')
        if numpy.any((a > a + b) | (c > c + d)) or numpy.any(numpy.array([a, b, c, d]) < 0):
            raise metafopy.errors.InputError('One or more counts are negative or larger than the corresponding group size.')
        k_all = int(a.shape[0])
        ids = numpy.arange(k_all)
        id0 = (a == 0) | (b == 0) | (c == 0) | (d == 0)
        id00 = ((a == 0) & (c == 0)) | ((b == 0) & (d == 0))
        id0, id00 = id0 & ~numpy.isnan(a + b + c + d), id00 & ~numpy.isnan(a + b + c + d)
        if drop00:
            for arr in (a, b, c, d):
                arr[id00] = numpy.nan
        if mask is not None:
            sel = numpy.flatnonzero(mask) if mask.dtype == numpy.bool_ else mask.astype(numpy.intp)
            a, b, c, d = a[sel], b[sel], c[sel], d[sel]
        frame = metafopy.escalc('OR', ai=a, bi=b, ci=c, di=d, drop00=drop00, onlyo1=onlyo1, addyi=addyi, addvi=addvi)
        yv, vv = frame['yi'].to_numpy(dtype=float), frame['vi'].to_numpy(dtype=float)
        counts = (a, b, c, d)

    def per_study(v: typing.Any, name: str) -> list[typing.Any] | None:
        if v is None:
            return None
        out = [v] * k_all if isinstance(v, str) or numpy.ndim(v) == 0 else list(v)
        if len(out) != k_all:
            raise metafopy.errors.InputError(f"Length of the '{name}' argument ({len(out)}) does not match the length of the data ({k_all}).")
        return out

    lty_l, lwd_l, col_l = per_study(lty, 'lty'), per_study(lwd, 'lwd'), per_study(col, 'col')
    if mask is not None:
        sel = numpy.flatnonzero(mask) if mask.dtype == numpy.bool_ else mask.astype(numpy.intp)
        if measure == 'GEN':
            yv, vv = yv[sel], vv[sel]
        ids = ids[sel]
        lty_l, lwd_l, col_l = ([x[i] for i in sel] if x is not None else None for x in (lty_l, lwd_l, col_l))
        if id0 is not None and id00 is not None:
            id0, id00 = id0[sel], id00[sel]
    has_na = numpy.isnan(yv) | numpy.isnan(vv)
    if counts is not None:
        has_na = numpy.isnan(counts[0]) | numpy.isnan(counts[1]) | numpy.isnan(counts[2]) | numpy.isnan(counts[3])
    if has_na.any():
        ok = ~has_na
        yv, vv = yv[ok], vv[ok]
        if counts is not None:
            counts = (counts[0][ok], counts[1][ok], counts[2][ok], counts[3][ok])
            assert id0 is not None
            assert id00 is not None
            id0, id00 = id0[ok], id00[ok]
        lty_l, lwd_l, col_l = ([x[i] for i, o in enumerate(ok) if o] if x is not None else None for x in (lty_l, lwd_l, col_l))
        warnings.warn(f'{int(has_na.sum())} {"studies" if has_na.sum() > 1 else "study"} with NAs omitted from plotting.', stacklevel=2)
    k = int(yv.shape[0])
    if k < 1:
        raise metafopy.errors.InputError('Processing terminated since k = 0.')
    if lty_l is None:
        if counts is None:
            lty_l = ['solid'] * k
        else:
            assert id0 is not None
            assert id00 is not None
            lty_l = ['dotted' if z else ('dashed' if o else 'solid') for o, z in zip(id0, id00, strict=True)]
    rank_vi = r_rank(vv)
    if lwd_l is None:
        lwd_l = list(numpy.linspace(4, 0.4, k)[rank_vi])
    if col_l is None:
        shades = [metafopy.plot.rutils.coladj(-float(v)) for v in numpy.linspace(0.8, 0.2, k)]
        col_l = [shades[i] for i in rank_vi]
    zcrit = float(scipy.stats.norm.isf(alpha / 2))
    ci_lb, ci_ub = yv - zcrit * numpy.sqrt(vv), yv + zcrit * numpy.sqrt(vv)
    xl = (float(numpy.nanmin(ci_lb)), float(numpy.nanmax(ci_ub))) if xlim is None else (float(min(xlim)), float(max(xlim)))
    xs = numpy.linspace(xl[0], xl[1], int(xvals))
    if counts is None:
        lls = scipy.stats.norm.pdf(yv[:, None], loc=xs[None, :], scale=numpy.sqrt(vv)[:, None])
    else:
        lls = odds_ratio_likelihood(*counts, xs)
    outside = (xs[None, :] < ci_lb[:, None]) | (xs[None, :] > ci_ub[:, None])
    if scale:
        for i in range(k):
            lls[i] = lls[i] / numpy.trapezoid(lls[i], xs)
    lls[outside] = numpy.nan
    yl = (0.0, float(numpy.nanmax(lls))) if ylim is None else (float(min(ylim)), float(max(ylim)))
    xlab_c = ('Observed Outcome' if measure == 'GEN' else 'Log Odds Ratio') if xlab is None else xlab
    ylab_c = ('Scaled Likelihood' if scale else 'Likelihood') if ylab is None else ylab
    metafopy.plot.plain.plain_plot(axx, xl, yl, xlab_c, ylab_c, thin_ylabels=True)
    for i in numpy.argsort(1 / vv, kind='stable'):
        metafopy.plot.draw.line(axx, xs, lls[i], lty=lty_l[i], col=col_l[i], lw=float(lwd_l[i]))
    if refline is not None and not math.isnan(refline):
        metafopy.plot.draw.vline(axx, refline, lty='solid', lw=2.0)
    return LlplotResult(axx, xs, lls)
