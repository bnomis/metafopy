"""Plots of cumulative meta-analyses (metafor's ``forest.cumul.rma`` and ``plot.cumul.rma``).

``forest_cumul`` (also reached as ``forest(cumul(res))``) draws the pooled estimate after each added study as a row of a forest plot;
``plot_cumul`` plots the heterogeneity statistic against the pooled estimate, joining the successive results by a line that fades from
light to dark. See https://wviechtb.github.io/metafor/reference/forest.cumul.rma.html,
https://wviechtb.github.io/metafor/reference/plot.cumul.rma.html and Lau et al. (1992), doi:10.1056/NEJM199207023270206.
"""

import collections.abc
import dataclasses
import typing

import matplotlib.axes
import numpy
import numpy.typing
import pandas

import metafopy.errors
import metafopy.methods.common
import metafopy.methods.leave1out
import metafopy.plot
import metafopy.plot.draw
import metafopy.plot.rutils

__all__ = ['PlotCumulResult', 'forest_cumul', 'plot_cumul']

type FloatArray = numpy.typing.NDArray[numpy.float64]
type Func = collections.abc.Callable[..., typing.Any]

YAXES = ('tau2', 'tau', 'I2', 'H2')
# auditme: accept[non-alphabetical-dict-keys] order follows metafor's documented order of the yaxis options
YLABELS = {
    'tau2': r'Amount of Heterogeneity $(\tau^2)$',
    'tau': r'Amount of Heterogeneity $(\tau)$',
    'I2': r'Percentage of Variability due to Heterogeneity $(I^2)$',
    'H2': r'Ratio of Total Variability to Sampling Variability $(H^2)$',
}


@dataclasses.dataclass(frozen=True, slots=True)
class PlotCumulResult:
    """The plotted points of a plot of cumulative results (``estim`` the pooled estimates, ``yval`` the heterogeneity statistic, ``col`` their colours)."""

    ax: matplotlib.axes.Axes
    estim: FloatArray
    yval: FloatArray
    col: tuple[str, ...]

    def to_frame(self) -> pandas.DataFrame:
        return pandas.DataFrame({'estim': self.estim, 'yval': self.yval, 'col': list(self.col)})


def forest_cumul(
    x: metafopy.methods.leave1out.CumulResult,
    *,
    annotate: bool = True,
    header: bool | str | collections.abc.Sequence[str] = True,
    xlim: collections.abc.Sequence[float] | None = None,
    alim: collections.abc.Sequence[float] | None = None,
    olim: collections.abc.Sequence[float] | None = None,
    ylim: collections.abc.Sequence[float] | None = None,
    at: numpy.typing.ArrayLike | None = None,
    steps: int = 5,
    refline: float | None = 0,
    digits: int | collections.abc.Sequence[float] | numpy.typing.NDArray[numpy.floating] = 2,
    width: int | collections.abc.Sequence[int] | None = None,
    xlab: str | collections.abc.Sequence[str] | None = None,
    ilab: typing.Any = None,
    ilab_lab: str | collections.abc.Sequence[str] | None = None,
    ilab_xpos: numpy.typing.ArrayLike | None = None,
    ilab_pos: int | collections.abc.Sequence[int] | None = None,
    transf: Func | None = None,
    atransf: Func | None = None,
    targs: typing.Any = None,
    rows: numpy.typing.ArrayLike | float | None = None,
    efac: float | collections.abc.Sequence[float] = 1,
    pch: int | collections.abc.Sequence[int] = 15,
    psize: float | numpy.typing.ArrayLike = 1,
    col: str | collections.abc.Sequence[str] | None = None,
    colci: str | collections.abc.Sequence[str] | None = None,
    shade: bool | str | numpy.typing.ArrayLike | None = None,
    colshade: str | None = None,
    lty: str | collections.abc.Sequence[str] | None = None,
    fonts: typing.Any = None,
    cex: float | None = None,
    cex_lab: float | None = None,
    cex_axis: float | None = None,
    ax: matplotlib.axes.Axes | None = None,
    annosym: collections.abc.Sequence[str] | None = None,
    tabfig: int | None = None,
    textpos: collections.abc.Sequence[float] | None = None,
    rowadj: float | collections.abc.Sequence[float] | None = None,
    top: float = 3,
    at_lab: collections.abc.Sequence[str] | None = None,
    xlabadj: float | collections.abc.Sequence[float] | None = None,
    xlabfont: int | collections.abc.Sequence[int] = 1,
) -> metafopy.plot.ForestResult:
    """Cumulative forest plot (``forest.cumul.rma``): one row per step of a :func:`~metafopy.cumul` analysis.

    The first row is labelled with the first study, the others with ``+`` and the added study. Arguments are those of
    :func:`~metafopy.forest` for estimates with confidence intervals (``psize`` is 1 and ``pch`` a square by default, and there
    is no summary polygon). ``forest(cumul(res), ...)`` is the same call.

    Raises:
        metafopy.errors.InputError: an argument is invalid.
    """
    if not isinstance(x, metafopy.methods.leave1out.CumulResult):
        raise metafopy.errors.InputError('forest_cumul() needs the result of cumul().')
    k = int(x.estimate.shape[0])
    if x.slab_null:
        labels = [f'+ Study {s}' for s in x.slab]
        labels[0] = f'Study {x.slab[0]}'
    else:
        labels = [f'+ {s}' for s in x.slab]
        labels[0] = x.slab[0]
    return metafopy.plot.forest(
        x.estimate,
        ci_lb=x.ci_lb,
        ci_ub=x.ci_ub,
        annotate=annotate,
        header=header,
        xlim=xlim,
        alim=alim,
        olim=olim,
        ylim=ylim,
        at=at,
        steps=steps,
        level=100 * (1 - x.level),
        refline=refline,
        digits=digits,
        width=width,
        xlab=metafopy.plot.rutils.setlab(x.measure, transf, atransf, 2) if xlab is None else xlab,
        slab=labels,
        ilab=ilab,
        ilab_lab=ilab_lab,
        ilab_xpos=ilab_xpos,
        ilab_pos=ilab_pos,
        transf=transf,
        atransf=atransf,
        targs=targs,
        rows=rows,
        efac=efac,
        pch=pch,
        psize=numpy.full(k, psize) if numpy.ndim(psize) == 0 else psize,
        col=col,
        colci=colci,
        shade=shade,
        colshade=colshade,
        lty=lty,
        fonts=fonts,
        cex=cex,
        cex_lab=cex_lab,
        cex_axis=cex_axis,
        ax=ax,
        annosym=annosym,
        tabfig=tabfig,
        textpos=textpos,
        rowadj=rowadj,
        top=top,
        at_lab=at_lab,
        xlabadj=xlabadj,
        xlabfont=xlabfont,
        measure=x.measure,
        slab_null=x.slab_null,
    )


def plot_cumul(
    x: metafopy.methods.leave1out.CumulResult,
    yaxis: str | None = None,
    xlim: collections.abc.Sequence[float] | None = None,
    ylim: collections.abc.Sequence[float] | None = None,
    xlab: str | None = None,
    ylab: str | None = None,
    at: numpy.typing.ArrayLike | None = None,
    transf: Func | None = None,
    atransf: Func | None = None,
    targs: typing.Any = None,
    digits: int | collections.abc.Sequence[int] | None = None,
    cols: collections.abc.Sequence[str] | None = None,
    grid: bool | str = True,
    pch: int = 19,
    cex: float = 1.0,
    lwd: float = 2.0,
    *,
    las: int = 0,
    ax: matplotlib.axes.Axes | None = None,
    addgrid: bool | str | None = None,
    na_action: metafopy.methods.common.NAAction = 'omit',
) -> PlotCumulResult:
    """Plot of the results of a cumulative meta-analysis (``plot.cumul.rma``).

    The pooled estimates (horizontal axis) are plotted against the amount of heterogeneity (``yaxis`` ``'tau2'``, ``'tau'``, ``'I2'`` or
    ``'H2'``; ``'tau2'`` unless the model is a fixed-effects one), joined in order by a line whose colour runs from the first of ``cols`` to
    the last (light to dark by default). ``transf`` transforms the estimates, ``atransf`` only the labels of the horizontal axis,
    ``digits`` are the digits of the labels of the two axes and ``grid`` draws the dotted grid (a colour for the lines of the grid; ``addgrid`` is the
    old spelling that metafor reads from ``...`` and wins).

    ``na_action`` stands for R's ``options(na.action=)``: an unknown value raises, the others
    give the same result (metafor sets ``na.pass`` while it computes). With ``'fail'`` missing values in the results stop the plot.

    Returns a :class:`PlotCumulResult` with the plotted values and their colours.

    Raises:
        metafopy.errors.InputError: an argument is invalid.
    """
    metafopy.methods.common.check_na_action(na_action)
    if not isinstance(x, metafopy.methods.leave1out.CumulResult):
        raise metafopy.errors.InputError('plot_cumul() needs the result of cumul().')
    if transf is not None and atransf is not None:
        raise metafopy.errors.InputError("Use either 'transf' or 'atransf' to specify a transformation (not both).")
    if yaxis is None:
        yaxis = 'I2' if x.tau2 is None else 'tau2'
    else:
        if yaxis not in YAXES:
            raise metafopy.errors.InputError(f'yaxis must be one of {YAXES}')
        if x.tau2 is None:
            raise metafopy.errors.InputError(f'Cannot use yaxis="{yaxis}" for equal/fixed-effects models.')
    if addgrid is not None:
        grid = addgrid
    axx = metafopy.plot.rutils.get_ax(ax)
    cols_c = [metafopy.plot.rutils.coladj(-0.2), metafopy.plot.rutils.coladj(-0.8)] if cols is None else list(cols)
    if digits is None:
        dig = (2, 3) if yaxis in ('tau2', 'tau') else (2, 1)
    elif isinstance(digits, int | float):
        dig = (int(digits), int(digits))
    else:
        dig = (int(digits[0]), int(digits[1]))
    yvalue: FloatArray
    if yaxis == 'tau2':
        assert x.tau2 is not None
        yvalue = x.tau2
    elif yaxis == 'tau':
        assert x.tau2 is not None
        yvalue = numpy.sqrt(x.tau2)
    elif yaxis == 'I2':
        yvalue = x.I2
    else:
        yvalue = x.H2
    estim = x.estimate.copy()
    keep = ~(numpy.isnan(estim) | numpy.isnan(yvalue))
    if na_action == 'fail' and not keep.all():
        raise metafopy.errors.InputError('Missing values in results.')
    estim, yvalue = estim[keep], yvalue[keep]
    k = int(estim.shape[0])
    if transf is not None:
        estim = numpy.asarray(metafopy.methods.common.apply_each(estim, transf, targs), dtype=float)
    xl = (float(estim.min()), float(estim.max())) if xlim is None else (float(min(xlim)), float(max(xlim)))
    yl = (float(yvalue.min()), float(yvalue.max())) if ylim is None else (float(min(ylim)), float(max(ylim)))
    at_in = None if at is None else numpy.asarray(at, dtype=float)
    if at_in is not None:
        xl = (min(xl[0], float(at_in.min())), max(xl[1], float(at_in.max())))
    ux = metafopy.plot.rutils.usr_range(*xl)
    uy = metafopy.plot.rutils.usr_range(*yl)
    xt = metafopy.plot.rutils.ax_ticks(*ux) if at_in is None else at_in[(at_in > ux[0]) & (at_in < ux[1])]
    yt = metafopy.plot.rutils.ax_ticks(*uy)
    xvalues = metafopy.methods.common.apply_each(xt, atransf, targs) if atransf is not None else xt
    xlabels = [metafopy.plot.rutils.fmtx1(float(v), dig[0], True) for v in xvalues]
    ylabels = [metafopy.plot.rutils.fmtx1(float(v), dig[1], True) for v in yt]
    metafopy.plot.draw.plain_axes(axx, ux, uy, xt, yt, xlabels, ylabels, bty='o')
    if las == 0:
        axx.tick_params(axis='y', labelrotation=90)
        for label_ in axx.get_yticklabels():
            label_.set_verticalalignment('center')
    axx.set_xlabel(metafopy.plot.rutils.setlab(x.measure, transf, atransf, 2) if xlab is None else xlab, fontsize=metafopy.plot.draw.base_fontsize())
    axx.set_ylabel(YLABELS[yaxis] if ylab is None else ylab, fontsize=metafopy.plot.draw.base_fontsize())
    if grid:
        # R recycles the two grid colours over the lines (so every other line is drawn in a colour that is invisible on white)
        grid_cols = [grid, grid] if isinstance(grid, str) else [metafopy.plot.rutils.coladj(-0.2), metafopy.plot.rutils.coladj(0.6)]
        for i, v in enumerate(xt):
            metafopy.plot.draw.grid(axx, [float(v)], [], grid_cols[i % 2])
        for i, v in enumerate(yt):
            metafopy.plot.draw.grid(axx, [], [float(v)], grid_cols[i % 2])
    point_cols = metafopy.plot.rutils.color_ramp(cols_c[0], cols_c[-1], k) if k > 1 else [cols_c[0]]
    steps = 50
    for i in range(k - 1):
        xs = numpy.linspace(estim[i], estim[i + 1], steps)
        ys = numpy.linspace(yvalue[i], yvalue[i + 1], steps)
        line_cols = metafopy.plot.rutils.color_ramp(point_cols[i], point_cols[i + 1], steps)
        for j in range(steps - 1):
            metafopy.plot.draw.seg(axx, float(xs[j]), float(ys[j]), float(xs[j + 1]), float(ys[j + 1]), col=line_cols[j], lw=lwd)
    for i in range(k):
        metafopy.plot.draw.pts(axx, float(estim[i]), float(yvalue[i]), pch=pch, cex=cex, col=point_cols[i])
    return PlotCumulResult(axx, estim, yvalue, tuple(point_cols))
