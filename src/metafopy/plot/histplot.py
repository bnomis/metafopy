"""Histograms drawn the way R's ``hist()`` does, with the tail shading of metafor's ``.coltail()`` (shared by the permutest and VIF plots)."""

import contextlib
import dataclasses
import typing

import matplotlib
import matplotlib.axes
import numpy
import numpy.typing

import metafopy.plot.draw
import metafopy.plot.infl
import metafopy.plot.rutils

__all__ = ['Hist', 'draw_histogram', 'draw_tail', 'panels']

type FloatArray = numpy.typing.NDArray[numpy.float64]

COL = '#B3B3B3'
BORDER = '#E5E5E5'
COL_OUT = '#FF000080'
COL_REF = '#4C4C4C'
COL_DENSITY = 'blue'


@dataclasses.dataclass(frozen=True, slots=True)
class Hist:
    """What ``hist(plot = FALSE)`` returns: the ``breaks`` and the ``counts`` and ``density`` of each class."""

    breaks: FloatArray
    counts: FloatArray
    density: FloatArray


def histogram(values: FloatArray, breaks: typing.Any) -> Hist:
    """``hist(values, breaks, plot = FALSE)`` for the finite values."""
    finite = values[numpy.isfinite(values)]
    brk = metafopy.plot.rutils.hist_breaks(finite, breaks)
    density = metafopy.plot.rutils.hist_density(finite, brk)
    return Hist(brk, density * finite.shape[0] * numpy.diff(brk), density)


def panels(ax: matplotlib.axes.Axes, n: int) -> tuple[list[matplotlib.axes.Axes], contextlib.AbstractContextManager[None]]:
    """``n`` panels in the area of ``ax`` (a grid as ``par(mfrow = n2mfrow(n))`` makes it) and a context in which the text has the size R gives it there."""
    if n == 1:
        return [ax], contextlib.nullcontext()
    nrow, ncol = metafopy.plot.infl.n2mfrow(n)
    cex = metafopy.plot.infl.mfrow_cex(nrow, ncol)
    grid = metafopy.plot.draw.subaxes(ax, nrow, ncol, hspace=metafopy.plot.infl.grid_hspace(ax, nrow, cex), wspace=0.3)
    flat = [a for row in grid for a in row]
    for a in flat[n:]:
        a.set_axis_off()
    return flat[:n], matplotlib.rc_context({'font.size': metafopy.plot.draw.base_fontsize() * cex})


def draw_histogram(
    ax: matplotlib.axes.Axes,
    values: FloatArray,
    breaks: typing.Any,
    *,
    col: str,
    border: str,
    main: str,
    xlab: str,
    freq: bool,
    ylim: tuple[float, float] | None = None,
    xaxis: bool = True,
) -> Hist:
    """``hist(values, breaks, col, border, main, xlab, freq, ylim)``: bars with R's axes (an axis line from the first to the last tick, no box)."""
    h = histogram(values, breaks)
    height = h.counts if freq else h.density
    xl = (float(h.breaks[0]), float(h.breaks[-1]))
    yl = (0.0, float(numpy.nanmax(height))) if ylim is None else ylim
    ux = metafopy.plot.rutils.usr_range(*xl)
    uy = metafopy.plot.rutils.usr_range(*yl)
    xt = metafopy.plot.rutils.ax_ticks(*ux)
    yt = metafopy.plot.rutils.ax_ticks(*uy)
    metafopy.plot.draw.plain_axes(ax, ux, uy, None, None, bty='n')
    for i in range(h.breaks.shape[0] - 1):
        metafopy.plot.draw.rect(ax, float(h.breaks[i]), 0.0, float(h.breaks[i + 1]), float(height[i]), col=col, border=border)
    if xaxis:
        metafopy.plot.draw.xaxis(ax, xt, metafopy.plot.rutils.rformat(xt))
    metafopy.plot.draw.yaxis(ax, yt, metafopy.plot.rutils.rformat(yt))
    fontsize = metafopy.plot.draw.base_fontsize()
    ax.set_xlabel(xlab, fontsize=fontsize)
    ax.set_ylabel('Frequency' if freq else 'Density', fontsize=fontsize)
    if main:
        ax.set_title(main, fontsize=fontsize * 1.2, fontweight='bold')
    return h


def draw_tail(
    ax: matplotlib.axes.Axes, h: Hist, val: float, *, tail: str = 'upper', mult: float = 1.0, col: str = COL_OUT, border: str = BORDER, freq: bool = False
) -> None:
    """metafor's ``.coltail()``: draw the bars beyond ``val`` again in ``col`` (the class containing ``val`` is cut at ``val``)."""
    breaks = h.breaks.copy()
    height = (h.counts if freq else h.density) * mult
    if tail == 'lower':
        above = numpy.flatnonzero(breaks > val)
        last = int(above[0]) if above.size else breaks.shape[0] - 1
        if above.size:
            breaks[last] = val
        keep, heights = breaks[: last + 1], height[:last]
    else:
        below = numpy.flatnonzero(breaks < val)
        first = int(below[-1]) if below.size else 0
        if below.size:
            breaks[first] = val
        keep, heights = breaks[first:], height[first:]
    for i in range(keep.shape[0] - 1):
        metafopy.plot.draw.rect(ax, float(keep[i]), 0.0, float(keep[i + 1]), float(heights[i]), col=col, border=border)


def apply_trim(values: FloatArray, trim: float, *, two_sided: bool) -> FloatArray:
    """Drop the largest ``trim`` of the values (or, ``two_sided``, ``trim / 2`` of each tail), as R's ``quantile()`` finds the bounds."""
    finite = values[numpy.isfinite(values)]
    if trim <= 0:
        return finite
    if two_sided:
        lo, hi = numpy.quantile(finite, [trim / 2, 1 - trim / 2])
        return numpy.asarray(finite[(finite >= lo) & (finite <= hi)], dtype=numpy.float64)
    return numpy.asarray(finite[finite <= numpy.quantile(finite, 1 - trim)], dtype=numpy.float64)
