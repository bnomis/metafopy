"""R's default ``plot()`` frame (4% extended user coordinates, ``axTicks`` ticks and labels, labels and a title)."""

import collections.abc
import math

import matplotlib.axes
import matplotlib.ticker
import numpy
import numpy.typing

import metafopy.plot
import metafopy.plot.draw
import metafopy.plot.rutils

type FloatArray = numpy.typing.NDArray[numpy.float64]


def plain_plot(
    ax: matplotlib.axes.Axes,
    xlim: collections.abc.Sequence[float],
    ylim: collections.abc.Sequence[float],
    xlab: str = '',
    ylab: str = '',
    main: str = '',
    *,
    bty: str = 'o',
    xaxt: bool = True,
    yaxt: bool = True,
    logy: bool = False,
    xaxs_i: bool = False,
    yaxs_i: bool = False,
    xat: collections.abc.Sequence[float] | numpy.typing.NDArray[numpy.float64] | None = None,
    xlabels: collections.abc.Sequence[str] | None = None,
    cex_lab: float = 1.0,
    cex_axis: float = 1.0,
    thin_xlabels: bool = True,
    thin_ylabels: bool = False,
) -> tuple[FloatArray | None, FloatArray | None]:
    """``plot(NA, xlim, ylim, xlab, ylab, main, bty, xaxt, yaxt)``; returns the ticks (``axTicks(1)``, ``axTicks(2)``) drawn."""
    ux = (float(xlim[0]), float(xlim[1])) if xaxs_i else metafopy.plot.rutils.usr_range(float(xlim[0]), float(xlim[1]))
    if logy:
        ly = metafopy.plot.rutils.usr_range(math.log10(ylim[0]), math.log10(ylim[1]))
        ax.set_yscale('log')
        uy = (10.0 ** ly[0], 10.0 ** ly[1])
    else:
        uy = (float(ylim[0]), float(ylim[1])) if yaxs_i else metafopy.plot.rutils.usr_range(float(ylim[0]), float(ylim[1]))
    xt = None
    if xaxt:
        xt = metafopy.plot.rutils.ax_ticks(ux[0], ux[1]) if xat is None else numpy.asarray(xat, dtype=float)
    yt = None
    if yaxt:
        yt = metafopy.plot.rutils.log_ticks(float(ylim[0]), float(ylim[1])) if logy else metafopy.plot.rutils.ax_ticks(uy[0], uy[1])
    metafopy.plot.draw.plain_axes(
        ax,
        ux,
        uy,
        xt,
        yt,
        None if xt is None else (xlabels if xlabels is not None else metafopy.plot.rutils.rformat(xt)),
        None if yt is None else metafopy.plot.rutils.rformat(yt),
        bty=bty,
        cex_axis=cex_axis,
        thin_xlabels=thin_xlabels,
        thin_ylabels=thin_ylabels and not logy,
    )
    if logy:
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_ylim(uy)
    fs = metafopy.plot.draw.base_fontsize() * cex_lab
    ax.set_xlabel(xlab, fontsize=fs)
    ax.set_ylabel(ylab, fontsize=fs)
    if main:
        ax.set_title(main, fontsize=metafopy.plot.draw.base_fontsize() * 1.2, fontweight='bold')
    return xt, yt
