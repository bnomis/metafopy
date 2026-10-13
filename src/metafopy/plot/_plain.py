"""R's default ``plot()`` frame (4% extended user coordinates, ``axTicks`` ticks and labels, labels and a title)."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import NullLocator
from numpy.typing import NDArray

from metafopy.plot import _draw as d
from metafopy.plot._rutils import ax_ticks, log_ticks, rformat, usr_range

type FloatArray = NDArray[np.float64]


def plain_plot(
    ax: Axes,
    xlim: Sequence[float],
    ylim: Sequence[float],
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
    xat: Sequence[float] | NDArray[np.float64] | None = None,
    xlabels: Sequence[str] | None = None,
    cex_lab: float = 1.0,
    cex_axis: float = 1.0,
) -> tuple[FloatArray | None, FloatArray | None]:
    """``plot(NA, xlim, ylim, xlab, ylab, main, bty, xaxt, yaxt)``; returns the ticks (``axTicks(1)``, ``axTicks(2)``) drawn."""
    ux = (float(xlim[0]), float(xlim[1])) if xaxs_i else usr_range(float(xlim[0]), float(xlim[1]))
    if logy:
        ly = usr_range(math.log10(ylim[0]), math.log10(ylim[1]))
        ax.set_yscale('log')
        uy = (10.0 ** ly[0], 10.0 ** ly[1])
    else:
        uy = (float(ylim[0]), float(ylim[1])) if yaxs_i else usr_range(float(ylim[0]), float(ylim[1]))
    xt = None
    if xaxt:
        xt = ax_ticks(ux[0], ux[1]) if xat is None else np.asarray(xat, dtype=float)
    yt = None
    if yaxt:
        yt = log_ticks(float(ylim[0]), float(ylim[1])) if logy else ax_ticks(uy[0], uy[1])
    d.plain_axes(
        ax,
        ux,
        uy,
        xt,
        yt,
        None if xt is None else (xlabels if xlabels is not None else rformat(xt)),
        None if yt is None else rformat(yt),
        bty=bty,
        cex_axis=cex_axis,
    )
    if logy:
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_ylim(uy)
    fs = d.base_fontsize() * cex_lab
    ax.set_xlabel(xlab, fontsize=fs)
    ax.set_ylabel(ylab, fontsize=fs)
    if main:
        ax.set_title(main, fontsize=d.base_fontsize() * 1.2, fontweight='bold')
    return xt, yt
