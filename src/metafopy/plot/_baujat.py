"""Baujat plot (metafor's ``baujat.rma``).

Plots each study's contribution to the overall heterogeneity (squared Pearson residual) against its influence on the
pooled result (the standardised squared change in the fitted value when the study is left out), see Baujat et al.
(2002), doi:10.1002/sim.1221, and https://wviechtb.github.io/metafor/reference/baujat.html.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from numpy.typing import NDArray

from metafopy.methods._common import refit
from metafopy.models._tau2 import FIXED_EFFECTS
from metafopy.models.rma_uni import RMAResult
from metafopy.plot import _draw as d
from metafopy.plot._plain import plain_plot
from metafopy.plot._rutils import coladj, get_ax

__all__ = ['BaujatResult', 'baujat']

type FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class BaujatResult:
    """The plotted coordinates of a Baujat plot (what metafor's ``baujat()`` returns invisibly) and the axes drawn on."""

    ax: Axes
    x: FloatArray
    y: FloatArray
    ids: tuple[int, ...]
    slab: tuple[str, ...]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({'x': self.x, 'y': self.y, 'ids': list(self.ids), 'slab': list(self.slab)})


def baujat(
    x: RMAResult,
    xlim: Sequence[float] | None = None,
    ylim: Sequence[float] | None = None,
    xlab: str | None = None,
    ylab: str | None = None,
    cex: float | None = None,
    symbol: str | int | Sequence[int] = 'ids',
    grid: bool | str = True,
    *,
    ax: Axes | None = None,
) -> BaujatResult:
    """Baujat plot of a fitted model (``baujat.rma``).

    ``symbol`` is ``'ids'`` (the 1-based study numbers), ``'slab'`` (the study labels) or one plotting symbol
    (a number, or one per study); ``grid`` adds a grid (a colour string sets its colour). Each study is
    left out in turn and the model refitted; the x-axis shows the study's squared Pearson residual and the y-axis
    the squared change in its fitted value, standardised by the variance of the prediction without the study.

    Returns a :class:`BaujatResult`.
    """
    if x.k == 1:
        raise ValueError('Stopped because k = 1.')
    axx = get_ax(ax)
    grid_col = grid if isinstance(grid, str) else coladj(-0.2)
    k = x.k
    pred_full = x.X @ x.beta
    delpred = np.full(k, np.nan)
    vdelpred = np.full(k, np.nan)
    for i in range(k):
        res = refit(x, np.delete(np.arange(k), i))
        if res is None:
            continue
        xi = x.X[i]
        delpred[i] = float(xi @ res.beta)
        vdelpred[i] = float(xi @ res.vb @ xi)
    yhati = (delpred - pred_full) ** 2 / vdelpred
    resid = x.yi - pred_full
    xhati = resid**2 / (x.tau2 + x.vi)

    cex_ = 0.8 if cex is None else cex
    if xlab is None:
        if x.method in FIXED_EFFECTS:
            xlab = 'Contribution to Overall Heterogeneity' if x.int_only else 'Contribution to Residual Heterogeneity'
        else:
            xlab = 'Squared Pearson Residual'
    if ylab is None:
        ylab = 'Influence on Overall Result' if x.int_only else 'Influence on Fitted Value'
    xl = (float(np.nanmin(xhati)), float(np.nanmax(xhati))) if xlim is None else (float(xlim[0]), float(xlim[1]))
    yl = (float(np.nanmin(yhati)), float(np.nanmax(yhati))) if ylim is None else (float(ylim[0]), float(ylim[1]))
    xt, yt = plain_plot(axx, xl, yl, xlab, ylab)
    if grid is not False:  # True or a colour
        assert xt is not None
        assert yt is not None
        d.grid(axx, xt, yt, grid_col)

    ids = tuple(int(i) + 1 for i in np.flatnonzero(x.data.keep))
    slab = tuple(x.slab)
    if not isinstance(symbol, str):
        sym = np.broadcast_to(np.asarray(symbol, dtype=int), (k,)) if np.ndim(symbol) == 0 else np.asarray(symbol, dtype=int)
        if sym.shape[0] == x.data.k_all and x.data.k_all != k:
            sym = sym[x.data.keep]
        if sym.shape[0] != k:
            raise ValueError(f"Length of the 'symbol' argument ({sym.shape[0]}) does not correspond to the size of the original dataset ({x.data.k_all}).")
        for i in range(k):
            if not (np.isnan(xhati[i]) or np.isnan(yhati[i])):
                d.pts(axx, xhati[i], yhati[i], pch=int(sym[i]), cex=cex_, col=d.FG)
    elif symbol == 'ids':
        for i in range(k):
            d.txt(axx, float(xhati[i]), float(yhati[i]), str(ids[i]), cex=cex_)
    elif symbol == 'slab':
        for i in range(k):
            d.txt(axx, float(xhati[i]), float(yhati[i]), slab[i], cex=cex_)
    return BaujatResult(axx, xhati, yhati, ids, slab)
