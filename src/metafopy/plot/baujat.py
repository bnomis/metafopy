"""Baujat plot (metafor's ``baujat.rma``).

Plots each study's contribution to the overall heterogeneity (squared Pearson residual) against its influence on the
pooled result (the standardised squared change in the fitted value when the study is left out), see Baujat et al.
(2002), doi:10.1002/sim.1221, and https://wviechtb.github.io/metafor/reference/baujat.html.
"""

import collections.abc
import dataclasses

import matplotlib.axes
import numpy
import numpy.typing
import pandas

import metafopy.errors
import metafopy.methods.binary
import metafopy.methods.common
import metafopy.methods.fitview
import metafopy.models.binary
import metafopy.models.rma_uni
import metafopy.models.tau2
import metafopy.plot
import metafopy.plot.draw
import metafopy.plot.plain
import metafopy.plot.rutils

__all__ = ['BaujatResult', 'baujat']

type FloatArray = numpy.typing.NDArray[numpy.float64]


@dataclasses.dataclass(frozen=True, slots=True)
class BaujatResult:
    """The plotted coordinates of a Baujat plot (what metafor's ``baujat()`` returns invisibly) and the axes drawn on."""

    ax: matplotlib.axes.Axes
    x: FloatArray
    y: FloatArray
    ids: tuple[int, ...]
    slab: tuple[str, ...]

    def to_frame(self) -> pandas.DataFrame:
        return pandas.DataFrame({'x': self.x, 'y': self.y, 'ids': list(self.ids), 'slab': list(self.slab)})


def baujat(
    x: metafopy.models.rma_uni.RMAResult | metafopy.models.binary.FixedBinaryResult,
    xlim: collections.abc.Sequence[float] | None = None,
    ylim: collections.abc.Sequence[float] | None = None,
    xlab: str | None = None,
    ylab: str | None = None,
    cex: float | None = None,
    symbol: str | int | collections.abc.Sequence[int] = 'ids',
    grid: bool | str = True,
    *,
    ax: matplotlib.axes.Axes | None = None,
    bty: str = 'o',
    las: int | None = None,
) -> BaujatResult:
    """Baujat plot of a fitted ``rma_uni``, ``rma_mh`` or ``rma_peto`` model (``baujat.rma``).

    ``symbol`` is ``'ids'`` (the 1-based study numbers), ``'slab'`` (the study labels) or one plotting symbol
    (a number, or one per study); ``grid`` adds a grid (a colour string sets its colour). Each study is
    left out in turn and the model refitted; the x-axis shows the study's squared Pearson residual and the y-axis
    the squared change in its fitted value, standardised by the variance of the prediction without the study.
    ``bty`` (the type of box around the plot: ``'o'``, ``'l'``, ``'7'``, ``'c'``, ``'u'``, ``']'`` or ``'n'``) and ``las`` (the
    orientation of the tick labels, 0 to 3) are passed on to ``plot()`` by metafor.

    Returns a :class:`BaujatResult`.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    metafopy.methods.fitview.check_class(x, notav=metafopy.methods.fitview.FIXED_UNI_ONLY)
    if x.k == 1:
        raise metafopy.errors.InputError('Stopped because k = 1.')
    axx = metafopy.plot.rutils.get_ax(ax)
    grid_col = grid if isinstance(grid, str) else metafopy.plot.rutils.coladj(-0.2)
    if isinstance(x, metafopy.models.binary.FixedBinaryResult):
        xhati, yhati, rows, k_all = binary_coordinates(x)
    else:
        xhati, yhati, rows, k_all = uni_coordinates(x)
    k = int(rows.shape[0])
    int_only = x.int_only
    ids = tuple(int(i) + 1 for i in rows)
    slab = tuple(x.slab) if isinstance(x, metafopy.models.rma_uni.RMAResult) else tuple(s for s, ok in zip(x.tables.slab, x.tables.not_na, strict=True) if ok)
    keep = numpy.zeros(k_all, dtype=numpy.bool_)
    keep[rows] = True

    cex_ = 0.8 if cex is None else cex
    if xlab is None:
        if x.method in metafopy.models.tau2.FIXED_EFFECTS:
            xlab = 'Contribution to Overall Heterogeneity' if int_only else 'Contribution to Residual Heterogeneity'
        else:
            xlab = 'Squared Pearson Residual'
    if ylab is None:
        ylab = 'Influence on Overall Result' if int_only else 'Influence on Fitted Value'
    xl = (float(numpy.nanmin(xhati)), float(numpy.nanmax(xhati))) if xlim is None else (float(xlim[0]), float(xlim[1]))
    yl = (float(numpy.nanmin(yhati)), float(numpy.nanmax(yhati))) if ylim is None else (float(ylim[0]), float(ylim[1]))
    xt, yt = metafopy.plot.plain.plain_plot(axx, xl, yl, xlab, ylab, bty=bty, las=las)
    if grid is not False:  # True or a colour
        assert xt is not None
        assert yt is not None
        metafopy.plot.draw.grid(axx, xt, yt, grid_col)

    if not isinstance(symbol, str):
        sym = numpy.broadcast_to(numpy.asarray(symbol, dtype=int), (k,)) if numpy.ndim(symbol) == 0 else numpy.asarray(symbol, dtype=int)
        if sym.shape[0] == k_all and k_all != k:
            sym = sym[keep]
        if sym.shape[0] != k:
            raise metafopy.errors.InputError(
                f"Length of the 'symbol' argument ({sym.shape[0]}) does not correspond to the size of the original dataset ({k_all})."
            )
        for i in range(k):
            if not (numpy.isnan(xhati[i]) or numpy.isnan(yhati[i])):
                metafopy.plot.draw.pts(axx, xhati[i], yhati[i], pch=int(sym[i]), cex=cex_, col=metafopy.plot.draw.FG)
    elif symbol == 'ids':
        for i in range(k):
            metafopy.plot.draw.txt(axx, float(xhati[i]), float(yhati[i]), str(ids[i]), cex=cex_)
    elif symbol == 'slab':
        for i in range(k):
            metafopy.plot.draw.txt(axx, float(xhati[i]), float(yhati[i]), slab[i], cex=cex_)
    return BaujatResult(axx, xhati, yhati, ids, slab)


def uni_coordinates(x: metafopy.models.rma_uni.RMAResult) -> tuple[FloatArray, FloatArray, numpy.typing.NDArray[numpy.intp], int]:
    """The squared Pearson residuals, the influence values, the 0-based rows of the studies and ``k_all`` of an ``rma_uni`` fit."""
    k = x.k
    pred_full = x.X @ x.beta
    delpred = numpy.full(k, numpy.nan)
    vdelpred = numpy.full(k, numpy.nan)
    for i in range(k):
        res = metafopy.methods.common.refit(x, numpy.delete(numpy.arange(k), i))
        if res is None:
            continue
        xi = x.X[i]
        delpred[i] = float(xi @ res.beta)
        vdelpred[i] = float(xi @ res.vb @ xi)
    yhati = (delpred - pred_full) ** 2 / vdelpred
    resid = x.yi - pred_full
    xhati = resid**2 / (x.tau2 + x.vi)
    return xhati, yhati, numpy.flatnonzero(x.data.keep), x.data.k_all


def binary_coordinates(x: metafopy.models.binary.FixedBinaryResult) -> tuple[FloatArray, FloatArray, numpy.typing.NDArray[numpy.intp], int]:
    """As :func:`uni_coordinates` for an ``rma_mh`` or ``rma_peto`` fit: each table with complete counts is left out in turn (``baujat.rma``)."""
    t = x.tables
    beta = float(x.beta[0])
    delpred = numpy.full(t.k_f, numpy.nan)
    vdelpred = numpy.full(t.k_f, numpy.nan)
    for i in numpy.flatnonzero(t.not_na):
        sub = numpy.ones(t.k_f, dtype=numpy.bool_)
        sub[i] = False
        res = metafopy.methods.binary.refit_tables(x, t.cells_f, sub)
        if res is None:
            continue
        delpred[i] = float(res.beta[0])
        vdelpred[i] = float(res.vb[0, 0])
    with numpy.errstate(all='ignore'):
        yhati = (delpred - beta) ** 2 / vdelpred
        xhati = (t.yi_f - beta) ** 2 / t.vi_f
    return xhati[t.not_na], yhati[t.not_na], t.ids[t.not_na], t.k_all
