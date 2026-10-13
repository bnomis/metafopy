"""Plot of the results of a GOSH analysis (metafor's ``plot.gosh.rma``).

For models without moderators the estimates are plotted against a heterogeneity statistic with marginal histograms and
kernel density estimates; otherwise a scatterplot matrix of the heterogeneity statistic and the coefficients is drawn.
See Olkin, Dahabreh & Trikalinos (2012), doi:10.1002/jrsm.1033, and
https://wviechtb.github.io/metafor/reference/plot.gosh.rma.html.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import matplotlib.colors as mcolors
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray

from metafopy.methods.gosh import GoshResult
from metafopy.plot import _draw as d
from metafopy.plot._plain import plain_plot
from metafopy.plot._rutils import ax_ticks, coladj, density_r, get_ax, hist_breaks, hist_density, rformat, setlab, usr_range

__all__ = ['PlotGoshResult', 'plot_gosh']

type FloatArray = NDArray[np.float64]

_HET = {'QE': 'QE', 'I2': 'I2', 'I^2': 'I2', 'H2': 'H2', 'H^2': 'H2', 'tau2': 'tau2', 'tau^2': 'tau2', 'tau': 'tau'}
_HET_LABEL = {'QE': r'$Q_E$', 'I2': r'$I^2$', 'H2': r'$H^2$', 'tau2': r'$\tau^2$', 'tau': r'$\tau$'}


@dataclass(frozen=True, slots=True)
class PlotGoshResult:
    """The axes of the plot: ``main`` is the scatterplot (``None`` for the scatterplot matrix), ``xhist``/``yhist`` the
    marginal histograms (if drawn) and ``panels`` the axes of the scatterplot matrix (row by row)."""

    main: Axes | None
    xhist: Axes | None
    yhist: Axes | None
    panels: tuple[tuple[Axes, ...], ...] = ()

    @property
    def axes(self) -> tuple[Axes, ...]:
        found = [a for a in (self.xhist, self.main, self.yhist) if a is not None]
        return tuple(found) if found else tuple(a for row in self.panels for a in row)


def _hex(col: str, alpha: float) -> str:
    """``rgb(col2rgb(col)/255, alpha)``: an eight-digit hex colour."""
    r, g, b = mcolors.to_rgb(d.rcolor(col))
    return '#' + ''.join(f'{int(255 * v + 0.5):02X}' for v in (r, g, b, alpha))


def _quantile(x: FloatArray, p: float) -> float:
    return float(np.nanquantile(x, p))


def _expand(v: Any, n: int) -> list[Any]:
    if isinstance(v, str) or np.ndim(v) == 0:
        return [v] * n
    out = list(v)
    return out + [out[-1]] * (n - len(out)) if len(out) < n else out[:n]


def plot_gosh(
    x: GoshResult,
    het: str = 'I2',
    pch: int = 16,
    cex: float | None = None,
    out: int | None = None,
    col: str | Sequence[str] | None = None,
    alpha: float | Sequence[float] | None = None,
    border: str | Sequence[str] | None = None,
    xlim: Sequence[float] | None = None,
    ylim: Sequence[float] | None = None,
    xhist: bool = True,
    yhist: bool = True,
    hh: float | Sequence[float] = 0.3,
    breaks: Any = None,
    adjust: float | Sequence[float] | None = None,
    lwd: float | Sequence[float] | None = None,
    labels: Sequence[str] | None = None,
    trim: float | Sequence[float | Sequence[float]] | None = None,
    *,
    ax: Axes | None = None,
) -> PlotGoshResult:
    """Plot the results of :func:`~metafopy.gosh` (``plot.gosh.rma``).

    For a model without moderators the heterogeneity statistic ``het`` (``'QE'``, ``'I2'``, ``'H2'``, ``'tau2'`` or
    ``'tau'``) is plotted against the estimate, with marginal histograms and kernel density estimates (``xhist``,
    ``yhist``; ``hh`` the relative sizes of the marginal plots, ``breaks``, ``adjust`` and ``lwd`` their settings, each
    a single value or a pair for the two margins); with moderators a scatterplot matrix of ``het`` and the
    coefficients is drawn. ``out`` (0-based) highlights the subsets that do (``col[0]``) or do not include that study,
    ``alpha`` gives the transparency of the points, histogram and density line, ``trim`` removes the most extreme
    subsets (a proportion or a list of proportions per column). The panels fill the area of ``ax``.

    Returns a :class:`PlotGoshResult` with the axes.
    """
    het = _HET.get(het, het)
    if het not in _HET_LABEL:
        raise ValueError(f'het must be one of {", ".join(_HET)}')
    if het in ('tau2', 'tau') and x.method in ('FE', 'EE', 'CE'):
        raise ValueError("Cannot plot 'tau2' for equal/fixed-effects models.")
    area = get_ax(ax)
    res = x.res
    incl = x.incl
    cex_ = 0.5 if cex is None else cex
    coef_cols = list(res.columns[6:])
    cols_x = [het, *coef_cols]
    if trim is not None:
        trims: list[Any] = [trim] * len(cols_x) if isinstance(trim, int | float) else list(trim)
        if len(trims) < len(cols_x):
            trims = trims + [trims[-1]] * (len(cols_x) - len(trims))
        dele = np.zeros(len(res), dtype=bool)
        for i, c in enumerate(cols_x):
            v = res[c].to_numpy(dtype=float)
            t = np.atleast_1d(np.asarray(trims[i], dtype=float))
            with np.errstate(invalid='ignore'):
                dele |= (v < _quantile(v, float(t[0]))) | (v > _quantile(v, 1 - float(t[-1])))
        res = res.loc[~dele].reset_index(drop=True)
        incl = incl[~dele]
    n = len(res)

    alpha_v = n ** (-0.2) if alpha is None else alpha
    a = list(np.atleast_1d(alpha_v).astype(float))
    if len(a) == 1:
        a = [a[0], 0.5, 0.9]
    elif len(a) == 2:
        a = [a[0], a[1], 0.9]
    if out is None:
        base = d.FG if col is None else (col if isinstance(col, str) else col[0])
        col_pnts: str | list[str] = _hex(base, a[0])
        cols = (_hex(base, a[1]), _hex(base, a[2]))
        cols_hist = cols_line = ('', '')
        isout = None
    else:
        if not 0 <= int(out) < x.k:
            raise ValueError('Non-existing study chosen as potential outlier.')
        cc = ['red', 'blue'] if col is None else ([col] if isinstance(col, str) else list(col))
        if len(cc) != 2:
            raise ValueError("Argument 'col' should specify two colors when argument 'out' is used.")
        isout = incl[:, int(out)]
        col_o, col_i = cc
        col_pnts = [_hex(col_o, a[0]) if f else _hex(col_i, a[0]) for f in isout]
        cols_hist = (_hex(col_o, a[1]), _hex(col_i, a[1]))
        cols_line = (_hex(col_o, a[2]), _hex(col_i, a[2]))
        cols = ('', '')
    border_c = coladj(-0.1) if border is None else border
    bd = _expand(border_c, 2)
    hhv = [float(v) for v in _expand(hh, 2)]
    if x.int_only and any(v < 0 or v > 1 for v in hhv):
        raise ValueError("Invalid value(s) specified for 'hh' argument.")
    if breaks is None or isinstance(breaks, str) or np.ndim(breaks) == 0:
        br: list[Any] = [breaks or 'Sturges'] * 2
    else:
        br = list(breaks)[:2]
        if len(br) == 1:
            br = br * 2
    adj = [float(v) for v in _expand(1.0 if adjust is None else adjust, 2)]
    lw = [float(v) for v in _expand(2.0 if lwd is None else lwd, 2)]

    if labels is None:
        if x.int_only:
            labels = [setlab(x.measure, None, None, 2), _HET_LABEL['QE'].replace('Q_E', 'Q') if het == 'QE' else _HET_LABEL[het]]
        else:
            labels = [_HET_LABEL[het], *coef_cols]
    labels = list(labels)

    if x.int_only:
        return _plot_int_only(area, res, het, isout, pch, cex_, col_pnts, cols, cols_hist, cols_line, bd, hhv, br, adj, lw, labels, xlim, ylim, xhist, yhist)
    return _plot_pairs(area, res, cols_x, isout, pch, cex_, col_pnts, cols, cols_hist, cols_line, bd, hhv, br, adj, lw, labels)


def _hist_panel_data(v: FloatArray, brk: Any, adjust: float) -> tuple[FloatArray, FloatArray, tuple[FloatArray, FloatArray]]:
    h = hist_breaks(v, brk)
    return h, hist_density(v, h), density_r(v, adjust)


def _plot_int_only(
    area: Axes,
    res: Any,
    het: str,
    isout: NDArray[np.bool_] | None,
    pch: int,
    cex: float,
    col_pnts: str | list[str],
    cols: tuple[str, str],
    cols_hist: tuple[Any, Any],
    cols_line: tuple[Any, Any],
    bd: list[Any],
    hh: list[float],
    br: list[Any],
    adj: list[float],
    lw: list[float],
    labels: list[str],
    xlim: Sequence[float] | None,
    ylim: Sequence[float] | None,
    xhist: bool,
    yhist: bool,
) -> PlotGoshResult:
    est = res['estimate'].to_numpy(dtype=float)
    hv = res[het].to_numpy(dtype=float)
    hx_breaks = hist_breaks(est, br[0])
    hy_breaks = hist_breaks(hv, br[1])
    rows, cols_n = (2 if xhist else 1), (2 if yhist else 1)
    kw: dict[str, Any] = {'hspace': 0.0, 'wspace': 0.0}
    if xhist:
        kw['height_ratios'] = [hh[0], 1 - hh[0]]
    if yhist:
        kw['width_ratios'] = [1 - hh[1], hh[1]]
    grid = d.subaxes(area, rows, cols_n, **kw)
    ax_x = grid[0][0] if xhist else None
    ax_main = grid[rows - 1][0]
    ax_y = grid[rows - 1][cols_n - 1] if yhist else None
    if xhist and yhist:
        grid[0][1].set_axis_off()

    if isout is None:
        xl = (float(hx_breaks.min()), float(hx_breaks.max())) if xlim is None else (float(xlim[0]), float(xlim[1]))
        yl = (float(hy_breaks.min()), float(hy_breaks.max())) if ylim is None else (float(ylim[0]), float(ylim[1]))
    else:
        xl = (float(hx_breaks.min()), float(hx_breaks.max())) if xlim is None else (float(xlim[0]), float(xlim[1]))
        yl = (float(hy_breaks.min()), float(hy_breaks.max())) if ylim is None else (float(ylim[0]), float(ylim[1]))

    def margin(axm: Axes, v: FloatArray, brk: FloatArray, a_: float, lim: tuple[float, float], horizontal: bool, bdr: str, lwd_: float) -> None:
        if isout is None:
            groups = [(v, cols[0], cols[1])]
        else:
            groups = [(v[~isout], cols_hist[1], cols_line[1]), (v[isout], cols_hist[0], cols_line[0])]
        dens = [(hist_density(g, brk), density_r(g, a_), ch, cl) for g, ch, cl in groups]
        top = max(float(np.nanmax(np.concatenate([h, dn[1]]))) for h, dn, _, _ in dens)
        if horizontal:
            plain_plot(axm, lim, (0.0, top), bty='n', xaxt=False, yaxt=False)
        else:
            plain_plot(axm, (0.0, top), lim, bty='n', xaxt=False, yaxt=False)
        for h, _, ch, _ in dens:
            for i in range(brk.shape[0] - 1):
                if horizontal:
                    d.rect(axm, float(brk[i]), 0.0, float(brk[i + 1]), float(h[i]), col=ch, border=bdr)
                else:
                    d.rect(axm, 0.0, float(brk[i]), float(h[i]), float(brk[i + 1]), col=ch, border=bdr)
        if lwd_ > 0:
            for _, (dx, dy), _, cl in dens:
                d.line(axm, dx, dy, col=cl, lw=lwd_ * 0.75) if horizontal else d.line(axm, dy, dx, col=cl, lw=lwd_ * 0.75)

    if ax_x is not None:
        margin(ax_x, est, hx_breaks, adj[0], xl, True, bd[0], lw[0])
    plain_plot(ax_main, xl, yl, labels[0], labels[1], bty='l')
    d.scatter(ax_main, est, hv, pch=pch, cex=cex, cols=col_pnts)
    if ax_y is not None:
        margin(ax_y, hv, hy_breaks, adj[1], yl, False, bd[1], lw[1])
    return PlotGoshResult(ax_main, ax_x, ax_y)


def _plot_pairs(
    area: Axes,
    res: Any,
    cols_x: list[str],
    isout: NDArray[np.bool_] | None,
    pch: int,
    cex: float,
    col_pnts: str | list[str],
    cols: tuple[str, str],
    cols_hist: tuple[Any, Any],
    cols_line: tuple[Any, Any],
    bd: list[Any],
    hh: list[float],
    br: list[Any],
    adj: list[float],
    lw: list[float],
    labels: list[str],
) -> PlotGoshResult:
    X = np.column_stack([res[c].to_numpy(dtype=float) for c in cols_x])
    nc = X.shape[1]
    grid = d.subaxes(area, nc, nc, hspace=0.0, wspace=0.0)
    fig = area.get_figure(root=True)
    assert fig is not None
    for i in range(nc):
        for j in range(nc):
            axp = grid[i][j]
            fin_j, fin_i = X[:, j][np.isfinite(X[:, j])], X[:, i][np.isfinite(X[:, i])]
            ux = usr_range(float(fin_j.min()), float(fin_j.max()))
            uy = usr_range(float(fin_i.min()), float(fin_i.max()))
            axp.set_xlim(ux)
            axp.set_ylim(uy)
            axp.set_autoscale_on(False)
            axp.set_xticks([])
            axp.set_yticks([])
            j_odd, i_odd = (j + 1) % 2 == 1, (i + 1) % 2 == 1
            if i == 0 and not j_odd:
                _side_axis(axp, 'top', ax_ticks(*ux))
            if i == nc - 1 and j_odd:
                _side_axis(axp, 'bottom', ax_ticks(*ux))
            if j == 0 and not i_odd:
                _side_axis(axp, 'left', ax_ticks(*uy))
            if j == nc - 1 and i_odd and i != j:
                _side_axis(axp, 'right', ax_ticks(*uy))
            if i == j:
                _diag_hist(axp, X[:, i], isout, cols, cols_hist, cols_line, bd[0], hh[0], br[0], adj[0], lw[0])
                w = 0.55 * max(len(t) for t in labels) * d.base_fontsize() / 72
                panel_in = max(axp.get_position().width * fig.get_size_inches()[0], 1e-6)
                cl = max(0.8, min(2.0, 0.9 / (w / panel_in)))
                t = axp.text(0.5, 0.5 + 1 / 3, labels[i], transform=axp.transAxes, ha='center', va='center', fontsize=d.base_fontsize() * cl)
                t.set_gid('mtext')
            else:
                d.scatter(axp, X[:, j], X[:, i], pch=pch, cex=cex, cols=col_pnts)
    return PlotGoshResult(None, None, None, tuple(tuple(r) for r in grid))


def _side_axis(ax: Axes, side: str, at: FloatArray) -> None:
    labels = rformat(at)
    if side == 'top' or side == 'bottom':
        ax.set_xticks(at)
        ax.set_xticklabels(labels, fontsize=d.base_fontsize() * 0.8)
        ax.xaxis.set_ticks_position('top' if side == 'top' else 'bottom')
    else:
        ax.set_yticks(at)
        ax.set_yticklabels(labels, fontsize=d.base_fontsize() * 0.8)
        ax.yaxis.set_ticks_position('left' if side == 'left' else 'right')


def _diag_hist(
    ax: Axes,
    v: FloatArray,
    isout: NDArray[np.bool_] | None,
    cols: tuple[str, str],
    cols_hist: tuple[Any, Any],
    cols_line: tuple[Any, Any],
    bdr: str,
    hh: float,
    brk_spec: Any,
    adj: float,
    lwd: float,
) -> None:
    """``panel.hist`` of plot.gosh.rma: a histogram scaled to a maximum of 1 with a density line, on ``ylim = c(0, 1.2 + hh)``."""
    ax.set_ylim(0.0, 1.2 + hh)
    brk = hist_breaks(v, brk_spec)
    if isout is None:
        h = hist_density(v, brk)
        z = h / np.nanmax(h)
        for i in range(brk.shape[0] - 1):
            d.rect(ax, float(brk[i]), 0.0, float(brk[i + 1]), float(z[i]), col=cols[0], border=bdr)
        dx, dy = density_r(v, adj)
        if lwd > 0:
            d.line(ax, dx, dy / np.nanmax(h), col=cols[1], lw=lwd * 0.75)
    else:
        h_o, h_i = hist_density(v[isout], brk), hist_density(v[~isout], brk)
        m = max(np.nanmax(h_o), np.nanmax(h_i))
        for h, ch in ((h_i, cols_hist[1]), (h_o, cols_hist[0])):
            z = h / m
            for i in range(brk.shape[0] - 1):
                d.rect(ax, float(brk[i]), 0.0, float(brk[i + 1]), float(z[i]), col=ch, border=bdr)
        if lwd > 0:
            for vv, cl in ((v[~isout], cols_line[1]), (v[isout], cols_line[0])):
                dx, dy = density_r(vv, adj)
                d.line(ax, dx, dy / m, col=cl, lw=lwd * 0.75)
