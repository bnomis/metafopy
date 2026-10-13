"""Thin drawing helpers that mimic R's base graphics primitives on a matplotlib ``Axes``.

Every artist gets a ``gid`` naming the primitive it stands for (``'segment'``, ``'polygon'``,
``'rect'``, ``'text'``, ``'point'``, ``'line'``, ``'hline'``, ``'vline'``) so that the geometry of a
plot can be read back (see :func:`collect`); this is how the tests compare the drawing with R.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import matplotlib as mpl
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle
from matplotlib.text import Text
from matplotlib.transforms import ScaledTranslation
from numpy.typing import ArrayLike

LTY = {
    'solid': '-',
    'dashed': '--',
    'dotted': ':',
    'dotdash': '-.',
    'longdash': (0, (8, 4)),
    'twodash': (0, (2, 2, 6, 2)),
    'blank': 'None',
    1: '-',
    2: '--',
    3: ':',
    4: '-.',
    5: (0, (8, 4)),
    6: (0, (2, 2, 6, 2)),
    0: 'None',
}

# R plotting symbols -> (marker, filled with the foreground colour, filled with ``bg``)
PCH: dict[int, tuple[str, bool, bool]] = {
    0: ('s', False, False),
    1: ('o', False, False),
    2: ('^', False, False),
    3: ('+', False, False),
    4: ('x', False, False),
    5: ('D', False, False),
    6: ('v', False, False),
    15: ('s', True, False),
    16: ('o', True, False),
    17: ('^', True, False),
    18: ('D', True, False),
    19: ('o', True, False),
    20: ('o', True, False),
    21: ('o', False, True),
    22: ('s', False, True),
    23: ('D', False, True),
    24: ('^', False, True),
    25: ('v', False, True),
}

FG = 'black'


_R_NAMES = {
    'gray': '#BEBEBE',
    'grey': '#BEBEBE',
    'green': '#00FF00',
    'lightgray': '#D3D3D3',
    'lightgrey': '#D3D3D3',
    'darkgray': '#A9A9A9',
    'darkgrey': '#A9A9A9',
}


def rcolor(col: Any) -> Any:
    """Translate R colour names that differ from matplotlib's (``gray``, ``grayNN``/``greyNN``, ``green``) to hex."""
    if not isinstance(col, str):
        return col
    name = col.lower()
    if name in _R_NAMES:
        return _R_NAMES[name]
    m = re.fullmatch(r'gr[ae]y(\d{1,3})', name)
    if m and int(m.group(1)) <= 100:
        v = int(int(m.group(1)) * 255 / 100 + 0.5)
        return f'#{v:02X}{v:02X}{v:02X}'
    return col


def _meta(art: Any, **kw: Any) -> None:
    """Record the R-level arguments (``pch``, ``cex``, ``lty``, ``pos``, ...) of a primitive on its artist."""
    art.meta = kw


def base_fontsize() -> float:
    return float(mpl.rcParams['font.size'])


def seg(ax: Axes, x0: float, y0: float, x1: float, y1: float, *, lty: Any = 'solid', col: str = FG, lw: float = 1.0, clip: bool = True) -> Line2D:
    (ln,) = ax.plot([x0, x1], [y0, y1], linestyle=LTY.get(lty, lty), color=rcolor(col), linewidth=lw, solid_capstyle='butt', clip_on=clip, label='_nolegend_')
    ln.set_gid('segment')
    _meta(ln, lty=lty, col=col)
    return ln


def hline(ax: Axes, y: float, *, lty: Any = 'solid', col: str = FG, lw: float = 1.0) -> Line2D:
    ln = ax.axhline(y, linestyle=LTY.get(lty, lty), color=rcolor(col), linewidth=lw)
    ln.set_gid('hline')
    _meta(ln, lty=lty, col=col)
    return ln


def vline(ax: Axes, x: float, *, lty: Any = 'solid', col: str = FG, lw: float = 1.0) -> Line2D:
    ln = ax.axvline(x, linestyle=LTY.get(lty, lty), color=rcolor(col), linewidth=lw)
    ln.set_gid('vline')
    _meta(ln, lty=lty, col=col)
    return ln


def line(ax: Axes, xs: ArrayLike, ys: ArrayLike, *, lty: Any = 'solid', col: str = FG, lw: float = 1.0, gid: str = 'line') -> Line2D:
    (ln,) = ax.plot(np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), linestyle=LTY.get(lty, lty), color=rcolor(col), linewidth=lw, label='_nolegend_')
    ln.set_gid(gid)
    _meta(ln, lty=lty, col=col)
    return ln


def poly(ax: Axes, xs: ArrayLike, ys: ArrayLike, *, col: str | None, border: str | None = None, lw: float = 1.0, zorder: float = 1.0) -> Polygon:
    verts = np.column_stack([np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)])
    p = Polygon(
        verts,
        closed=True,
        facecolor='none' if col is None else rcolor(col),
        edgecolor='none' if border is None else rcolor(border),
        linewidth=lw,
        zorder=zorder,
    )
    p.set_gid('polygon')
    _meta(p, xy=verts, col=col, border=border)
    ax.add_patch(p)
    return p


def rect(ax: Axes, xl: float, yb: float, xr: float, yt: float, *, col: str | None, border: str | None = None, zorder: float = 1.0) -> Rectangle:
    r = Rectangle(
        (min(xl, xr), min(yb, yt)),
        abs(xr - xl),
        abs(yt - yb),
        facecolor='none' if col is None else rcolor(col),
        edgecolor='none' if border is None else rcolor(border),
        zorder=zorder,
    )
    r.set_gid('rect')
    _meta(r, xy=(xl, yb, xr, yt), col=col)
    ax.add_patch(r)
    return r


def txt(
    ax: Axes,
    x: float,
    y: float,
    s: str,
    *,
    pos: int | None = None,
    cex: float = 1.0,
    bold: bool = False,
    offset: float = 0.5,
    col: str = FG,
    family: str | None = None,
    rotation: float = 0.0,
) -> Text:
    """``text(x, y, s, pos, cex, offset)``: ``pos`` 1/2/3/4 = below/left/above/right of the point, ``None`` = centred."""
    fs = base_fontsize() * cex
    ha, va = 'center', 'center'
    dx = dy = 0.0
    off = offset * fs / 72.0  # inches
    if pos == 1:
        va, dy = 'top', -off
    elif pos == 2:
        ha, dx = 'right', -off
    elif pos == 3:
        va, dy = 'bottom', off
    elif pos == 4:
        ha, dx = 'left', off
    t = ax.text(
        x,
        y,
        s,
        ha=ha,
        va=va if pos in (1, 3) else 'center_baseline',
        fontsize=fs,
        fontweight='bold' if bold else 'normal',
        color=rcolor(col),
        family=family,
        rotation=rotation,
        transform=ax.transData + ScaledTranslation(dx, dy, ax.figure.dpi_scale_trans),
        clip_on=False,
    )
    t.set_gid('text')
    _meta(t, pos=pos, cex=cex, bold=bold, col=col)
    t.set_in_layout(False)
    return t


def pts(
    ax: Axes,
    x: ArrayLike,
    y: ArrayLike,
    *,
    pch: int = 19,
    cex: float = 1.0,
    col: str = FG,
    bg: str | None = None,
    size: float = 5.5,
    alpha: float | None = None,
    zorder: float = 3.0,
) -> list[Line2D]:
    """``points(x, y, pch, cex, col, bg)``; every point is its own ``Line2D`` (gid ``'point'``)."""
    marker, filled, bg_filled = PCH.get(int(pch), ('o', True, False))
    xs = np.atleast_1d(np.asarray(x, dtype=float))
    ys = np.atleast_1d(np.asarray(y, dtype=float))
    out: list[Line2D] = []
    for xi, yi in zip(xs, ys, strict=True):
        fc = rcolor(col) if filled else (rcolor(bg) if bg_filled and bg is not None else 'none')
        (ln,) = ax.plot(
            [xi],
            [yi],
            linestyle='None',
            marker=marker,
            markersize=size * cex,
            markerfacecolor=fc,
            markeredgecolor=rcolor(col),
            markeredgewidth=0.8,
            alpha=alpha,
            zorder=zorder,
            clip_on=False,
            label='_nolegend_',
        )
        ln.set_gid('point')
        _meta(ln, pch=int(pch), cex=cex, col=col, bg=bg)
        out.append(ln)
    return out


def xaxis(ax: Axes, at: Sequence[float] | np.ndarray, labels: Sequence[str], *, cex: float = 1.0) -> None:
    """R's ``axis(side = 1, at, labels)``: a line from the first to the last tick, with ticks and labels."""
    at_a = np.asarray(at, dtype=float)
    sp = ax.spines['bottom']
    sp.set_visible(True)
    sp.set_bounds(float(at_a.min()), float(at_a.max()))
    ax.set_xticks(at_a)
    ax.set_xticklabels(list(labels), fontsize=base_fontsize() * cex)
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis='x', length=4.0)


def yaxis(ax: Axes, at: Sequence[float] | np.ndarray, labels: Sequence[str], *, cex: float = 1.0) -> None:
    """R's ``axis(side = 2, at, labels)``."""
    at_a = np.asarray(at, dtype=float)
    sp = ax.spines['left']
    sp.set_visible(True)
    sp.set_bounds(float(at_a.min()), float(at_a.max()))
    ax.set_yticks(at_a)
    ax.set_yticklabels(list(labels), fontsize=base_fontsize() * cex)
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='y', length=4.0)


def box_l(ax: Axes) -> None:
    """``bty = "l"``: only the left and bottom borders."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def only_xaxis(ax: Axes) -> None:
    """``bty = "n"`` with an x axis only (forest plots)."""
    for name in ('top', 'right', 'left'):
        ax.spines[name].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.yaxis.set_visible(False)


def xlab_below(ax: Axes, x: float, s: str, *, cex: float = 1.0, line: float = 2.5, ha: str = 'center', bold: bool = False) -> Text:
    """``mtext(s, side = 1, at = x, line = line)``."""
    fs = base_fontsize() * cex
    off = line * 0.2 * (base_fontsize() / 12.0)
    tr = ax.get_xaxis_transform() + ScaledTranslation(0.0, -off, ax.figure.dpi_scale_trans)
    t = ax.text(x, 0, s, ha=ha, va='top', fontsize=fs, fontweight='bold' if bold else 'normal', transform=tr, clip_on=False)
    t.set_gid('mtext')
    _meta(t, cex=cex)
    t.set_in_layout(False)
    return t


def collect(ax: Axes | Sequence[Axes]) -> dict[str, list[Any]]:
    """The artists drawn on ``ax`` by gid (see the module docstring), in drawing order; their R-level arguments are in ``artist.meta``."""
    out: dict[str, list[Any]] = {g: [] for g in ('segment', 'polygon', 'rect', 'text', 'point', 'line', 'hline', 'vline', 'mtext', 'abline', 'grid')}
    for one in [ax] if isinstance(ax, Axes) else ax:
        for coll in (one.lines, one.patches, one.texts):
            for a in coll:
                gid = a.get_gid()
                if gid in out:
                    out[gid].append(a)
    return out


def abline(ax: Axes, a: float, b: float, *, lty: Any = 'solid', col: str = FG, lw: float = 1.0) -> Line2D:
    """``abline(a, b)``: a line across the whole plot region (gid ``'abline'``)."""
    x0, x1 = ax.get_xlim()
    (ln,) = ax.plot(
        [x0, x1], [a + b * x0, a + b * x1], linestyle=LTY.get(lty, lty), color=rcolor(col), linewidth=lw, scalex=False, scaley=False, label='_nolegend_'
    )
    ln.set_gid('abline')
    _meta(ln, a=a, b=b, lty=lty, col=col)
    return ln


def grid(ax: Axes, xat: ArrayLike, yat: ArrayLike, col: str) -> None:
    """``grid(col = col)``: dotted lines at the axis ticks (gid ``'grid'``)."""
    for v in np.asarray(xat, dtype=float):
        ln = ax.axvline(v, linestyle=':', color=rcolor(col), linewidth=1.0, zorder=0.8)
        ln.set_gid('grid')
        _meta(ln, v=float(v))
    for v in np.asarray(yat, dtype=float):
        ln = ax.axhline(v, linestyle=':', color=rcolor(col), linewidth=1.0, zorder=0.8)
        ln.set_gid('grid')
        _meta(ln, h=float(v))


def plain_axes(
    ax: Axes,
    ux: tuple[float, float],
    uy: tuple[float, float],
    xat: Sequence[float] | np.ndarray | None,
    yat: Sequence[float] | np.ndarray | None,
    xlabels: Sequence[str] | None = None,
    ylabels: Sequence[str] | None = None,
    *,
    bty: str = 'o',
    cex_axis: float = 1.0,
) -> None:
    """The axes of R's ``plot()``: user coordinates ``ux``/``uy``, ticks at ``xat``/``yat`` (``None`` = no axis), and a box (``bty``)."""
    ax.set_xlim(ux)
    ax.set_ylim(uy)
    ax.set_autoscale_on(False)
    for name in ('top', 'right', 'left', 'bottom'):
        ax.spines[name].set_visible(False)
    if bty == 'o':
        for name in ('top', 'right', 'left', 'bottom'):
            ax.spines[name].set_visible(True)
    elif bty == 'l':
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
    if xat is None:
        ax.set_xticks([])
    else:
        at = np.asarray(xat, dtype=float)
        ax.set_xticks(at)
        if xlabels is not None:
            ax.set_xticklabels(list(xlabels), fontsize=base_fontsize() * cex_axis)
        ax.tick_params(axis='x', length=4.0, labelsize=base_fontsize() * cex_axis)
    if yat is None:
        ax.set_yticks([])
    else:
        at = np.asarray(yat, dtype=float)
        ax.set_yticks(at)
        if ylabels is not None:
            ax.set_yticklabels(list(ylabels), fontsize=base_fontsize() * cex_axis)
        ax.tick_params(axis='y', length=4.0, labelsize=base_fontsize() * cex_axis)


def subaxes(ax: Axes, nrow: int, ncol: int, **kw: Any) -> list[list[Axes]]:
    """Replace ``ax`` by an ``nrow`` x ``ncol`` grid of axes filling its area (R's ``par(mfrow)`` / ``layout()``)."""
    fig = ax.figure
    pos = ax.get_position()
    ax.set_axis_off()
    gs = fig.add_gridspec(nrow, ncol, left=pos.x0, right=pos.x1, bottom=pos.y0, top=pos.y1, **kw)
    return [[fig.add_subplot(gs[i, j]) for j in range(ncol)] for i in range(nrow)]


def mtext_left(ax: Axes, s: str, y: float, *, line: float, cex: float = 1.0, ha: str = 'right') -> Text:
    """``mtext(s, side = 2, line = line, at = y, las = 1)``: horizontal text left of the plot region (gid ``'mtext'``)."""
    fs = base_fontsize() * cex
    off = line * 0.2 * (base_fontsize() / 12.0)
    tr = ax.get_yaxis_transform() + ScaledTranslation(-off, 0.0, ax.figure.dpi_scale_trans)
    t = ax.text(0, y, s, ha=ha, va='center', fontsize=fs, transform=tr, clip_on=False)
    t.set_gid('mtext')
    _meta(t, cex=cex)
    t.set_in_layout(False)
    return t
