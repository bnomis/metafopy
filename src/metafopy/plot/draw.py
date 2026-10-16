"""Thin drawing helpers that mimic R's base graphics primitives on a matplotlib ``Axes``.

Every artist gets a ``gid`` naming the primitive it stands for (``'segment'``, ``'polygon'``,
``'rect'``, ``'text'``, ``'point'``, ``'line'``, ``'hline'``, ``'vline'``) so that the geometry of a
plot can be read back (see :func:`collect`); this is how the tests compare the drawing with R.
"""

import collections.abc
import math
import re
import types
import typing

import matplotlib
import matplotlib.axes
import matplotlib.collections
import matplotlib.colors
import matplotlib.lines
import matplotlib.patches
import matplotlib.text
import matplotlib.textpath
import matplotlib.transforms
import numpy
import numpy.typing

import metafopy.errors

LTY = types.MappingProxyType(
    {
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
)

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

# R's device-independent font families (and the PostScript names it knows) -> matplotlib generic families
FAMILIES = types.MappingProxyType(
    {
        'sans': 'sans-serif',
        'serif': 'serif',
        'mono': 'monospace',
        'Helvetica': 'sans-serif',
        'AvantGarde': 'sans-serif',
        'Times': 'serif',
        'Bookman': 'serif',
        'NewCenturySchoolbook': 'serif',
        'Palatino': 'serif',
        'Courier': 'monospace',
    }
)

type Font = tuple[str, int]
type FontSpec = str | Font


def mpl_family(family: str | None) -> str | None:
    """The matplotlib font family for an R family name (``''`` is the device default, i.e. matplotlib's own default)."""
    if not family:
        return None
    return FAMILIES.get(family, family)


def parse_fonts(
    fonts: FontSpec | collections.abc.Sequence[FontSpec] | collections.abc.Mapping[str, int] | None,
    n: int,
) -> list[Font]:
    """metafor's ``fonts`` argument as ``n`` pairs ``(family, font)``.

    ``fonts`` is a family name (``'sans'``, ``'serif'``, ``'mono'``, ...), a pair ``(family, font)`` with ``font`` 1 (plain), 2 (bold),
    3 (italic) or 4 (bold italic), a mapping ``{family: font}`` (R's named vector) or a sequence of those; they are used for the study
    labels, the annotations and (for ``n == 3``) the ``ilab`` columns. A single entry is used for all, and for ``n == 3`` two entries are
    completed with the first for the third. ``None`` is the device default.

    Raises:
        metafopy.errors.InputError: the font specification is invalid.
    """
    if fonts is None:
        return [('', 1)] * n
    items: list[FontSpec]
    if isinstance(fonts, str):
        items = [fonts]
    elif isinstance(fonts, collections.abc.Mapping):
        items = [(str(k), int(v)) for k, v in fonts.items()]
    elif isinstance(fonts, tuple) and len(fonts) == 2 and isinstance(fonts[0], str) and isinstance(fonts[1], int | numpy.integer):
        items = [(fonts[0], int(fonts[1]))]
    else:
        items = list(fonts)
    if not items:
        raise metafopy.errors.InputError("Argument 'fonts' must not be empty.")
    if len(items) == 1:
        items = items * n
    if n == 3 and len(items) == 2:
        items = [*items, items[0]]
    out: list[Font] = []
    for it in items[:n]:
        fam, fnt = (it, 1) if isinstance(it, str) else (it[0], int(it[1]))
        if not isinstance(fam, str) or fnt not in (1, 2, 3, 4):
            raise metafopy.errors.InputError("Each entry of 'fonts' must be a family name or a (family, font) pair with font 1, 2, 3 or 4.")
        out.append((fam, fnt))
    if len(out) < n:
        raise metafopy.errors.InputError(f"Argument 'fonts' must have length 1 to {n}.")
    return out


# auditme: accept[non-alphabetical-dict-keys] order follows the R script that generates the fixtures
R_NAMES = {
    'gray': '#BEBEBE',
    'grey': '#BEBEBE',
    'green': '#00FF00',
    'lightgray': '#D3D3D3',
    'lightgrey': '#D3D3D3',
    'darkgray': '#A9A9A9',
    'darkgrey': '#A9A9A9',
}


def rcolor(col: typing.Any) -> typing.Any:
    """Translate R colour names that differ from matplotlib's (``gray``, ``grayNN``/``greyNN``, ``green``) to hex."""
    if not isinstance(col, str):
        return col
    name = col.lower()
    if name in R_NAMES:
        return R_NAMES[name]
    m = re.fullmatch(r'gr[ae]y(\d{1,3})', name)
    if m and int(m.group(1)) <= 100:
        v = int(int(m.group(1)) * 255 / 100 + 0.5)
        return f'#{v:02X}{v:02X}{v:02X}'
    return col


def artist_meta(art: typing.Any, **kw: typing.Any) -> None:
    """Record the R-level arguments (``pch``, ``cex``, ``lty``, ``pos``, ...) of a primitive on its artist."""
    art.meta = kw


def base_fontsize() -> float:
    return float(matplotlib.rcParams['font.size'])


def seg(
    ax: matplotlib.axes.Axes, x0: float, y0: float, x1: float, y1: float, *, lty: typing.Any = 'solid', col: str = FG, lw: float = 1.0, clip: bool = True
) -> matplotlib.lines.Line2D:
    (ln,) = ax.plot([x0, x1], [y0, y1], linestyle=LTY.get(lty, lty), color=rcolor(col), linewidth=lw, solid_capstyle='butt', clip_on=clip, label='_nolegend_')
    ln.set_gid('segment')
    artist_meta(ln, lty=lty, col=col)
    return ln


def hline(ax: matplotlib.axes.Axes, y: float, *, lty: typing.Any = 'solid', col: str = FG, lw: float = 1.0) -> matplotlib.lines.Line2D:
    ln = ax.axhline(y, linestyle=LTY.get(lty, lty), color=rcolor(col), linewidth=lw)
    ln.set_gid('hline')
    artist_meta(ln, lty=lty, col=col)
    return ln


def vline(ax: matplotlib.axes.Axes, x: float, *, lty: typing.Any = 'solid', col: str = FG, lw: float = 1.0) -> matplotlib.lines.Line2D:
    ln = ax.axvline(x, linestyle=LTY.get(lty, lty), color=rcolor(col), linewidth=lw)
    ln.set_gid('vline')
    artist_meta(ln, lty=lty, col=col)
    return ln


def line(
    ax: matplotlib.axes.Axes,
    xs: numpy.typing.ArrayLike,
    ys: numpy.typing.ArrayLike,
    *,
    lty: typing.Any = 'solid',
    col: str = FG,
    lw: float = 1.0,
    gid: str = 'line',
) -> matplotlib.lines.Line2D:
    (ln,) = ax.plot(
        numpy.asarray(xs, dtype=float), numpy.asarray(ys, dtype=float), linestyle=LTY.get(lty, lty), color=rcolor(col), linewidth=lw, label='_nolegend_'
    )
    ln.set_gid(gid)
    artist_meta(ln, lty=lty, col=col)
    return ln


def poly(
    ax: matplotlib.axes.Axes,
    xs: numpy.typing.ArrayLike,
    ys: numpy.typing.ArrayLike,
    *,
    col: str | None,
    border: str | None = None,
    lw: float = 1.0,
    zorder: float = 1.0,
) -> matplotlib.patches.Polygon:
    verts = numpy.column_stack([numpy.asarray(xs, dtype=float), numpy.asarray(ys, dtype=float)])
    p = matplotlib.patches.Polygon(
        verts,
        closed=True,
        facecolor='none' if col is None else rcolor(col),
        edgecolor='none' if border is None else rcolor(border),
        linewidth=lw,
        zorder=zorder,
    )
    p.set_gid('polygon')
    artist_meta(p, xy=verts, col=col, border=border)
    ax.add_artist(p)
    return p


def rect(
    ax: matplotlib.axes.Axes, xl: float, yb: float, xr: float, yt: float, *, col: str | None, border: str | None = None, zorder: float = 1.0
) -> matplotlib.patches.Rectangle:
    r = matplotlib.patches.Rectangle(
        (min(xl, xr), min(yb, yt)),
        abs(xr - xl),
        abs(yt - yb),
        facecolor='none' if col is None else rcolor(col),
        edgecolor='none' if border is None else rcolor(border),
        zorder=zorder,
    )
    r.set_gid('rect')
    artist_meta(r, xy=(xl, yb, xr, yt), col=col)
    ax.add_artist(r)
    return r


def rects(
    ax: matplotlib.axes.Axes,
    xl: numpy.typing.ArrayLike,
    yb: float,
    xr: numpy.typing.ArrayLike,
    yt: float,
    *,
    cols: collections.abc.Sequence[str],
    zorder: float = 1.0,
) -> matplotlib.collections.PolyCollection:
    """``rect(xl, yb, xr, yt, col = cols, border = cols)`` for many rectangles between the same ``yb`` and ``yt`` (gid ``'rects'``)."""
    left, right = numpy.asarray(xl, dtype=float), numpy.asarray(xr, dtype=float)
    verts = [[(a, yb), (a, yt), (b, yt), (b, yb)] for a, b in zip(left, right, strict=True)]
    rgba = [matplotlib.colors.to_rgba(rcolor(c)) for c in cols]
    pc = matplotlib.collections.PolyCollection(verts, facecolors=rgba, edgecolors=rgba, linewidths=1.0, zorder=zorder)
    pc.set_gid('rects')
    artist_meta(pc, xl=left, yb=yb, xr=right, yt=yt, cols=list(cols))
    ax.add_collection(pc, autolim=False)
    return pc


def txt(
    ax: matplotlib.axes.Axes,
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
    font: int = 1,
    rotation: float = 0.0,
) -> matplotlib.text.Text:
    """``text(x, y, s, pos, cex, offset)``: ``pos`` 1/2/3/4 = below/left/above/right of the point, ``None`` = centred.

    ``family`` is an R family name (see :func:`parse_fonts`) and ``font`` the R font (1 plain, 2 bold, 3 italic, 4 bold italic;
    ``bold=True`` is the same as ``font=2``).
    """
    if bold and font not in (2, 4):
        font = 4 if font == 3 else 2
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
        fontweight='bold' if font in (2, 4) else 'normal',
        fontstyle='italic' if font in (3, 4) else 'normal',
        color=rcolor(col),
        family=mpl_family(family),
        rotation=rotation,
        transform=ax.transData + matplotlib.transforms.ScaledTranslation(dx, dy, ax.figure.dpi_scale_trans),
        clip_on=False,
    )
    t.set_gid('text')
    artist_meta(t, pos=pos, cex=cex, bold=font in (2, 4), col=col, family=family or '', font=font)
    t.set_in_layout(False)
    return t


def pts(
    ax: matplotlib.axes.Axes,
    x: numpy.typing.ArrayLike,
    y: numpy.typing.ArrayLike,
    *,
    pch: int = 19,
    cex: float = 1.0,
    col: str = FG,
    bg: str | None = None,
    size: float = 5.5,
    alpha: float | None = None,
    zorder: float = 3.0,
) -> list[matplotlib.lines.Line2D]:
    """``points(x, y, pch, cex, col, bg)``; every point is its own ``Line2D`` (gid ``'point'``)."""
    marker, filled, bg_filled = PCH.get(int(pch), ('o', True, False))
    xs = numpy.atleast_1d(numpy.asarray(x, dtype=float))
    ys = numpy.atleast_1d(numpy.asarray(y, dtype=float))
    out: list[matplotlib.lines.Line2D] = []
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
        artist_meta(ln, pch=int(pch), cex=cex, col=col, bg=bg)
        out.append(ln)
    return out


def xaxis(
    ax: matplotlib.axes.Axes,
    at: collections.abc.Sequence[float] | numpy.ndarray,
    labels: collections.abc.Sequence[str],
    *,
    cex: float = 1.0,
    family: str | None = None,
) -> None:
    """R's ``axis(side = 1, at, labels)``: a line from the first to the last tick, with ticks and labels (in the R font ``family``)."""
    at_a = numpy.asarray(at, dtype=float)
    sp = ax.spines['bottom']
    sp.set_visible(True)
    sp.set_bounds(float(at_a.min()), float(at_a.max()))
    ax.set_xticks(at_a)
    ax.set_xticklabels(list(labels), fontsize=base_fontsize() * cex, family=mpl_family(family))
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(axis='x', length=4.0)


def yaxis(ax: matplotlib.axes.Axes, at: collections.abc.Sequence[float] | numpy.ndarray, labels: collections.abc.Sequence[str], *, cex: float = 1.0) -> None:
    """R's ``axis(side = 2, at, labels)``."""
    at_a = numpy.asarray(at, dtype=float)
    sp = ax.spines['left']
    sp.set_visible(True)
    sp.set_bounds(float(at_a.min()), float(at_a.max()))
    ax.set_yticks(at_a)
    ax.set_yticklabels(list(labels), fontsize=base_fontsize() * cex)
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='y', length=4.0)


def box_l(ax: matplotlib.axes.Axes) -> None:
    """``bty = "l"``: only the left and bottom borders."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def only_xaxis(ax: matplotlib.axes.Axes) -> None:
    """``bty = "n"`` with an x axis only (forest plots)."""
    for name in ('top', 'right', 'left'):
        ax.spines[name].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.yaxis.set_visible(False)


def xlab_below(
    ax: matplotlib.axes.Axes,
    x: float,
    s: str,
    *,
    cex: float = 1.0,
    line: float = 2.5,
    ha: str = 'center',
    bold: bool = False,
    family: str | None = None,
    font: int = 1,
) -> matplotlib.text.Text:
    """``mtext(s, side = 1, at = x, line = line)`` in the R font ``family`` and ``font`` (``bold=True`` is ``font=2``)."""
    if bold and font not in (2, 4):
        font = 4 if font == 3 else 2
    fs = base_fontsize() * cex
    off = line * 0.2 * (base_fontsize() / 12.0)
    tr = ax.get_xaxis_transform() + matplotlib.transforms.ScaledTranslation(0.0, -off, ax.figure.dpi_scale_trans)
    t = ax.text(
        x,
        0,
        s,
        ha=ha,
        va='top',
        fontsize=fs,
        fontweight='bold' if font in (2, 4) else 'normal',
        fontstyle='italic' if font in (3, 4) else 'normal',
        family=mpl_family(family),
        transform=tr,
        clip_on=False,
    )
    t.set_gid('mtext')
    artist_meta(t, cex=cex, family=family or '', font=font)
    t.set_in_layout(False)
    return t


def collect(ax: matplotlib.axes.Axes | collections.abc.Sequence[matplotlib.axes.Axes]) -> dict[str, list[typing.Any]]:
    """The artists drawn on ``ax`` by gid (see the module docstring), in drawing order; their R-level arguments are in ``artist.meta``."""
    out: dict[str, list[typing.Any]] = {
        g: [] for g in ('segment', 'polygon', 'rect', 'rects', 'text', 'point', 'line', 'hline', 'vline', 'mtext', 'abline', 'grid', 'scatter')
    }
    for one in [ax] if isinstance(ax, matplotlib.axes.Axes) else ax:
        for coll in (one.lines, one.patches, one.texts, one.collections):
            for a in coll:
                gid = a.get_gid()
                if gid in out:
                    out[gid].append(a)
    return out


def abline(ax: matplotlib.axes.Axes, a: float, b: float, *, lty: typing.Any = 'solid', col: str = FG, lw: float = 1.0) -> matplotlib.lines.Line2D:
    """``abline(a, b)``: a line across the whole plot region (gid ``'abline'``)."""
    x0, x1 = ax.get_xlim()
    (ln,) = ax.plot(
        [x0, x1], [a + b * x0, a + b * x1], linestyle=LTY.get(lty, lty), color=rcolor(col), linewidth=lw, scalex=False, scaley=False, label='_nolegend_'
    )
    ln.set_gid('abline')
    artist_meta(ln, a=a, b=b, lty=lty, col=col)
    return ln


def grid(ax: matplotlib.axes.Axes, xat: numpy.typing.ArrayLike, yat: numpy.typing.ArrayLike, col: str) -> None:
    """``grid(col = col)``: dotted lines at the axis ticks (gid ``'grid'``)."""
    for v in numpy.asarray(xat, dtype=float):
        ln = ax.axvline(v, linestyle=':', color=rcolor(col), linewidth=1.0, zorder=0.8)
        ln.set_gid('grid')
        artist_meta(ln, v=float(v))
    for v in numpy.asarray(yat, dtype=float):
        ln = ax.axhline(v, linestyle=':', color=rcolor(col), linewidth=1.0, zorder=0.8)
        ln.set_gid('grid')
        artist_meta(ln, h=float(v))


def overlapping_labels(
    ax: matplotlib.axes.Axes,
    at: numpy.ndarray,
    labels: collections.abc.Sequence[str],
    limits: tuple[float, float],
    fontsize: float,
    *,
    vertical: bool = False,
) -> list[bool]:
    """Which tick labels are left out because they would overlap: R's ``axis()`` skips a label that comes closer than an "m" to the previous
    one drawn (the ticks stay)."""
    box = ax.get_position()
    fig_inches = (ax.figure.bbox.height if vertical else ax.figure.bbox.width) / ax.figure.dpi
    points_per_unit = (box.height if vertical else box.width) * fig_inches * 72.0 / (limits[1] - limits[0])

    def extent(s: str) -> float:
        if not s.strip():
            return 0.0
        box = matplotlib.textpath.TextPath((0, 0), s, size=fontsize).get_extents()
        return float(box.height if vertical else box.width)

    gap = float(matplotlib.textpath.TextPath((0, 0), 'M', size=fontsize).get_extents().width)
    hidden: list[bool] = []
    previous_end = -math.inf
    for x, label in zip(at, labels, strict=True):
        half = extent(label) / 2
        centre = float(x) * points_per_unit
        overlaps = centre - half < previous_end + gap
        hidden.append(overlaps)
        if not overlaps:
            previous_end = centre + half
    return hidden


# R's keywords for the position of a legend as matplotlib's ``loc``
LEGEND_POSITIONS: types.MappingProxyType[str, str] = types.MappingProxyType(
    {
        'bottom': 'lower center',
        'bottomleft': 'lower left',
        'bottomright': 'lower right',
        'center': 'center',
        'left': 'center left',
        'right': 'center right',
        'top': 'upper center',
        'topleft': 'upper left',
        'topright': 'upper right',
    }
)


# the sides of the box drawn by R's ``bty`` (``"n"`` and unknown values draw none)
BOX_SIDES: types.MappingProxyType[str, tuple[str, ...]] = types.MappingProxyType(
    {
        ']': ('top', 'right', 'bottom'),
        '7': ('top', 'right'),
        'c': ('top', 'left', 'bottom'),
        'l': ('left', 'bottom'),
        'o': ('top', 'right', 'left', 'bottom'),
        'u': ('left', 'bottom', 'right'),
    }
)


def plain_axes(
    ax: matplotlib.axes.Axes,
    ux: tuple[float, float],
    uy: tuple[float, float],
    xat: collections.abc.Sequence[float] | numpy.ndarray | None,
    yat: collections.abc.Sequence[float] | numpy.ndarray | None,
    xlabels: collections.abc.Sequence[str] | None = None,
    ylabels: collections.abc.Sequence[str] | None = None,
    *,
    bty: str = 'o',
    cex_axis: float = 1.0,
    thin_xlabels: bool = True,
    thin_ylabels: bool = False,
) -> None:
    """The axes of R's ``plot()``: user coordinates ``ux``/``uy``, ticks at ``xat``/``yat`` (``None`` = no axis), and a box (``bty``)."""
    ax.set_xlim(ux)
    ax.set_ylim(uy)
    ax.set_autoscale_on(False)
    for name in ('top', 'right', 'left', 'bottom'):
        ax.spines[name].set_visible(name in BOX_SIDES.get(bty, ()))
    if xat is None:
        ax.set_xticks([])
    else:
        at = numpy.asarray(xat, dtype=float)
        ax.set_xticks(at)
        if xlabels is not None:
            ax.set_xticklabels(list(xlabels), fontsize=base_fontsize() * cex_axis)
            texts = ax.get_xticklabels()
            if thin_xlabels:
                hidden_x = overlapping_labels(ax, at, xlabels, ux, base_fontsize() * cex_axis)
                for text in texts:
                    text.set_visible(not hidden_x[int(numpy.argmin(numpy.abs(at - text.get_position()[0])))])
        ax.tick_params(axis='x', length=4.0, labelsize=base_fontsize() * cex_axis)
    if yat is None:
        ax.set_yticks([])
    else:
        at = numpy.asarray(yat, dtype=float)
        ax.set_yticks(at)
        if ylabels is not None:
            ax.set_yticklabels(list(ylabels), fontsize=base_fontsize() * cex_axis)
            texts = ax.get_yticklabels()
            if thin_ylabels:
                hidden_y = overlapping_labels(ax, at, ylabels, uy, base_fontsize() * cex_axis, vertical=True)
                for text in texts:
                    text.set_visible(not hidden_y[int(numpy.argmin(numpy.abs(at - text.get_position()[1])))])
        ax.tick_params(axis='y', length=4.0, labelsize=base_fontsize() * cex_axis)


def subaxes(ax: matplotlib.axes.Axes, nrow: int, ncol: int, **kw: typing.Any) -> list[list[matplotlib.axes.Axes]]:
    """Replace ``ax`` by an ``nrow`` x ``ncol`` grid of axes filling its area (R's ``par(mfrow)`` / ``layout()``)."""
    fig = ax.figure
    pos = ax.get_position()
    ax.set_axis_off()
    gs = fig.add_gridspec(nrow, ncol, left=pos.x0, right=pos.x1, bottom=pos.y0, top=pos.y1, **kw)
    return [[fig.add_subplot(gs[i, j]) for j in range(ncol)] for i in range(nrow)]


def mtext_left(ax: matplotlib.axes.Axes, s: str, y: float, *, line: float, cex: float = 1.0, ha: str = 'right') -> matplotlib.text.Text:
    """``mtext(s, side = 2, line = line, at = y, las = 1)``: horizontal text left of the plot region (gid ``'mtext'``)."""
    fs = base_fontsize() * cex
    off = line * 0.2 * (base_fontsize() / 12.0)
    tr = ax.get_yaxis_transform() + matplotlib.transforms.ScaledTranslation(-off, 0.0, ax.figure.dpi_scale_trans)
    t = ax.text(0, y, s, ha=ha, va='center', fontsize=fs, transform=tr, clip_on=False)
    t.set_gid('mtext')
    artist_meta(t, cex=cex)
    t.set_in_layout(False)
    return t


def scatter(
    ax: matplotlib.axes.Axes,
    x: numpy.typing.ArrayLike,
    y: numpy.typing.ArrayLike,
    *,
    pch: int,
    cex: float,
    cols: collections.abc.Sequence[str] | str,
    size: float = 5.5,
) -> typing.Any:
    """Many points drawn at once (``plot(x, y, pch, cex, col)``); the collection has gid ``'scatter'`` and ``meta`` with ``x``, ``y`` and ``cols``."""
    xs = numpy.asarray(x, dtype=float)
    ys = numpy.asarray(y, dtype=float)
    marker, filled, _ = PCH.get(int(pch), ('o', True, False))
    col_list = [cols] * xs.shape[0] if isinstance(cols, str) else list(cols)
    rgba = [matplotlib.colors.to_rgba(rcolor(c)) for c in col_list]
    s = (size * cex) ** 2
    if filled:
        sc = ax.scatter(xs, ys, s=s, marker=marker, c=rgba, linewidths=0.0, zorder=3.0, clip_on=False)
    else:
        sc = ax.scatter(xs, ys, s=s, marker=marker, facecolors='none', edgecolors=rgba, linewidths=0.8, zorder=3.0, clip_on=False)
    sc.set_gid('scatter')
    artist_meta(sc, x=xs, y=ys, cols=col_list, pch=int(pch), cex=cex)
    return sc
