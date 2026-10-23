"""Plot of the results of a GOSH analysis (metafor's ``plot.gosh.rma``).

For models without moderators the estimates are plotted against a heterogeneity statistic with marginal histograms and
kernel density estimates; otherwise a scatterplot matrix of the heterogeneity statistic and the coefficients is drawn.
See Olkin, Dahabreh & Trikalinos (2012), doi:10.1002/jrsm.1053, and
https://wviechtb.github.io/metafor/reference/plot.gosh.rma.html.
"""

import collections.abc
import dataclasses
import typing

import matplotlib.axes
import matplotlib.colors
import numpy
import numpy.typing

import metafopy.errors
import metafopy.methods.gosh
import metafopy.plot
import metafopy.plot.draw
import metafopy.plot.plain
import metafopy.plot.rutils

__all__ = ['PlotGoshResult', 'plot_gosh']

type FloatArray = numpy.typing.NDArray[numpy.float64]

# auditme: accept[non-alphabetical-dict-keys] order follows the R script that generates the fixtures
HET = {'QE': 'QE', 'I2': 'I2', 'I^2': 'I2', 'H2': 'H2', 'H^2': 'H2', 'tau2': 'tau2', 'tau^2': 'tau2', 'tau': 'tau'}
# auditme: accept[non-alphabetical-dict-keys] order follows the R script that generates the fixtures
HET_LABEL = {'QE': r'$Q_E$', 'I2': r'$I^2$', 'H2': r'$H^2$', 'tau2': r'$\tau^2$', 'tau': r'$\tau$'}


@dataclasses.dataclass(frozen=True, slots=True)
class PlotGoshResult:
    """The axes of the plot: ``main`` is the scatterplot (``None`` for the scatterplot matrix), ``xhist``/``yhist`` the
    marginal histograms (if drawn) and ``panels`` the axes of the scatterplot matrix (row by row)."""

    main: matplotlib.axes.Axes | None
    xhist: matplotlib.axes.Axes | None
    yhist: matplotlib.axes.Axes | None
    panels: tuple[tuple[matplotlib.axes.Axes, ...], ...] = ()

    @property
    def axes(self) -> tuple[matplotlib.axes.Axes, ...]:
        found = [a for a in (self.xhist, self.main, self.yhist) if a is not None]
        return tuple(found) if found else tuple(a for row in self.panels for a in row)


def hex_color(col: str, alpha: float) -> str:
    """``rgb(col2rgb(col)/255, alpha)``: an eight-digit hex colour."""
    r, g, b = matplotlib.colors.to_rgb(metafopy.plot.draw.rcolor(col))
    return '#' + ''.join(f'{int(255 * v + 0.5):02X}' for v in (r, g, b, alpha))


def quantile(x: FloatArray, p: float) -> float:
    return float(numpy.nanquantile(x, p))


def expand(v: typing.Any, n: int) -> list[typing.Any]:
    if isinstance(v, str) or numpy.ndim(v) == 0:
        return [v] * n
    out = list(v)
    return out + [out[-1]] * (n - len(out)) if len(out) < n else out[:n]


def plot_gosh(
    x: metafopy.methods.gosh.GoshResult,
    het: str = 'I2',
    pch: int = 16,
    cex: float | None = None,
    out: int | None = None,
    col: str | collections.abc.Sequence[str] | None = None,
    alpha: float | collections.abc.Sequence[float] | None = None,
    border: str | collections.abc.Sequence[str] | None = None,
    xlim: collections.abc.Sequence[float] | None = None,
    ylim: collections.abc.Sequence[float] | None = None,
    xhist: bool = True,
    yhist: bool = True,
    hh: float | collections.abc.Sequence[float] = 0.3,
    breaks: typing.Any = None,
    adjust: float | collections.abc.Sequence[float] | None = None,
    lwd: float | collections.abc.Sequence[float] | None = None,
    labels: collections.abc.Sequence[str] | None = None,
    trim: float | collections.abc.Sequence[float | collections.abc.Sequence[float]] | None = None,
    *,
    ax: matplotlib.axes.Axes | None = None,
) -> PlotGoshResult:
    """Plot the results of [gosh][metafopy.gosh] (``plot.gosh.rma``).

    Mirrors metafor's ``plot.gosh.rma()``.

    For a model without moderators the heterogeneity statistic ``het`` (``'QE'``, ``'I2'``, ``'H2'``, ``'tau2'`` or
    ``'tau'``) is plotted against the estimate, with marginal histograms and kernel density estimates (``xhist``,
    ``yhist``; ``hh`` the relative sizes of the marginal plots, ``breaks``, ``adjust`` and ``lwd`` their settings, each
    a single value or a pair for the two margins); with moderators a scatterplot matrix of ``het`` and the
    coefficients is drawn. ``out`` (0-based) highlights the subsets that do (``col[0]``) or do not include that study,
    ``alpha`` gives the transparency of the points, histogram and density line, ``trim`` removes the most extreme
    subsets (a proportion or a list of proportions per column). The panels fill the area of ``ax``.

    Args:
        x: The result of [gosh][metafopy.gosh].
        het: Heterogeneity statistic to plot: ``'QE'``, ``'I2'`` (the default), ``'H2'``, ``'tau2'`` or ``'tau'`` (``'I^2'``, ``'H^2'`` and ``'tau^2'``
            are accepted too). ``'tau2'`` and ``'tau'`` are not available for equal-effects models.
        pch: Plotting symbol of the points (R's ``pch`` code). Default: 16, a filled circle without border.
        cex: Expansion factor of the points. Default: 0.5.
        out: 0-based index of a study to highlight (metafor's ``out`` counts from 1): the subsets that include it are drawn in ``col[0]``, those
            without in ``col[1]``. Default: no highlighting.
        col: Colour of the points. Default: black. With ``out`` it must be two colours (subsets with the ``out`` study, subsets without it), default
            ``('red', 'blue')``.
        alpha: Transparency (0 transparent to 1 opaque) of the points, of the histogram bars and of the density lines: one value for the points
            (the histograms then get 0.5 and the density lines 0.9), two for points and histograms, or all three. Default: ``n ** -0.2`` for ``n`` subsets.
        border: Border colour of the histogram bars, one for both or a pair (horizontal, vertical margin). Default: light grey. (metafor's ``FALSE``,
            which omits the borders, is not supported.)
        xlim: Limits of the horizontal axis as (low, high); models without moderators only. Default: the range of the histogram breaks of the estimates.
        ylim: Limits of the vertical axis as (low, high); models without moderators only. Default: the range of the histogram breaks of ``het``.
        xhist: Draw the marginal histogram and density of the estimates; models without moderators only. Default: ``True``.
        yhist: Draw the marginal histogram and density of the heterogeneity statistic; models without moderators only. Default: ``True``.
        hh: Relative size of the marginal plots, between 0 and 1 (checked for models without moderators); a single value or a pair for the two margins.
            For models with moderators it stretches the height of the histograms on the diagonal. Default: 0.3.
        breaks: Histogram breaks of the marginal plots, as for R's ``hist``; a single value or a pair. Default: ``'Sturges'``.
        adjust: Bandwidth adjustment of the kernel density estimates (larger is smoother); a single value or a pair. Default: 1.
        lwd: Width of the density lines, 0 omits them; a single value or a pair. Default: 2.
        labels: The labels of the horizontal and vertical axes (no moderators) or the names of the variables of the scatterplot matrix. Default: derived
            from the effect size measure, ``het`` and the coefficient names.
        trim: Remove the most extreme subsets before plotting (an undocumented argument of metafor, passed through ``...``): the proportion to cut
            from each tail of every variable (``het`` and the coefficients), or a list with one proportion, or a (lower, upper) pair, per variable.
        ax: Matplotlib axes whose area the panels fill. Default: the current axes.

    Returns:
        A ``PlotGoshResult``: for a model without moderators ``main`` is the scatterplot and ``xhist`` and ``yhist`` the marginal histograms (``None`` if
        not drawn); for a scatterplot matrix ``main``, ``xhist`` and ``yhist`` are ``None`` and ``panels`` holds the axes row by row. ``axes`` lists
        whichever were drawn.

    Raises:
        metafopy.errors.InputError: ``het`` is unknown or ``'tau2'``/``'tau'`` is used for an equal-effects model, ``out`` is not a study, ``col`` does not
            have two colours when ``out`` is used, or ``hh`` is outside 0 to 1.

    Metafor reference:
        [plot.gosh.rma](https://wviechtb.github.io/metafor/reference/plot.gosh.rma.html)
    """
    het = HET.get(het, het)
    if het not in HET_LABEL:
        raise metafopy.errors.InputError(f'het must be one of {", ".join(HET)}')
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    if het in ('tau2', 'tau') and x.method in ('FE', 'EE', 'CE'):
        raise metafopy.errors.InputError("Cannot plot 'tau2' for equal/fixed-effects models.")
    area = metafopy.plot.rutils.get_ax(ax)
    res = x.res
    incl = x.incl
    cex_ = 0.5 if cex is None else cex
    coef_cols = list(res.columns[6:])
    cols_x = [het, *coef_cols]
    if trim is not None:
        trims: list[typing.Any] = [trim] * len(cols_x) if isinstance(trim, int | float) else list(trim)
        if len(trims) < len(cols_x):
            trims = trims + [trims[-1]] * (len(cols_x) - len(trims))
        dele = numpy.zeros(len(res), dtype=bool)
        for i, c in enumerate(cols_x):
            v = res[c].to_numpy(dtype=float)
            t = numpy.atleast_1d(numpy.asarray(trims[i], dtype=float))
            with numpy.errstate(invalid='ignore'):
                dele |= (v < quantile(v, float(t[0]))) | (v > quantile(v, 1 - float(t[-1])))
        res = res.loc[~dele].reset_index(drop=True)
        incl = incl[~dele]
    n = len(res)

    alpha_v = n ** (-0.2) if alpha is None else alpha
    a = list(numpy.atleast_1d(alpha_v).astype(float))
    if len(a) == 1:
        a = [a[0], 0.5, 0.9]
    elif len(a) == 2:
        a = [a[0], a[1], 0.9]
    if out is None:
        base = metafopy.plot.draw.FG if col is None else (col if isinstance(col, str) else col[0])
        col_pnts: str | list[str] = hex_color(base, a[0])
        cols = (hex_color(base, a[1]), hex_color(base, a[2]))
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        cols_hist = cols_line = ('', '')
        isout = None
    else:
        if not 0 <= int(out) < x.k:
            raise metafopy.errors.InputError('Non-existing study chosen as potential outlier.')
        cc = ['red', 'blue'] if col is None else ([col] if isinstance(col, str) else list(col))
        if len(cc) != 2:
            raise metafopy.errors.InputError("Argument 'col' should specify two colors when argument 'out' is used.")
        isout = incl[:, int(out)]
        col_o, col_i = cc
        col_pnts = [hex_color(col_o, a[0]) if f else hex_color(col_i, a[0]) for f in isout]
        cols_hist = (hex_color(col_o, a[1]), hex_color(col_i, a[1]))
        cols_line = (hex_color(col_o, a[2]), hex_color(col_i, a[2]))
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        cols = ('', '')
    border_c = metafopy.plot.rutils.coladj(-0.1) if border is None else border
    bd = expand(border_c, 2)
    hhv = [float(v) for v in expand(hh, 2)]
    if x.int_only and any(v < 0 or v > 1 for v in hhv):
        raise metafopy.errors.InputError("Invalid value(s) specified for 'hh' argument.")
    if breaks is None or isinstance(breaks, str) or numpy.ndim(breaks) == 0:
        br: list[typing.Any] = [breaks or 'Sturges'] * 2
    else:
        br = list(breaks)[:2]
        if len(br) == 1:
            br = br * 2
    adj = [float(v) for v in expand(1.0 if adjust is None else adjust, 2)]
    lw = [float(v) for v in expand(2.0 if lwd is None else lwd, 2)]

    if labels is None:
        if x.int_only:
            labels = [metafopy.plot.rutils.setlab(x.measure, None, None, 2), HET_LABEL['QE'].replace('Q_E', 'Q') if het == 'QE' else HET_LABEL[het]]
        else:
            labels = [HET_LABEL[het], *coef_cols]
    labels = list(labels)

    if x.int_only:
        return plot_int_only(area, res, het, isout, pch, cex_, col_pnts, cols, cols_hist, cols_line, bd, hhv, br, adj, lw, labels, xlim, ylim, xhist, yhist)
    return plot_pairs(area, res, cols_x, isout, pch, cex_, col_pnts, cols, cols_hist, cols_line, bd, hhv, br, adj, lw, labels)


def hist_panel_data(v: FloatArray, brk: typing.Any, adjust: float) -> tuple[FloatArray, FloatArray, tuple[FloatArray, FloatArray]]:
    h = metafopy.plot.rutils.hist_breaks(v, brk)
    return h, metafopy.plot.rutils.hist_density(v, h), metafopy.plot.rutils.density_r(v, adjust)


def plot_int_only(
    area: matplotlib.axes.Axes,
    res: typing.Any,
    het: str,
    isout: numpy.typing.NDArray[numpy.bool_] | None,
    pch: int,
    cex: float,
    col_pnts: str | list[str],
    cols: tuple[str, str],
    cols_hist: tuple[typing.Any, typing.Any],
    cols_line: tuple[typing.Any, typing.Any],
    bd: list[typing.Any],
    hh: list[float],
    br: list[typing.Any],
    adj: list[float],
    lw: list[float],
    labels: list[str],
    xlim: collections.abc.Sequence[float] | None,
    ylim: collections.abc.Sequence[float] | None,
    xhist: bool,
    yhist: bool,
) -> PlotGoshResult:
    est = res['estimate'].to_numpy(dtype=float)
    hv = res[het].to_numpy(dtype=float)
    hx_breaks = metafopy.plot.rutils.hist_breaks(est, br[0])
    hy_breaks = metafopy.plot.rutils.hist_breaks(hv, br[1])
    rows, cols_n = (2 if xhist else 1), (2 if yhist else 1)
    kw: dict[str, typing.Any] = {'hspace': 0.0, 'wspace': 0.0}
    if xhist:
        kw['height_ratios'] = [hh[0], 1 - hh[0]]
    if yhist:
        kw['width_ratios'] = [1 - hh[1], hh[1]]
    grid = metafopy.plot.draw.subaxes(area, rows, cols_n, **kw)
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

    def margin(axm: matplotlib.axes.Axes, v: FloatArray, brk: FloatArray, a_: float, lim: tuple[float, float], horizontal: bool, bdr: str, lwd_: float) -> None:
        if isout is None:
            groups = [(v, cols[0], cols[1])]
        else:
            groups = [(v[~isout], cols_hist[1], cols_line[1]), (v[isout], cols_hist[0], cols_line[0])]
        dens = [(metafopy.plot.rutils.hist_density(g, brk), metafopy.plot.rutils.density_r(g, a_), ch, cl) for g, ch, cl in groups]
        top = max(float(numpy.nanmax(numpy.concatenate([h, dn[1]]))) for h, dn, _, _ in dens)
        if horizontal:
            metafopy.plot.plain.plain_plot(axm, lim, (0.0, top), bty='n', xaxt=False, yaxt=False)
        else:
            metafopy.plot.plain.plain_plot(axm, (0.0, top), lim, bty='n', xaxt=False, yaxt=False)
        for h, _, ch, _ in dens:
            for i in range(brk.shape[0] - 1):
                if horizontal:
                    metafopy.plot.draw.rect(axm, float(brk[i]), 0.0, float(brk[i + 1]), float(h[i]), col=ch, border=bdr)
                else:
                    metafopy.plot.draw.rect(axm, 0.0, float(brk[i]), float(h[i]), float(brk[i + 1]), col=ch, border=bdr)
        if lwd_ > 0:
            for _, (dx, dy), _, cl in dens:
                metafopy.plot.draw.line(axm, dx, dy, col=cl, lw=lwd_ * 0.75) if horizontal else metafopy.plot.draw.line(axm, dy, dx, col=cl, lw=lwd_ * 0.75)

    if ax_x is not None:
        margin(ax_x, est, hx_breaks, adj[0], xl, True, bd[0], lw[0])
    metafopy.plot.plain.plain_plot(ax_main, xl, yl, labels[0], labels[1], bty='l')
    metafopy.plot.draw.scatter(ax_main, est, hv, pch=pch, cex=cex, cols=col_pnts)
    if ax_y is not None:
        margin(ax_y, hv, hy_breaks, adj[1], yl, False, bd[1], lw[1])
    return PlotGoshResult(ax_main, ax_x, ax_y)


def plot_pairs(
    area: matplotlib.axes.Axes,
    res: typing.Any,
    cols_x: list[str],
    isout: numpy.typing.NDArray[numpy.bool_] | None,
    pch: int,
    cex: float,
    col_pnts: str | list[str],
    cols: tuple[str, str],
    cols_hist: tuple[typing.Any, typing.Any],
    cols_line: tuple[typing.Any, typing.Any],
    bd: list[typing.Any],
    hh: list[float],
    br: list[typing.Any],
    adj: list[float],
    lw: list[float],
    labels: list[str],
) -> PlotGoshResult:
    X = numpy.column_stack([res[c].to_numpy(dtype=float) for c in cols_x])
    nc = X.shape[1]
    grid = metafopy.plot.draw.subaxes(area, nc, nc, hspace=0.0, wspace=0.0)
    fig = area.get_figure(root=True)
    assert fig is not None
    for i in range(nc):
        for j in range(nc):
            axp = grid[i][j]
            fin_j, fin_i = X[:, j][numpy.isfinite(X[:, j])], X[:, i][numpy.isfinite(X[:, i])]
            ux = metafopy.plot.rutils.usr_range(float(fin_j.min()), float(fin_j.max()))
            uy = metafopy.plot.rutils.usr_range(float(fin_i.min()), float(fin_i.max()))
            axp.set_xlim(ux)
            axp.set_ylim(uy)
            axp.set_autoscale_on(False)
            axp.set_xticks([])
            axp.set_yticks([])
            j_odd, i_odd = (j + 1) % 2 == 1, (i + 1) % 2 == 1
            if i == 0 and not j_odd:
                side_axis(axp, 'top', metafopy.plot.rutils.ax_ticks(*ux))
            if i == nc - 1 and j_odd:
                side_axis(axp, 'bottom', metafopy.plot.rutils.ax_ticks(*ux))
            if j == 0 and not i_odd:
                side_axis(axp, 'left', metafopy.plot.rutils.ax_ticks(*uy))
            if j == nc - 1 and i_odd and i != j:
                side_axis(axp, 'right', metafopy.plot.rutils.ax_ticks(*uy))
            if i == j:
                diag_hist(axp, X[:, i], isout, cols, cols_hist, cols_line, bd[0], hh[0], br[0], adj[0], lw[0])
                w = 0.55 * max(len(t) for t in labels) * metafopy.plot.draw.base_fontsize() / 72
                panel_in = max(axp.get_position().width * fig.get_size_inches()[0], 1e-6)
                cl = max(0.8, min(2.0, 0.9 / (w / panel_in)))
                t = axp.text(0.5, 0.5 + 1 / 3, labels[i], transform=axp.transAxes, ha='center', va='center', fontsize=metafopy.plot.draw.base_fontsize() * cl)
                t.set_gid('mtext')
            else:
                metafopy.plot.draw.scatter(axp, X[:, j], X[:, i], pch=pch, cex=cex, cols=col_pnts)
    return PlotGoshResult(None, None, None, tuple(tuple(r) for r in grid))


def side_axis(ax: matplotlib.axes.Axes, side: str, at: FloatArray) -> None:
    labels = metafopy.plot.rutils.rformat(at)
    if side == 'top' or side == 'bottom':
        ax.set_xticks(at)
        ax.set_xticklabels(labels, fontsize=metafopy.plot.draw.base_fontsize() * 0.8)
        ax.xaxis.set_ticks_position('top' if side == 'top' else 'bottom')
    else:
        ax.set_yticks(at)
        ax.set_yticklabels(labels, fontsize=metafopy.plot.draw.base_fontsize() * 0.8)
        ax.yaxis.set_ticks_position('left' if side == 'left' else 'right')


def diag_hist(
    ax: matplotlib.axes.Axes,
    v: FloatArray,
    isout: numpy.typing.NDArray[numpy.bool_] | None,
    cols: tuple[str, str],
    cols_hist: tuple[typing.Any, typing.Any],
    cols_line: tuple[typing.Any, typing.Any],
    bdr: str,
    hh: float,
    brk_spec: typing.Any,
    adj: float,
    lwd: float,
) -> None:
    """``panel.hist`` of plot.gosh.rma: a histogram scaled to a maximum of 1 with a density line, on ``ylim = c(0, 1.2 + hh)``."""
    ax.set_ylim(0.0, 1.2 + hh)
    brk = metafopy.plot.rutils.hist_breaks(v, brk_spec)
    if isout is None:
        h = metafopy.plot.rutils.hist_density(v, brk)
        z = h / numpy.nanmax(h)
        for i in range(brk.shape[0] - 1):
            metafopy.plot.draw.rect(ax, float(brk[i]), 0.0, float(brk[i + 1]), float(z[i]), col=cols[0], border=bdr)
        dx, dy = metafopy.plot.rutils.density_r(v, adj)
        if lwd > 0:
            metafopy.plot.draw.line(ax, dx, dy / numpy.nanmax(h), col=cols[1], lw=lwd * 0.75)
    else:
        h_o, h_i = metafopy.plot.rutils.hist_density(v[isout], brk), metafopy.plot.rutils.hist_density(v[~isout], brk)
        m = max(numpy.nanmax(h_o), numpy.nanmax(h_i))
        for h, ch in ((h_i, cols_hist[1]), (h_o, cols_hist[0])):
            z = h / m
            for i in range(brk.shape[0] - 1):
                metafopy.plot.draw.rect(ax, float(brk[i]), 0.0, float(brk[i + 1]), float(z[i]), col=ch, border=bdr)
        if lwd > 0:
            for vv, cl in ((v[~isout], cols_line[1]), (v[isout], cols_line[0])):
                dx, dy = metafopy.plot.rutils.density_r(vv, adj)
                metafopy.plot.draw.line(ax, dx, dy / m, col=cl, lw=lwd * 0.75)
