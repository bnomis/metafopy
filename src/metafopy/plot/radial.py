"""Radial (Galbraith) plot (metafor's ``radial.rma``).

Plots the standardised effects ``z_i = y_i / sqrt(v_i + tau^2)`` against the precisions ``x_i = 1 / sqrt(v_i + tau^2)``;
the slope of the line through the origin and a point is the estimate, shown on the arc to the right. See
Galbraith (1988), doi:10.1080/00401706.1988.10488400, and
https://wviechtb.github.io/metafor/reference/radial.html.
"""

import collections.abc
import dataclasses
import math
import typing

import matplotlib.axes
import numpy
import numpy.typing
import pandas
import scipy.stats

import metafopy.errors
import metafopy.methods.common
import metafopy.methods.fitview
import metafopy.models.rma_uni
import metafopy.models.tau2
import metafopy.plot
import metafopy.plot.draw
import metafopy.plot.plain
import metafopy.plot.rutils

__all__ = ['RadialResult', 'galbraith', 'radial']

type FloatArray = numpy.typing.NDArray[numpy.float64]


@dataclasses.dataclass(frozen=True, slots=True)
class RadialResult:
    """The plotted coordinates of a radial plot (metafor's invisible return value) and the axes drawn on."""

    ax: matplotlib.axes.Axes
    x: FloatArray
    y: FloatArray
    ids: tuple[int, ...]
    slab: tuple[str, ...]

    def to_frame(self) -> pandas.DataFrame:
        return pandas.DataFrame({'x': self.x, 'y': self.y, 'ids': list(self.ids), 'slab': list(self.slab)})


def asch(v: float) -> str:
    """``as.character()`` of a number."""
    return f'{v:.15g}'


def arc(len_: float, at: FloatArray, asp: float) -> tuple[FloatArray, FloatArray]:
    """Points on the arc of radius ``len_`` (in the metric of the plot) at the slopes ``at``."""
    xis = numpy.sqrt(len_**2 / (1 + (at / asp) ** 2))
    return xis, xis * at


def radial(
    x: metafopy.methods.fitview.Fit,
    center: bool = False,
    xlim: collections.abc.Sequence[float] | None = None,
    zlim: collections.abc.Sequence[float] | None = None,
    xlab: str | None = None,
    zlab: str | None = None,
    atz: numpy.typing.ArrayLike | None = None,
    aty: numpy.typing.ArrayLike | None = None,
    steps: int = 7,
    level: float | None = None,
    digits: int = 2,
    transf: collections.abc.Callable[..., typing.Any] | None = None,
    targs: typing.Any = None,
    pch: int = 21,
    col: str | None = None,
    bg: str | None = None,
    back: str | None = None,
    arc_res: int | collections.abc.Sequence[int] = 100,
    cex: float | None = None,
    cex_lab: float | None = None,
    cex_axis: float | None = None,
    *,
    ax: matplotlib.axes.Axes | None = None,
) -> RadialResult:
    """Radial (Galbraith) plot of a model without moderators (``radial.rma``); ``metafopy.plot.galbraith`` is an alias.

    Mirrors metafor's ``radial()``, see <https://wviechtb.github.io/metafor/reference/radial.html>.

    ``center`` plots the deviations from the pooled estimate, ``atz``/``aty`` set the ticks of the z-axis and
    the values on the arc (``aty`` in the units of the effect sizes, ``transf`` is applied to the arc labels),
    ``steps`` the number of arc ticks, ``level`` the confidence level of the shaded band (default: the model's),
    ``digits`` the digits of the arc labels and ``arc_res`` the resolution of the arcs (a second number sets that
    of the confidence interval arc, default a quarter).

    Args:
        x: A fitted model without moderators (not a robust, ``rma_mv``, location-scale or selection model).
        center: Plot the standardized deviations from the pooled estimate, which centers the plot horizontally at the pooled estimate. Default: ``False``.
        xlim: Limits of the horizontal axis (the precision). Default: 0 to 1.3 times the largest precision.
        zlim: Limits of the vertical axis (the standardized effects); these are the actual limits of the plotting region. Default: derived from the data,
            at least -5 to 5.
        xlab: Label of the horizontal axis. Default: derived from the model (a math-text label).
        zlab: Label of the vertical axis. Default: derived from the model and ``center`` (a math-text label).
        atz: Positions of the tick marks and labels of the vertical axis (those outside ``zlim`` are dropped). Default: -2, 0 and 2, with unlabelled
            marks at -4 to 4.
        aty: Positions (in the units of the effect sizes, before ``transf``) of the tick marks on the arc; ``steps`` is then ignored. Default: ``steps``
            marks spread from the smallest to the largest estimate.
        steps: Number of tick marks on the arc. Default: 7.
        level: Confidence level of the shaded band (the z-axis error region), between 0 and 100 (95 or 0.95 for 95 percent). Default: the level of the model.
        digits: Number of decimals of the arc labels. Default: 2.
        transf: Function applied to the arc labels.
        targs: Additional arguments passed to ``transf``.
        pch: Plotting symbol of the points (R's ``pch`` code). Default: 21, a circle with border ``col`` and fill ``bg``.
        col: Border colour of the points (a matplotlib colour). Default: the foreground colour (black).
        bg: Fill colour of the points for symbols with a separate fill. Default: a grey adjusted from the background.
        back: Colour of the shaded band. Default: a light grey. (metafor's ``NA``, which suppresses the shading, is not supported.)
        arc_res: Number of points used to draw the arc of the estimates; a second value sets that of the confidence interval arc. Default: 100 and,
            when one number is given, a quarter of it (rounded up) for the confidence interval arc.
        cex: Expansion factor of the point symbols only (as in metafor). Default: 1.
        cex_lab: Expansion factor of the axis labels. Default: 1.
        cex_axis: Expansion factor of the tick labels (the labels of the arc are drawn at 0.85 times this). Default: 1.
        ax: Matplotlib axes to draw into. Default: the current axes.

    Returns:
        The plotted values: ``x`` (precisions), ``y`` (standardized effects), ``ids`` and ``slab``, and the axes.

    Raises:
        metafopy.errors.InputError: the model has moderators, ``level`` is outside 0 to 100, or an estimate or precision is infinite and ``xlim`` and
            ``zlim`` are not given.
        metafopy.errors.NotSupportedError: ``x`` is a robust, ``rma_mv``, location-scale or selection-model fit.
        metafopy.errors.ArgumentError: ``x`` is not a fitted model.
    """
    metafopy.methods.fitview.check_class(x, notav=('robust.rma', 'rma.mv', 'rma.ls', 'rma.gen', 'rma.uni.selmodel'))
    view = metafopy.methods.fitview.fit_view(x)
    if not view.int_only:
        raise metafopy.errors.InputError('Radial plots only available for models without moderators.')
    axx = metafopy.plot.rutils.get_ax(ax)
    back_c = metafopy.plot.rutils.coladj(-0.1) if back is None else back
    col_c = metafopy.plot.draw.FG if col is None else col
    bg_c = metafopy.plot.rutils.coladj(-0.35) if bg is None else bg

    yi = view.yi.copy()
    yi_c = yi.copy()
    vi = view.vi
    beta = float(view.beta[0])
    ci_lb, ci_ub = float(x.ci_lb[0]), float(x.ci_ub[0])
    tau2 = view.tau2
    if aty is None:
        atyis_rng = (float(yi.min()), float(yi.max()))
        aty_c = None
        aty_a = None
    else:
        aty_c = numpy.asarray(aty, dtype=float)
        aty_a = aty_c.copy()
        atyis_rng = (float(aty_a.min()), float(aty_a.max()))
    if center:
        yi = yi - beta
        ci_lb -= beta
        ci_ub -= beta
        atyis_rng = (atyis_rng[0] - beta, atyis_rng[1] - beta)
        if aty_a is not None:
            aty_a = aty_a - beta
        beta = 0.0
    alpha = view.level if level is None else metafopy.plot.rutils.level_to_alpha(level)
    zcrit = float(scipy.stats.norm.isf(alpha / 2))
    zi = yi / numpy.sqrt(vi + tau2)
    xi = 1 / numpy.sqrt(vi + tau2)
    if numpy.any(numpy.isinf(xi)) or numpy.any(numpy.isinf(zi)):
        raise metafopy.errors.InputError("Setting 'xlim' and 'zlim' automatically not possible (must set axis limits manually).")
    xlims = (0.0, 1.3 * float(xi.max())) if xlim is None else (float(min(xlim)), float(max(xlim)))
    ci_xpos = xlims[1] + 0.12 * (xlims[1] - xlims[0])
    ya_xpos = xlims[1] + 0.14 * (xlims[1] - xlims[0])
    xmax = xlims[1]
    if zlim is None:
        zlims = (
            min(-5.0, 1.1 * float(zi.min()), 1.1 * ci_lb * ci_xpos, 1.1 * atyis_rng[0] * ya_xpos, 1.1 * float(yi.min()) * ya_xpos, -1.1 * zcrit + xmax * beta),
            max(5.0, 1.1 * float(zi.max()), 1.1 * ci_ub * ci_xpos, 1.1 * atyis_rng[1] * ya_xpos, 1.1 * float(yi.max()) * ya_xpos, 1.1 * zcrit + xmax * beta),
        )
    else:
        zlims = (float(min(zlim)), float(max(zlim)))

    fixed = view.method in metafopy.models.tau2.FIXED_EFFECTS
    if xlab is None:
        xlab = r'$x_i = 1/\sqrt{v_i}$' if fixed else r'$x_i = 1/\sqrt{v_i + \tau^2}$'
    cex_ = 1.0 if cex is None else cex
    cex_lab_ = 1.0 if cex_lab is None else cex_lab
    cex_axis_ = 1.0 if cex_axis is None else cex_axis

    axx.set_box_aspect(1)
    pos = axx.get_position()  # room for the z-axis label on the left and the arcs on the right (metafor widens the margins)
    axx.set_position((pos.x0 + 0.2 * pos.width, pos.y0, 0.58 * pos.width, pos.height))
    metafopy.plot.plain.plain_plot(axx, xlims, zlims, xlab, '', bty='n', xaxt=False, yaxt=False, xaxs_i=True, yaxs_i=True, cex_lab=cex_lab_)
    metafopy.plot.draw.poly(axx, [0, xmax, xmax, 0], [zcrit, zcrit + xmax * beta, -zcrit + xmax * beta, -zcrit], col=back_c, border=None, zorder=0.6)
    metafopy.plot.draw.seg(axx, 0, 0, xmax, xmax * beta, lty='solid')
    metafopy.plot.draw.seg(axx, 0, -zcrit, xmax, -zcrit + xmax * beta, lty='dotted')
    metafopy.plot.draw.seg(axx, 0, zcrit, xmax, zcrit + xmax * beta, lty='dotted')
    # axes
    xt = metafopy.plot.rutils.ax_ticks(xlims[0], xlims[1])
    metafopy.plot.draw.xaxis(axx, xt, metafopy.plot.rutils.rformat(xt), cex=cex_axis_)
    if atz is None:
        minor = numpy.linspace(-4, 4, 9)
        minor = minor[(minor >= zlims[0]) & (minor <= zlims[1])]
        yat = numpy.array([v for v in numpy.linspace(-2, 2, 3) if zlims[0] <= v <= zlims[1]])
        metafopy.plot.draw.yaxis(axx, yat, metafopy.plot.rutils.rformat(yat), cex=cex_axis_)
        axx.set_yticks(minor, minor=True)
        axx.tick_params(axis='y', which='minor', length=2.0, labelleft=False)
        if minor.size:
            axx.spines['left'].set_bounds(float(minor.min()), float(minor.max()))
    else:
        yat = numpy.asarray(atz, dtype=float)
        yat = yat[(yat >= zlims[0]) & (yat <= zlims[1])]
        metafopy.plot.draw.yaxis(axx, yat, [asch(float(v)) for v in yat], cex=cex_axis_)
    if zlab is None:
        if center:
            zl = r'$z_i = \frac{y_i - \hat{\theta}}{\sqrt{v_i}}$' if fixed else r'$z_i = \frac{y_i - \hat{\mu}}{\sqrt{v_i + \tau^2}}$'
        else:
            zl = r'$z_i = \frac{y_i}{\sqrt{v_i}}$' if fixed else r'$z_i = \frac{y_i}{\sqrt{v_i + \tau^2}}$'
        metafopy.plot.draw.mtext_left(axx, zl, 0.0, line=5.0, cex=cex_lab_)
    else:
        metafopy.plot.draw.mtext_left(axx, zlab, 0.0, line=4.0, cex=cex_lab_, ha='center')

    asp = (zlims[1] - zlims[0]) / (xlims[1] - xlims[0])
    arcs = [int(arc_res), math.ceil(int(arc_res) / 4)] if isinstance(arc_res, int | numpy.integer) else [int(v) for v in arc_res][:2]
    seq_rng = atyis_rng if aty_a is None else (float(aty_a.min()), float(aty_a.max()))
    at_arc = numpy.linspace(float(yi.min()), float(yi.max()), arcs[0]) if aty_a is None else numpy.linspace(seq_rng[0], seq_rng[1], arcs[0])
    xis, zis = arc(ya_xpos, at_arc, asp)
    valid = (zis > zlims[0]) & (zis < zlims[1])
    metafopy.plot.draw.line(axx, xis[valid], zis[valid]).set_clip_on(False)
    # ticks on the arc
    atyis = numpy.linspace(float(yi.min()), float(yi.max()), steps) if aty_a is None else aty_a
    len_l, len_u = ya_xpos, ya_xpos + 0.015 * (xlims[1] - xlims[0])
    xl_, zl_ = arc(len_l, atyis, asp)
    xu_, zu_ = arc(len_u, atyis, asp)
    valid = (zl_ > zlims[0]) & (zu_ > zlims[0]) & (zl_ < zlims[1]) & (zu_ < zlims[1])
    for xa, za, xb, zb in zip(xl_[valid], zl_[valid], xu_[valid], zu_[valid], strict=True):
        metafopy.plot.draw.seg(axx, float(xa), float(za), float(xb), float(zb), clip=False)
    # labels of the arc
    atyis_lab = numpy.linspace(float(yi_c.min()), float(yi_c.max()), steps) if aty_c is None else aty_c
    xl2, zl2 = arc(ya_xpos + 0.02 * (xlims[1] - xlims[0]), atyis, asp)
    if transf is not None:
        atyis_lab = metafopy.methods.common.apply_each(atyis_lab, transf, targs)
    valid = (zl2 > zlims[0]) & (zl2 < zlims[1])
    if valid.any():
        for xa, za, lab in zip(xl2[valid], zl2[valid], metafopy.plot.rutils.fmtx(atyis_lab[valid], digits), strict=True):
            metafopy.plot.draw.txt(axx, float(xa), float(za), lab, pos=4, cex=cex_axis_ * 0.85, offset=0.25)
    # arc of the confidence interval
    at_ci = numpy.linspace(ci_lb, ci_ub, arcs[1])
    xis, zis = arc(ci_xpos, at_ci, asp)
    valid = (zis > zlims[0]) & (zis < zlims[1])
    if valid.any():
        metafopy.plot.draw.line(axx, xis[valid], zis[valid]).set_clip_on(False)
    at3 = numpy.array([ci_lb, beta, ci_ub])
    xl3, zl3 = arc(ci_xpos - 0.007 * (xlims[1] - xlims[0]), at3, asp)
    xu3, zu3 = arc(ci_xpos + 0.007 * (xlims[1] - xlims[0]), at3, asp)
    valid = (zl3 > zlims[0]) & (zu3 > zlims[0]) & (zl3 < zlims[1]) & (zu3 < zlims[1])
    for xa, za, xb, zb in zip(xl3[valid], zl3[valid], xu3[valid], zu3[valid], strict=True):
        metafopy.plot.draw.seg(axx, float(xa), float(za), float(xb), float(zb), clip=False)
    for a_, b_ in zip(xi, zi, strict=True):
        metafopy.plot.draw.pts(axx, float(a_), float(b_), pch=pch, cex=cex_, col=col_c, bg=bg_c)
    ids = tuple(int(i) + 1 for i in view.ids)
    return RadialResult(axx, numpy.asarray(xi, dtype=float), numpy.asarray(zi, dtype=float), ids, view.slab)


galbraith = radial
