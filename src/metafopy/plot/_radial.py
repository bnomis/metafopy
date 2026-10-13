"""Radial (Galbraith) plot (metafor's ``radial.rma``).

Plots the standardised effects ``z_i = y_i / sqrt(v_i + tau^2)`` against the precisions ``x_i = 1 / sqrt(v_i + tau^2)``;
the slope of the line through the origin and a point is the estimate, shown on the arc to the right. See
Rex Galbraith (1988), doi:10.1080/00401706.1988.10488373, and
https://wviechtb.github.io/metafor/reference/radial.html.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from metafopy.methods._common import apply_each
from metafopy.models._tau2 import FIXED_EFFECTS
from metafopy.models.rma_uni import RMAResult
from metafopy.plot import _draw as d
from metafopy.plot._plain import plain_plot
from metafopy.plot._rutils import ax_ticks, coladj, fmtx, get_ax, level_to_alpha, rformat

__all__ = ['RadialResult', 'galbraith', 'radial']

type FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class RadialResult:
    """The plotted coordinates of a radial plot (metafor's invisible return value) and the axes drawn on."""

    ax: Axes
    x: FloatArray
    y: FloatArray
    ids: tuple[int, ...]
    slab: tuple[str, ...]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({'x': self.x, 'y': self.y, 'ids': list(self.ids), 'slab': list(self.slab)})


def _asch(v: float) -> str:
    """``as.character()`` of a number."""
    return f'{v:.15g}'


def _arc(len_: float, at: FloatArray, asp: float) -> tuple[FloatArray, FloatArray]:
    """Points on the arc of radius ``len_`` (in the metric of the plot) at the slopes ``at``."""
    xis = np.sqrt(len_**2 / (1 + (at / asp) ** 2))
    return xis, xis * at


def radial(
    x: RMAResult,
    center: bool = False,
    xlim: Sequence[float] | None = None,
    zlim: Sequence[float] | None = None,
    xlab: str | None = None,
    zlab: str | None = None,
    atz: ArrayLike | None = None,
    aty: ArrayLike | None = None,
    steps: int = 7,
    level: float | None = None,
    digits: int = 2,
    transf: Callable[..., Any] | None = None,
    targs: Any = None,
    pch: int = 21,
    col: str | None = None,
    bg: str | None = None,
    back: str | None = None,
    arc_res: int | Sequence[int] = 100,
    cex: float | None = None,
    cex_lab: float | None = None,
    cex_axis: float | None = None,
    *,
    ax: Axes | None = None,
) -> RadialResult:
    """Radial (Galbraith) plot of a model without moderators (``radial.rma``); ``galbraith`` is an alias.

    ``center`` plots the deviations from the pooled estimate, ``atz``/``aty`` set the ticks of the z-axis and
    the values on the arc (``aty`` in the units of the effect sizes, ``transf`` is applied to the arc labels),
    ``steps`` the number of arc ticks, ``level`` the confidence level of the shaded band (default: the model's),
    ``digits`` the digits of the arc labels and ``arc_res`` the resolution of the arcs (a second number sets that
    of the confidence interval arc, default a quarter).

    Returns a :class:`RadialResult` with ``x`` (precisions), ``y`` (standardised effects), ``ids`` and ``slab``.
    """
    if not x.int_only:
        raise ValueError('Radial plots only available for models without moderators.')
    axx = get_ax(ax)
    back_c = coladj(-0.1) if back is None else back
    col_c = d.FG if col is None else col
    bg_c = coladj(-0.35) if bg is None else bg

    yi = x.yi.copy()
    yi_c = yi.copy()
    vi = x.vi
    beta = float(x.beta[0])
    ci_lb, ci_ub = float(x.ci_lb[0]), float(x.ci_ub[0])
    tau2 = x.tau2
    if aty is None:
        atyis_rng = (float(yi.min()), float(yi.max()))
        aty_c = None
        aty_a = None
    else:
        aty_c = np.asarray(aty, dtype=float)
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
    alpha = x.level if level is None else level_to_alpha(level)
    zcrit = float(stats.norm.isf(alpha / 2))
    zi = yi / np.sqrt(vi + tau2)
    xi = 1 / np.sqrt(vi + tau2)
    if np.any(np.isinf(xi)) or np.any(np.isinf(zi)):
        raise ValueError("Setting 'xlim' and 'zlim' automatically not possible (must set axis limits manually).")
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

    fixed = x.method in FIXED_EFFECTS
    if xlab is None:
        xlab = r'$x_i = 1/\sqrt{v_i}$' if fixed else r'$x_i = 1/\sqrt{v_i + \tau^2}$'
    cex_ = 1.0 if cex is None else cex
    cex_lab_ = 1.0 if cex_lab is None else cex_lab
    cex_axis_ = 1.0 if cex_axis is None else cex_axis

    axx.set_box_aspect(1)
    pos = axx.get_position()  # room for the z-axis label on the left and the arcs on the right (metafor widens the margins)
    axx.set_position((pos.x0 + 0.2 * pos.width, pos.y0, 0.58 * pos.width, pos.height))
    plain_plot(axx, xlims, zlims, xlab, '', bty='n', xaxt=False, yaxt=False, xaxs_i=True, yaxs_i=True, cex_lab=cex_lab_)
    d.poly(axx, [0, xmax, xmax, 0], [zcrit, zcrit + xmax * beta, -zcrit + xmax * beta, -zcrit], col=back_c, border=None, zorder=0.6)
    d.seg(axx, 0, 0, xmax, xmax * beta, lty='solid')
    d.seg(axx, 0, -zcrit, xmax, -zcrit + xmax * beta, lty='dotted')
    d.seg(axx, 0, zcrit, xmax, zcrit + xmax * beta, lty='dotted')
    # axes
    xt = ax_ticks(xlims[0], xlims[1])
    d.xaxis(axx, xt, rformat(xt), cex=cex_axis_)
    if atz is None:
        minor = np.linspace(-4, 4, 9)
        minor = minor[(minor >= zlims[0]) & (minor <= zlims[1])]
        yat = np.array([v for v in np.linspace(-2, 2, 3) if zlims[0] <= v <= zlims[1]])
        d.yaxis(axx, yat, rformat(yat), cex=cex_axis_)
        axx.set_yticks(minor, minor=True)
        axx.tick_params(axis='y', which='minor', length=2.0, labelleft=False)
        if minor.size:
            axx.spines['left'].set_bounds(float(minor.min()), float(minor.max()))
    else:
        yat = np.asarray(atz, dtype=float)
        yat = yat[(yat >= zlims[0]) & (yat <= zlims[1])]
        d.yaxis(axx, yat, [_asch(float(v)) for v in yat], cex=cex_axis_)
    if zlab is None:
        if center:
            zl = r'$z_i = \frac{y_i - \hat{\theta}}{\sqrt{v_i}}$' if fixed else r'$z_i = \frac{y_i - \hat{\mu}}{\sqrt{v_i + \tau^2}}$'
        else:
            zl = r'$z_i = \frac{y_i}{\sqrt{v_i}}$' if fixed else r'$z_i = \frac{y_i}{\sqrt{v_i + \tau^2}}$'
        d.mtext_left(axx, zl, 0.0, line=5.0, cex=cex_lab_)
    else:
        d.mtext_left(axx, zlab, 0.0, line=4.0, cex=cex_lab_, ha='center')

    asp = (zlims[1] - zlims[0]) / (xlims[1] - xlims[0])
    arcs = [int(arc_res), math.ceil(int(arc_res) / 4)] if isinstance(arc_res, int | np.integer) else [int(v) for v in arc_res][:2]
    seq_rng = atyis_rng if aty_a is None else (float(aty_a.min()), float(aty_a.max()))
    at_arc = np.linspace(float(yi.min()), float(yi.max()), arcs[0]) if aty_a is None else np.linspace(seq_rng[0], seq_rng[1], arcs[0])
    xis, zis = _arc(ya_xpos, at_arc, asp)
    valid = (zis > zlims[0]) & (zis < zlims[1])
    d.line(axx, xis[valid], zis[valid]).set_clip_on(False)
    # ticks on the arc
    atyis = np.linspace(float(yi.min()), float(yi.max()), steps) if aty_a is None else aty_a
    len_l, len_u = ya_xpos, ya_xpos + 0.015 * (xlims[1] - xlims[0])
    xl_, zl_ = _arc(len_l, atyis, asp)
    xu_, zu_ = _arc(len_u, atyis, asp)
    valid = (zl_ > zlims[0]) & (zu_ > zlims[0]) & (zl_ < zlims[1]) & (zu_ < zlims[1])
    for xa, za, xb, zb in zip(xl_[valid], zl_[valid], xu_[valid], zu_[valid], strict=True):
        d.seg(axx, float(xa), float(za), float(xb), float(zb), clip=False)
    # labels of the arc
    atyis_lab = np.linspace(float(yi_c.min()), float(yi_c.max()), steps) if aty_c is None else aty_c
    xl2, zl2 = _arc(ya_xpos + 0.02 * (xlims[1] - xlims[0]), atyis, asp)
    if transf is not None:
        atyis_lab = apply_each(atyis_lab, transf, targs)
    valid = (zl2 > zlims[0]) & (zl2 < zlims[1])
    if valid.any():
        for xa, za, lab in zip(xl2[valid], zl2[valid], fmtx(atyis_lab[valid], digits), strict=True):
            d.txt(axx, float(xa), float(za), lab, pos=4, cex=cex_axis_ * 0.85, offset=0.25)
    # arc of the confidence interval
    at_ci = np.linspace(ci_lb, ci_ub, arcs[1])
    xis, zis = _arc(ci_xpos, at_ci, asp)
    valid = (zis > zlims[0]) & (zis < zlims[1])
    if valid.any():
        d.line(axx, xis[valid], zis[valid]).set_clip_on(False)
    at3 = np.array([ci_lb, beta, ci_ub])
    xl3, zl3 = _arc(ci_xpos - 0.007 * (xlims[1] - xlims[0]), at3, asp)
    xu3, zu3 = _arc(ci_xpos + 0.007 * (xlims[1] - xlims[0]), at3, asp)
    valid = (zl3 > zlims[0]) & (zu3 > zlims[0]) & (zl3 < zlims[1]) & (zu3 < zlims[1])
    for xa, za, xb, zb in zip(xl3[valid], zl3[valid], xu3[valid], zu3[valid], strict=True):
        d.seg(axx, float(xa), float(za), float(xb), float(zb), clip=False)
    for a_, b_ in zip(xi, zi, strict=True):
        d.pts(axx, float(a_), float(b_), pch=pch, cex=cex_, col=col_c, bg=bg_c)
    ids = tuple(int(i) + 1 for i in np.flatnonzero(x.data.keep))
    return RadialResult(axx, np.asarray(xi, dtype=float), np.asarray(zi, dtype=float), ids, tuple(x.slab))


galbraith = radial
