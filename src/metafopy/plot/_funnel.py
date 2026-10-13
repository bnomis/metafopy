"""Funnel plots (metafor's ``funnel.rma`` and ``funnel.default``).

Plots the observed effects against a measure of their precision (standard error, variance, their
inverses, sample size, weight, ...); for ``sei``, ``vi``, ``seinv`` and ``vinv`` the pseudo
confidence-interval regions around the reference line are drawn (``level`` may be a vector for
contour-enhanced funnel plots, with ``shade`` the colours of the regions). For a fitted model with
moderators the residuals are plotted; a trim-and-fill result (an :class:`~metafopy.RMAResult` with a
boolean ``fill`` attribute) shows the imputed studies as open symbols.

See https://wviechtb.github.io/metafor/reference/funnel.html, Sterne & Egger (2001),
doi:10.1016/S0895-4356(01)00377-8 and Peters et al. (2008), doi:10.1001/jama.299.15.2037.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from metafopy.methods._common import apply_each
from metafopy.methods.influence import rstandard, rstudent
from metafopy.models.rma_uni import RMAResult
from metafopy.plot import _draw as d
from metafopy.plot._rutils import ax_ticks, coladj, fmtx1, get_ax, level_to_alpha, setlab, usr_range

__all__ = ['FunnelResult', 'funnel']

type FloatArray = NDArray[np.float64]
type Func = Callable[..., Any]

_YAXES = ('sei', 'vi', 'seinv', 'vinv', 'ni', 'ninv', 'sqrtni', 'sqrtninv', 'lni', 'wi')
_YLAB = {
    'sei': 'Standard Error',
    'vi': 'Variance',
    'seinv': 'Inverse Standard Error',
    'vinv': 'Inverse Variance',
    'ni': 'Sample Size',
    'ninv': 'Inverse Sample Size',
    'sqrtni': 'Square Root Sample Size',
    'sqrtninv': 'Inverse Square Root Sample Size',
    'lni': 'Log Sample Size',
    'wi': 'Weight (in %)',
}


@dataclass(frozen=True, slots=True)
class FunnelResult:
    """The plotted coordinates (what metafor's ``funnel()`` returns invisibly) and the axes drawn on.

    ``x`` holds the observed outcomes (or residuals), ``y`` the values on the vertical axis,
    ``fill`` marks imputed studies of a trim-and-fill result (``None`` otherwise).
    """

    ax: Axes
    x: FloatArray
    y: FloatArray
    slab: tuple[str, ...]
    fill: NDArray[np.bool_] | None = None

    def to_frame(self) -> pd.DataFrame:
        out = pd.DataFrame({'x': self.x, 'y': self.y, 'slab': list(self.slab)})
        if self.fill is not None:
            out['fill'] = self.fill
        return out


def _color_ramp(c1: str, c2: str, n: int) -> list[str]:
    """``colorRampPalette(c(c1, c2))(n)`` for hex colours (linear interpolation in RGB)."""
    a = np.array([int(c1[i : i + 2], 16) for i in (1, 3, 5)], dtype=float)
    b = np.array([int(c2[i : i + 2], 16) for i in (1, 3, 5)], dtype=float)
    out: list[str] = []
    for t in np.linspace(0, 1, n):
        v = a + (b - a) * t
        r, g, bl = (math.floor(x + 0.5) for x in v)
        out.append(f'#{r:02X}{g:02X}{bl:02X}')
    return out


def _yvalues(yaxis: str, sei: FloatArray, vi: FloatArray, ni: FloatArray | None, weights: FloatArray | None) -> FloatArray:
    """The values plotted on the vertical axis."""
    if yaxis == 'sei':
        return sei
    if yaxis == 'vi':
        return vi
    if yaxis == 'seinv':
        return np.asarray(1 / sei, dtype=np.float64)
    if yaxis == 'vinv':
        return np.asarray(1 / vi, dtype=np.float64)
    if yaxis == 'wi':
        assert weights is not None
        return weights
    assert ni is not None
    return np.asarray({'ni': ni, 'ninv': 1 / ni, 'sqrtni': np.sqrt(ni), 'sqrtninv': 1 / np.sqrt(ni), 'lni': np.log(ni)}[yaxis], dtype=np.float64)


def _vec(v: Any, n: int, name: str) -> tuple[list[Any], bool]:
    if isinstance(v, str) or np.ndim(v) == 0:
        return [v] * n, False
    out = list(v)
    if len(out) != n:
        raise ValueError(f"Length of the '{name}' argument ({len(out)}) does not correspond to the number of outcomes ({n}).")
    return out, True


def funnel(
    x: RMAResult | ArrayLike,
    vi: ArrayLike | None = None,
    sei: ArrayLike | None = None,
    ni: ArrayLike | None = None,
    subset: ArrayLike | None = None,
    yaxis: str = 'sei',
    xlim: Sequence[float] | None = None,
    ylim: Sequence[float] | None = None,
    xlab: str | None = None,
    ylab: str | None = None,
    slab: Sequence[Any] | None = None,
    steps: int = 5,
    at: ArrayLike | None = None,
    atransf: Func | None = None,
    targs: Any = None,
    digits: int | Sequence[int] | None = None,
    level: float | Sequence[float] | None = None,
    addtau2: bool = False,
    type: str = 'rstandard',
    back: str | None = None,
    shade: str | Sequence[str] | None = None,
    hlines: str | None = None,
    refline: float | None = None,
    lty: Any = 3,
    pch: int | Sequence[int] | None = None,
    pch_fill: int = 21,
    col: str | Sequence[str] | None = None,
    bg: str | Sequence[str] | None = None,
    label: bool | int | str = False,
    offset: float = 0.4,
    legend: bool | str | dict[str, Any] = False,
    *,
    ax: Axes | None = None,
    refline2: float | None = None,
    level2: float | None = None,
    lty2: Any = 3,
    ci_res: int = 1000,
    colref: str | None = None,
    colci: str | None = None,
    colbox: str | None = None,
    at_lab: Sequence[str] | None = None,
    measure: str | None = None,
) -> FunnelResult:
    """Funnel plot of a fitted model (``funnel.rma``) or of estimates with their variances (``funnel.default``).

    ``yaxis`` is one of ``'sei'`` (default), ``'vi'``, ``'seinv'``, ``'vinv'``, ``'ni'``,
    ``'ninv'``, ``'sqrtni'``, ``'sqrtninv'``, ``'lni'`` or ``'wi'``. ``level`` (a percentage; a
    vector draws several CI regions, a contour-enhanced funnel plot) defaults to the confidence
    level of the model (95 for the numeric form), ``shade`` colours the regions (``back`` the
    background outside), ``refline`` is the reference line (the pooled estimate, or 0 for models
    with moderators and the numeric form), ``addtau2`` adds the model's ``tau^2`` to the sampling
    variances in the CI regions, ``label`` (``True``/``'all'``, ``'out'`` or the number of the
    most extreme studies) labels points with ``slab``, and ``type`` selects the residuals plotted
    for models with moderators (``'rstandard'`` or ``'rstudent'``). Pass ``ni`` (or fit the model with
    ``ni``) for the sample-size axes. For a trim-and-fill result the imputed studies are drawn
    with symbol ``pch_fill`` (open by default). ``legend`` is only approximated (a matplotlib legend).

    Returns a :class:`FunnelResult` with the plotted coordinates and the axes.
    """
    if yaxis not in _YAXES:
        raise ValueError(f'yaxis must be one of {_YAXES}')
    if type not in ('rstandard', 'rstudent'):
        raise ValueError("type must be 'rstandard' or 'rstudent'")
    axx = get_ax(ax)
    is_fit = isinstance(x, RMAResult)
    fill_all: NDArray[np.bool_] | None = None
    fill_attr = getattr(x, 'fill', None) if is_fit else None
    if fill_attr is not None:
        fill_all = np.asarray(fill_attr, dtype=bool)
    tf = fill_all is not None

    fg = d.FG
    back_c = coladj(-0.2) if back is None else back
    hlines_c = coladj(1.0) if hlines is None else hlines
    colref_c = coladj(-0.6) if colref is None else colref
    colci_c = coladj(-0.6) if colci is None else colci
    colbox_c = coladj(-0.6) if colbox is None else colbox
    tau2 = 0.0
    k_in: int
    int_only = True
    if is_fit:
        res = x
        assert isinstance(res, RMAResult)
        lvl_alpha = [res.level] if level is None else [level_to_alpha(v) for v in np.atleast_1d(level)]
        lvl2 = res.level if level2 is None else level_to_alpha(level2)
        int_only = res.int_only
        k_in = res.k
        ni_all = res.data.ni
        if ni_all is not None and ni_all.shape[0] == res.data.k_all:
            ni_all = ni_all[res.data.keep] if ni_all.shape[0] != res.k else ni_all
        slab_all = list(res.slab) if slab is None else [str(s) for s in slab]
        if slab is not None and len(slab_all) == res.data.k_all and res.data.k_all != res.k:
            slab_all = [s for s, kp in zip(slab_all, res.data.keep, strict=True) if kp]
        if len(slab_all) != k_in:
            raise ValueError(f"Length of the 'slab' argument ({len(slab_all)}) does not correspond to the size of the original dataset.")
        if int_only:
            ref = float(res.beta[0]) if refline is None else refline
            yi, vi_a = res.yi.copy(), res.vi.copy()
            ni_a = None if ni_all is None else np.asarray(ni_all, dtype=float)
            sei_a = np.sqrt(vi_a)
            xlab_c = setlab(res.measure, None, atransf, 1) if xlab is None else xlab
            tau2 = res.tau2 if addtau2 else 0.0
        else:
            ref = 0.0 if refline is None else refline
            rr = rstandard(res) if type == 'rstandard' else rstudent(res)
            ok = ~np.isnan(rr.resid)
            yi, sei_a = rr.resid[ok], rr.se[ok]
            vi_a = sei_a**2
            ni_a = None if ni_all is None else np.asarray(ni_all, dtype=float)[ok]
            slab_all = [s for s, o in zip(slab_all, ok, strict=True) if o]
            xlab_c = 'Residual Value' if xlab is None else xlab
        weights = res.weights_() if yaxis == 'wi' else None
        if yaxis in ('ni', 'ninv', 'sqrtni', 'sqrtninv', 'lni') and ni_a is None:
            raise ValueError('No sample size information stored in model object.')
        pch_vals, pch_vec = _vec(19 if pch is None else pch, k_in, 'pch')
        if tf:
            assert fill_all is not None
            colv = [fg, fg] if col is None else ([col, fg] if isinstance(col, str) else list(col))
            bgv = [coladj(-0.6), coladj(-0.1)] if bg is None else ([bg, coladj(-0.1)] if isinstance(bg, str) else list(bg))
            col_vals: list[Any] = []
            bg_vals: list[Any] = []
            col_vec = bg_vec = False
            fill = fill_all[: yi.shape[0]] if int_only else None
        else:
            col_vals, col_vec = _vec(fg if col is None else col, k_in, 'col')
            bg_vals, bg_vec = _vec(coladj(-0.1) if bg is None else bg, k_in, 'bg')
            colv, bgv = [], []
            fill = None
        if int_only and k_in != yi.shape[0]:
            raise ValueError('inconsistent number of studies')
    else:
        yi = np.atleast_1d(np.asarray(x, dtype=np.float64)).copy()
        k_in = int(yi.shape[0])
        lvl_alpha = [level_to_alpha(95 if level is None else v) for v in np.atleast_1d(95 if level is None else level)]
        lvl2 = level_to_alpha(95 if level2 is None else level2)
        attrs = getattr(x, 'attrs', {})
        mes = measure if measure is not None else (str(attrs['measure']) if isinstance(attrs, dict) and attrs.get('measure') else None)
        ni_a = None if ni is None else np.atleast_1d(np.asarray(ni, dtype=float))
        if ni_a is None and isinstance(attrs, dict) and attrs.get('ni') is not None:
            ni_a = np.asarray(attrs['ni'], dtype=float)
        if vi is None and sei is not None:
            vi_in: FloatArray | None = np.asarray(sei, dtype=float) ** 2
        else:
            vi_in = None if vi is None else np.asarray(vi, dtype=float)
        if yaxis in ('sei', 'vi', 'seinv', 'vinv', 'wi') and vi_in is None:
            raise ValueError("Must specify the 'vi' or 'sei' argument.")
        if vi_in is not None and vi_in.shape[0] != k_in:
            raise ValueError("Length of 'yi' and 'vi' (or 'sei') are not the same.")
        vi_a = np.zeros(k_in) if vi_in is None else np.where(vi_in < 0, 0.0, vi_in)
        sei_a = np.sqrt(vi_a)
        slab_all = [str(i + 1) for i in range(k_in)] if slab is None else [str(s) for s in slab]
        if len(slab_all) != k_in:
            raise ValueError(f"Length of the 'slab' argument ({len(slab_all)}) does not correspond to the number of outcomes ({k_in}).")
        pch_vals, pch_vec = _vec(19 if pch is None else pch, k_in, 'pch')
        col_vals, col_vec = _vec(fg if col is None else col, k_in, 'col')
        bg_vals, bg_vec = _vec(coladj(-0.1) if bg is None else bg, k_in, 'bg')
        colv, bgv = [], []
        ref = 0.0 if refline is None else refline
        fill = None
        weights = None
        if subset is not None:
            sa = np.asarray(subset)
            sel = np.flatnonzero(sa) if sa.dtype == np.bool_ else sa.astype(np.intp)
            yi, vi_a, sei_a = yi[sel], vi_a[sel], sei_a[sel]
            if ni_a is not None:
                ni_a = ni_a[sel]
            slab_all = [slab_all[i] for i in sel]
            pch_vals, col_vals, bg_vals = ([v[i] for i in sel] for v in (pch_vals, col_vals, bg_vals))
        has_na = np.isnan(yi)
        if yaxis in ('sei', 'seinv', 'vi', 'vinv'):
            has_na |= np.isnan(vi_a)
        if yaxis in ('ni', 'ninv', 'sqrtni', 'sqrtninv', 'lni'):
            if ni_a is None:
                raise ValueError('No sample size information was specified (ni).')
            has_na |= np.isnan(ni_a)
        if has_na.any():
            ok = ~has_na
            yi, vi_a, sei_a = yi[ok], vi_a[ok], sei_a[ok]
            if ni_a is not None:
                ni_a = ni_a[ok]
            slab_all = [s for s, o in zip(slab_all, ok, strict=True) if o]
            pch_vals, col_vals, bg_vals = ([v for v, o in zip(vs, ok, strict=True) if o] for vs in (pch_vals, col_vals, bg_vals))
        xlab_c = setlab(mes, None, atransf, 1) if xlab is None else xlab
        if yi.shape[0] < 2:
            raise ValueError('Plotting terminated since k < 2.')
        if yaxis == 'wi':
            if np.any(vi_a <= 0):
                raise ValueError('Cannot plot weights when there are non-positive sampling variances in the data.')
            w0 = 1 / vi_a
            weights = w0 / w0.sum() * 100

    shade_c: list[str]
    if shade is None:
        base = coladj(0.0)
        shade_c = [base]
    else:
        shade_c = [shade] if isinstance(shade, str) else list(shade)
    if len(lvl_alpha) > 1 and len(shade_c) == 1:
        shade_c = _color_ramp(shade_c[0], coladj(-0.5), len(lvl_alpha))
        shade_c = [shade_c[0], *shade_c[:0:-1]] if len(shade_c) > 1 else shade_c

    dg = [2, 3] if digits is None else ([digits, digits] if isinstance(digits, int) else list(digits))
    if digits is None:
        dg = {'ni': [2, 0], 'wi': [2, 2]}.get(yaxis, [2, 3])
    drop0 = all(isinstance(v, int | np.integer) for v in dg)
    ylab_c = _YLAB[yaxis] if ylab is None else ylab
    lt = [lty, lty] if np.ndim(lty) == 0 or isinstance(lty, str) else list(lty)

    sei_ok = sei_a
    # y limits
    if ylim is None:
        if yaxis == 'sei':
            yl = [float(np.max(sei_ok)), 0.0]
        elif yaxis == 'vi':
            yl = [float(np.max(vi_a)), 0.0]
        elif yaxis == 'seinv':
            yl = [float(np.min(1 / sei_ok)), float(np.max(1 / sei_ok))]
        elif yaxis == 'vinv':
            yl = [float(np.min(1 / vi_a)), float(np.max(1 / vi_a))]
        else:
            assert ni_a is not None or yaxis == 'wi'
            if yaxis == 'ni':
                yl = [float(np.nanmin(ni_a)), float(np.nanmax(ni_a))]  # type: ignore[arg-type]
            elif yaxis == 'ninv':
                yl = [float(np.nanmax(1 / ni_a)), float(np.nanmin(1 / ni_a))]  # type: ignore[operator]
            elif yaxis == 'sqrtni':
                yl = [float(np.nanmin(np.sqrt(ni_a))), float(np.nanmax(np.sqrt(ni_a)))]  # type: ignore[arg-type]
            elif yaxis == 'sqrtninv':
                yl = [float(np.nanmax(1 / np.sqrt(ni_a))), float(np.nanmin(1 / np.sqrt(ni_a)))]  # type: ignore[arg-type]
            elif yaxis == 'lni':
                yl = [float(np.nanmin(np.log(ni_a))), float(np.nanmax(np.log(ni_a)))]  # type: ignore[arg-type]
            else:
                assert weights is not None
                yl = [float(np.min(weights)), float(np.max(weights))]
        if any(math.isinf(v) for v in yl):
            raise ValueError("Setting 'ylim' automatically not possible (must set y-axis limits manually).")
    else:
        lo_, hi_ = float(min(ylim)), float(max(ylim))
        yl = [hi_, lo_] if yaxis in ('sei', 'vi', 'ninv', 'sqrtninv') else [lo_, hi_]
        if yaxis in ('sei', 'vi', 'ni', 'ninv', 'sqrtni', 'sqrtninv', 'lni', 'wi') and (yl[0] < 0 or yl[1] < 0):
            raise ValueError('Both y-axis limits must be >= 0.')
        if yaxis in ('seinv', 'vinv') and (yl[0] <= 0 or yl[1] <= 0):
            raise ValueError('Both y-axis limits must be > 0.')

    zcrit_min = float(stats.norm.isf(min(lvl_alpha) / 2))
    lvals = len(lvl_alpha)
    ci_axes = yaxis in ('sei', 'vi', 'seinv', 'vinv')
    if ci_axes:
        bot = {
            'sei': math.sqrt(yl[0] ** 2 + tau2),
            'vi': math.sqrt(yl[0] + tau2),
            'seinv': math.sqrt(1 / yl[0] ** 2 + tau2),
            'vinv': math.sqrt(1 / yl[0] + tau2),
        }[yaxis]
        x_lb_bot, x_ub_bot = ref - zcrit_min * bot, ref + zcrit_min * bot
    if xlim is None:
        if ci_axes:
            xl = [min(x_lb_bot, float(np.nanmin(yi))), max(x_ub_bot, float(np.nanmax(yi)))]
        else:
            xl = [float(np.min(yi)), float(np.max(yi))]
        rx = xl[1] - xl[0]
        xl = [xl[0] - rx * 0.1, xl[1] + rx * 0.1]
    else:
        xl = [float(min(xlim)), float(max(xlim))]
    at_in = None if at is None else np.asarray(at, dtype=float)
    if at_in is not None:
        xl[0] = float(min(xl[0], np.nanmin(at_in)))
        xl[1] = float(max(xl[1], np.nanmax(at_in)))

    # -- drawing -------------------------------------------------------------------------------------------------
    ux = usr_range(xl[0], xl[1])
    uy = usr_range(yl[0], yl[1])
    axx.set_xlim(ux)
    axx.set_ylim(uy)
    axx.set_autoscale_on(False)
    axx.set_xlabel(xlab_c)
    axx.set_ylabel(ylab_c)
    for name in ('top', 'right'):
        axx.spines[name].set_visible(False)
    for name in ('left', 'bottom'):
        axx.spines[name].set_visible(True)
        axx.spines[name].set_color(colbox_c)
    d.rect(axx, ux[0], uy[0], ux[1], uy[1], col=back_c, border=None, zorder=0.4)
    yat = np.linspace(yl[0], yl[1], steps)
    d.yaxis(axx, yat, [fmtx1(float(v), dg[1], drop0) for v in yat])
    for v in yat:
        d.hline(axx, float(v), col=hlines_c)
        axx.lines[-1].set_zorder(0.5)

    if ci_axes:
        if yaxis in ('sei', 'vi'):
            rylim = yl[0] - yl[1]
            yl[0] = yl[0] + rylim * 0.1
            yl[1] = max(0.0, yl[1] - rylim * 0.1)
        else:
            rylim = yl[1] - yl[0]
            yl[1] = yl[1] + rylim * 0.1
        yvals = np.linspace(yl[0], yl[1], ci_res)
        with np.errstate(divide='ignore'):
            vvals = np.asarray({'sei': yvals**2, 'vi': yvals, 'seinv': 1 / yvals**2, 'vinv': 1 / yvals}[yaxis], dtype=np.float64)
        for m in range(lvals - 1, -1, -1):
            zc = float(stats.norm.isf(lvl_alpha[m] / 2))
            ci_left = ref - zc * np.sqrt(vvals + tau2)
            ci_right = ref + zc * np.sqrt(vvals + tau2)
            d.poly(axx, np.concatenate([ci_left, ci_right[::-1]]), np.concatenate([yvals, yvals[::-1]]), col=shade_c[m], border=None, zorder=0.6)
            d.line(axx, ci_left, yvals, lty=lt[0], col=colci_c).set_zorder(0.7)
            d.line(axx, ci_right, yvals, lty=lt[0], col=colci_c).set_zorder(0.7)
        if refline2 is not None:
            zc2 = float(stats.norm.isf(lvl2 / 2))
            d.line(axx, refline2 - zc2 * np.sqrt(vvals + tau2), yvals, lty=lty2, col=colci_c).set_zorder(0.7)
            d.line(axx, refline2 + zc2 * np.sqrt(vvals + tau2), yvals, lty=lty2, col=colci_c).set_zorder(0.7)
        d.seg(axx, ref, yl[0], ref, yl[1], lty=lt[1], col=colref_c).set_zorder(0.7)
        if refline2 is not None:
            d.seg(axx, refline2, yl[0], refline2, yl[1], lty=lty2, col=colref_c).set_zorder(0.7)
    else:
        d.vline(axx, ref, lty=lt[1], col=colref_c)
        if refline2 is not None:
            d.vline(axx, refline2, lty=lty2, col=colref_c)

    xvals = yi
    with np.errstate(divide='ignore'):
        yvals_pts = _yvalues(yaxis, sei_ok, vi_a, ni_a, weights)
    if tf and is_fit and int_only:
        assert fill is not None
        for xi_, yi_ in zip(xvals[~fill], yvals_pts[~fill], strict=True):
            d.pts(axx, xi_, yi_, pch=int(pch_vals[0]), col=colv[0], bg=bgv[0])
        for xi_, yi_ in zip(xvals[fill], yvals_pts[fill], strict=True):
            d.pts(axx, xi_, yi_, pch=int(pch_fill), col=colv[1], bg=bgv[1])
    else:
        for i in range(xvals.shape[0]):
            if not np.isnan(xvals[i]) and not np.isnan(yvals_pts[i]):
                d.pts(axx, xvals[i], yvals_pts[i], pch=int(pch_vals[i]), col=col_vals[i], bg=bg_vals[i])

    if at_in is None:
        at_a = ax_ticks(ux[0], ux[1])
    else:
        at_a = at_in[(at_in > ux[0]) & (at_in < ux[1])]
    if at_lab is None:
        vals = apply_each(at_a, atransf, targs) if atransf is not None else at_a
        at_lab_l = [fmtx1(float(v), dg[0], drop0) for v in vals]
    else:
        at_lab_l = list(at_lab)
    d.xaxis(axx, at_a, at_lab_l)
    axx.spines['bottom'].set_bounds(float(at_a.min()), float(at_a.max())) if at_a.size else None

    k = int(xvals.shape[0])
    sel_label: list[int] = []
    if label is not False and label is not None:
        ref_l = ref
        if math.isnan(ref_l):
            ref_l = float(np.nanmean(yi))
        if isinstance(label, str):
            if label == 'all':
                sel_label = list(range(k))
            elif label == 'out':
                if not ci_axes:
                    sel_label = list(range(k))
                else:
                    sel_label = [i for i in range(k) if abs(yi[i] - ref_l) / math.sqrt(vi_a[i] + tau2) >= zcrit_min]
        elif label is True:
            sel_label = list(range(k))
        elif isinstance(label, int | float | np.integer):
            nlab = min(max(round(float(label)), 0), k)
            sel_label = [int(i) for i in np.argsort(-np.abs(yi - ref_l), kind='stable')[:nlab]]
        for i in sel_label:
            d.txt(axx, float(yi[i]), float(yvals_pts[i]), slab_all[i], pos=4 if yi[i] - ref_l >= 0 else 2, offset=offset)

    _legend(
        axx,
        legend,
        lvl_alpha,
        shade_c,
        back_c,
        yaxis,
        tf,
        pch_vals,
        col_vals if not tf else colv,
        bg_vals if not tf else bgv,
        pch_fill,
        pch_vec,
        col_vec,
        bg_vec,
        colci_c,
    )
    return FunnelResult(axx, np.asarray(xvals, dtype=float), np.asarray(yvals_pts, dtype=float), tuple(slab_all), fill)


def _legend(
    ax: Axes,
    legend: bool | str | dict[str, Any],
    level: list[float],
    shade: list[str],
    back: str,
    yaxis: str,
    trimfill: bool,
    pch: list[Any],
    col: list[Any],
    bg: list[Any],
    pch_fill: int,
    pch_vec: bool,
    col_vec: bool,
    bg_vec: bool,
    colci: str,
) -> None:
    """An approximation of metafor's funnel legend (CI-region p-value/CI entries and the study symbols)."""
    opts: dict[str, Any] = {'x': 'upper right', 'show': 'pvals', 'studies': True}
    if isinstance(legend, dict):
        opts.update(legend)
        on = True
    elif isinstance(legend, str):
        opts['x'] = {'topright': 'upper right', 'topleft': 'upper left', 'bottomright': 'lower right', 'bottomleft': 'lower left'}.get(legend, legend)
        on = True
    else:
        on = bool(legend)
    if not on:
        return
    if yaxis not in ('sei', 'vi', 'seinv', 'vinv'):
        opts['show'] = None
    if pch_vec or col_vec or bg_vec:
        opts['studies'] = False
    if not opts['studies'] and opts['show'] is None:
        return
    handles: list[Any] = []
    labels: list[str] = []
    if opts['show'] == 'pvals':
        lv = [*level, 0.0]
        nd = max(len(f'{v:.8g}'.rstrip('0')) for v in lv) - 2
        nd = max(0, nd)
        for i, v in enumerate(lv):
            hi = 1.0 if i == 0 else lv[i - 1]
            labels.append(f'{v:.{nd}f} < p <= {hi:.{nd}f}')
            handles.append(Patch(facecolor=[*shade, back][i], edgecolor=colci))
    elif opts['show'] == 'cis':
        for v, sc in zip(level, shade, strict=False):
            labels.append(f'{100 - 100 * v:g}% CI Region')
            handles.append(Patch(facecolor=sc, edgecolor=colci))
    if opts['studies']:
        labels.append('Observed Studies' if trimfill else 'Studies')
        handles.append(Line2D([], [], linestyle='None', marker='o', markerfacecolor=str(bg[0]), markeredgecolor=str(col[0])))
        if trimfill:
            labels.append('Imputed Studies')
            handles.append(Line2D([], [], linestyle='None', marker='o', markerfacecolor=str(bg[1]), markeredgecolor=str(col[1])))
    ax.legend(handles, labels, loc=opts['x'], framealpha=1.0)
