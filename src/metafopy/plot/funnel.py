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

import collections.abc
import dataclasses
import math
import typing

import matplotlib.axes
import matplotlib.lines
import matplotlib.patches
import numpy
import numpy.typing
import pandas
import scipy.stats

import metafopy.errors
import metafopy.methods.common
import metafopy.methods.influence
import metafopy.models.rma_uni
import metafopy.plot
import metafopy.plot.draw
import metafopy.plot.rutils

__all__ = ['FunnelResult', 'funnel']

type FloatArray = numpy.typing.NDArray[numpy.float64]
type Func = collections.abc.Callable[..., typing.Any]

# auditme: accept[non-alphabetical-literal] order is metafor's documented order and is meaningful
YAXES = ('sei', 'vi', 'seinv', 'vinv', 'ni', 'ninv', 'sqrtni', 'sqrtninv', 'lni', 'wi')
# auditme: accept[non-alphabetical-dict-keys] order follows the R script that generates the fixtures
YLAB = {
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


@dataclasses.dataclass(frozen=True, slots=True)
class FunnelResult:
    """The plotted coordinates (what metafor's ``funnel()`` returns invisibly) and the axes drawn on.

    ``x`` holds the observed outcomes (or residuals), ``y`` the values on the vertical axis,
    ``fill`` marks imputed studies of a trim-and-fill result (``None`` otherwise).
    """

    ax: matplotlib.axes.Axes
    x: FloatArray
    y: FloatArray
    slab: tuple[str, ...]
    fill: numpy.typing.NDArray[numpy.bool_] | None = None

    def to_frame(self) -> pandas.DataFrame:
        out = pandas.DataFrame({'x': self.x, 'y': self.y, 'slab': list(self.slab)})
        if self.fill is not None:
            out['fill'] = self.fill
        return out


def yvalues(yaxis: str, sei: FloatArray, vi: FloatArray, ni: FloatArray | None, weights: FloatArray | None) -> FloatArray:
    """The values plotted on the vertical axis."""
    if yaxis == 'sei':
        return sei
    if yaxis == 'vi':
        return vi
    if yaxis == 'seinv':
        return numpy.asarray(1 / sei, dtype=numpy.float64)
    if yaxis == 'vinv':
        return numpy.asarray(1 / vi, dtype=numpy.float64)
    if yaxis == 'wi':
        assert weights is not None
        return weights
    assert ni is not None
    return numpy.asarray({'ni': ni, 'ninv': 1 / ni, 'sqrtni': numpy.sqrt(ni), 'sqrtninv': 1 / numpy.sqrt(ni), 'lni': numpy.log(ni)}[yaxis], dtype=numpy.float64)


def vec(v: typing.Any, n: int, name: str) -> tuple[list[typing.Any], bool]:
    if isinstance(v, str) or numpy.ndim(v) == 0:
        return [v] * n, False
    out = list(v)
    if len(out) != n:
        raise metafopy.errors.InputError(f"Length of the '{name}' argument ({len(out)}) does not correspond to the number of outcomes ({n}).")
    return out, True


def funnel(
    x: metafopy.models.rma_uni.RMAResult | numpy.typing.ArrayLike,
    vi: numpy.typing.ArrayLike | None = None,
    sei: numpy.typing.ArrayLike | None = None,
    ni: numpy.typing.ArrayLike | None = None,
    subset: numpy.typing.ArrayLike | None = None,
    yaxis: str = 'sei',
    xlim: collections.abc.Sequence[float] | None = None,
    ylim: collections.abc.Sequence[float] | None = None,
    xlab: str | None = None,
    ylab: str | None = None,
    slab: collections.abc.Sequence[typing.Any] | None = None,
    steps: int = 5,
    at: numpy.typing.ArrayLike | None = None,
    atransf: Func | None = None,
    targs: typing.Any = None,
    digits: int | collections.abc.Sequence[int] | None = None,
    level: float | collections.abc.Sequence[float] | None = None,
    addtau2: bool = False,
    type: str = 'rstandard',
    back: str | None = None,
    shade: str | collections.abc.Sequence[str] | None = None,
    hlines: str | None = None,
    refline: float | None = None,
    lty: typing.Any = 3,
    pch: int | collections.abc.Sequence[int] | None = None,
    pch_fill: int = 21,
    col: str | collections.abc.Sequence[str] | None = None,
    bg: str | collections.abc.Sequence[str] | None = None,
    label: bool | int | str = False,
    offset: float = 0.4,
    legend: bool | str | dict[str, typing.Any] = False,
    *,
    ax: matplotlib.axes.Axes | None = None,
    refline2: float | None = None,
    level2: float | None = None,
    lty2: typing.Any = 3,
    ci_res: int = 1000,
    colref: str | None = None,
    colci: str | None = None,
    colbox: str | None = None,
    at_lab: collections.abc.Sequence[str] | None = None,
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

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    if yaxis not in YAXES:
        raise metafopy.errors.InputError(f'yaxis must be one of {YAXES}')
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    if type not in ('rstandard', 'rstudent'):
        raise metafopy.errors.InputError("type must be 'rstandard' or 'rstudent'")
    axx = metafopy.plot.rutils.get_ax(ax)
    is_fit = isinstance(x, metafopy.models.rma_uni.RMAResult)
    fill_all: numpy.typing.NDArray[numpy.bool_] | None = None
    fill_attr = getattr(x, 'fill', None) if is_fit else None
    if fill_attr is not None:
        fill_all = numpy.asarray(fill_attr, dtype=bool)
    tf = fill_all is not None

    fg = metafopy.plot.draw.FG
    back_c = metafopy.plot.rutils.coladj(-0.2) if back is None else back
    hlines_c = metafopy.plot.rutils.coladj(1.0) if hlines is None else hlines
    colref_c = metafopy.plot.rutils.coladj(-0.6) if colref is None else colref
    colci_c = metafopy.plot.rutils.coladj(-0.6) if colci is None else colci
    colbox_c = metafopy.plot.rutils.coladj(-0.6) if colbox is None else colbox
    tau2 = 0.0
    k_in: int
    int_only = True
    if is_fit:
        res = x
        assert isinstance(res, metafopy.models.rma_uni.RMAResult)
        lvl_alpha = [res.level] if level is None else [metafopy.plot.rutils.level_to_alpha(v) for v in numpy.atleast_1d(level)]
        lvl2 = res.level if level2 is None else metafopy.plot.rutils.level_to_alpha(level2)
        int_only = res.int_only
        k_in = res.k
        ni_all = res.data.ni
        if ni_all is not None and ni_all.shape[0] == res.data.k_all:
            ni_all = ni_all[res.data.keep] if ni_all.shape[0] != res.k else ni_all
        slab_all = list(res.slab) if slab is None else [str(s) for s in slab]
        if slab is not None and len(slab_all) == res.data.k_all and res.data.k_all != res.k:
            slab_all = [s for s, kp in zip(slab_all, res.data.keep, strict=True) if kp]
        if len(slab_all) != k_in:
            raise metafopy.errors.InputError(f"Length of the 'slab' argument ({len(slab_all)}) does not correspond to the size of the original dataset.")
        if int_only:
            ref = float(res.beta[0]) if refline is None else refline
            yi, vi_a = res.yi.copy(), res.vi.copy()
            ni_a = None if ni_all is None else numpy.asarray(ni_all, dtype=float)
            sei_a = numpy.sqrt(vi_a)
            xlab_c = metafopy.plot.rutils.setlab(res.measure, None, atransf, 1) if xlab is None else xlab
            tau2 = res.tau2 if addtau2 else 0.0
        else:
            ref = 0.0 if refline is None else refline
            rr = metafopy.methods.influence.rstandard(res) if type == 'rstandard' else metafopy.methods.influence.rstudent(res)
            ok = ~numpy.isnan(rr.resid)
            yi, sei_a = rr.resid[ok], rr.se[ok]
            vi_a = sei_a**2
            ni_a = None if ni_all is None else numpy.asarray(ni_all, dtype=float)[ok]
            slab_all = [s for s, o in zip(slab_all, ok, strict=True) if o]
            xlab_c = 'Residual Value' if xlab is None else xlab
        weights = res.weights_() if yaxis == 'wi' else None
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        if yaxis in ('ni', 'ninv', 'sqrtni', 'sqrtninv', 'lni') and ni_a is None:
            raise metafopy.errors.InputError('No sample size information stored in model object.')
        pch_vals, pch_vec = vec(19 if pch is None else pch, k_in, 'pch')
        if tf:
            assert fill_all is not None
            colv = [fg, fg] if col is None else ([col, fg] if isinstance(col, str) else list(col))
            bgv = (
                [metafopy.plot.rutils.coladj(-0.6), metafopy.plot.rutils.coladj(-0.1)]
                if bg is None
                else ([bg, metafopy.plot.rutils.coladj(-0.1)] if isinstance(bg, str) else list(bg))
            )
            col_vals: list[typing.Any] = []
            bg_vals: list[typing.Any] = []
            col_vec = bg_vec = False
            fill = fill_all[: yi.shape[0]] if int_only else None
        else:
            col_vals, col_vec = vec(fg if col is None else col, k_in, 'col')
            bg_vals, bg_vec = vec(metafopy.plot.rutils.coladj(-0.1) if bg is None else bg, k_in, 'bg')
            colv, bgv = [], []
            fill = None
        if int_only and k_in != yi.shape[0]:
            raise metafopy.errors.InputError('inconsistent number of studies')
    else:
        yi = numpy.atleast_1d(numpy.asarray(x, dtype=numpy.float64)).copy()
        k_in = int(yi.shape[0])
        lvl_alpha = [metafopy.plot.rutils.level_to_alpha(95 if level is None else v) for v in numpy.atleast_1d(95 if level is None else level)]
        lvl2 = metafopy.plot.rutils.level_to_alpha(95 if level2 is None else level2)
        attrs = getattr(x, 'attrs', {})
        mes = measure if measure is not None else (str(attrs['measure']) if isinstance(attrs, dict) and attrs.get('measure') else None)
        ni_a = None if ni is None else numpy.atleast_1d(numpy.asarray(ni, dtype=float))
        if ni_a is None and isinstance(attrs, dict) and attrs.get('ni') is not None:
            ni_a = numpy.asarray(attrs['ni'], dtype=float)
        if vi is None and sei is not None:
            vi_in: FloatArray | None = numpy.asarray(sei, dtype=float) ** 2
        else:
            vi_in = None if vi is None else numpy.asarray(vi, dtype=float)
        if yaxis in ('sei', 'vi', 'seinv', 'vinv', 'wi') and vi_in is None:
            raise metafopy.errors.InputError("Must specify the 'vi' or 'sei' argument.")
        if vi_in is not None and vi_in.shape[0] != k_in:
            raise metafopy.errors.InputError("Length of 'yi' and 'vi' (or 'sei') are not the same.")
        vi_a = numpy.zeros(k_in) if vi_in is None else numpy.where(vi_in < 0, 0.0, vi_in)
        sei_a = numpy.sqrt(vi_a)
        slab_all = [str(i + 1) for i in range(k_in)] if slab is None else [str(s) for s in slab]
        if len(slab_all) != k_in:
            raise metafopy.errors.InputError(f"Length of the 'slab' argument ({len(slab_all)}) does not correspond to the number of outcomes ({k_in}).")
        pch_vals, pch_vec = vec(19 if pch is None else pch, k_in, 'pch')
        col_vals, col_vec = vec(fg if col is None else col, k_in, 'col')
        bg_vals, bg_vec = vec(metafopy.plot.rutils.coladj(-0.1) if bg is None else bg, k_in, 'bg')
        colv, bgv = [], []
        ref = 0.0 if refline is None else refline
        fill = None
        weights = None
        if subset is not None:
            sa = numpy.asarray(subset)
            sel = numpy.flatnonzero(sa) if sa.dtype == numpy.bool_ else sa.astype(numpy.intp)
            yi, vi_a, sei_a = yi[sel], vi_a[sel], sei_a[sel]
            if ni_a is not None:
                ni_a = ni_a[sel]
            slab_all = [slab_all[i] for i in sel]
            pch_vals, col_vals, bg_vals = ([v[i] for i in sel] for v in (pch_vals, col_vals, bg_vals))
        has_na = numpy.isnan(yi)
        if yaxis in ('sei', 'seinv', 'vi', 'vinv'):
            has_na |= numpy.isnan(vi_a)
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        if yaxis in ('ni', 'ninv', 'sqrtni', 'sqrtninv', 'lni'):
            if ni_a is None:
                raise metafopy.errors.InputError('No sample size information was specified (ni).')
            has_na |= numpy.isnan(ni_a)
        if has_na.any():
            ok = ~has_na
            yi, vi_a, sei_a = yi[ok], vi_a[ok], sei_a[ok]
            if ni_a is not None:
                ni_a = ni_a[ok]
            slab_all = [s for s, o in zip(slab_all, ok, strict=True) if o]
            pch_vals, col_vals, bg_vals = ([v for v, o in zip(vs, ok, strict=True) if o] for vs in (pch_vals, col_vals, bg_vals))
        xlab_c = metafopy.plot.rutils.setlab(mes, None, atransf, 1) if xlab is None else xlab
        if yi.shape[0] < 2:
            raise metafopy.errors.InputError('Plotting terminated since k < 2.')
        if yaxis == 'wi':
            if numpy.any(vi_a <= 0):
                raise metafopy.errors.InputError('Cannot plot weights when there are non-positive sampling variances in the data.')
            w0 = 1 / vi_a
            weights = w0 / w0.sum() * 100

    shade_c: list[str]
    if shade is None:
        base = metafopy.plot.rutils.coladj(0.0)
        shade_c = [base]
    else:
        shade_c = [shade] if isinstance(shade, str) else list(shade)
    if len(lvl_alpha) > 1 and len(shade_c) == 1:
        shade_c = metafopy.plot.rutils.color_ramp(shade_c[0], metafopy.plot.rutils.coladj(-0.5), len(lvl_alpha))
        shade_c = [shade_c[0], *shade_c[:0:-1]] if len(shade_c) > 1 else shade_c

    dg = [2, 3] if digits is None else ([digits, digits] if isinstance(digits, int) else list(digits))
    if digits is None:
        dg = {'ni': [2, 0], 'wi': [2, 2]}.get(yaxis, [2, 3])
    drop0 = all(isinstance(v, int | numpy.integer) for v in dg)
    ylab_c = YLAB[yaxis] if ylab is None else ylab
    lt = [lty, lty] if numpy.ndim(lty) == 0 or isinstance(lty, str) else list(lty)

    sei_ok = sei_a
    # y limits
    if ylim is None:
        # auditme: accept[assert-never-missing] the value is a str that is validated against the allowed names earlier
        if yaxis == 'sei':
            yl = [float(numpy.max(sei_ok)), 0.0]
        # auditme: accept[assert-never-missing] the value is a str that is validated against the allowed names earlier
        elif yaxis == 'vi':
            yl = [float(numpy.max(vi_a)), 0.0]
        elif yaxis == 'seinv':
            yl = [float(numpy.min(1 / sei_ok)), float(numpy.max(1 / sei_ok))]
        elif yaxis == 'vinv':
            yl = [float(numpy.min(1 / vi_a)), float(numpy.max(1 / vi_a))]
        else:
            assert ni_a is not None or yaxis == 'wi'
            # auditme: accept[assert-never-missing] the value is a str that is validated against the allowed names earlier
            if yaxis == 'ni':
                yl = [float(numpy.nanmin(ni_a)), float(numpy.nanmax(ni_a))]  # type: ignore[arg-type]
            # auditme: accept[assert-never-missing] the value is a str that is validated against the allowed names earlier
            elif yaxis == 'ninv':
                yl = [float(numpy.nanmax(1 / ni_a)), float(numpy.nanmin(1 / ni_a))]  # type: ignore[operator]
            # auditme: accept[assert-never-missing] the value is a str that is validated against the allowed names earlier
            elif yaxis == 'sqrtni':
                yl = [float(numpy.nanmin(numpy.sqrt(ni_a))), float(numpy.nanmax(numpy.sqrt(ni_a)))]  # type: ignore[arg-type]
            elif yaxis == 'sqrtninv':
                yl = [float(numpy.nanmax(1 / numpy.sqrt(ni_a))), float(numpy.nanmin(1 / numpy.sqrt(ni_a)))]  # type: ignore[arg-type]
            elif yaxis == 'lni':
                yl = [float(numpy.nanmin(numpy.log(ni_a))), float(numpy.nanmax(numpy.log(ni_a)))]  # type: ignore[arg-type]
            else:
                assert weights is not None
                yl = [float(numpy.min(weights)), float(numpy.max(weights))]
        if any(math.isinf(v) for v in yl):
            raise metafopy.errors.InputError("Setting 'ylim' automatically not possible (must set y-axis limits manually).")
    else:
        lo_, hi_ = float(min(ylim)), float(max(ylim))
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        yl = [hi_, lo_] if yaxis in ('sei', 'vi', 'ninv', 'sqrtninv') else [lo_, hi_]
        if yaxis in ('sei', 'vi', 'ni', 'ninv', 'sqrtni', 'sqrtninv', 'lni', 'wi') and (yl[0] < 0 or yl[1] < 0):
            raise metafopy.errors.InputError('Both y-axis limits must be >= 0.')
        if yaxis in ('seinv', 'vinv') and (yl[0] <= 0 or yl[1] <= 0):
            raise metafopy.errors.InputError('Both y-axis limits must be > 0.')

    zcrit_min = float(scipy.stats.norm.isf(min(lvl_alpha) / 2))
    lvals = len(lvl_alpha)
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
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
            xl = [min(x_lb_bot, float(numpy.nanmin(yi))), max(x_ub_bot, float(numpy.nanmax(yi)))]
        else:
            xl = [float(numpy.min(yi)), float(numpy.max(yi))]
        rx = xl[1] - xl[0]
        xl = [xl[0] - rx * 0.1, xl[1] + rx * 0.1]
    else:
        xl = [float(min(xlim)), float(max(xlim))]
    at_in = None if at is None else numpy.asarray(at, dtype=float)
    if at_in is not None:
        xl[0] = float(min(xl[0], numpy.nanmin(at_in)))
        xl[1] = float(max(xl[1], numpy.nanmax(at_in)))

    # -- drawing -------------------------------------------------------------------------------------------------
    ux = metafopy.plot.rutils.usr_range(xl[0], xl[1])
    uy = metafopy.plot.rutils.usr_range(yl[0], yl[1])
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
    metafopy.plot.draw.rect(axx, ux[0], uy[0], ux[1], uy[1], col=back_c, border=None, zorder=0.4)
    yat = numpy.linspace(yl[0], yl[1], steps)
    metafopy.plot.draw.yaxis(axx, yat, [metafopy.plot.rutils.fmtx1(float(v), dg[1], drop0) for v in yat])
    for v in yat:
        metafopy.plot.draw.hline(axx, float(v), col=hlines_c)
        axx.lines[-1].set_zorder(0.5)

    if ci_axes:
        # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
        if yaxis in ('sei', 'vi'):
            rylim = yl[0] - yl[1]
            yl[0] = yl[0] + rylim * 0.1
            yl[1] = max(0.0, yl[1] - rylim * 0.1)
        else:
            rylim = yl[1] - yl[0]
            yl[1] = yl[1] + rylim * 0.1
        yvals = numpy.linspace(yl[0], yl[1], ci_res)
        with numpy.errstate(divide='ignore'):
            vvals = numpy.asarray({'sei': yvals**2, 'vi': yvals, 'seinv': 1 / yvals**2, 'vinv': 1 / yvals}[yaxis], dtype=numpy.float64)
        for m in range(lvals - 1, -1, -1):
            zc = float(scipy.stats.norm.isf(lvl_alpha[m] / 2))
            ci_left = ref - zc * numpy.sqrt(vvals + tau2)
            ci_right = ref + zc * numpy.sqrt(vvals + tau2)
            metafopy.plot.draw.poly(
                axx, numpy.concatenate([ci_left, ci_right[::-1]]), numpy.concatenate([yvals, yvals[::-1]]), col=shade_c[m], border=None, zorder=0.6
            )
            metafopy.plot.draw.line(axx, ci_left, yvals, lty=lt[0], col=colci_c).set_zorder(0.7)
            metafopy.plot.draw.line(axx, ci_right, yvals, lty=lt[0], col=colci_c).set_zorder(0.7)
        if refline2 is not None:
            zc2 = float(scipy.stats.norm.isf(lvl2 / 2))
            metafopy.plot.draw.line(axx, refline2 - zc2 * numpy.sqrt(vvals + tau2), yvals, lty=lty2, col=colci_c).set_zorder(0.7)
            metafopy.plot.draw.line(axx, refline2 + zc2 * numpy.sqrt(vvals + tau2), yvals, lty=lty2, col=colci_c).set_zorder(0.7)
        metafopy.plot.draw.seg(axx, ref, yl[0], ref, yl[1], lty=lt[1], col=colref_c).set_zorder(0.7)
        if refline2 is not None:
            metafopy.plot.draw.seg(axx, refline2, yl[0], refline2, yl[1], lty=lty2, col=colref_c).set_zorder(0.7)
    else:
        metafopy.plot.draw.vline(axx, ref, lty=lt[1], col=colref_c)
        if refline2 is not None:
            metafopy.plot.draw.vline(axx, refline2, lty=lty2, col=colref_c)

    xvals = yi
    with numpy.errstate(divide='ignore'):
        yvals_pts = yvalues(yaxis, sei_ok, vi_a, ni_a, weights)
    if tf and is_fit and int_only:
        assert fill is not None
        for xi_, yi_ in zip(xvals[~fill], yvals_pts[~fill], strict=True):
            metafopy.plot.draw.pts(axx, xi_, yi_, pch=int(pch_vals[0]), col=colv[0], bg=bgv[0])
        for xi_, yi_ in zip(xvals[fill], yvals_pts[fill], strict=True):
            metafopy.plot.draw.pts(axx, xi_, yi_, pch=int(pch_fill), col=colv[1], bg=bgv[1])
    else:
        for i in range(xvals.shape[0]):
            if not numpy.isnan(xvals[i]) and not numpy.isnan(yvals_pts[i]):
                metafopy.plot.draw.pts(axx, xvals[i], yvals_pts[i], pch=int(pch_vals[i]), col=col_vals[i], bg=bg_vals[i])

    if at_in is None:
        at_a = metafopy.plot.rutils.ax_ticks(ux[0], ux[1])
    else:
        at_a = at_in[(at_in > ux[0]) & (at_in < ux[1])]
    if at_lab is None:
        vals = metafopy.methods.common.apply_each(at_a, atransf, targs) if atransf is not None else at_a
        at_lab_l = [metafopy.plot.rutils.fmtx1(float(v), dg[0], drop0) for v in vals]
    else:
        at_lab_l = list(at_lab)
    metafopy.plot.draw.xaxis(axx, at_a, at_lab_l)
    axx.spines['bottom'].set_bounds(float(at_a.min()), float(at_a.max())) if at_a.size else None

    k = int(xvals.shape[0])
    sel_label: list[int] = []
    if label is not False and label is not None:
        ref_l = ref
        if math.isnan(ref_l):
            ref_l = float(numpy.nanmean(yi))
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
        elif isinstance(label, int | float | numpy.integer):
            nlab = min(max(round(float(label)), 0), k)
            sel_label = [int(i) for i in numpy.argsort(-numpy.abs(yi - ref_l), kind='stable')[:nlab]]
        for i in sel_label:
            metafopy.plot.draw.txt(axx, float(yi[i]), float(yvals_pts[i]), slab_all[i], pos=4 if yi[i] - ref_l >= 0 else 2, offset=offset)

    funnel_legend(
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
    return FunnelResult(axx, numpy.asarray(xvals, dtype=float), numpy.asarray(yvals_pts, dtype=float), tuple(slab_all), fill)


def funnel_legend(
    ax: matplotlib.axes.Axes,
    legend: bool | str | dict[str, typing.Any],
    level: list[float],
    shade: list[str],
    back: str,
    yaxis: str,
    trimfill: bool,
    pch: list[typing.Any],
    col: list[typing.Any],
    bg: list[typing.Any],
    pch_fill: int,
    pch_vec: bool,
    col_vec: bool,
    bg_vec: bool,
    colci: str,
) -> None:
    """An approximation of metafor's funnel legend (CI-region p-value/CI entries and the study symbols)."""
    opts: dict[str, typing.Any] = {'x': 'upper right', 'show': 'pvals', 'studies': True}
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
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    if yaxis not in ('sei', 'vi', 'seinv', 'vinv'):
        opts['show'] = None
    if pch_vec or col_vec or bg_vec:
        opts['studies'] = False
    if not opts['studies'] and opts['show'] is None:
        return
    handles: list[typing.Any] = []
    labels: list[str] = []
    if opts['show'] == 'pvals':
        lv = [*level, 0.0]
        nd = max(len(f'{v:.8g}'.rstrip('0')) for v in lv) - 2
        nd = max(0, nd)
        for i, v in enumerate(lv):
            hi = 1.0 if i == 0 else lv[i - 1]
            labels.append(f'{v:.{nd}f} < p <= {hi:.{nd}f}')
            handles.append(matplotlib.patches.Patch(facecolor=metafopy.plot.draw.rcolor([*shade, back][i]), edgecolor=metafopy.plot.draw.rcolor(colci)))
    elif opts['show'] == 'cis':
        for v, sc in zip(level, shade, strict=False):
            labels.append(f'{100 - 100 * v:g}% CI Region')
            handles.append(matplotlib.patches.Patch(facecolor=metafopy.plot.draw.rcolor(sc), edgecolor=metafopy.plot.draw.rcolor(colci)))
    if opts['studies']:
        labels.append('Observed Studies' if trimfill else 'Studies')
        handles.append(
            matplotlib.lines.Line2D(
                [],
                [],
                linestyle='None',
                marker='o',
                markerfacecolor=metafopy.plot.draw.rcolor(str(bg[0])),
                markeredgecolor=metafopy.plot.draw.rcolor(str(col[0])),
            )
        )
        if trimfill:
            labels.append('Imputed Studies')
            handles.append(
                matplotlib.lines.Line2D(
                    [],
                    [],
                    linestyle='None',
                    marker='o',
                    markerfacecolor=metafopy.plot.draw.rcolor(str(bg[1])),
                    markeredgecolor=metafopy.plot.draw.rcolor(str(col[1])),
                )
            )
    ax.legend(handles, labels, loc=opts['x'], framealpha=1.0)
