"""Plot of the estimated selection function (metafor's ``plot.rma.uni.selmodel``).

See https://wviechtb.github.io/metafor/reference/plot.rma.uni.selmodel.html and
https://wviechtb.github.io/metafor/reference/selmodel.html.
"""

import collections.abc
import dataclasses
import math
import typing

import matplotlib.axes
import numpy
import numpy.typing
import pandas

import metafopy.errors
import metafopy.models.sel_result
import metafopy.models.sel_spec
import metafopy.plot.draw
import metafopy.plot.plain
import metafopy.plot.rutils

__all__ = ['SelPlotResult', 'plot_selmodel']

type FloatArray = numpy.typing.NDArray[numpy.float64]

BOUNDED = ('halfnorm', 'halfnorm2', 'logistic', 'logistic2', 'negexp', 'negexp2', 'negexppow', 'power', 'power2')
SOLID = 'solid'
DOTTED = 'dotted'
# auditme: accept[non-alphabetical-literal] solid line first, as in metafor
LINE_STYLES = (SOLID, DOTTED)


@dataclasses.dataclass(frozen=True, slots=True)
class SelPlotResult:
    """The curve drawn by :func:`plot_selmodel`: p-values ``xs``, the selection function ``ys`` and the bounds ``ys_lb``/``ys_ub`` (NaN without a CI)."""

    xs: FloatArray
    ys: FloatArray
    ys_lb: FloatArray
    ys_ub: FloatArray
    ax: matplotlib.axes.Axes | None = None

    def to_frame(self) -> pandas.DataFrame:
        return pandas.DataFrame({'xs': self.xs, 'ys': self.ys, 'ys.lb': self.ys_lb, 'ys.ub': self.ys_ub})


def plot_selmodel(
    x: metafopy.models.sel_result.SelmodelResult,
    xlim: collections.abc.Sequence[float] | None = None,
    ylim: collections.abc.Sequence[float] | None = None,
    n: int = 1000,
    prec: str | float = 'max',
    scale: bool = False,
    ci: bool | str = False,
    reps: int = 1000,
    shade: bool | str = True,
    rug: bool = True,
    add: bool = False,
    lty: collections.abc.Sequence[str] = LINE_STYLES,
    lwd: collections.abc.Sequence[float] = (2.0, 1.0),
    seed: int | None = None,
    draw: bool = True,
    *,
    ax: matplotlib.axes.Axes | None = None,
    bty: str = 'o',
    col: str | None = None,
) -> SelPlotResult:
    """Draw the selection function of a selection model against the p-value.

    ``prec`` fixes the precision at which the function is drawn (``'min'``, ``'max'``, ``'mean'``, ``'median'`` or a number).
    ``ci`` adds a confidence region: ``'wald'`` transforms the Wald bounds of the selection parameters, ``'boot'`` (or ``True``)
    simulates ``reps`` parameter vectors from their estimated multivariate normal distribution (with NumPy's generator seeded by
    ``seed``, so the bounds agree with metafor only up to simulation error). ``scale`` rescales the curve to the unit interval,
    ``shade`` fills the region, ``rug`` marks the observed p-values and ``add`` draws into existing axes. ``draw=False`` only
    computes the curve. ``bty`` (the type of box around the plot) and ``col`` (the colour of the lines) are passed on to ``plot()`` and ``lines()`` by
    metafor. The curve is returned as :class:`SelPlotResult`.

    Raises:
        metafopy.errors.InputError: an argument is invalid or the model has no selection function.
    """
    if x.type in metafopy.models.sel_spec.TRUNCATED or x.spec.weight is None:
        raise metafopy.errors.InputError('Cannot draw the selection function for this type of selection model.')
    citype = 'boot'
    do_ci = bool(ci)
    if isinstance(ci, str):
        citype = ci.lower()
        do_ci = True
    if citype not in ('boot', 'wald'):
        raise metafopy.errors.InputError('Unknown confidence interval type specified.')
    if xlim is not None and len(xlim) != 2:
        raise metafopy.errors.InputError("Argument 'xlim' should be a vector of length 2.")
    xl = (x.pval_min, 1 - x.pval_min) if xlim is None else (float(min(xlim)), float(max(xlim)))
    if xl[0] < 0 or xl[1] > 1:
        raise metafopy.errors.InputError("Values for 'xlim' should be between 0 and 1.")
    if isinstance(prec, str):
        if prec not in ('min', 'max', 'mean', 'median'):
            raise metafopy.errors.InputError("Unknown options specified for the 'prec' argument.")
        preci = x.precis[prec]
    else:
        preci = float(prec) if x.precspec else 1.0

    delta = x.delta
    weight = x.spec.weight
    ps = numpy.linspace(xl[0], xl[1], n)
    stepwise = x.type in metafopy.models.sel_spec.STEPWISE
    if stepwise:
        ps = numpy.unique(numpy.concatenate([ps, x.steps]))
        ps = ps[(ps >= xl[0]) & (ps <= xl[1])]

    def curve(d: FloatArray) -> FloatArray:
        return numpy.asarray(weight(ps, d, x.yi, x.vi, preci, x.alternative, x.steps), dtype=numpy.float64)

    ys = curve(delta)
    if do_ci and citype == 'boot' and bool(numpy.all(numpy.isnan(x.vd))):
        do_ci = False
    if do_ci and citype == 'wald' and bool(numpy.all(numpy.isnan(x.ci_lb_delta))) and bool(numpy.all(numpy.isnan(x.ci_ub_delta))):
        do_ci = False
    if do_ci and citype == 'wald' and not stepwise and sum(not f for f in x.delta_fix) >= 2:
        raise metafopy.errors.InputError('Cannot compute Wald-type confidence intervals for this selection model.')

    ys_lb = numpy.full_like(ys, numpy.nan)
    ys_ub = numpy.full_like(ys, numpy.nan)
    if do_ci:
        if citype == 'boot':
            vd = numpy.array(x.vd, dtype=numpy.float64)
            na = numpy.isnan(numpy.diag(vd))
            vd[na, :] = 0
            vd[:, na] = 0
            dsim = numpy.random.default_rng(seed).multivariate_normal(delta, vd, size=reps)
            dsim = numpy.clip(dsim, x.delta_min, x.delta_max)
            sims = numpy.array([curve(d) for d in dsim])
            lo, hi = numpy.quantile(sims, [x.level / 2, 1 - x.level / 2], axis=0)
            ys_lb, ys_ub = numpy.asarray(lo), numpy.asarray(hi)
        else:
            lb_d = numpy.array(x.ci_lb_delta, dtype=numpy.float64)
            ub_d = numpy.array(x.ci_ub_delta, dtype=numpy.float64)
            if stepwise:
                fixed = numpy.asarray(x.delta_fix)
                lb_d[fixed] = delta[fixed]
                ub_d[fixed] = delta[fixed]
            ys_lb, ys_ub = curve(lb_d), curve(ub_d)

    if scale:
        ys = numpy.where(numpy.isinf(ys), numpy.nan, ys)
        lo_y, hi_y = float(numpy.nanmin(ys)), float(numpy.nanmax(ys))
        rng = hi_y - lo_y
        if rng > math.sqrt(numpy.finfo(float).eps):
            ys = (ys - lo_y) / rng
            ys_lb = (ys_lb - lo_y) / rng
            ys_ub = (ys_ub - lo_y) / rng
    ys = numpy.where(ys < 0, 0.0, ys)
    ys_lb = numpy.where(ys_lb < 0, 0.0, ys_lb)
    ys_ub = numpy.where(ys_ub < 0, 0.0, ys_ub)

    if ylim is None:
        if x.type in BOUNDED:
            yl = (0.0, 1.0)
        else:
            fin = ys[numpy.isfinite(ys)]
            if do_ci:
                yl = (
                    float(min(ys_lb[numpy.isfinite(ys_lb)].min(initial=math.inf), fin.min(initial=math.inf))),
                    float(max(ys_ub[numpy.isfinite(ys_ub)].max(initial=-math.inf), fin.max(initial=-math.inf))),
                )
            else:
                yl = (float(fin.min()), float(fin.max()))
    else:
        if len(ylim) != 2:
            raise metafopy.errors.InputError("Argument 'ylim' should be a vector of length 2.")
        yl = (float(min(ylim)), float(max(ylim)))

    if not draw:
        return SelPlotResult(xs=ps, ys=ys, ys_lb=ys_lb, ys_ub=ys_ub)

    axx = metafopy.plot.rutils.get_ax(ax)
    if not add:
        metafopy.plot.plain.plain_plot(axx, xl, yl, 'p-value', 'Relative Likelihood of Selection', '', bty=bty)
    line_col = metafopy.plot.draw.FG if col is None else col
    style: typing.Literal['steps-post', 'default'] = 'steps-post' if stepwise else 'default'
    if do_ci:
        if shade:
            shade_col = shade if isinstance(shade, str) else metafopy.plot.rutils.coladj(-0.1)
            xs_poly = numpy.concatenate([ps, ps[::-1]])
            ys_poly = numpy.concatenate([ys_lb, ys_ub[::-1]])
            if stepwise:
                xs_poly = numpy.repeat(ps, 2)[1:]
                lb_s = numpy.repeat(ys_lb, 2)[:-1]
                ub_s = numpy.repeat(ys_ub, 2)[:-1]
                xs_poly = numpy.concatenate([xs_poly, xs_poly[::-1]])
                ys_poly = numpy.concatenate([lb_s, ub_s[::-1]])
            ok = numpy.isfinite(ys_poly)
            metafopy.plot.draw.poly(axx, xs_poly[ok], ys_poly[ok], col=shade_col)
        for bound in (ys_lb, ys_ub):
            ln = metafopy.plot.draw.line(axx, ps, bound, lty=lty[1], lw=float(lwd[1]), col=line_col)
            ln.set_drawstyle(style)
    if rug and not add:
        y0, y1 = axx.get_ylim()
        for pv in x.pvals:
            metafopy.plot.draw.seg(axx, float(pv), y0, float(pv), y0 + 0.03 * (y1 - y0))
    ln = metafopy.plot.draw.line(axx, ps, ys, lty=lty[0], lw=float(lwd[0]), col=line_col)
    ln.set_drawstyle(style)
    return SelPlotResult(xs=ps, ys=ys, ys_lb=ys_lb, ys_ub=ys_ub, ax=axx)
