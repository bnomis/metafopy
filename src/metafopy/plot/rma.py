"""The four diagnostic plots of a fitted model (metafor's ``plot.rma.uni``, ``plot.rma.mh`` and ``plot.rma.peto``).

A forest plot, a funnel plot (of the residuals for a meta-regression), a radial plot (or, for a meta-regression,
the standardized residuals against the fitted values) and the standardized residuals of the studies (or a normal
Q-Q plot with ``qqplot=True``), see https://github.com/wviechtb/metafor/blob/master/R/plot.rma.uni.r and
https://wviechtb.github.io/metafor/reference/plot.rma.html.
"""

import dataclasses
import typing

import matplotlib.axes
import numpy
import numpy.typing
import scipy.stats

import metafopy.errors
import metafopy.methods.common
import metafopy.methods.fitted
import metafopy.methods.fitview
import metafopy.methods.influence
import metafopy.models.binary
import metafopy.models.rma_uni
import metafopy.models.sel_result
import metafopy.plot
import metafopy.plot.draw
import metafopy.plot.infl
import metafopy.plot.plain
import metafopy.plot.rutils
import metafopy.plot.selmodel

__all__ = ['PlotRmaResult', 'plot_rma']

type FloatArray = numpy.typing.NDArray[numpy.float64]


@dataclasses.dataclass(frozen=True, slots=True)
class PlotRmaResult:
    """The panels drawn by :func:`plot_rma` and what each of them returns.

    ``axes`` are the four panels (forest, funnel, radial or fitted vs. residuals, residuals or Q-Q plot). ``radial`` is
    ``None`` for a meta-regression, ``qqnorm`` unless ``qqplot=True``. ``z`` and ``ids`` are the standardized residuals
    and the 1-based study numbers of the last panel (empty with ``qqplot=True``), ``fitted`` the fitted values plotted
    against ``z`` in the third panel of a meta-regression (empty otherwise).
    """

    axes: tuple[matplotlib.axes.Axes, ...]
    forest: metafopy.plot.ForestResult
    funnel: metafopy.plot.FunnelResult
    radial: metafopy.plot.RadialResult | None
    qqnorm: metafopy.plot.QQNormResult | None
    z: FloatArray
    ids: tuple[int, ...]
    fitted: FloatArray


def title(ax: matplotlib.axes.Axes, main: str) -> None:
    """R's ``title(main)``."""
    ax.set_title(main, fontsize=metafopy.plot.draw.base_fontsize() * 1.2, fontweight='bold')


def residual_bounds(z: FloatArray) -> tuple[float, float]:
    """``c(min(z, -2, na.rm=TRUE), max(z, 2, na.rm=TRUE))``."""
    return (min(float(numpy.nanmin(z)), -2.0), max(float(numpy.nanmax(z)), 2.0))


def reference_lines(ax: matplotlib.axes.Axes) -> None:
    metafopy.plot.draw.hline(ax, 0.0, lty='dashed')
    for q in (0.025, 0.975):
        metafopy.plot.draw.hline(ax, float(scipy.stats.norm.ppf(q)), lty='dotted')


def residual_panel(ax: matplotlib.axes.Axes, z: FloatArray, ids: tuple[int, ...]) -> None:
    """The standardized residuals in study order, joined by a line (the last panel of ``plot.rma.uni``)."""
    k = int(z.shape[0])
    pos = numpy.arange(1, k + 1, dtype=float)
    metafopy.plot.plain.plain_plot(ax, (1.0, float(k)), residual_bounds(z), 'Study', '', bty='l', xat=pos, xlabels=[str(i) for i in ids], thin_xlabels=True)
    not_na = ~numpy.isnan(z)
    metafopy.plot.draw.line(ax, pos[not_na], z[not_na], col=metafopy.plot.rutils.coladj(-0.2))
    metafopy.plot.draw.line(ax, pos, z)
    for xp, yp in zip(pos, z, strict=True):
        if not numpy.isnan(yp):
            metafopy.plot.draw.pts(ax, float(xp), float(yp), pch=21, bg=metafopy.plot.rutils.coladj(-0.35))
    reference_lines(ax)
    title(ax, 'Standardized Residuals')


def fitted_panel(ax: matplotlib.axes.Axes, pred: FloatArray, z: FloatArray) -> None:
    """The standardized residuals against the fitted values (the third panel of ``plot.rma.uni`` for a meta-regression)."""
    metafopy.plot.plain.plain_plot(
        ax, (float(numpy.nanmin(pred)), float(numpy.nanmax(pred))), residual_bounds(z), 'Fitted Value', 'Standardized Residual', bty='l'
    )
    reference_lines(ax)
    for xp, yp in zip(pred, z, strict=True):
        if not (numpy.isnan(xp) or numpy.isnan(yp)):
            metafopy.plot.draw.pts(ax, float(xp), float(yp), pch=21, bg=metafopy.plot.rutils.coladj(-0.35))
    title(ax, 'Fitted vs. Standardized Residuals')


def plot_rma(
    x: metafopy.methods.fitview.Fit,
    qqplot: bool = False,
    *,
    seed: int | None = None,
    ax: matplotlib.axes.Axes | None = None,
    na_action: metafopy.methods.common.NAAction = 'omit',
    **kwargs: typing.Any,
) -> PlotRmaResult | metafopy.plot.selmodel.SelPlotResult:
    """Draw the forest, funnel, radial and residual plots of a model (metafor's ``plot()`` of an ``rma.uni``, ``rma.mh`` or ``rma.peto`` fit).

    For a model without moderators the panels are :func:`~metafopy.forest`, :func:`~metafopy.funnel`,
    :func:`~metafopy.radial` and the standardized residuals (:func:`~metafopy.rstandard`) of the studies in order;
    for a meta-regression the funnel plot shows the residuals and the radial plot is replaced by the standardized
    residuals against the fitted values. ``qqplot=True`` draws :func:`~metafopy.qqnorm` (with ``seed`` for its envelope)
    instead of the last panel. The panels fill ``ax`` (default: the current axes) in a 2 x 2 grid, as
    ``par(mfrow = c(2, 2))`` does. A selection model (``rma.uni.selmodel``) is drawn by :func:`~metafopy.plot_selmodel`,
    which gets ``kwargs``, as R's ``plot()`` dispatches to ``plot.rma.uni.selmodel``. ``na_action`` stands for R's ``options(na.action=)``: with
    ``'omit'`` (default) the last panel and the third panel of a meta-regression show the studies used in fitting, with ``'exclude'`` and ``'pass'`` every
    study of the subset (a gap where a value is missing), and ``'fail'`` stops when studies were left out of the fit.

    Raises:
        metafopy.errors.ArgumentError: ``kwargs`` are given for a model that is not a selection model.
        metafopy.errors.InputError: ``na_action`` is unknown, or ``'fail'`` and studies were left out of the fit.
        metafopy.errors.NotSupportedError: the model is an ``rma_mv``, ``rma_glmm``, location-scale or robust fit (metafor has no such method).
    """
    if isinstance(x, metafopy.models.sel_result.SelmodelResult):
        return metafopy.plot.selmodel.plot_selmodel(x, ax=ax, **kwargs)
    if kwargs:
        raise metafopy.errors.ArgumentError(f'unexpected arguments: {", ".join(sorted(kwargs))}')
    metafopy.methods.fitview.check_class(x, notav=('rma.mv', 'rma.glmm', 'robust.rma', 'rma.ls', 'rma.gen'))
    metafopy.methods.common.check_na_action(na_action)
    view = metafopy.methods.fitview.fit_view(x)
    if na_action == 'fail' and view.na_rows is not None and view.na_rows.any():
        raise metafopy.errors.InputError('Missing values in results.')
    every_row = na_action in ('exclude', 'pass')
    area = metafopy.plot.rutils.get_ax(ax)
    cex = metafopy.plot.infl.mfrow_cex(2, 2)
    grid = metafopy.plot.draw.subaxes(area, 2, 2, hspace=metafopy.plot.infl.grid_hspace(area, 2, cex), wspace=0.3)
    axes = (grid[0][0], grid[0][1], grid[1][0], grid[1][1])
    model = typing.cast(metafopy.models.rma_uni.RMAResult | metafopy.models.binary.FixedBinaryResult, x)
    radial: metafopy.plot.RadialResult | None = None
    qq: metafopy.plot.QQNormResult | None = None
    z = numpy.zeros(0)
    ids: tuple[int, ...] = ()
    fitted = numpy.zeros(0)
    with matplotlib.rc_context({'font.size': metafopy.plot.draw.base_fontsize() * cex}):
        fr = metafopy.plot.forest(x, ax=axes[0])
        title(axes[0], 'Forest Plot')
        fu = metafopy.plot.funnel(x, ax=axes[1])
        title(axes[1], 'Funnel Plot' if view.int_only else 'Residual Funnel Plot')
        z_all = numpy.asarray(metafopy.methods.influence.rstandard(x, na_action='pass' if every_row else 'omit').z, dtype=numpy.float64)
        if view.int_only:
            radial = metafopy.plot.radial(x, ax=axes[2])
            title(axes[2], 'Radial Plot')
        else:
            fitted = metafopy.methods.fitted.fitted_values(x, 'pass') if every_row else view.fitted
            fitted_panel(axes[2], fitted, z_all)
        if qqplot:
            qq = metafopy.plot.qqnorm(model, seed=seed, ax=axes[3], na_action=na_action)
        else:
            if every_row:
                subset = view.keep if view.na_rows is None else view.keep | view.na_rows
                z = z_all
                ids = tuple(int(i) + 1 for i in numpy.flatnonzero(subset))
            else:
                ok = ~numpy.isnan(z_all)
                z = z_all[ok]
                ids = tuple(int(i) + 1 for i, o in zip(view.ids, ok, strict=True) if o)
            residual_panel(axes[3], z, ids)
    return PlotRmaResult(axes=axes, forest=fr, funnel=fu, radial=radial, qqnorm=qq, z=z, ids=ids, fitted=fitted)
