"""Plot of a profile likelihood or of the list of profiles of a model with several components (metafor's ``plot.profile.rma``).

See https://wviechtb.github.io/metafor/reference/profile.rma.html.
"""

import collections.abc
import math
import typing

import matplotlib.axes
import numpy
import numpy.typing
import scipy.stats

import metafopy.errors
import metafopy.methods.ls_profile
import metafopy.methods.mv_profile
import metafopy.methods.profile
import metafopy.methods.selmodel
import metafopy.plot
import metafopy.plot.draw
import metafopy.plot.histplot
import metafopy.plot.plain
import metafopy.plot.rutils

__all__ = ['plot_profile']

# the symbols of the rma.mv components and whether they are squared
MV_SYMBOLS = {'gamma2': (r'\gamma', True), 'phi': (r'\phi', False), 'rho': (r'\rho', False), 'sigma2': (r'\sigma', True), 'tau2': (r'\tau', True)}


type Profile = metafopy.methods.profile.ProfileResult | metafopy.methods.ls_profile.LSProfileResult


def grid_of(x: Profile) -> numpy.typing.NDArray[numpy.float64]:
    return numpy.asarray(x.alpha if isinstance(x, metafopy.methods.ls_profile.LSProfileResult) else x.tau2, dtype=float)


def symbol_of(x: Profile) -> str | None:
    """The mathtext symbol of the profiled component (``None`` for ``tau^2`` or ``tau`` of an ``rma.uni`` fit)."""
    if isinstance(x, metafopy.methods.ls_profile.LSProfileResult):
        return r'\alpha' if x.alphas == 1 else rf'\alpha_{{{x.position + 1 - int(x.sub1)}}}'
    if isinstance(x, metafopy.methods.selmodel.SelProfileResult):
        if x.comp == 'delta':
            return r'\delta' if x.deltas == 1 else rf'\delta_{{{x.delta_pos + 1}}}'
        return r'\tau^2'
    if isinstance(x, metafopy.methods.mv_profile.MVProfileResult):
        base, square = MV_SYMBOLS[x.comp]
        return (base if x.comp_n == 1 else f'{base}_{{{x.comp_pos + 1}}}') + ('^2' if square else '')
    return None


def draw_one(
    x: Profile,
    xlim: collections.abc.Sequence[float] | None,
    ylim: collections.abc.Sequence[float] | None,
    pch: int,
    xlab: str | None,
    ylab: str | None,
    main: str | None,
    refline: bool,
    cline: bool | float,
    ax: matplotlib.axes.Axes,
    log: str | None = None,
    cex: float = 1.0,
    lwd: float = 1.0,
) -> matplotlib.axes.Axes:
    xl = x.xlim if xlim is None else (float(xlim[0]), float(xlim[1]))
    yl = x.ylim if ylim is None else (float(ylim[0]), float(ylim[1]))
    sym = symbol_of(x)
    sqrt = not isinstance(x, metafopy.methods.ls_profile.LSProfileResult) and x.sqrt
    if xlab is None:
        xlab = rf'${sym}$ Value' if sym is not None else (r'$\tau$ Value' if sqrt else r'$\tau^2$ Value')
    if ylab is None:
        ylab = ('Restricted ' if x.method == 'REML' else '') + ('Likelihood' if x.exp else 'Log-Likelihood')
    if main is None:
        main = rf'Profile Plot for ${sym}$' if sym is not None else (r'Profile Plot for $\tau$' if sqrt else r'Profile Plot for $\tau^2$')
    xv, yv = grid_of(x), numpy.asarray(x.ll, dtype=float)
    if xv.min() <= x.vc <= xv.max():
        pos = int(numpy.flatnonzero(xv >= x.vc)[0])
        xv = numpy.concatenate([xv[:pos], [x.vc], xv[pos:]])
        yv = numpy.concatenate([yv[:pos], [x.maxll], yv[pos:]])
    metafopy.plot.plain.plain_plot(ax, xl, yl, xlab, ylab, main, bty='l', logx='x' in (log or ''), logy='y' in (log or ''))
    if refline:
        metafopy.plot.draw.vline(ax, x.vc, lty='dotted')
        metafopy.plot.draw.hline(ax, x.maxll, lty='dotted')
    if cline is not False and cline is not None:
        c = 0.05 if cline is True else metafopy.plot.rutils.level_to_alpha(float(cline))
        crit = float(scipy.stats.chi2.ppf(1 - c, 1)) / 2
        hval = math.exp(math.log(x.maxll) - crit) if x.exp else x.maxll - crit
        metafopy.plot.draw.hline(ax, hval, lty='dotted')
    ok = numpy.isfinite(xv) & numpy.isfinite(yv)
    metafopy.plot.draw.line(ax, xv[ok], yv[ok], gid='oline', lw=lwd)
    for a, b in zip(xv, yv, strict=True):
        if math.isfinite(a) and math.isfinite(b):
            metafopy.plot.draw.pts(ax, float(a), float(b), pch=pch, cex=cex, col=metafopy.plot.draw.FG)
    return ax


def per_panel(value: str | collections.abc.Sequence[str] | None, n: int, what: str) -> list[str | None]:
    """metafor's ``.expand1()``: one value for every panel, or one per panel.

    Raises:
        metafopy.errors.InputError: ``value`` has neither one entry nor one per panel.
    """
    if value is None or isinstance(value, str):
        return [value] * n
    if len(value) == 1:
        return [value[0]] * n
    if len(value) != n:
        raise metafopy.errors.InputError(f"Length of the '{what}' argument ({len(value)}) does not match the number of profiles ({n}).")
    return list(value)


@typing.overload
def plot_profile(
    x: Profile,
    xlim: collections.abc.Sequence[float] | None = None,
    ylim: collections.abc.Sequence[float] | None = None,
    pch: int = 19,
    xlab: str | None = None,
    ylab: str | None = None,
    main: str | None = None,
    refline: bool = True,
    cline: bool | float = False,
    *,
    ax: matplotlib.axes.Axes | None = None,
    log: str | None = None,
    cex: float = 1.0,
    lwd: float = 1.0,
) -> matplotlib.axes.Axes: ...


@typing.overload
def plot_profile(
    x: collections.abc.Sequence[Profile],
    xlim: collections.abc.Sequence[float] | None = None,
    ylim: collections.abc.Sequence[float] | None = None,
    pch: int = 19,
    xlab: str | collections.abc.Sequence[str] | None = None,
    ylab: str | collections.abc.Sequence[str] | None = None,
    main: str | collections.abc.Sequence[str] | None = None,
    refline: bool = True,
    cline: bool | float = False,
    *,
    ax: matplotlib.axes.Axes | None = None,
    log: str | None = None,
    cex: float = 1.0,
    lwd: float = 1.0,
) -> list[matplotlib.axes.Axes]: ...


def plot_profile(
    x: Profile | collections.abc.Sequence[Profile],
    xlim: collections.abc.Sequence[float] | None = None,
    ylim: collections.abc.Sequence[float] | None = None,
    pch: int = 19,
    xlab: str | collections.abc.Sequence[str] | None = None,
    ylab: str | collections.abc.Sequence[str] | None = None,
    main: str | collections.abc.Sequence[str] | None = None,
    refline: bool = True,
    cline: bool | float = False,
    *,
    ax: matplotlib.axes.Axes | None = None,
    log: str | None = None,
    cex: float = 1.0,
    lwd: float = 1.0,
) -> matplotlib.axes.Axes | list[matplotlib.axes.Axes]:
    """Plot the profile (log-)likelihood computed by [profile][metafopy.profile] (``plot.profile.rma``).

    Mirrors metafor's ``plot.profile.rma()``, which is documented with ``profile.rma``.

    ``refline`` marks the estimate and its (log-)likelihood with dotted lines, ``cline`` (``True`` for a significance
    level of 0.05, or a number as for metafor's ``level`` arguments) adds a horizontal line at the likelihood-ratio test's critical value. The
    estimate itself is added to the profile if it lies within the range of the grid. Returns the axes.

    ``x`` may also be the list of profiles that [profile][metafopy.profile] returns for models with several variance
    components (``rma_mv``, location-scale and selection models): one panel per component is drawn in a grid, as
    ``par(mfrow = n2mfrow(comps))`` makes it (inside ``ax`` if given), and the list of axes is returned. ``xlab``,
    ``ylab`` and ``main`` are then one string for all panels or one per panel. Unlike metafor, ``refline`` also applies to
    the panels of a list. ``log`` (``'x'``, ``'y'`` or ``'xy'``) draws logarithmic axes, ``cex`` is the size of the points and ``lwd`` the
    width of the line, which metafor passes on through ``...``.

    Args:
        x: A profile from [profile][metafopy.profile], or the list of profiles it returns for models with several variance components (``rma_mv``,
            location-scale and selection models; one panel per component).
        xlim: Limits of the horizontal axis as (low, high). Default: those stored in the profile (``x.xlim``).
        ylim: Limits of the vertical axis as (low, high). Default: those stored in the profile (``x.ylim``).
        pch: Plotting symbol of the points (R's ``pch`` code). Default: 19.
        xlab: Label of the horizontal axis; for a list of profiles one string for all panels or one per panel. Default: derived from the profiled
            component (for example ``tau^2 Value``).
        ylab: Label of the vertical axis; for a list of profiles one string for all panels or one per panel. Default: ``'Log-Likelihood'`` (or
            ``'Likelihood'`` if the profile was computed with ``exp``), preceded by ``'Restricted '`` for REML.
        main: Title; for a list of profiles one string for all panels or one per panel. Default: ``'Profile Plot for'`` and the component.
        refline: Mark the estimate and its (log-)likelihood with dotted lines (unlike metafor, also applies to the panels of a list). Default: ``True``.
        cline: Add a horizontal dotted line at the critical value of the likelihood-ratio test: ``True`` for a significance level of 0.05, or a number
            read as metafor's ``level`` is (95 or 0.95 for 95 percent, 0.05 for a significance level of 5 percent). Default: ``False``.
        ax: Matplotlib axes to draw into (for a list of profiles, the area the panels fill). Default: the current axes.
        log: Logarithmic axes: ``'x'``, ``'y'`` or ``'xy'``. Default: linear axes.
        cex: Size of the points relative to the default. Default: 1.
        lwd: Width of the line. Default: 1.

    Returns:
        The axes drawn into (a list of axes, one per component, for a list of profiles).

    Raises:
        metafopy.errors.InputError: ``xlab``, ``ylab`` or ``main`` does not have one entry per profile (or is not a single string for a single
            profile), a limit of a logarithmic axis is not positive, or ``cline`` is outside 0 to 100.

    Metafor reference:
        [profile.rma](https://wviechtb.github.io/metafor/reference/profile.rma.html)
    """
    if isinstance(x, collections.abc.Sequence):
        n = len(x)
        xlabs, ylabs, mains = per_panel(xlab, n, 'xlab'), per_panel(ylab, n, 'ylab'), per_panel(main, n, 'main')
        panels, context = metafopy.plot.histplot.panels(metafopy.plot.rutils.get_ax(ax), n)
        with context:
            for j, (r, a) in enumerate(zip(x, panels, strict=True)):
                draw_one(r, xlim, ylim, pch, xlabs[j], ylabs[j], mains[j], refline, cline, a, log, cex, lwd)
        return panels
    if not (xlab is None or isinstance(xlab, str)) or not (ylab is None or isinstance(ylab, str)) or not (main is None or isinstance(main, str)):
        raise metafopy.errors.InputError("Arguments 'xlab', 'ylab' and 'main' must be single strings for a single profile.")
    return draw_one(x, xlim, ylim, pch, xlab, ylab, main, refline, cline, metafopy.plot.rutils.get_ax(ax), log, cex, lwd)
