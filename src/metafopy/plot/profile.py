"""Plot of a profile likelihood (metafor's ``plot.profile.rma``).

See https://wviechtb.github.io/metafor/reference/profile.rma.html.
"""

import collections.abc
import math

import matplotlib.axes
import numpy
import scipy.stats

import metafopy.methods.mv_profile
import metafopy.methods.profile
import metafopy.methods.selmodel
import metafopy.plot
import metafopy.plot.draw
import metafopy.plot.plain
import metafopy.plot.rutils

__all__ = ['plot_profile']

# the symbols of the rma.mv components and whether they are squared
MV_SYMBOLS = {'gamma2': (r'\gamma', True), 'phi': (r'\phi', False), 'rho': (r'\rho', False), 'sigma2': (r'\sigma', True), 'tau2': (r'\tau', True)}


def plot_profile(
    x: metafopy.methods.profile.ProfileResult,
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
) -> matplotlib.axes.Axes:
    """Plot the profile (log-)likelihood computed by :func:`~metafopy.profile` (``plot.profile.rma``).

    ``refline`` marks the estimate and its (log-)likelihood with dotted lines, ``cline`` (``True`` for 0.05, or the
    significance level) adds a horizontal line at the likelihood-ratio test's critical value. The estimate itself is
    added to the profile if it lies within the range of the grid. Returns the axes.
    """
    axx = metafopy.plot.rutils.get_ax(ax)
    xl = x.xlim if xlim is None else (float(xlim[0]), float(xlim[1]))
    yl = x.ylim if ylim is None else (float(ylim[0]), float(ylim[1]))
    sel = x if isinstance(x, metafopy.methods.selmodel.SelProfileResult | metafopy.methods.mv_profile.MVProfileResult) else None
    sym = r'\tau^2'
    if isinstance(x, metafopy.methods.selmodel.SelProfileResult) and x.comp == 'delta':
        sym = r'\delta' if x.deltas == 1 else rf'\delta_{{{x.delta_pos + 1}}}'
    if isinstance(x, metafopy.methods.mv_profile.MVProfileResult):
        base, square = MV_SYMBOLS[x.comp]
        sym = (base if x.comp_n == 1 else f'{base}_{{{x.comp_pos + 1}}}') + ('^2' if square else '')
    if xlab is None:
        xlab = rf'${sym}$ Value' if sel is not None else (r'$\tau$ Value' if x.sqrt else r'$\tau^2$ Value')
    if ylab is None:
        ylab = ('Restricted ' if x.method == 'REML' else '') + ('Likelihood' if x.exp else 'Log-Likelihood')
    if main is None:
        main = rf'Profile Plot for ${sym}$' if sel is not None else (r'Profile Plot for $\tau$' if x.sqrt else r'Profile Plot for $\tau^2$')
    xv, yv = numpy.asarray(x.tau2, dtype=float), numpy.asarray(x.ll, dtype=float)
    if xv.min() <= x.vc <= xv.max():
        pos = int(numpy.flatnonzero(xv >= x.vc)[0])
        xv = numpy.concatenate([xv[:pos], [x.vc], xv[pos:]])
        yv = numpy.concatenate([yv[:pos], [x.maxll], yv[pos:]])
    metafopy.plot.plain.plain_plot(axx, xl, yl, xlab, ylab, main, bty='l')
    if refline:
        metafopy.plot.draw.vline(axx, x.vc, lty='dotted')
        metafopy.plot.draw.hline(axx, x.maxll, lty='dotted')
    if cline is not False and cline is not None:
        c = 0.05 if cline is True else metafopy.plot.rutils.level_to_alpha(float(cline))
        crit = float(scipy.stats.chi2.ppf(1 - c, 1)) / 2
        hval = math.exp(math.log(x.maxll) - crit) if x.exp else x.maxll - crit
        metafopy.plot.draw.hline(axx, hval, lty='dotted')
    ok = numpy.isfinite(xv) & numpy.isfinite(yv)
    metafopy.plot.draw.line(axx, xv[ok], yv[ok], gid='oline')
    for a, b in zip(xv, yv, strict=True):
        if math.isfinite(a) and math.isfinite(b):
            metafopy.plot.draw.pts(axx, float(a), float(b), pch=pch, col=metafopy.plot.draw.FG)
    return axx
