"""Plot of the case-deletion diagnostics (metafor's ``plot.infl.rma.uni``).

See Viechtbauer & Cheung (2010), doi:10.1002/jrsm.11, and
https://wviechtb.github.io/metafor/reference/plot.infl.rma.uni.html.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray
from scipy import stats

from metafopy.methods.influence import InfluenceResult
from metafopy.plot import _draw as d
from metafopy.plot._plain import plain_plot
from metafopy.plot._rutils import coladj, get_ax

__all__ = ['PlotInflResult', 'plot_infl']

type FloatArray = NDArray[np.float64]

_MFROW = {1: (1, 1), 2: (2, 1), 3: (3, 1), 4: (2, 2), 5: (5, 1), 6: (3, 2), 7: (7, 1), 8: (4, 2)}
_NAMES = ('rstudent', 'dffits', 'cook.d', 'cov.r', 'tau2.del', 'QE.del', 'hat', 'weight')


@dataclass(frozen=True, slots=True)
class PlotInflResult:
    """The axes of the influence plots: ``axes`` for the diagnostics and ``dfbs_axes`` for the DFBETAS (if drawn)."""

    axes: tuple[Axes, ...]
    dfbs_axes: tuple[Axes, ...] = ()


def _n2mfrow(n: int) -> tuple[int, int]:
    """R's ``n2mfrow()``."""
    if n <= 3:
        return n, 1
    if n <= 6:
        return (n + 1) // 2, 2
    if n <= 12:
        return (n + 2) // 3, 3
    ncol = math.ceil(math.sqrt(n))
    return math.ceil(n / ncol), ncol


def _abbreviate(s: str, minlength: int = 4) -> str:
    """A simplified ``abbreviate()``: drop lower-case vowels, then lower-case letters, from the end until the string is short enough."""
    if len(s) <= minlength:
        return s
    chars = list(s)
    for pool in ('aeiou', 'abcdefghijklmnopqrstuvwxyz'):
        for i in range(len(chars) - 1, 0, -1):
            if len(chars) <= minlength:
                break
            if chars[i] in pool and chars[i - 1] != ' ':
                del chars[i]
    return ''.join(chars).replace(' ', '')


def _panel(
    ax: Axes,
    zi_all: FloatArray,
    is_infl_all: NDArray[np.bool_],
    ids: list[str],
    *,
    main: str,
    ylim: tuple[float, float],
    hlines: Sequence[tuple[float, str]],
    na_action: str,
    las: int,
    pch: int,
    bg: str,
    bg_infl: str,
    col_na: str,
    logy: bool = False,
) -> None:
    not_na = ~np.isnan(zi_all)
    if na_action == 'omit':
        zi = zi_all[not_na]
        infl = is_infl_all[not_na]
        labs = [s for s, ok in zip(ids, not_na, strict=True) if ok]
        n = int(zi.shape[0])
    else:
        zi, infl, labs, n = zi_all, is_infl_all, ids, int(zi_all.shape[0])
    pos = np.arange(1, n + 1, dtype=float)
    plain_plot(ax, (1.0, float(n)), ylim, '', '', main, xaxt=True, yaxt=True, logy=logy, xat=pos, xlabels=labs)
    if las in (2, 3):
        ax.tick_params(axis='x', labelrotation=90)
    for h, lty in hlines:
        d.hline(ax, h, lty=lty)
    if na_action != 'omit':
        sel = ~np.isnan(zi)
        d.line(ax, pos[sel], zi[sel], col=col_na)
    d.line(ax, pos, zi)
    for xp, yp in zip(pos, zi, strict=True):
        if not math.isnan(yp):
            d.pts(ax, float(xp), float(yp), pch=pch, bg=bg)
    for xp, yp, f in zip(pos, zi, infl, strict=True):
        if f and not math.isnan(yp):
            d.pts(ax, float(xp), float(yp), pch=pch, bg=bg_infl)


def plot_infl(
    x: InfluenceResult,
    plotinf: bool | int | Sequence[int] = True,
    plotdfbs: bool | int | Sequence[int] = False,
    dfbsnew: bool = False,
    logcov: bool = True,
    slab_style: int = 1,
    las: int = 0,
    pch: int = 21,
    bg: str | None = None,
    bg_infl: str = 'red',
    col_na: str | None = None,
    *,
    ids: Sequence[int | str] | None = None,
    na_action: str = 'omit',
    ax: Axes | None = None,
) -> PlotInflResult:
    """Plot the case-deletion diagnostics returned by :func:`~metafopy.influence` (``plot.infl.rma.uni``).

    ``plotinf`` selects the diagnostics (``True`` for all eight, or their numbers 1-8: ``rstudent``, ``dffits``,
    ``cook.d``, ``cov.r``, ``tau2.del``, ``QE.del``, ``hat`` and ``weight``) and ``plotdfbs`` the DFBETAS (``True``
    for all, or the numbers of the coefficients, starting at 1 as in metafor). Influential studies are drawn in
    ``bg_infl``. ``slab_style`` chooses the labels of the x-axis (1 = ``ids``, 2 = study labels, 3 = abbreviated
    study labels); ``ids`` defaults to ``1..k``. ``na_action`` is ``'omit'`` (leave out studies with missing
    diagnostics) or ``'exclude'``/``'pass'`` (keep them, joining the other points with a line in ``col_na``).
    The diagnostics are drawn into a grid of panels filling ``ax``; the DFBETAS are drawn on a new figure if the
    diagnostics are also plotted (or ``dfbsnew`` is true), as metafor starts a new page for them.

    Returns a :class:`PlotInflResult` with the axes.
    """
    if na_action not in ('omit', 'exclude', 'pass', 'fail'):
        raise ValueError("Unknown 'na_action' specified.")
    bg_c = coladj(-0.35) if bg is None else bg
    col_na_c = coladj(-0.2) if col_na is None else col_na
    n_all = int(x.rstudent.shape[0])
    values = {
        'rstudent': x.rstudent,
        'dffits': x.dffits,
        'cook.d': x.cook_d,
        'cov.r': x.cov_r,
        'tau2.del': x.tau2_del,
        'QE.del': x.QE_del,
        'hat': x.hat,
        'weight': x.weight,
    }
    any_na = any(np.isnan(v).any() for v in values.values()) or np.isnan(x.dfbs).any()
    if any_na and na_action == 'fail':
        raise ValueError('Missing values in results.')
    if na_action == 'fail':
        na_action = 'omit'

    which_inf: list[int] = []
    if plotinf is True:
        which_inf = list(range(1, 9))
    elif plotinf is not False:
        which_inf = list(dict.fromkeys(round(v) for v in np.atleast_1d(plotinf) if 1 <= v <= 8))
        if not which_inf:
            raise ValueError("Incorrect specification of the 'plotinf' argument.")
    which_dfbs: list[int] = []
    if plotdfbs is True:
        which_dfbs = list(range(1, x.p + 1))
    elif plotdfbs is not False:
        which_dfbs = list(dict.fromkeys(round(v) for v in np.atleast_1d(plotdfbs) if 1 <= v <= x.p))
        if not which_dfbs:
            raise ValueError("Incorrect specification of the 'plotdfbs' argument.")
    if not which_inf and not which_dfbs:
        raise ValueError("At least one of the arguments 'plotinf' or 'plotdfbs' must be TRUE.")
    if not which_inf:
        dfbsnew = False

    if slab_style == 1:
        id_list = [str(i + 1) for i in range(n_all)] if ids is None else [str(i) for i in ids]
    elif slab_style == 2:
        id_list = list(x.slab)
    else:
        id_list = [_abbreviate(s) for s in x.slab]
    if len(id_list) != n_all:
        raise ValueError("Length of 'ids' does not match the number of studies.")

    common: dict[str, Any] = {'na_action': na_action, 'las': las, 'pch': pch, 'bg': bg_c, 'bg_infl': bg_infl, 'col_na': col_na_c}
    infl = np.asarray(x.is_infl, dtype=bool)
    k, p = x.k, x.p
    qn = float(stats.norm.ppf(0.975))
    area = get_ax(ax)

    def draw(axes: list[Axes], which: int) -> None:
        name = _NAMES[which - 1]
        zi = np.asarray(values[name], dtype=float)
        a = axes[0]
        if np.isnan(zi).all():
            plain_plot(a, (0.0, 1.0), (0.0, 1.0), '', '', name, xaxt=False, yaxt=False)
            return
        zmin, zmax = float(np.nanmin(zi)), float(np.nanmax(zi))
        hl: list[tuple[float, str]]
        if which == 1:
            yl = (min(min(zmin, -2.0), -qn) * 1.05, max(max(zmax, 2.0), qn) * 1.05)
            hl = [(0.0, 'dashed'), (-qn, 'dotted'), (qn, 'dotted')]
        elif which == 2:
            lim = 3 * math.sqrt(p / (k - p))
            yl = (min(zmin, -lim) * 1.05, max(zmax, lim) * 1.05)
            hl = [(0.0, 'dashed'), (lim, 'dotted'), (-lim, 'dotted')]
        elif which == 3:
            yl = (0.0, zmax * 1.05)
            hl = [(float(stats.chi2.ppf(0.5, df=x.m)), 'dotted')]
        elif which == 4:
            yl = (zmin, zmax)
            hl = [(1.0, 'dashed')]
        elif which == 5:
            yl = (zmin, zmax)
            hl = [(x.tau2, 'dashed')]
        elif which == 6:
            yl = (zmin, zmax)
            hl = [(x.QE, 'dashed'), (float(k - p), 'dotted')]
        elif which == 7:
            yl = (0.0, max(zmax, 3 * p / k) * 1.05)
            hl = [(p / k, 'dashed'), (3 * p / k, 'dotted')]
        else:
            yl = (0.0, zmax * 1.05)
            hl = [(100 / k, 'dashed')]
        _panel(a, zi, infl, id_list, main=name, ylim=yl, hlines=hl, logy=(which == 4 and logcov), **common)

    axes_inf: list[Axes] = []
    if which_inf:
        nr, nc = _MFROW[len(which_inf)]
        grid = d.subaxes(area, nr, nc, hspace=0.6, wspace=0.3)
        flat = [a for row in grid for a in row]
        for a, which in zip(flat, which_inf, strict=False):
            draw([a], which)
            axes_inf.append(a)
        for a in flat[len(which_inf) :]:
            a.set_axis_off()
    axes_dfbs: list[Axes] = []
    if which_dfbs:
        nr, nc = _n2mfrow(len(which_dfbs))
        # with the diagnostics also drawn (or dfbsnew) metafor starts a new page for the DFBETAS
        target = plt.figure().add_subplot(111) if (which_inf or dfbsnew) else area
        grid = d.subaxes(target, nr, nc, hspace=0.6, wspace=0.3)
        flat = [a for row in grid for a in row]
        names = list(x.coef_names)
        for a, j in zip(flat, which_dfbs, strict=False):
            zi = np.asarray(x.dfbs[:, j - 1], dtype=float)
            _panel(
                a,
                zi,
                infl,
                id_list,
                main=f'dfbs:  {names[j - 1]}',
                ylim=(float(np.nanmin(zi)) * 1.05, float(np.nanmax(zi)) * 1.05),
                hlines=[(0.0, 'dashed'), (1.0, 'dotted'), (-1.0, 'dotted')],
                **common,
            )
            axes_dfbs.append(a)
        for a in flat[len(which_dfbs) :]:
            a.set_axis_off()
    return PlotInflResult(tuple(axes_inf), tuple(axes_dfbs))
