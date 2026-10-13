"""Plot of the case-deletion diagnostics (metafor's ``plot.infl.rma.uni``).

See Viechtbauer & Cheung (2010), doi:10.1002/jrsm.11, and
https://wviechtb.github.io/metafor/reference/plot.infl.rma.uni.html.
"""

import collections.abc
import dataclasses
import math
import typing

import matplotlib.axes
import matplotlib.pyplot
import numpy
import numpy.typing
import scipy.stats

import metafopy.errors
import metafopy.methods.influence
import metafopy.plot
import metafopy.plot.draw
import metafopy.plot.plain
import metafopy.plot.rutils

__all__ = ['PlotInflResult', 'plot_infl']

type FloatArray = numpy.typing.NDArray[numpy.float64]

MFROW = {1: (1, 1), 2: (2, 1), 3: (3, 1), 4: (2, 2), 5: (5, 1), 6: (3, 2), 7: (7, 1), 8: (4, 2)}
# auditme: accept[non-alphabetical-literal] order is metafor's documented order and is meaningful
NAMES = ('rstudent', 'dffits', 'cook.d', 'cov.r', 'tau2.del', 'QE.del', 'hat', 'weight')


@dataclasses.dataclass(frozen=True, slots=True)
class PlotInflResult:
    """The axes of the influence plots: ``axes`` for the diagnostics and ``dfbs_axes`` for the DFBETAS (if drawn)."""

    axes: tuple[matplotlib.axes.Axes, ...]
    dfbs_axes: tuple[matplotlib.axes.Axes, ...] = ()


def mfrow_cex(nrow: int, ncol: int) -> float:
    """The factor by which R's ``par(mfrow = c(nrow, ncol))`` reduces the text size: 0.83 for 2 x 2 panels and 0.66 for more rows or columns."""
    if nrow > 2 or ncol > 2:
        return 0.66
    return 0.83 if nrow == 2 and ncol == 2 else 1.0


def n2mfrow(n: int) -> tuple[int, int]:
    """R's ``n2mfrow()``."""
    if n <= 3:
        return n, 1
    if n <= 6:
        return (n + 1) // 2, 2
    if n <= 12:
        return (n + 2) // 3, 3
    ncol = math.ceil(math.sqrt(n))
    return math.ceil(n / ncol), ncol


def abbreviate(s: str, minlength: int = 4) -> str:
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


def panel(
    ax: matplotlib.axes.Axes,
    zi_all: FloatArray,
    is_infl_all: numpy.typing.NDArray[numpy.bool_],
    ids: list[str],
    *,
    main: str,
    ylim: tuple[float, float],
    hlines: collections.abc.Sequence[tuple[float, str]],
    na_action: str,
    las: int,
    pch: int,
    bg: str,
    bg_infl: str,
    col_na: str,
    logy: bool = False,
) -> None:
    not_na = ~numpy.isnan(zi_all)
    if na_action == 'omit':
        zi = zi_all[not_na]
        infl = is_infl_all[not_na]
        labs = [s for s, ok in zip(ids, not_na, strict=True) if ok]
        n = int(zi.shape[0])
    else:
        zi, infl, labs, n = zi_all, is_infl_all, ids, int(zi_all.shape[0])
    pos = numpy.arange(1, n + 1, dtype=float)
    metafopy.plot.plain.plain_plot(
        ax, (1.0, float(n)), ylim, '', '', main, xaxt=True, yaxt=True, logy=logy, xat=pos, xlabels=labs, thin_xlabels=las not in (2, 3)
    )
    if las in (2, 3):
        ax.tick_params(axis='x', labelrotation=90)
    for h, lty in hlines:
        metafopy.plot.draw.hline(ax, h, lty=lty)
    if na_action != 'omit':
        sel = ~numpy.isnan(zi)
        metafopy.plot.draw.line(ax, pos[sel], zi[sel], col=col_na)
    metafopy.plot.draw.line(ax, pos, zi)
    for xp, yp in zip(pos, zi, strict=True):
        if not math.isnan(yp):
            metafopy.plot.draw.pts(ax, float(xp), float(yp), pch=pch, bg=bg)
    for xp, yp, f in zip(pos, zi, infl, strict=True):
        if f and not math.isnan(yp):
            metafopy.plot.draw.pts(ax, float(xp), float(yp), pch=pch, bg=bg_infl)


def plot_infl(
    x: metafopy.methods.influence.InfluenceResult,
    plotinf: bool | int | collections.abc.Sequence[int] = True,
    plotdfbs: bool | int | collections.abc.Sequence[int] = False,
    dfbsnew: bool = False,
    logcov: bool = True,
    slab_style: int = 1,
    las: int = 0,
    pch: int = 21,
    bg: str | None = None,
    bg_infl: str = 'red',
    col_na: str | None = None,
    *,
    ids: collections.abc.Iterable[int | str] | None = None,
    na_action: str = 'omit',
    mfrow: tuple[int, int] | None = None,
    ax: matplotlib.axes.Axes | None = None,
) -> PlotInflResult:
    """Plot the case-deletion diagnostics returned by :func:`~metafopy.influence` (``plot.infl.rma.uni``).

    ``plotinf`` selects the diagnostics (``True`` for all eight, or their numbers 1-8: ``rstudent``, ``dffits``,
    ``cook.d``, ``cov.r``, ``tau2.del``, ``QE.del``, ``hat`` and ``weight``) and ``plotdfbs`` the DFBETAS (``True``
    for all, or the numbers of the coefficients, starting at 1 as in metafor). Influential studies are drawn in
    ``bg_infl``. ``slab_style`` chooses the labels of the x-axis (1 = ``ids``, 2 = study labels, 3 = abbreviated
    study labels); ``ids`` defaults to ``1..k``. ``na_action`` is ``'omit'`` (leave out studies with missing
    diagnostics) or ``'exclude'``/``'pass'`` (keep them, joining the other points with a line in ``col_na``).
    The diagnostics are drawn into a grid of panels filling ``ax``, with the number of rows and columns of metafor's default
    (``par(mfrow = ...)``) unless ``mfrow=(rows, columns)`` is given, e.g. ``(8, 1)`` for one column of panels; the DFBETAS are drawn on a new figure if the
    diagnostics are also plotted (or ``dfbsnew`` is true), as metafor starts a new page for them.

    Returns a :class:`PlotInflResult` with the axes.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    if na_action not in ('omit', 'exclude', 'pass', 'fail'):
        raise metafopy.errors.InputError("Unknown 'na_action' specified.")
    bg_c = metafopy.plot.rutils.coladj(-0.35) if bg is None else bg
    col_na_c = metafopy.plot.rutils.coladj(-0.2) if col_na is None else col_na
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
    any_na = any(numpy.isnan(v).any() for v in values.values()) or numpy.isnan(x.dfbs).any()
    if any_na and na_action == 'fail':
        raise metafopy.errors.InputError('Missing values in results.')
    if na_action == 'fail':
        na_action = 'omit'

    which_inf: list[int] = []
    if plotinf is True:
        which_inf = list(range(1, 9))
    elif plotinf is not False:
        which_inf = list(dict.fromkeys(round(v) for v in numpy.atleast_1d(plotinf) if 1 <= v <= 8))
        if not which_inf:
            raise metafopy.errors.InputError("Incorrect specification of the 'plotinf' argument.")
    which_dfbs: list[int] = []
    if plotdfbs is True:
        which_dfbs = list(range(1, x.p + 1))
    elif plotdfbs is not False:
        which_dfbs = list(dict.fromkeys(round(v) for v in numpy.atleast_1d(plotdfbs) if 1 <= v <= x.p))
        if not which_dfbs:
            raise metafopy.errors.InputError("Incorrect specification of the 'plotdfbs' argument.")
    if not which_inf and not which_dfbs:
        raise metafopy.errors.InputError("At least one of the arguments 'plotinf' or 'plotdfbs' must be TRUE.")
    if not which_inf:
        dfbsnew = False

    if slab_style == 1:
        id_list = [str(i + 1) for i in range(n_all)] if ids is None else [str(i) for i in ids]
    elif slab_style == 2:
        id_list = list(x.slab)
    else:
        id_list = [abbreviate(s) for s in x.slab]
    if len(id_list) != n_all:
        raise metafopy.errors.InputError("Length of 'ids' does not match the number of studies.")

    common: dict[str, typing.Any] = {'na_action': na_action, 'las': las, 'pch': pch, 'bg': bg_c, 'bg_infl': bg_infl, 'col_na': col_na_c}
    infl = numpy.asarray(x.is_infl, dtype=bool)
    k, p = x.k, x.p
    qn = float(scipy.stats.norm.ppf(0.975))
    area = metafopy.plot.rutils.get_ax(ax)

    def draw(axes: list[matplotlib.axes.Axes], which: int) -> None:
        name = NAMES[which - 1]
        zi = numpy.asarray(values[name], dtype=float)
        a = axes[0]
        if numpy.isnan(zi).all():
            metafopy.plot.plain.plain_plot(a, (0.0, 1.0), (0.0, 1.0), '', '', name, xaxt=False, yaxt=False)
            return
        zmin, zmax = float(numpy.nanmin(zi)), float(numpy.nanmax(zi))
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
            hl = [(float(scipy.stats.chi2.ppf(0.5, df=x.m)), 'dotted')]
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
        panel(a, zi, infl, id_list, main=name, ylim=yl, hlines=hl, logy=(which == 4 and logcov), **common)

    axes_inf: list[matplotlib.axes.Axes] = []
    if which_inf:
        nr, nc = MFROW[len(which_inf)] if mfrow is None else mfrow
        if nr * nc < len(which_inf):
            raise metafopy.errors.InputError("Argument 'mfrow' has too few panels for the diagnostics.")
        grid = metafopy.plot.draw.subaxes(area, nr, nc, hspace=0.6, wspace=0.3)
        flat = [a for row in grid for a in row]
        with matplotlib.rc_context({'font.size': metafopy.plot.draw.base_fontsize() * mfrow_cex(nr, nc)}):
            for a, which in zip(flat, which_inf, strict=False):
                draw([a], which)
                axes_inf.append(a)
        for a in flat[len(which_inf) :]:
            a.set_axis_off()
    axes_dfbs: list[matplotlib.axes.Axes] = []
    if which_dfbs:
        nr, nc = n2mfrow(len(which_dfbs))
        # with the diagnostics also drawn (or dfbsnew) metafor starts a new page for the DFBETAS
        target = matplotlib.pyplot.figure().add_subplot(111) if (which_inf or dfbsnew) else area
        grid = metafopy.plot.draw.subaxes(target, nr, nc, hspace=0.6, wspace=0.3)
        flat = [a for row in grid for a in row]
        names = list(x.coef_names)
        with matplotlib.rc_context({'font.size': metafopy.plot.draw.base_fontsize() * mfrow_cex(nr, nc)}):
            for a, j in zip(flat, which_dfbs, strict=False):
                zi = numpy.asarray(x.dfbs[:, j - 1], dtype=float)
                panel(
                    a,
                    zi,
                    infl,
                    id_list,
                    main=f'dfbs:  {names[j - 1]}',
                    ylim=(float(numpy.nanmin(zi)) * 1.05, float(numpy.nanmax(zi)) * 1.05),
                    hlines=[(0.0, 'dashed'), (1.0, 'dotted'), (-1.0, 'dotted')],
                    **common,
                )
                axes_dfbs.append(a)
        for a in flat[len(which_dfbs) :]:
            a.set_axis_off()
    return PlotInflResult(tuple(axes_inf), tuple(axes_dfbs))
