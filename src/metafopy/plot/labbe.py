"""L'Abbe plot (metafor's ``labbe.rma``).

Plots the arm-level outcomes (log risks, log odds, risks, arcsine-transformed risks or the corresponding incidence
rate measures) of the two groups of each study against each other, with a line for the estimated effect, see
L'Abbe et al. (1987), doi:10.7326/0003-4819-107-2-224 and
https://wviechtb.github.io/metafor/reference/labbe.html.
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

import metafopy.effects
import metafopy.errors
import metafopy.methods.common
import metafopy.methods.predict
import metafopy.models.rma_uni
import metafopy.plot
import metafopy.plot.draw
import metafopy.plot.plain
import metafopy.plot.rutils

__all__ = ['LabbeResult', 'labbe']

type FloatArray = numpy.typing.NDArray[numpy.float64]

# auditme: accept[non-alphabetical-dict-keys] order follows the R script that generates the fixtures
ARM_MEASURE = {'RR': 'PLN', 'OR': 'PLO', 'RD': 'PR', 'AS': 'PAS', 'IRR': 'IRLN', 'IRD': 'IR', 'IRSD': 'IRS'}


@dataclasses.dataclass(frozen=True, slots=True)
class LabbeResult:
    """The plotted arm-level outcomes and point attributes (metafor's invisible return value) and the axes drawn on."""

    ax: matplotlib.axes.Axes
    x: FloatArray
    y: FloatArray
    cex: FloatArray
    pch: tuple[int, ...]
    col: tuple[str, ...]
    bg: tuple[str, ...]
    ids: tuple[int, ...]
    slab: tuple[str, ...]

    def to_frame(self) -> pandas.DataFrame:
        return pandas.DataFrame(
            {
                'x': self.x,
                'y': self.y,
                'cex': self.cex,
                'pch': list(self.pch),
                'col': list(self.col),
                'bg': list(self.bg),
                'ids': list(self.ids),
                'slab': list(self.slab),
            }
        )


def column(x: metafopy.models.rma_uni.RMAResult, name: str, value: numpy.typing.ArrayLike | str | None) -> FloatArray:
    """The counts ``name`` for the ``k`` studies of the model (a column of the model's data, or an array).

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    k, k_all = x.k, x.data.k_all
    frame = x.data.data
    if value is None:
        if frame is None or name not in frame.columns:
            raise metafopy.errors.InputError(f"Information needed to construct the plot is not available in the model object (pass '{name}').")
        value = name
    if isinstance(value, str):
        if frame is None or value not in frame.columns:
            raise metafopy.errors.InputError(f"Column '{value}' not found in the data of the model.")
        arr = frame[value].to_numpy(dtype=float)
    else:
        arr = numpy.atleast_1d(numpy.asarray(value, dtype=float))
    if arr.shape[0] == k_all and k_all != k:
        arr = arr[x.data.keep]
    if arr.shape[0] != k:
        raise metafopy.errors.InputError(f"Length of '{name}' ({arr.shape[0]}) does not correspond to the number of studies.")
    return numpy.asarray(arr, dtype=numpy.float64)


def per_study(v: typing.Any, x: metafopy.models.rma_uni.RMAResult, name: str) -> list[typing.Any]:
    if isinstance(v, str) or numpy.ndim(v) == 0:
        return [v] * x.k
    out = list(v)
    if len(out) == x.data.k_all and x.data.k_all != x.k:
        out = [o for o, kp in zip(out, x.data.keep, strict=True) if kp]
    if len(out) != x.k:
        raise metafopy.errors.InputError(
            f"Length of the '{name}' argument ({len(out)}) does not correspond to the size of the original dataset ({x.data.k_all})."
        )
    return out


def labbe(
    x: metafopy.models.rma_uni.RMAResult,
    xlim: collections.abc.Sequence[float] | None = None,
    ylim: collections.abc.Sequence[float] | None = None,
    lim: collections.abc.Sequence[float] | None = None,
    xlab: str | None = None,
    ylab: str | None = None,
    flip: bool = False,
    ci: bool | str = False,
    pi: bool | str = False,
    grid: bool | str = False,
    legend: bool | str | dict[str, typing.Any] = False,
    add: float | None = None,
    to: str = 'only0',
    transf: collections.abc.Callable[..., typing.Any] | None = None,
    targs: typing.Any = None,
    pch: int | collections.abc.Sequence[int] = 21,
    psize: float | collections.abc.Sequence[float] | None = None,
    plim: collections.abc.Sequence[float] = (0.5, 3.5),
    col: str | collections.abc.Sequence[str] | None = None,
    bg: str | collections.abc.Sequence[str] | None = None,
    lty: str | collections.abc.Sequence[str] | None = None,
    *,
    ai: numpy.typing.ArrayLike | str | None = None,
    bi: numpy.typing.ArrayLike | str | None = None,
    ci_counts: numpy.typing.ArrayLike | str | None = None,
    di: numpy.typing.ArrayLike | str | None = None,
    x1i: numpy.typing.ArrayLike | str | None = None,
    x2i: numpy.typing.ArrayLike | str | None = None,
    t1i: numpy.typing.ArrayLike | str | None = None,
    t2i: numpy.typing.ArrayLike | str | None = None,
    llim: collections.abc.Sequence[float] | None = None,
    addyi: bool = True,
    addvi: bool = True,
    ax: matplotlib.axes.Axes | None = None,
) -> LabbeResult:
    """L'Abbe plot of a model without moderators fitted to 2x2 table data or incidence rates (``labbe.rma``).

    The model's ``measure`` must be ``'RR'``, ``'OR'``, ``'RD'``, ``'AS'``, ``'IRR'``, ``'IRD'`` or ``'IRSD'``.
    A fitted :class:`~metafopy.RMAResult` does not keep the raw counts, so they are taken from the columns
    ``ai``, ``bi``, ``ci``, ``di`` (``x1i``, ``t1i``, ``x2i``, ``t2i`` for the rate measures) of the data the model
    was fitted to, or passed as the keyword arguments ``ai``, ``bi``, ``ci_counts`` (the ``ci`` counts; ``ci``
    itself draws the confidence region as in metafor), ``di``, ``x1i``, ``x2i``, ``t1i`` and ``t2i`` (column names or
    arrays). ``ci`` and ``pi`` shade the confidence and prediction interval of the line of the estimated effect
    (a colour string sets the colour), ``flip`` swaps the groups, ``psize``/``plim`` set the point sizes (default:
    proportional to the precision between ``plim``), ``transf`` is applied to the arm-level outcomes, and
    ``add``, ``to`` (defaults 1/2, 0 for ``'AS'`` and ``'IRSD'``, and ``'only0'``), ``lim``/``llim`` as in metafor.

    Returns a :class:`LabbeResult`.

    Raises:
        metafopy.errors.InputError: an argument or the data is invalid.
    """
    if not x.int_only:
        raise metafopy.errors.InputError("L'Abbe plots can only be drawn for models without moderators.")
    if x.measure not in ARM_MEASURE:
        raise metafopy.errors.InputError("Argument 'measure' must have been set to one of the following: 'RR','OR','RD','AS','IRR','IRD','IRSD'.")
    # auditme: accept[literal-duplication] branches mirror metafor, which repeats these option lists
    if to not in ('all', 'only0', 'if0all', 'none'):
        raise metafopy.errors.InputError("Unknown 'to' argument specified.")
    axx = metafopy.plot.rutils.get_ax(ax)
    add_ = (0.0 if x.measure in ('AS', 'IRSD') else 0.5) if add is None else add
    lt = ['solid', 'dashed'] if lty is None else ([lty, lty] if isinstance(lty, str | int) else list(lty))
    ci_col = ci if isinstance(ci, str) else metafopy.plot.rutils.coladj(-0.15)
    pi_col = pi if isinstance(pi, str) else metafopy.plot.rutils.coladj(-0.05)
    grid_col = grid if isinstance(grid, str) else metafopy.plot.rutils.coladj(-0.2)
    show_ci, show_pi, show_grid = bool(ci), bool(pi), bool(grid)

    k = x.k
    pch_v = per_study(pch, x, 'pch')
    col_v = per_study(metafopy.plot.draw.FG if col is None else col, x, 'col')
    bg_v = per_study(metafopy.plot.rutils.coladj(-0.35) if bg is None else bg, x, 'bg')
    psize_v = None if psize is None else numpy.asarray(per_study(psize, x, 'psize'), dtype=float)

    measure = ARM_MEASURE[x.measure]
    if x.measure in ('RR', 'OR', 'RD', 'AS'):
        a, b, c, dd = (column(x, n, v) for n, v in (('ai', ai), ('bi', bi), ('ci', ci_counts), ('di', di)))
        first, second = ((a, b), (c, dd)) if flip else ((c, dd), (a, b))
        kw_x: dict[str, typing.Any] = {'xi': first[0], 'mi': first[1]}
        kw_y: dict[str, typing.Any] = {'xi': second[0], 'mi': second[1]}
    else:
        a1, tt1, a2, tt2 = (column(x, n, v) for n, v in (('x1i', x1i), ('t1i', t1i), ('x2i', x2i), ('t2i', t2i)))
        first, second = ((a1, tt1), (a2, tt2)) if flip else ((a2, tt2), (a1, tt1))
        kw_x = {'xi': first[0], 'ti': first[1]}
        kw_y = {'xi': second[0], 'ti': second[1]}
    opts: dict[str, typing.Any] = {'add': add_, 'to': to, 'addyi': addyi, 'addvi': addvi, 'cutoff': math.nan}
    dat_x = metafopy.effects.escalc(measure, **kw_x, **opts)
    dat_y = metafopy.effects.escalc(measure, **kw_y, **opts)
    yx, vx = dat_x['yi'].to_numpy(dtype=float), dat_x['vi'].to_numpy(dtype=float)
    yy, vy = dat_y['yi'].to_numpy(dtype=float), dat_y['vi'].to_numpy(dtype=float)
    has_na = numpy.isnan(yx) | numpy.isnan(vx) | numpy.isnan(yy) | numpy.isnan(vy)
    not_na = ~has_na
    ids = tuple(int(i) + 1 for i in numpy.flatnonzero(x.data.keep)[not_na])
    slab = tuple(numpy.asarray(x.slab, dtype=object)[not_na].tolist())
    pch_v = [p for p, ok in zip(pch_v, not_na, strict=True) if ok]
    col_v = [p for p, ok in zip(col_v, not_na, strict=True) if ok]
    bg_v = [p for p, ok in zip(bg_v, not_na, strict=True) if ok]
    if psize_v is not None:
        psize_v = psize_v[not_na]
    yx, vx, yy, vy = yx[not_na], vx[not_na], yy[not_na], vy[not_na]
    if yx.size == 0 or yy.size == 0:
        raise metafopy.errors.InputError('No information in object to compute the arm-level outcomes.')
    vi = vx + vy
    k = int(vi.shape[0])

    if psize_v is None:
        if len(plim) < 2:
            raise metafopy.errors.InputError("Argument 'plim' must be of length 2 or 3.")
        wi = numpy.sqrt(1 / vi)
        p0, p1 = plim[0], plim[1]
        psz = numpy.full(k, numpy.nan)
        if not math.isnan(p0) and not math.isnan(p1):
            rng = float(wi.max() - wi.min())
            psz = numpy.ones(k) if rng <= math.sqrt(numpy.finfo(float).eps) else (wi - wi.min()) / rng * (p1 - p0) + p0
        if math.isnan(p0) and not math.isnan(p1):
            psz = wi / wi.max() * p1
            if len(plim) == 3:
                psz[psz <= plim[2]] = plim[2]
        if not math.isnan(p0) and math.isnan(p1):
            psz = wi / wi.min() * p0
            if len(plim) == 3:
                psz[psz >= plim[2]] = plim[2]
        if numpy.all(numpy.isnan(psz)):
            psz = numpy.ones(k)
    else:
        psz = psize_v

    min_yi, max_yi = float(min(yx.min(), yy.min())), float(max(yx.max(), yy.max()))
    rng_yi = max_yi - min_yi
    ln = 10000
    intrcpt = float(x.beta[0])
    if llim is None:
        m = x.measure
        if m in ('RD', 'IRD', 'IRSD'):
            x_vals = numpy.linspace(0 if intrcpt > 0 else -intrcpt, 1 - intrcpt if intrcpt > 0 else 1, ln)
        elif m in ('RR', 'IRR'):
            x_vals = numpy.linspace(min_yi - rng_yi, -intrcpt if intrcpt > 0 else 0, ln)
        elif m == 'OR':
            x_vals = numpy.linspace(min_yi - rng_yi, max_yi + rng_yi, ln)
        else:
            x_vals = numpy.linspace(0 if intrcpt > 0 else -intrcpt, math.asin(1) - intrcpt if intrcpt > 0 else math.asin(1), ln)
    else:
        if len(llim) != 2:
            raise metafopy.errors.InputError("Argument 'llim' must be of length 2.")
        x_vals = numpy.linspace(llim[0], llim[1], ln)
    y_vals = intrcpt + 1 * x_vals
    ci_lb = ci_ub = pi_lb = pi_ub = None
    if show_ci or show_pi:
        pr = metafopy.methods.predict.predict(x)
        ci_lb, ci_ub = pr.ci_lb[0] + x_vals, pr.ci_ub[0] + x_vals
        pi_lb, pi_ub = pr.pi_lb[0] + x_vals, pr.pi_ub[0] + x_vals

    if transf is not None:
        yx, yy = metafopy.methods.common.apply_each(yx, transf, targs), metafopy.methods.common.apply_each(yy, transf, targs)
        x_vals, y_vals = metafopy.methods.common.apply_each(x_vals, transf, targs), metafopy.methods.common.apply_each(y_vals, transf, targs)
        if ci_lb is not None and ci_ub is not None and pi_lb is not None and pi_ub is not None:
            ci_lb, ci_ub = metafopy.methods.common.apply_each(ci_lb, transf, targs), metafopy.methods.common.apply_each(ci_ub, transf, targs)
            pi_lb, pi_ub = metafopy.methods.common.apply_each(pi_lb, transf, targs), metafopy.methods.common.apply_each(pi_ub, transf, targs)
    min_yi, max_yi = float(min(yx.min(), yy.min())), float(max(yx.max(), yy.max()))
    if lim is None:
        xl = (min_yi, max_yi) if xlim is None else (float(xlim[0]), float(xlim[1]))
        yl = (min_yi, max_yi) if ylim is None else (float(ylim[0]), float(ylim[1]))
    else:
        xl = yl = (float(lim[0]), float(lim[1]))

    order = numpy.argsort(-psz, kind='stable')
    if xlab is None:
        xlab = metafopy.plot.rutils.setlab(measure, transf, None, 1) + ' (Group 1)'
    if ylab is None:
        ylab = metafopy.plot.rutils.setlab(measure, transf, None, 1) + ' (Group 2)'
    xt, yt = metafopy.plot.plain.plain_plot(axx, xl, yl, xlab, ylab)
    if show_pi and pi_lb is not None and pi_ub is not None:
        metafopy.plot.draw.poly(axx, numpy.concatenate([x_vals, x_vals[::-1]]), numpy.concatenate([pi_lb, pi_ub[::-1]]), col=pi_col, border=None, zorder=0.6)
    if show_ci and ci_lb is not None and ci_ub is not None:
        metafopy.plot.draw.poly(axx, numpy.concatenate([x_vals, x_vals[::-1]]), numpy.concatenate([ci_lb, ci_ub[::-1]]), col=ci_col, border=None, zorder=0.6)
    if show_grid:
        assert xt is not None
        assert yt is not None
        metafopy.plot.draw.grid(axx, xt, yt, grid_col)
    metafopy.plot.draw.seg(axx, float(x_vals.min()), float(x_vals.min()), float(x_vals.max()), float(x_vals.max()), lty=lt[0])
    metafopy.plot.draw.line(axx, x_vals, y_vals, lty=lt[1])
    for i in order:
        metafopy.plot.draw.pts(axx, float(yx[i]), float(yy[i]), pch=int(pch_v[i]), cex=float(psz[i]), col=col_v[i], bg=bg_v[i])
    labbe_legend(axx, legend, intrcpt, lt, show_ci, show_pi, ci_col, pi_col, 100 * (1 - x.level))
    return LabbeResult(
        axx,
        numpy.asarray(yx, dtype=float),
        numpy.asarray(yy, dtype=float),
        numpy.asarray(psz, dtype=float),
        tuple(int(p) for p in pch_v),
        tuple(str(c) for c in col_v),
        tuple(str(c) for c in bg_v),
        ids,
        slab,
    )


def labbe_legend(
    ax: matplotlib.axes.Axes,
    legend: bool | str | dict[str, typing.Any],
    intrcpt: float,
    lty: list[typing.Any],
    ci: bool,
    pi: bool,
    ci_col: str,
    pi_col: str,
    lvl: float,
) -> None:
    """An approximation of metafor's legend (reference line, line of the estimated effect and the CI/PI regions)."""
    pos = 'lower right' if intrcpt > 0 else 'upper left'
    if isinstance(legend, dict):
        pos = str(legend.get('x', pos))
    elif isinstance(legend, str):
        pos = {'topright': 'upper right', 'topleft': 'upper left', 'bottomright': 'lower right', 'bottomleft': 'lower left'}.get(legend, legend)
    elif not legend:
        return
    handles: list[typing.Any] = []
    labels: list[str] = []
    for name, ls in (('Reference Line of No Effect', lty[0]), ('Line for the Estimated Effect', lty[1])):
        if ls not in ('blank', 0):
            handles.append(matplotlib.lines.Line2D([], [], color=metafopy.plot.draw.FG, linestyle=metafopy.plot.draw.LTY.get(ls, ls)))
            labels.append(name)
    shown = f'{lvl:g}'
    if ci:
        handles.append(matplotlib.patches.Patch(facecolor=metafopy.plot.draw.rcolor(ci_col), edgecolor='black'))
        labels.append(f'{shown}% Confidence Interval')
    if pi:
        handles.append(matplotlib.patches.Patch(facecolor=metafopy.plot.draw.rcolor(pi_col), edgecolor='black'))
        labels.append(f'{shown}% Prediction Interval')
    if handles:
        ax.legend(handles, labels, loc=typing.cast(typing.Any, pos), framealpha=1.0)
