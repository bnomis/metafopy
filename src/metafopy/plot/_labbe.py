"""L'Abbe plot (metafor's ``labbe.rma``).

Plots the arm-level outcomes (log risks, log odds, risks, arcsine-transformed risks or the corresponding incidence
rate measures) of the two groups of each study against each other, with a line for the estimated effect, see
L'Abbe et al. (1987), doi:10.7326/0003-4819-107-2-224 and
https://wviechtb.github.io/metafor/reference/labbe.html.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from numpy.typing import ArrayLike, NDArray

from metafopy.effects import escalc
from metafopy.methods._common import apply_each
from metafopy.methods.predict import predict
from metafopy.models.rma_uni import RMAResult
from metafopy.plot import _draw as d
from metafopy.plot._plain import plain_plot
from metafopy.plot._rutils import coladj, get_ax, setlab

__all__ = ['LabbeResult', 'labbe']

type FloatArray = NDArray[np.float64]

_ARM_MEASURE = {'RR': 'PLN', 'OR': 'PLO', 'RD': 'PR', 'AS': 'PAS', 'IRR': 'IRLN', 'IRD': 'IR', 'IRSD': 'IRS'}


@dataclass(frozen=True, slots=True)
class LabbeResult:
    """The plotted arm-level outcomes and point attributes (metafor's invisible return value) and the axes drawn on."""

    ax: Axes
    x: FloatArray
    y: FloatArray
    cex: FloatArray
    pch: tuple[int, ...]
    col: tuple[str, ...]
    bg: tuple[str, ...]
    ids: tuple[int, ...]
    slab: tuple[str, ...]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
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


def _column(x: RMAResult, name: str, value: ArrayLike | str | None) -> FloatArray:
    """The counts ``name`` for the ``k`` studies of the model (a column of the model's data, or an array)."""
    k, k_all = x.k, x.data.k_all
    frame = x.data.data
    if value is None:
        if frame is None or name not in frame.columns:
            raise ValueError(f"Information needed to construct the plot is not available in the model object (pass '{name}').")
        value = name
    if isinstance(value, str):
        if frame is None or value not in frame.columns:
            raise ValueError(f"Column '{value}' not found in the data of the model.")
        arr = frame[value].to_numpy(dtype=float)
    else:
        arr = np.atleast_1d(np.asarray(value, dtype=float))
    if arr.shape[0] == k_all and k_all != k:
        arr = arr[x.data.keep]
    if arr.shape[0] != k:
        raise ValueError(f"Length of '{name}' ({arr.shape[0]}) does not correspond to the number of studies.")
    return np.asarray(arr, dtype=np.float64)


def _per_study(v: Any, x: RMAResult, name: str) -> list[Any]:
    if isinstance(v, str) or np.ndim(v) == 0:
        return [v] * x.k
    out = list(v)
    if len(out) == x.data.k_all and x.data.k_all != x.k:
        out = [o for o, kp in zip(out, x.data.keep, strict=True) if kp]
    if len(out) != x.k:
        raise ValueError(f"Length of the '{name}' argument ({len(out)}) does not correspond to the size of the original dataset ({x.data.k_all}).")
    return out


def labbe(
    x: RMAResult,
    xlim: Sequence[float] | None = None,
    ylim: Sequence[float] | None = None,
    lim: Sequence[float] | None = None,
    xlab: str | None = None,
    ylab: str | None = None,
    flip: bool = False,
    ci: bool | str = False,
    pi: bool | str = False,
    grid: bool | str = False,
    legend: bool | str | dict[str, Any] = False,
    add: float | None = None,
    to: str = 'only0',
    transf: Callable[..., Any] | None = None,
    targs: Any = None,
    pch: int | Sequence[int] = 21,
    psize: float | Sequence[float] | None = None,
    plim: Sequence[float] = (0.5, 3.5),
    col: str | Sequence[str] | None = None,
    bg: str | Sequence[str] | None = None,
    lty: str | Sequence[str] | None = None,
    *,
    ai: ArrayLike | str | None = None,
    bi: ArrayLike | str | None = None,
    ci_counts: ArrayLike | str | None = None,
    di: ArrayLike | str | None = None,
    x1i: ArrayLike | str | None = None,
    x2i: ArrayLike | str | None = None,
    t1i: ArrayLike | str | None = None,
    t2i: ArrayLike | str | None = None,
    llim: Sequence[float] | None = None,
    addyi: bool = True,
    addvi: bool = True,
    ax: Axes | None = None,
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
    """
    if not x.int_only:
        raise ValueError("L'Abbe plots can only be drawn for models without moderators.")
    if x.measure not in _ARM_MEASURE:
        raise ValueError("Argument 'measure' must have been set to one of the following: 'RR','OR','RD','AS','IRR','IRD','IRSD'.")
    if to not in ('all', 'only0', 'if0all', 'none'):
        raise ValueError("Unknown 'to' argument specified.")
    axx = get_ax(ax)
    add_ = (0.0 if x.measure in ('AS', 'IRSD') else 0.5) if add is None else add
    lt = ['solid', 'dashed'] if lty is None else ([lty, lty] if isinstance(lty, str | int) else list(lty))
    ci_col = ci if isinstance(ci, str) else coladj(-0.15)
    pi_col = pi if isinstance(pi, str) else coladj(-0.05)
    grid_col = grid if isinstance(grid, str) else coladj(-0.2)
    show_ci, show_pi, show_grid = bool(ci), bool(pi), bool(grid)

    k = x.k
    pch_v = _per_study(pch, x, 'pch')
    col_v = _per_study(d.FG if col is None else col, x, 'col')
    bg_v = _per_study(coladj(-0.35) if bg is None else bg, x, 'bg')
    psize_v = None if psize is None else np.asarray(_per_study(psize, x, 'psize'), dtype=float)

    measure = _ARM_MEASURE[x.measure]
    if x.measure in ('RR', 'OR', 'RD', 'AS'):
        a, b, c, dd = (_column(x, n, v) for n, v in (('ai', ai), ('bi', bi), ('ci', ci_counts), ('di', di)))
        first, second = ((a, b), (c, dd)) if flip else ((c, dd), (a, b))
        kw_x: dict[str, Any] = {'xi': first[0], 'mi': first[1]}
        kw_y: dict[str, Any] = {'xi': second[0], 'mi': second[1]}
    else:
        a1, tt1, a2, tt2 = (_column(x, n, v) for n, v in (('x1i', x1i), ('t1i', t1i), ('x2i', x2i), ('t2i', t2i)))
        first, second = ((a1, tt1), (a2, tt2)) if flip else ((a2, tt2), (a1, tt1))
        kw_x = {'xi': first[0], 'ti': first[1]}
        kw_y = {'xi': second[0], 'ti': second[1]}
    opts: dict[str, Any] = {'add': add_, 'to': to, 'addyi': addyi, 'addvi': addvi, 'cutoff': math.nan}
    dat_x = escalc(measure, **kw_x, **opts)
    dat_y = escalc(measure, **kw_y, **opts)
    yx, vx = dat_x['yi'].to_numpy(dtype=float), dat_x['vi'].to_numpy(dtype=float)
    yy, vy = dat_y['yi'].to_numpy(dtype=float), dat_y['vi'].to_numpy(dtype=float)
    has_na = np.isnan(yx) | np.isnan(vx) | np.isnan(yy) | np.isnan(vy)
    not_na = ~has_na
    ids = tuple(int(i) + 1 for i in np.flatnonzero(x.data.keep)[not_na])
    slab = tuple(np.asarray(x.slab, dtype=object)[not_na].tolist())
    pch_v = [p for p, ok in zip(pch_v, not_na, strict=True) if ok]
    col_v = [p for p, ok in zip(col_v, not_na, strict=True) if ok]
    bg_v = [p for p, ok in zip(bg_v, not_na, strict=True) if ok]
    if psize_v is not None:
        psize_v = psize_v[not_na]
    yx, vx, yy, vy = yx[not_na], vx[not_na], yy[not_na], vy[not_na]
    if yx.size == 0 or yy.size == 0:
        raise ValueError('No information in object to compute the arm-level outcomes.')
    vi = vx + vy
    k = int(vi.shape[0])

    if psize_v is None:
        if len(plim) < 2:
            raise ValueError("Argument 'plim' must be of length 2 or 3.")
        wi = np.sqrt(1 / vi)
        p0, p1 = plim[0], plim[1]
        psz = np.full(k, np.nan)
        if not math.isnan(p0) and not math.isnan(p1):
            rng = float(wi.max() - wi.min())
            psz = np.ones(k) if rng <= math.sqrt(np.finfo(float).eps) else (wi - wi.min()) / rng * (p1 - p0) + p0
        if math.isnan(p0) and not math.isnan(p1):
            psz = wi / wi.max() * p1
            if len(plim) == 3:
                psz[psz <= plim[2]] = plim[2]
        if not math.isnan(p0) and math.isnan(p1):
            psz = wi / wi.min() * p0
            if len(plim) == 3:
                psz[psz >= plim[2]] = plim[2]
        if np.all(np.isnan(psz)):
            psz = np.ones(k)
    else:
        psz = psize_v

    min_yi, max_yi = float(min(yx.min(), yy.min())), float(max(yx.max(), yy.max()))
    rng_yi = max_yi - min_yi
    ln = 10000
    intrcpt = float(x.beta[0])
    if llim is None:
        m = x.measure
        if m in ('RD', 'IRD', 'IRSD'):
            x_vals = np.linspace(0 if intrcpt > 0 else -intrcpt, 1 - intrcpt if intrcpt > 0 else 1, ln)
        elif m in ('RR', 'IRR'):
            x_vals = np.linspace(min_yi - rng_yi, -intrcpt if intrcpt > 0 else 0, ln)
        elif m == 'OR':
            x_vals = np.linspace(min_yi - rng_yi, max_yi + rng_yi, ln)
        else:
            x_vals = np.linspace(0 if intrcpt > 0 else -intrcpt, math.asin(1) - intrcpt if intrcpt > 0 else math.asin(1), ln)
    else:
        if len(llim) != 2:
            raise ValueError("Argument 'llim' must be of length 2.")
        x_vals = np.linspace(llim[0], llim[1], ln)
    y_vals = intrcpt + 1 * x_vals
    ci_lb = ci_ub = pi_lb = pi_ub = None
    if show_ci or show_pi:
        pr = predict(x)
        ci_lb, ci_ub = pr.ci_lb[0] + x_vals, pr.ci_ub[0] + x_vals
        pi_lb, pi_ub = pr.pi_lb[0] + x_vals, pr.pi_ub[0] + x_vals

    if transf is not None:
        yx, yy = apply_each(yx, transf, targs), apply_each(yy, transf, targs)
        x_vals, y_vals = apply_each(x_vals, transf, targs), apply_each(y_vals, transf, targs)
        if ci_lb is not None and ci_ub is not None and pi_lb is not None and pi_ub is not None:
            ci_lb, ci_ub = apply_each(ci_lb, transf, targs), apply_each(ci_ub, transf, targs)
            pi_lb, pi_ub = apply_each(pi_lb, transf, targs), apply_each(pi_ub, transf, targs)
    min_yi, max_yi = float(min(yx.min(), yy.min())), float(max(yx.max(), yy.max()))
    if lim is None:
        xl = (min_yi, max_yi) if xlim is None else (float(xlim[0]), float(xlim[1]))
        yl = (min_yi, max_yi) if ylim is None else (float(ylim[0]), float(ylim[1]))
    else:
        xl = yl = (float(lim[0]), float(lim[1]))

    order = np.argsort(-psz, kind='stable')
    if xlab is None:
        xlab = setlab(measure, transf, None, 1) + ' (Group 1)'
    if ylab is None:
        ylab = setlab(measure, transf, None, 1) + ' (Group 2)'
    xt, yt = plain_plot(axx, xl, yl, xlab, ylab)
    if show_pi and pi_lb is not None and pi_ub is not None:
        d.poly(axx, np.concatenate([x_vals, x_vals[::-1]]), np.concatenate([pi_lb, pi_ub[::-1]]), col=pi_col, border=None, zorder=0.6)
    if show_ci and ci_lb is not None and ci_ub is not None:
        d.poly(axx, np.concatenate([x_vals, x_vals[::-1]]), np.concatenate([ci_lb, ci_ub[::-1]]), col=ci_col, border=None, zorder=0.6)
    if show_grid:
        assert xt is not None
        assert yt is not None
        d.grid(axx, xt, yt, grid_col)
    d.seg(axx, float(x_vals.min()), float(x_vals.min()), float(x_vals.max()), float(x_vals.max()), lty=lt[0])
    d.line(axx, x_vals, y_vals, lty=lt[1])
    for i in order:
        d.pts(axx, float(yx[i]), float(yy[i]), pch=int(pch_v[i]), cex=float(psz[i]), col=col_v[i], bg=bg_v[i])
    _legend(axx, legend, intrcpt, lt, show_ci, show_pi, ci_col, pi_col, 100 * (1 - x.level))
    return LabbeResult(
        axx,
        np.asarray(yx, dtype=float),
        np.asarray(yy, dtype=float),
        np.asarray(psz, dtype=float),
        tuple(int(p) for p in pch_v),
        tuple(str(c) for c in col_v),
        tuple(str(c) for c in bg_v),
        ids,
        slab,
    )


def _legend(ax: Axes, legend: bool | str | dict[str, Any], intrcpt: float, lty: list[Any], ci: bool, pi: bool, ci_col: str, pi_col: str, lvl: float) -> None:
    """An approximation of metafor's legend (reference line, line of the estimated effect and the CI/PI regions)."""
    pos = 'lower right' if intrcpt > 0 else 'upper left'
    if isinstance(legend, dict):
        pos = str(legend.get('x', pos))
    elif isinstance(legend, str):
        pos = {'topright': 'upper right', 'topleft': 'upper left', 'bottomright': 'lower right', 'bottomleft': 'lower left'}.get(legend, legend)
    elif not legend:
        return
    handles: list[Any] = []
    labels: list[str] = []
    for name, ls in (('Reference Line of No Effect', lty[0]), ('Line for the Estimated Effect', lty[1])):
        if ls not in ('blank', 0):
            handles.append(Line2D([], [], color=d.FG, linestyle=d.LTY.get(ls, ls)))
            labels.append(name)
    shown = f'{lvl:g}'
    if ci:
        handles.append(Patch(facecolor=ci_col, edgecolor='black'))
        labels.append(f'{shown}% Confidence Interval')
    if pi:
        handles.append(Patch(facecolor=pi_col, edgecolor='black'))
        labels.append(f'{shown}% Prediction Interval')
    if handles:
        ax.legend(handles, labels, loc=cast(Any, pos), framealpha=1.0)
